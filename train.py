import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.processor import ExprGraphProcessor
from data.symbol_manager import SymbolManager
from data.data_loader import LeanRetrievalDataset, PrecomputedLeanDataset, collate_fn
from models.hgt_model import LeanHGT

def get_full_metadata():
    node_types = ['expr', 'symbol', 'virtual']
    edge_types = []
    
    # AST
    ast_base = ['has_fn', 'has_arg', 'has_type', 'has_body', 'has_value', 'has_expr']
    for rel in ast_base:
        edge_types.append(('expr', rel, 'expr'))
        edge_types.append(('expr', f'rev_{rel}', 'expr'))
        
    # Symbol
    edge_types.append(('expr', 'is_instance_of', 'symbol'))
    edge_types.append(('symbol', 'rev_is_instance_of', 'expr'))
    
    # Global
    edge_types.append(('expr', 'to_virtual', 'virtual'))
    edge_types.append(('virtual', 'from_virtual', 'expr'))
    edge_types.append(('symbol', 'sym_to_virtual', 'virtual'))
    edge_types.append(('virtual', 'sym_from_virtual', 'symbol'))
    
    return node_types, edge_types

def multi_positive_infonce_loss(state_embs, premise_embs, positive_mask, temperature=0.07):
    """
    Adapted InfoNCE for multiple positives per state.
    L = log(sum(exp(all_sim))) - log(sum(exp(pos_sim)))
    """
    # [num_states, num_premises]
    logits = torch.matmul(state_embs, premise_embs.t()) / temperature
    
    # Denominator: logsumexp over all premises
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)
    
    # Numerator: logsumexp over positive premises only
    # Fill negatives with a very small value to effectively ignore them in logsumexp
    masked_logits = logits.masked_fill(~positive_mask, -1e9)
    log_sum_exp_pos = torch.logsumexp(masked_logits, dim=-1)
    
    return (log_sum_exp_all - log_sum_exp_pos).mean()



def main():
    parser = argparse.ArgumentParser(description="Train HGT for Lean 4 Premise Retrieval with Multi-Contrastive Loss.")
    # Paths
    parser.add_argument("--premises_db", type=str, default="premises.db", help="Path to the premises SQLite database.")
    parser.add_argument("--states_db", type=str, default="states.db", help="Path to the states SQLite database.")
    parser.add_argument("--vocab_path", type=str, default="symbol_vocab.json", help="Path to symbol vocab JSON.")
    parser.add_argument("--embeddings_path", type=str, default="symbol_embeddings.pt", help="Path to precomputed symbol embeddings.")
    
    # Model Hparams
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    
    # Training Hparams
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20, help="Epoch to train until in this session.")
    parser.add_argument("--max_epochs", type=int, default=20, help="Total epochs for the cosine decay schedule.")
    parser.add_argument("--temp", type=float, default=0.07)
    # Precomputation Flags
    parser.add_argument("--use_precomputed", action="store_true", help="Load dataset directly from RAM via precomputed tensors.")
    parser.add_argument("--precomputed_train_path", type=str, default="precomputed/states_list_train.pt")
    parser.add_argument("--precomputed_val_path", type=str, default="precomputed/states_list_val.pt")
    parser.add_argument("--precomputed_premises_path", type=str, default="precomputed/premises_dict.pt")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # DDP Initialization
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0
        world_size = 1

    if rank == 0:
        print(f"Using device: {device}, distributed: {is_distributed}")

    # 1. Initialize Managers
    symbol_manager = SymbolManager(vocab_path=args.vocab_path)
    if os.path.exists(args.embeddings_path):
        symbol_manager.load_embeddings(args.embeddings_path)
    else:
        if rank == 0:
            print(f"Warning: {args.embeddings_path} not found. Model will use random initialization for symbols.")
        
    processor = ExprGraphProcessor(symbol_to_id=symbol_manager.symbol_to_id)
    
    # 2. Setup Dataset
    if args.use_precomputed:
        train_dataset = PrecomputedLeanDataset(
            states_list_path=args.precomputed_train_path,
            premises_dict_path=args.precomputed_premises_path
        )
        val_dataset = PrecomputedLeanDataset(
            states_list_path=args.precomputed_val_path,
            premises_dict_path=args.precomputed_premises_path
        )
    else:
        # Fallback to sqlite (Note: this is just a placeholder, real split is not implemented for sqlite here)
        train_dataset = LeanRetrievalDataset(
            premises_db=args.premises_db,
            states_db=args.states_db,
            graph_processor=processor
        )
        val_dataset = None
    
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,            # Tận dụng 4 CPU cores của Kaggle
        pin_memory=True,          # Tăng tốc độ copy từ RAM -> GPU VRAM
        prefetch_factor=2         # Chuẩn bị sẵn 2 batch trong lúc GPU tính toán
    )
    
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    else:
        val_loader = None

    # 3. Setup Model
    metadata = get_full_metadata()
    if rank == 0:
        print(f"Model Metadata initialized with {len(metadata[1])} edge types.")
    
    model = LeanHGT(
        metadata=metadata,
        pretrained_symbol_embeddings=symbol_manager.embeddings if symbol_manager.embeddings is not None else None,
        hidden_channels=args.hidden_dim,
        out_channels=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)

    # 4. Resume from Checkpoint
    start_epoch = 0
    checkpoint_data = None
    if args.resume_from is not None:
        if os.path.exists(args.resume_from):
            if rank == 0:
                print(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint_data = torch.load(args.resume_from, map_location=device)
            # Support both old format (state_dict directly) and new format (dict with 'model_state_dict')
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['model_state_dict']
                # Remove 'module.' prefix if it exists to allow loading into base model
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                start_epoch = checkpoint_data.get('epoch', 0)
            else:
                state_dict = {k.replace('module.', ''): v for k, v in checkpoint_data.items()}
                model.load_state_dict(state_dict)
        else:
            if rank == 0:
                print(f"Warning: Checkpoint {args.resume_from} not found. Starting from scratch.")
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # T_max is the absolute end of the learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    # Load Optimizer and Scheduler states if they exist
    if checkpoint_data and isinstance(checkpoint_data, dict):
        if 'optimizer_state_dict' in checkpoint_data:
            try:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                if rank == 0:
                    print("Successfully loaded optimizer state.")
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Could not load optimizer state: {e}")
        
        if 'scheduler_state_dict' in checkpoint_data:
            try:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                if rank == 0:
                    print("Successfully loaded scheduler state.")
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Could not load scheduler state: {e}")

    # 5. Training Loop
    if start_epoch >= args.epochs:
        if rank == 0:
            print(f"Already reached target session epoch {args.epochs}. Nothing to do.")
        if is_distributed:
            dist.destroy_process_group()
        return

    if rank == 0:
        print(f"Starting session: Epoch {start_epoch+1} -> {args.epochs} (Full schedule: {args.max_epochs})")
        
    for epoch in range(start_epoch, args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0
        
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        else:
            pbar = train_loader
        
        for batch in pbar:
            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                continue
            states, prems, pos_mask = batch
            states = states.to(device)
            prems = prems.to(device)
            pos_mask = pos_mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            state_embs = model(states.x_dict, states.edge_index_dict)
            premise_embs = model(prems.x_dict, prems.edge_index_dict)
            
            loss = multi_positive_infonce_loss(state_embs, premise_embs, pos_mask, temperature=args.temp)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if rank == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.3f}"})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Loop
        avg_val_loss = 0.0
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                if rank == 0:
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]")
                else:
                    val_pbar = val_loader
                    
                for batch in val_pbar:
                    if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                        continue
                    states, prems, pos_mask = batch
                    states = states.to(device)
                    prems = prems.to(device)
                    pos_mask = pos_mask.to(device)
                    
                    state_embs = model(states.x_dict, states.edge_index_dict)
                    premise_embs = model(prems.x_dict, prems.edge_index_dict)
                    
                    val_loss = multi_positive_infonce_loss(state_embs, premise_embs, pos_mask, temperature=args.temp)
                    total_val_loss += val_loss.item()
                    
                    if rank == 0:
                        val_pbar.set_postfix({"Val Loss": f"{val_loss.item():.3f}"})
            
            # Aggregate validation loss across GPUs
            val_loss_tensor = torch.tensor([total_val_loss], device=device)
            if is_distributed:
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            
            avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)
            
            if rank == 0:
                print(f"Epoch {epoch+1} Complete. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        else:
            if rank == 0:
                print(f"Epoch {epoch+1} Complete. Train Loss: {avg_train_loss:.4f}")
            
        scheduler.step()
        
        if rank == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/hgt_epoch_{epoch+1}_val_loss_{avg_val_loss:.3f}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss
            }, checkpoint_path)

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
