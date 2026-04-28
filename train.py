import argparse
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.processor import ExprGraphProcessor
from data.symbol_manager import SymbolManager
from data.data_loader import LeanRetrievalDataset, collate_fn
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

def premise_cooccur_loss(premise_embs, cooccur_mask, temperature=0.07):
    """
    Loss to ensure premises used in the same state have similar embeddings.
    """
    if not cooccur_mask.any():
        return torch.tensor(0.0, device=premise_embs.device)
        
    logits = torch.matmul(premise_embs, premise_embs.t()) / temperature
    
    # For each row (premise i), cooccur_mask[i] identifies its positive pairs
    # Ignore diagonal (self-similarity) if possible, but standard InfoNCE includes it 
    # as a "hard negative" if not masked out.
    
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)
    masked_logits = logits.masked_fill(~cooccur_mask, -1e9)
    log_sum_exp_pos = torch.logsumexp(masked_logits, dim=-1)
    
    # Only compute loss for premises that HAVE co-occurring pairs in the batch
    valid_rows = cooccur_mask.any(dim=-1)
    if not valid_rows.any():
        return torch.tensor(0.0, device=premise_embs.device)
        
    return (log_sum_exp_all[valid_rows] - log_sum_exp_pos[valid_rows]).mean()

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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--lambda_cooccur", type=float, default=0.2, help="Weight for premise co-occurrence loss.")
    parser.add_argument("--preprocessed_dir", type=str, default="/kaggle/working/preprocessed", help="Path to preprocessed .pt files.")
    
    args = parser.parse_args()
    
    # Optimize SQLite Access: Copy to /tmp if on /kaggle/input and not preprocessed
    local_premises = args.premises_db
    local_states = args.states_db
    if "/kaggle/input" in args.premises_db and not os.path.exists(os.path.join(args.preprocessed_dir, "premises_processed.pt")):
        print("Copying databases to local /tmp for faster access...")
        local_premises = "/tmp/premises.db"
        local_states = "/tmp/states.db"
        if not os.path.exists(local_premises): shutil.copy(args.premises_db, local_premises)
        if not os.path.exists(local_states): shutil.copy(args.states_db, local_states)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize Managers
    symbol_manager = SymbolManager(vocab_path=args.vocab_path)
    if os.path.exists(args.embeddings_path):
        symbol_manager.load_embeddings(args.embeddings_path)
    else:
        print(f"Warning: {args.embeddings_path} not found. Model will use random initialization for symbols.")
        
    processor = ExprGraphProcessor(symbol_to_id=symbol_manager.symbol_to_id)
    
    # 2. Setup Dataset
    dataset = LeanRetrievalDataset(
        premises_db=local_premises,
        states_db=local_states,
        graph_processor=processor,
        preprocessed_dir=args.preprocessed_dir if os.path.exists(args.preprocessed_dir) else None
    )
    
    # IMPORTANT: If already preprocessed, do NOT use num_workers > 0
    # because each worker will duplicate the RAM cache (~10GB x 4 = OOM)
    if dataset.use_preprocessed:
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None
        print("Using preprocessed data. num_workers set to 0 to avoid RAM duplication.")
    else:
        num_workers = 4
        pin_memory = True
        persistent_workers = True
        prefetch_factor = 2
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    # 3. Setup Model
    metadata = get_full_metadata()
    print(f"Model Metadata initialized with {len(metadata[1])} edge types.")
    
    model = LeanHGT(
        metadata=metadata,
        pretrained_symbol_embeddings=symbol_manager.embeddings if symbol_manager.embeddings is not None else None,
        hidden_channels=args.hidden_dim,
        out_channels=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training Loop
    print(f"Starting MCL training (lambda_cooccur={args.lambda_cooccur})...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_t1 = 0
        total_t3 = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                print(f"DEBUG: Unexpected batch type {type(batch)} with length {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
            states, prems, pos_mask, cooccur_mask = batch
            states = states.to(device)
            prems = prems.to(device)
            pos_mask = pos_mask.to(device)
            cooccur_mask = cooccur_mask.to(device)
            
            optimizer.zero_grad()
            
            state_embs = model(states.x_dict, states.edge_index_dict)
            premise_embs = model(prems.x_dict, prems.edge_index_dict)
            
            l_t1 = multi_positive_infonce_loss(state_embs, premise_embs, pos_mask, temperature=args.temp)
            l_t3 = premise_cooccur_loss(premise_embs, cooccur_mask, temperature=args.temp)
            
            loss = l_t1 + args.lambda_cooccur * l_t3
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_t1 += l_t1.item()
            total_t3 += l_t3.item()
            pbar.set_postfix({
                "L": f"{loss.item():.3f}",
                "T1": f"{l_t1.item():.3f}",
                "T3": f"{l_t3.item():.3f}"
            })
            
        avg_loss = total_loss / len(loader)
        avg_t1 = total_t1 / len(loader)
        avg_t3 = total_t3 / len(loader)
        print(f"Epoch {epoch+1} Complete. Loss: {avg_loss:.4f} (T1: {avg_t1:.4f}, T3: {avg_t3:.4f})")
        scheduler.step()
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/hgt_mcl_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
