import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch_geometric.data import Batch

# Fix sys path to import models and data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.hgt_model import LeanHGT
from data.symbol_manager import SymbolManager

# Need get_full_metadata from train
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

class PremiseDataset(Dataset):
    def __init__(self, premises_dict):
        self.premises = list(premises_dict.items())
        
    def __len__(self):
        return len(self.premises)
        
    def __getitem__(self, idx):
        return self.premises[idx]

def collate_premises(batch):
    pids, graphs = zip(*batch)
    return list(pids), Batch.from_data_list(graphs)

class StateDataset(Dataset):
    def __init__(self, states_list):
        self.states = states_list
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx]

def collate_states(batch):
    graphs, target_pids_list = zip(*batch)
    return Batch.from_data_list(graphs), list(target_pids_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--precomputed_val_path", type=str, default="datatrain/precomputed_50k/states_list_val.pt")
    parser.add_argument("--precomputed_premises_path", type=str, default="datatrain/precomputed_50k/premises_dict.pt")
    parser.add_argument("--vocab_path", type=str, default="datatrain/symbol_vocab.json")
    parser.add_argument("--embeddings_path", type=str, default="datatrain/symbol_embeddings.pt")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--k", type=int, default=5, help="Top K for recall")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load symbol manager
    symbol_manager = SymbolManager(vocab_path=args.vocab_path)
    if os.path.exists(args.embeddings_path):
        symbol_manager.load_embeddings(args.embeddings_path)
        
    # Setup model
    model = LeanHGT(
        metadata=get_full_metadata(),
        pretrained_symbol_embeddings=symbol_manager.embeddings if symbol_manager.embeddings is not None else None,
        hidden_channels=args.hidden_dim,
        out_channels=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)
    
    # Load weights
    print(f"Loading checkpoint {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load Precomputed Data
    print("Loading precomputed data...")
    premises_dict = torch.load(args.precomputed_premises_path, weights_only=False)
    states_list = torch.load(args.precomputed_val_path, weights_only=False)
    
    premise_dataset = PremiseDataset(premises_dict)
    state_dataset = StateDataset(states_list)
    
    premise_loader = DataLoader(premise_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_premises, num_workers=4)
    state_loader = DataLoader(state_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_states, num_workers=4)
    
    # 1. Compute all premise embeddings
    print("Computing premise embeddings...")
    all_premise_embs = []
    all_pids = []
    
    with torch.no_grad():
        for pids, graphs in tqdm(premise_loader, desc="Premises"):
            graphs = graphs.to(device)
            embs = model(graphs.x_dict, graphs.edge_index_dict)
            all_premise_embs.append(embs.cpu())
            all_pids.extend(pids)
            
    P_matrix = torch.cat(all_premise_embs, dim=0).to(device) # [num_premises, hidden_dim]
    print(f"Premise matrix shape: {P_matrix.shape}")
    
    # 2. Compute state embeddings and evaluate
    print("Evaluating states...")
    recall_at_1 = 0
    recall_at_k = 0
    mrr = 0.0
    total_states = 0
    
    with torch.no_grad():
        for graphs, target_pids_list in tqdm(state_loader, desc="States"):
            graphs = graphs.to(device)
            state_embs = model(graphs.x_dict, graphs.edge_index_dict) # [batch_size, hidden_dim]
            
            # Compute similarity: state_embs @ P_matrix.T
            # Both are normalized, so dot product == cosine similarity
            sim_scores = torch.matmul(state_embs, P_matrix.t()) # [batch_size, num_premises]
            
            # Get top K indices
            # For MRR we need ranks up to num_premises
            sorted_indices = torch.argsort(sim_scores, dim=-1, descending=True) # [batch_size, num_premises]
            
            batch_size = len(target_pids_list)
            for i in range(batch_size):
                targets = set(target_pids_list[i])
                if len(targets) == 0:
                    continue
                    
                total_states += 1
                
                # Check top 1
                top1_pid = all_pids[sorted_indices[i, 0].item()]
                if top1_pid in targets:
                    recall_at_1 += 1
                    
                # Check top K
                topk_pids = [all_pids[idx.item()] for idx in sorted_indices[i, :args.k]]
                if any(pid in targets for pid in topk_pids):
                    recall_at_k += 1
                    
                # Calculate MRR
                for rank, idx in enumerate(sorted_indices[i]):
                    pid = all_pids[idx.item()]
                    if pid in targets:
                        mrr += 1.0 / (rank + 1)
                        break
                        
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Total Evaluated States: {total_states}")
    print(f"Recall@1: {recall_at_1 / total_states * 100:.2f}%")
    print(f"Recall@{args.k}: {recall_at_k / total_states * 100:.2f}%")
    print(f"MRR:      {mrr / total_states:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
