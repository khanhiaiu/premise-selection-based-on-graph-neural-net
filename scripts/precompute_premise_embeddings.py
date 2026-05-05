import os
import sys
import torch
import argparse
from tqdm import tqdm
from torch_geometric.data import Batch

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hgt_model import LeanHGT
from data.symbol_manager import SymbolManager
from utils.retriever import get_full_metadata

def main():
    parser = argparse.ArgumentParser(description="Precompute and save premise embeddings.")
    parser.add_argument("--model_path", type=str, default="checkpoints/hgt_epoch_29_val_loss_1.858.pt")
    parser.add_argument("--vocab_path", type=str, default="datatrain/symbol_vocab.json")
    parser.add_argument("--symbol_embeddings_path", type=str, default="datatrain/symbol_embeddings.pt")
    parser.add_argument("--precomputed_premises_path", type=str, default="datatrain/precomputed_50k/premises_dict.pt")
    parser.add_argument("--output_path", type=str, default="datatrain/precomputed_50k/premise_embeddings.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading Symbol Manager...")
    symbol_manager = SymbolManager(vocab_path=args.vocab_path)
    if os.path.exists(args.symbol_embeddings_path):
        symbol_manager.load_embeddings(args.symbol_embeddings_path)

    print("Initializing Model...")
    model = LeanHGT(
        metadata=get_full_metadata(),
        pretrained_symbol_embeddings=symbol_manager.embeddings if symbol_manager.embeddings is not None else None,
        hidden_channels=512,
        out_channels=512,
        num_heads=8,
        num_layers=4
    ).to(device)

    print(f"Loading checkpoint {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print(f"Loading precomputed premise graphs from {args.precomputed_premises_path}...")
    premises_dict = torch.load(args.precomputed_premises_path, map_location='cpu', weights_only=False)
    premises_list = list(premises_dict.items())
    
    print(f"Total premises to encode: {len(premises_list)}")
    
    all_premise_embs = []
    all_pids = []

    with torch.no_grad():
        for i in tqdm(range(0, len(premises_list), args.batch_size), desc="Computing Embeddings"):
            batch = premises_list[i:i+args.batch_size]
            pids, graphs = zip(*batch)
            batch_graph = Batch.from_data_list(graphs).to(device)
            
            embs = model(batch_graph.x_dict, batch_graph.edge_index_dict)
            all_premise_embs.append(embs.cpu())
            all_pids.extend(pids)

    P_matrix = torch.cat(all_premise_embs, dim=0) # [num_premises, hidden_dim]
    
    print(f"Saving premise matrix {P_matrix.shape} to {args.output_path}...")
    torch.save({
        'pids': all_pids,
        'embeddings': P_matrix
    }, args.output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
