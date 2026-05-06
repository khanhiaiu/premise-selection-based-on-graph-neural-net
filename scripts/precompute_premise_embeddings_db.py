import os
import sys
import torch
import sqlite3
import pickle
import argparse
from tqdm import tqdm
from torch_geometric.data import Batch

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hgt_model import LeanHGT
from data.symbol_manager import SymbolManager
from utils.retriever import get_full_metadata

def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings from a SQLite database of graphs.")
    parser.add_argument("--model_path", type=str, default="checkpoints/leanhgt.pt")
    parser.add_argument("--vocab_path", type=str, default="datatrain/symbol_vocab.json")
    parser.add_argument("--symbol_embeddings_path", type=str, default="datatrain/symbol_embeddings.pt")
    parser.add_argument("--input_db_path", type=str, default="datatrain/precomputed/premises_precomputed.db")
    parser.add_argument("--output_path", type=str, default="premise embedded/premise_embeddings.pt")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load symbol manager
    symbol_manager = SymbolManager(vocab_path=args.vocab_path)
    if os.path.exists(args.symbol_embeddings_path):
        symbol_manager.load_embeddings(args.symbol_embeddings_path)

    # Initialize Model
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

    # Connect to Input DB
    if not os.path.exists(args.input_db_path):
        print(f"Error: DB not found at {args.input_db_path}")
        return

    conn = sqlite3.connect(args.input_db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT count(*) FROM premises")
    total_count = cur.fetchone()[0]
    print(f"Total premises to encode: {total_count}")

    all_premise_embs = []
    all_pids = []

    # Process in batches directly from DB
    cur.execute("SELECT id, graph_data FROM premises")
    
    batch_graphs = []
    batch_pids = []
    
    pbar = tqdm(total=total_count, desc="Embedding generation")
    
    while True:
        rows = cur.fetchmany(args.batch_size)
        if not rows:
            break
            
        for pid, graph_blob in rows:
            graph = pickle.loads(graph_blob)
            batch_graphs.append(graph)
            batch_pids.append(pid)
            
        # Run inference
        with torch.no_grad():
            batch_data = Batch.from_data_list(batch_graphs).to(device)
            embs = model(batch_data.x_dict, batch_data.edge_index_dict)
            all_premise_embs.append(embs.cpu())
            all_pids.extend(batch_pids)
            
        # Reset batches
        batch_graphs = []
        batch_pids = []
        pbar.update(len(rows))

    pbar.close()
    conn.close()

    # Concatenate and save
    P_matrix = torch.cat(all_premise_embs, dim=0)
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    print(f"Saving final embeddings {P_matrix.shape} to {args.output_path}...")
    torch.save({
        'pids': all_pids,
        'embeddings': P_matrix
    }, args.output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
