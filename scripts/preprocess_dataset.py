import os
import sys
import torch
import argparse
from tqdm import tqdm

# Add root to path for imports
sys.path.append(os.getcwd())

from data.data_loader import LeanLibrary
from data.processor import ExprGraphProcessor
from data.symbol_manager import SymbolManager

def preprocess(premises_db, states_db, vocab_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    symbol_manager = SymbolManager(vocab_path=vocab_path)
    # Use same max_nodes as training (e.g., 512)
    processor = ExprGraphProcessor(symbol_to_id=symbol_manager.symbol_to_id, max_nodes=512)
    lib = LeanLibrary(premises_db, states_db)
    
    # 1. Pre-process all premises (~205K)
    print("Pre-processing premises...")
    premise_ids = lib.get_all_premise_ids()
    premise_cache = {}
    for pid in tqdm(premise_ids, desc="Premises"):
        p_json = lib.get_premise_json(pid)
        if p_json and 'graph' in p_json:
            try:
                premise_cache[pid] = processor.process_json_graph(p_json['graph'])
            except Exception as e:
                print(f"Error processing premise {pid}: {e}")
    
    premise_output = os.path.join(output_dir, "premises_processed.pt")
    torch.save(premise_cache, premise_output)
    print(f"Saved {len(premise_cache)} premises to {premise_output}")
    
    # Clear memory
    del premise_cache
    
    # 2. Pre-process all states (~746K)
    # WARNING: 746K states might exceed 16GB RAM if saved in one dict.
    # We will save them in chunks if needed, but let's try one file first.
    print("Pre-processing states...")
    state_ids = lib.get_all_state_ids()
    state_cache = {}
    for sid in tqdm(state_ids, desc="States"):
        entry = lib.get_state_json(sid)
        if entry and 'graph' in entry and entry.get('target_premises'):
            try:
                state_graph = processor.process_json_graph(entry['graph'])
                state_cache[sid] = {
                    'graph': state_graph,
                    'targets': entry['target_premises']
                }
            except Exception as e:
                pass # Skip corrupt states
    
    state_output = os.path.join(output_dir, "states_processed.pt")
    torch.save(state_cache, state_output)
    print(f"Saved {len(state_cache)} states to {state_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--premises_db", default="premises.db")
    parser.add_argument("--states_db", default="states.db")
    parser.add_argument("--vocab_path", default="symbol_vocab.json")
    parser.add_argument("--output_dir", default="preprocessed")
    args = parser.parse_args()
    
    preprocess(args.premises_db, args.states_db, args.vocab_path, args.output_dir)
