import os
import sqlite3
import json
import torch
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys

# Thêm parent directory vào path để import được data.processor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.processor import ExprGraphProcessor
from data.symbol_manager import SymbolManager

# Khởi tạo Global variables cho multiprocessing workers
_processor = None

def init_worker(vocab_path):
    global _processor
    symbol_manager = SymbolManager(vocab_path=vocab_path)
    _processor = ExprGraphProcessor(symbol_to_id=symbol_manager.symbol_to_id, max_nodes=512)

def process_premise_row(row):
    pid, json_str = row
    data = json.loads(json_str)
    try:
        graph = _processor.process_json_graph(data['graph'])
        return pid, graph
    except Exception as e:
        return pid, None

def process_state_row(row):
    sid, json_str = row
    data = json.loads(json_str)
    try:
        graph = _processor.process_json_graph(data['graph'])
        return graph, data.get('target_premises', [])
    except Exception as e:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states_db", default="states_50k.db", type=str)
    parser.add_argument("--premises_db", default="premises_50k.db", type=str)
    parser.add_argument("--vocab_path", default="symbol_vocab.json", type=str)
    parser.add_argument("--out_dir", default="precomputed", type=str)
    parser.add_argument("--num_workers", default=cpu_count(), type=int)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. PRECOMPUTE PREMISES
    print("Reading premises.db...")
    conn_p = sqlite3.connect(args.premises_db)
    cur_p = conn_p.cursor()
    cur_p.execute("SELECT id, json_data FROM premises")
    premise_rows = cur_p.fetchall()
    
    print(f"Precomputing {len(premise_rows)} premises using {args.num_workers} workers...")
    premises_dict = {}
    with Pool(args.num_workers, initializer=init_worker, initargs=(args.vocab_path,)) as pool:
        for pid, graph in tqdm(pool.imap_unordered(process_premise_row, premise_rows), total=len(premise_rows), desc="Premises"):
            if graph is not None:
                premises_dict[pid] = graph

    out_p = os.path.join(args.out_dir, "premises_dict.pt")
    torch.save(premises_dict, out_p)
    print(f"Saved premises dict to {out_p} ({os.path.getsize(out_p) / 1024 / 1024:.2f} MB)")

    # 2. PRECOMPUTE STATES
    print("\nReading states.db...")
    conn_s = sqlite3.connect(args.states_db)
    cur_s = conn_s.cursor()
    cur_s.execute("SELECT id, json_data FROM states")
    state_rows = cur_s.fetchall()

    print(f"Precomputing {len(state_rows)} states using {args.num_workers} workers...")
    states_list = []
    with Pool(args.num_workers, initializer=init_worker, initargs=(args.vocab_path,)) as pool:
        for graph, target_premises in tqdm(pool.imap_unordered(process_state_row, state_rows), total=len(state_rows), desc="States"):
            if graph is not None:
                states_list.append((graph, target_premises))

    out_s = os.path.join(args.out_dir, "states_list.pt")
    torch.save(states_list, out_s)
    print(f"Saved states list to {out_s} ({os.path.getsize(out_s) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()
