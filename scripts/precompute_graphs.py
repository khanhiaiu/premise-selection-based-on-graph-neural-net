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

def init_worker(vocab_path, max_nodes):
    global _processor
    symbol_manager = SymbolManager(vocab_path=vocab_path)
    _processor = ExprGraphProcessor(symbol_to_id=symbol_manager.symbol_to_id, max_nodes=max_nodes)

def process_premise_row(row):
    pid, json_str = row
    data = json.loads(json_str)
    # Lọc kích thước node trước khi xử lý
    if len(data.get('graph', [])) >= _processor.max_nodes:
        return pid, None
    try:
        graph = _processor.process_json_graph(data['graph'])
        return pid, graph
    except Exception as e:
        return pid, None

def process_state_row(row):
    sid, json_str = row
    data = json.loads(json_str)
    # Lọc kích thước node trước khi xử lý
    if len(data.get('graph', [])) >= _processor.max_nodes:
        return sid, None, None
    try:
        graph = _processor.process_json_graph(data['graph'])
        return sid, graph, data.get('target_premises', [])
    except Exception as e:
        return sid, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--premises_db", default="datatrain/premises.db", type=str)
    parser.add_argument("--states_db", default="datatrain/states_50k.db", type=str)
    parser.add_argument("--vocab_path", default="datatrain/symbol_vocab.json", type=str)
    parser.add_argument("--out_dir", default="datatrain/precomputed", type=str)
    parser.add_argument("--num_workers", default=cpu_count(), type=int)
    parser.add_argument("--max_nodes", default=512, type=int, help="Max nodes to include")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # # 1. PRECOMPUTE PREMISES
    # print(f"Reading premises from {args.premises_db}...")
    # conn_p = sqlite3.connect(args.premises_db)
    # cur_p = conn_p.cursor()
    
    # cur_p.execute("SELECT count(*) FROM premises")
    # total_rows = cur_p.fetchone()[0]
    
    # cur_p.execute("SELECT id, json_data FROM premises")
    # premise_rows = cur_p.fetchall()
    
    # print(f"Precomputing {len(premise_rows)} premises (max_nodes < {args.max_nodes}) using {args.num_workers} workers...")
    
    # # Tạo database tạm thời để lưu kết quả trung gian
    # temp_db = "temp_premises.db"
    # if os.path.exists(temp_db): os.remove(temp_db)
    # conn_temp = sqlite3.connect(temp_db)
    # cur_temp = conn_temp.cursor()
    # cur_temp.execute("CREATE TABLE IF NOT EXISTS temp_graphs (id TEXT, graph_data BLOB)")
    
    import pickle
    # count = 0
    # with Pool(args.num_workers, initializer=init_worker, initargs=(args.vocab_path, args.max_nodes)) as pool:
    #     for pid, graph in tqdm(pool.imap_unordered(process_premise_row, premise_rows), total=len(premise_rows), desc="Premises"):
    #         if graph is not None:
    #             cur_temp.execute("INSERT INTO temp_graphs VALUES (?, ?)", (pid, pickle.dumps(graph)))
    #             count += 1
    #             if count % 500 == 0:
    #                 conn_temp.commit()
    
    # conn_temp.commit()
    
    # # BƯỚC CUỐI: Chuyển từ DB tạm sang file .pt
    # print(f"\nĐang tập hợp {count} premises vào RAM để lưu file .pt...")
    # premises_dict = {}
    # cur_temp.execute("SELECT id, graph_data FROM temp_graphs")
    # for pid, graph_blob in cur_temp:
    #     premises_dict[pid] = pickle.loads(graph_blob)
    
    # conn_temp.close()
    # if os.path.exists(temp_db): os.remove(temp_db) # Xóa file tạm
    
    # out_p = os.path.join(args.out_dir, "premises_dict_full.pt")
    # print(f"Đang ghi file: {out_p}...")
    # torch.save(premises_dict, out_p)
    
    # print(f"DONE! Đã lưu {len(premises_dict)} premises vào {out_p}")
    # conn_p.close()

    # 2. PRECOMPUTE STATES
    if os.path.exists(args.states_db):
        print(f"\n--- PROCESSING STATES: {args.states_db} ---")
        conn_s = sqlite3.connect(args.states_db)
        cur_s = conn_s.cursor()
        cur_s.execute("SELECT id, json_data FROM states")
        state_rows = cur_s.fetchall()
        
        temp_s_db = "temp_states.db"
        if os.path.exists(temp_s_db): os.remove(temp_s_db)
        conn_temp_s = sqlite3.connect(temp_s_db)
        cur_temp_s = conn_temp_s.cursor()
        cur_temp_s.execute("CREATE TABLE temp_states (sid TEXT, graph_data BLOB, targets BLOB)")
        
        count_s = 0
        with Pool(args.num_workers, initializer=init_worker, initargs=(args.vocab_path, args.max_nodes)) as pool:
            for sid, graph, targets in tqdm(pool.imap_unordered(process_state_row, state_rows), total=len(state_rows), desc="States"):
                if graph is not None:
                    cur_temp_s.execute("INSERT INTO temp_states VALUES (?, ?, ?)", 
                                      (sid, pickle.dumps(graph), pickle.dumps(targets)))
                    count_s += 1
                    if count_s % 500 == 0: conn_temp_s.commit()
        
        conn_temp_s.commit()
        
        # Xuất ra file .pt cuối cùng kèm theo ID
        print(f"Đang tập hợp {count_s} states vào RAM...")
        states_list = []
        cur_temp_s.execute("SELECT sid, graph_data, targets FROM temp_states")
        for sid, g_blob, t_blob in cur_temp_s:
            states_list.append({
                "sid": sid,
                "graph": pickle.loads(g_blob),
                "target_premises": pickle.loads(t_blob)
            })
        
        conn_temp_s.close()
        if os.path.exists(temp_s_db): os.remove(temp_s_db)
        
        out_s = os.path.join(args.out_dir, "states_list_with_ids.pt")
        torch.save(states_list, out_s)
        print(f"DONE! Đã lưu {len(states_list)} states vào {out_s}")
        conn_s.close()

if __name__ == "__main__":
    main()
