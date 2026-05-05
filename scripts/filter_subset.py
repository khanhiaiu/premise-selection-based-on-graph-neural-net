import os
import json
import sqlite3
import argparse
from tqdm import tqdm

def setup_db(db_path, is_premises=False):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if is_premises:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS premises (
                id TEXT PRIMARY KEY,
                json_data TEXT
            )
        ''')
    else:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS states (
                id TEXT PRIMARY KEY,
                json_data TEXT
            )
        ''')
    conn.commit()
    return conn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states_db", default="datatrain/states.db", type=str)
    parser.add_argument("--premises_db", default="datatrain/premises.db", type=str)
    parser.add_argument("--out_states_db", default="datatrain/states_210k.db", type=str)
    parser.add_argument("--out_premises_db", default="datatrain/premises_210k.db", type=str)
    parser.add_argument("--num_states", default=210000, type=int)
    parser.add_argument("--process_all", action="store_true", help="Process all states in DB, ignoring num_states")
    parser.add_argument("--max_nodes", default=100, type=int, help="Max nodes in graph for both state and premise")
    args = parser.parse_args()

    print(f"Opening source databases...")
    conn_s_in = sqlite3.connect(args.states_db)
    conn_p_in = sqlite3.connect(args.premises_db)
    
    cur_s_in = conn_s_in.cursor()
    cur_p_in = conn_p_in.cursor()
    
    print(f"Creating target databases...")
    conn_s_out = setup_db(args.out_states_db, is_premises=False)
    conn_p_out = setup_db(args.out_premises_db, is_premises=True)
    
    cur_s_out = conn_s_out.cursor()
    cur_p_out = conn_p_out.cursor()

    saved_premises = set()
    states_saved = 0
    
    # Get total states for tqdm
    cur_s_in.execute("SELECT COUNT(*) FROM states")
    total_states = cur_s_in.fetchone()[0]
    
    cur_s_in.execute("SELECT id, json_data FROM states")
    
    pbar_total = total_states if args.process_all else args.num_states
    pbar = tqdm(total=pbar_total, desc="Filtering States & Premises")
    
    while True:
        if not args.process_all and states_saved >= args.num_states:
            break
            
        row = cur_s_in.fetchone()
        if row is None:
            if not args.process_all:
                print("Reached end of states.db before hitting target!")
            break
            
        if args.process_all:
            pbar.update(1)
            
        state_id, state_json_str = row
        state_data = json.loads(state_json_str)
        
        # 1. Check target_premises length and deduplicate to avoid IntegrityError
        target_premises = list(set(state_data.get('target_premises', [])))
        if not (1 <= len(target_premises) <= 10):
            continue
            
        # 2. Check state graph size
        if len(state_data.get('graph', [])) >= args.max_nodes:
            continue
            
        # 3. Check all target premises
        all_premises_valid = True
        premises_to_save = []
        
        for pid in target_premises:
            # Check if we already validated and saved this premise
            if pid in saved_premises:
                continue
                
            cur_p_in.execute("SELECT json_data FROM premises WHERE id = ?", (pid,))
            p_row = cur_p_in.fetchone()
            if not p_row:
                all_premises_valid = False
                break
                
            p_json_str = p_row[0]
            p_data = json.loads(p_json_str)
            
            # Check premise graph size
            if len(p_data.get('graph', [])) >= args.max_nodes:
                all_premises_valid = False
                break
                
            premises_to_save.append((pid, p_json_str))
            
        if not all_premises_valid:
            continue
            
        # If we reach here, state and all its new premises are valid
        cur_s_out.execute("INSERT INTO states (id, json_data) VALUES (?, ?)", (state_id, state_json_str))
        
        for pid, p_json_str in premises_to_save:
            cur_p_out.execute("INSERT INTO premises (id, json_data) VALUES (?, ?)", (pid, p_json_str))
            saved_premises.add(pid)
            
        states_saved += 1
        if not args.process_all:
            pbar.update(1)
        
        # Commit every 1000 states to avoid massive RAM usage for transaction
        if states_saved % 1000 == 0:
            conn_s_out.commit()
            conn_p_out.commit()
            
    pbar.close()
    conn_s_out.commit()
    conn_p_out.commit()
    
    print(f"Done! Saved {states_saved} states to {args.out_states_db}")
    print(f"Saved {len(saved_premises)} unique premises to {args.out_premises_db}")

if __name__ == "__main__":
    main()
