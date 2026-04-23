import os
import json
import sqlite3
import glob
from tqdm import tqdm
import argparse

def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS states (
            id TEXT PRIMARY KEY,
            json_data TEXT
        )
    ''')
    conn.commit()
    return conn

def index_states(conn, dataset_dir):
    files = sorted(glob.glob(os.path.join(dataset_dir, "*_dataset.jsonl")))
    cursor = conn.cursor()
    batch_size = 1000
    current_batch = []
    
    print(f"Indexing {len(files)} files into 'states' table...")
    for file_path in tqdm(files, desc="States"):
        fname = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if not line.strip(): continue
                    data = json.loads(line)
                    original_id = data.get('id', 'unknown')
                    # Generate a truly unique ID to avoid overwriting
                    unique_id = f"{fname}:{line_idx}:{original_id}"
                    
                    current_batch.append((unique_id, line.strip()))
                    
                    if len(current_batch) >= batch_size:
                        cursor.executemany("INSERT OR REPLACE INTO states (id, json_data) VALUES (?, ?)", current_batch)
                        conn.commit()
                        current_batch = []
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    if current_batch:
        cursor.executemany("INSERT OR REPLACE INTO states (id, json_data) VALUES (?, ?)", current_batch)
        conn.commit()

def main():
    parser = argparse.ArgumentParser(description="Index all state-premise pairs into SQLite.")
    parser.add_argument("--db_path", type=str, default="states.db")
    parser.add_argument("--dataset_dir", type=str, default="/media/hahaha/E/Trainingdata/mathlib_dataset_gnn")
    args = parser.parse_args()

    conn = setup_database(args.db_path)
    index_states(conn, args.dataset_dir)
    conn.close()
    print(f"Done! States indexed in {args.db_path}")

if __name__ == "__main__":
    main()
