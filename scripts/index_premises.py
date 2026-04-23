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
        CREATE TABLE IF NOT EXISTS premises (
            id TEXT PRIMARY KEY,
            json_data TEXT
        )
    ''')
    conn.commit()
    return conn

def index_premises(conn, graphs_dir):
    files = sorted(glob.glob(os.path.join(graphs_dir, "*_graphs.jsonl")))
    cursor = conn.cursor()
    batch_size = 1000
    current_batch = []
    
    print(f"Indexing {len(files)} files into 'premises' table...")
    for file_path in tqdm(files, desc="Premises"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    if data.get('id'):
                        current_batch.append((data['id'], line.strip()))
                    if len(current_batch) >= batch_size:
                        cursor.executemany("INSERT OR REPLACE INTO premises (id, json_data) VALUES (?, ?)", current_batch)
                        conn.commit()
                        current_batch = []
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    if current_batch:
        cursor.executemany("INSERT OR REPLACE INTO premises (id, json_data) VALUES (?, ?)", current_batch)
        conn.commit()

def main():
    parser = argparse.ArgumentParser(description="Index all premise graphs into SQLite.")
    parser.add_argument("--db_path", type=str, default="premises.db")
    parser.add_argument("--graphs_dir", type=str, default="/media/hahaha/E/Trainingdata/mathlib_graphs")
    args = parser.parse_args()

    conn = setup_database(args.db_path)
    index_premises(conn, args.graphs_dir)
    conn.close()
    print(f"Done! Premises indexed in {args.db_path}")

if __name__ == "__main__":
    main()
