import os
import json
import sqlite3
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Build symbol vocabulary from the indexed premise database.")
    parser.add_argument("--db_path", type=str, default="premises.db", help="Path to the premises SQLite database.")
    parser.add_argument("--output_path", type=str, default="symbol_vocab.json", help="Path to save the vocabulary JSON.")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"Error: Database {args.db_path} not found.")
        return

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()
    
    print("Reading premise graphs from database...")
    cursor.execute("SELECT json_data FROM premises")
    
    symbol_to_id = {"<UNK>": 0}
    
    # We use a set to avoid duplicates and then sort
    all_symbols = set()
    
    # Fetch all rows
    rows = cursor.fetchall()
    print(f"Found {len(rows)} premises. Extracting symbols...")
    
    for (json_str,) in tqdm(rows):
        data = json.loads(json_str)
        graph = data.get('graph', [])
        for node in graph:
            if node.get('kind') == 'const' and 'name' in node:
                all_symbols.add(node['name'])
                
    # Sort symbols for deterministic IDs
    sorted_symbols = sorted(list(all_symbols))
    for i, sym in enumerate(sorted_symbols):
        symbol_to_id[sym] = i + 1
        
    print(f"Extraction complete. Total unique symbols: {len(symbol_to_id)}")
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(symbol_to_id, f, indent=2)
    print(f"Vocabulary saved to {args.output_path}")

if __name__ == "__main__":
    main()
