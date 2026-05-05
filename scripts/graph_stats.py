import sqlite3
import json
import argparse
import numpy as np
from tqdm import tqdm

def analyze_db(db_path, table_name):
    print(f"Analyzing {table_name} in {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Kiểm tra bảng có tồn tại không
        cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cur.fetchone():
            print(f"Table '{table_name}' does not exist in {db_path}.\n")
            return
        
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        total = cur.fetchone()[0]
        
        cur.execute(f"SELECT json_data FROM {table_name}")
        
        lengths = []
        for row in tqdm(cur, total=total, desc=f"Processing {table_name}"):
            json_str = row[0]
            try:
                data = json.loads(json_str)
                # Graph là một list các object, mỗi object là 1 node
                graph = data.get('graph', [])
                lengths.append(len(graph))
            except Exception as e:
                continue
                
        if not lengths:
            print("No valid data found.")
            return
            
        lengths = np.array(lengths)
        
        print("\n" + "="*40)
        print(f"STATISTICS FOR: {table_name.upper()} (Total Records: {len(lengths)})")
        print("="*40)
        print(f"Mean (Trung bình): {np.mean(lengths):.2f}")
        print(f"Median (Trung vị): {np.median(lengths):.2f}")
        print(f"StdDev (Độ lệch chuẩn): {np.std(lengths):.2f}")
        print(f"Min: {np.min(lengths)}")
        print(f"Max: {np.max(lengths)}")
        print(f"75th Percentile: {np.percentile(lengths, 75):.2f}")
        print(f"90th Percentile: {np.percentile(lengths, 90):.2f}")
        print(f"95th Percentile: {np.percentile(lengths, 95):.2f}")
        print(f"99th Percentile: {np.percentile(lengths, 99):.2f}")
        
        # Thêm thống kê số lượng node dưới 100 và dưới 512
        under_100 = np.sum(lengths < 100)
        under_512 = np.sum(lengths < 512)
        print(f"\nNodes < 100: {under_100} ({under_100 / len(lengths) * 100:.2f}%)")
        print(f"Nodes < 512: {under_512} ({under_512 / len(lengths) * 100:.2f}%)")
        print("="*40 + "\n")
        
    except sqlite3.OperationalError as e:
        print(f"Error accessing database: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="Calculate graph size statistics.")
    parser.add_argument("--states_db", default="datatrain/states.db", type=str)
    parser.add_argument("--premises_db", default="datatrain/premises.db", type=str)
    args = parser.parse_args()
    
    analyze_db(args.states_db, "states")
    analyze_db(args.premises_db, "premises")

if __name__ == "__main__":
    main()
