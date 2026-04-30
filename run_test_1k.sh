#!/bin/bash
set -e

echo "=== 1. Tạo tập test 1000 states ==="
python scripts/filter_subset.py \
    --states_db datatrain/states.db \
    --premises_db datatrain/premises.db \
    --out_states_db datatrain/states_50k.db \
    --out_premises_db datatrain/premises_50k.db \
    --num_states 50000 \
    --max_nodes 100

echo "=== 2. Pre-compute Graphs (8 Workers) ==="
python scripts/precompute_graphs.py \
    --states_db datatrain/states_50k.db \
    --premises_db datatrain/premises_50k.db \
    --vocab_path datatrain/symbol_vocab.json \
    --out_dir datatrain/precomputed_50k \
    --num_workers 8
