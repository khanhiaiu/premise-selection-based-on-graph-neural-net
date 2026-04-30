#!/bin/bash
set -e

echo "=== 1. Tạo tập test 1000 states ==="
python scripts/filter_subset.py \
    --states_db datatrain/states.db \
    --premises_db datatrain/premises.db \
    --out_states_db datatrain/states_1k.db \
    --out_premises_db datatrain/premises_1k.db \
    --num_states 1000 \
    --max_nodes 100

echo "=== 2. Pre-compute Graphs (8 Workers) ==="
python scripts/precompute_graphs.py \
    --states_db datatrain/states_1k.db \
    --premises_db datatrain/premises_1k.db \
    --vocab_path datatrain/symbol_vocab.json \
    --out_dir datatrain/precomputed_1k \
    --num_workers 8

echo "=== 3. Chạy Training ==="
python train.py \
    --batch_size 64 \
    --use_precomputed \
    --precomputed_dir datatrain/precomputed_1k
