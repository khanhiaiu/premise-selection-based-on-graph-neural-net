import os
import sys
import torch
import faiss
import argparse
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

# Thêm đường dẫn để import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.hgt_model import LeanHGT
from scripts.evaluate import PremiseDataset, StateDataset, collate_premises, collate_states, get_full_metadata
from data.symbol_manager import SymbolManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/leanhgt.pt")
    parser.add_argument("--vocab_path", type=str, default="datatrain/symbol_vocab.json")
    parser.add_argument("--embeddings_path", type=str, default="datatrain/symbol_embeddings.pt")
    parser.add_argument("--precomputed_premises_path", type=str, default="datatrain/precomputed/premises_dict_full.pt")
    parser.add_argument("--precomputed_states_path", type=str, default="datatrain/precomputed/states_list_with_ids.pt")
    parser.add_argument("--premise_embeddings_path", type=str, default="premise embedded/premise_embeddings.pt")
    parser.add_argument("--output_bench", type=str, default="datatrain/precomputed/benchmark_bench.pt")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_bench)), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load data
    print("Loading precomputed data...")
    states_list = torch.load(args.precomputed_states_path, weights_only=False)
    premise_data = torch.load(args.premise_embeddings_path, weights_only=False)
    P_matrix = premise_data['embeddings'].to(device)
    all_pids = premise_data['pids']
    pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}

    # 2. Load model
    symbol_manager = SymbolManager(vocab_path=args.vocab_path)
    if os.path.exists(args.embeddings_path): symbol_manager.load_embeddings(args.embeddings_path)
    
    model = LeanHGT(
        metadata=get_full_metadata(),
        pretrained_symbol_embeddings=symbol_manager.embeddings,
        hidden_channels=512, out_channels=512, num_heads=8, num_layers=4
    ).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    # 3. Setup FAISS
    print("Initializing FAISS...")
    d = P_matrix.shape[1]
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(d))
    index.add(P_matrix.cpu().numpy())

    # 4. Evaluate and Categorize
    state_dataset = StateDataset(states_list)
    state_loader = DataLoader(state_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_states)

    top1_correct = []
    top10_correct_only = [] # Đúng ở top 10 nhưng sai ở top 1
    incorrect_states = []    # Sai hoàn toàn ở top 10

    print("Evaluating 50k states to find matches for tiers...")
    state_idx = 0
    with torch.no_grad():
        for graphs, target_pids_list in tqdm(state_loader, desc="Scanning"):
            graphs = graphs.to(device)
            state_embs = model(graphs.x_dict, graphs.edge_index_dict)
            state_embs_np = state_embs.cpu().numpy()
            D, I = index.search(state_embs_np, k=10)

            for i in range(len(target_pids_list)):
                targets = target_pids_list[i]
                target_indices = set([pid_to_idx[p] for p in targets if p in pid_to_idx])
                
                if not target_indices:
                    state_idx += 1
                    continue

                top_indices = I[i]
                is_top1 = top_indices[0] in target_indices
                is_top10 = any(idx in target_indices for idx in top_indices)

                original_item = states_list[state_idx]
                if is_top1:
                    top1_correct.append(original_item)
                elif is_top10:
                    top10_correct_only.append(original_item)
                else:
                    incorrect_states.append(original_item)
                
                state_idx += 1

    print(f"\nStats found:")
    print(f"- K=1 correct: {len(top1_correct)}")
    print(f"- K=10 only correct: {len(top10_correct_only)}")
    print(f"- Incorrect at K=10: {len(incorrect_states)}")

    # 5. Sample to meet requirements (Total 2000)
    # Target: 294 (K=1) + 582 (K=10 only) + 1124 (Random) = 2000
    if len(top1_correct) < 294 or len(top10_correct_only) < 582:
        print("WARNING: Not enough states in correct categories. Taking maximum available.")
        n_top1 = min(len(top1_correct), 294)
        n_top10 = min(len(top10_correct_only), 582)
    else:
        n_top1 = 294
        n_top10 = 582

    bench_top1 = random.sample(top1_correct, n_top1)
    bench_top10 = random.sample(top10_correct_only, n_top10)
    
    # Tạo pool chứa tất cả các mẫu chưa được chọn (kể cả đúng hay sai)
    used_ids = set([id(item) for item in bench_top1 + bench_top10])
    remaining_pool = [item for item in states_list if id(item) not in used_ids]
    
    n_random = min(len(remaining_pool), 1124)
    bench_random = random.sample(remaining_pool, n_random)
    
    final_bench = bench_top1 + bench_top10 + bench_random
    random.shuffle(final_bench)

    # 6. Save
    print(f"\nFinal Bench Composition:")
    print(f"- Top-1 Correct: {len(bench_top1)}")
    print(f"- Top-10 Correct (Only): {len(bench_top10)}")
    print(f"- Random Samples: {len(bench_random)}")
    print(f"- TOTAL: {len(final_bench)}")
    
    print(f"\nSaving to {args.output_bench}...")
    torch.save(final_bench, args.output_bench)
    print("Success!")

if __name__ == "__main__":
    main()
