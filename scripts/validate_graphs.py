import sys
import os
import torch
from tqdm import tqdm

# Add parent directory to path to import data modules
sys.path.append(os.getcwd())

from data.processor import ExprGraphProcessor
from data.data_loader import LeanLibrary
from data.symbol_manager import SymbolManager

def validate():
    premises_db = "premises.db"
    states_db = "states.db"
    vocab_path = "symbol_vocab.json"
    
    if not os.path.exists(vocab_path):
        print(f"Error: {vocab_path} not found. Run build_vocab.py first.")
        return

    symbol_manager = SymbolManager(vocab_path)
    processor = ExprGraphProcessor(symbol_to_id=symbol_manager.symbol_to_id)
    lib = LeanLibrary(premises_db, states_db)
    
    print("Fetching sample IDs...")
    state_ids = lib.get_all_state_ids()[:500]
    premise_ids = lib.get_all_premise_ids()[:500]
    test_ids = [("state", sid) for sid in state_ids] + [("premise", pid) for pid in premise_ids]
    
    success_count = 0
    errors = []
    
    pbar = tqdm(test_ids, desc="Validating Graphs")
    for type, id in pbar:
        try:
            if type == "state":
                entry = lib.get_state_json(id)
            else:
                entry = lib.get_premise_json(id)
                
            if not entry or 'graph' not in entry:
                continue
                
            data = processor.process_json_graph(entry['graph'])
            
            # Validation Checks
            num_expr = data['expr'].x.size(0)
            num_sym = data['symbol'].x.size(0)
            
            # 1. Connectivity
            if num_expr > 0:
                # Check virtual node edges
                if ('expr', 'to_virtual', 'virtual') not in data.edge_index_dict:
                    errors.append(f"{type} {id}: Missing to_virtual edges")
                else:
                    edge_index = data['expr', 'to_virtual', 'virtual'].edge_index
                    if edge_index.size(1) != num_expr:
                        errors.append(f"{type} {id}: to_virtual edge count mismatch ({edge_index.size(1)} vs {num_expr})")

            # 2. Const to Symbol
            expr_kinds = data['expr'].x[:, :11] # One-hot
            const_indices = (expr_kinds[:, 2] == 1.0).nonzero(as_tuple=True)[0]
            if len(const_indices) > 0:
                if ('expr', 'is_instance_of', 'symbol') not in data.edge_index_dict:
                    errors.append(f"{type} {id}: Has consts but missing is_instance_of edges")
                else:
                    edge_index = data['expr', 'is_instance_of', 'symbol'].edge_index
                    src_nodes = set(edge_index[0].tolist())
                    for c_idx in const_indices.tolist():
                        if c_idx not in src_nodes:
                            errors.append(f"{type} {id}: Const at index {c_idx} missing symbol link")

            success_count += 1
            
        except Exception as e:
            errors.append(f"{type} {id}: Exception {str(e)}")
            pbar.set_postfix({"errors": len(errors)})

    print("\n" + "="*50)
    print(f"Validation Complete: {success_count}/{len(test_ids)} processed successfully.")
    print(f"Total Errors found: {len(errors)}")
    
    if errors:
        print("\nFirst 10 errors:")
        for err in errors[:10]:
            print(f"  - {err}")
    else:
        print("\nAll checks passed for 1000 samples!")
    print("="*50)

if __name__ == "__main__":
    validate()
