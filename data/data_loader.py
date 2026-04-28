import json
import sqlite3
import torch
import random
import os
from typing import List, Dict, Tuple, Any, Optional
from torch_geometric.data import Dataset, Batch
from .processor import ExprGraphProcessor

class LeanLibrary:
    """Manages separate SQLite databases for premises and states."""
    def __init__(self, premises_db: str, states_db: str):
        self.premises_db = premises_db
        self.states_db = states_db
        self.p_conn = None
        self.s_conn = None
        
    def _get_p_conn(self):
        if self.p_conn is None:
            if not os.path.exists(self.premises_db):
                raise FileNotFoundError(f"Premises database not found at {self.premises_db}")
            if "/kaggle/input" in self.premises_db:
                self.p_conn = sqlite3.connect(f"file:{self.premises_db}?mode=ro", uri=True)
            else:
                self.p_conn = sqlite3.connect(self.premises_db)
        return self.p_conn

    def _get_s_conn(self):
        if self.s_conn is None:
            if not os.path.exists(self.states_db):
                raise FileNotFoundError(f"States database not found at {self.states_db}")
            if "/kaggle/input" in self.states_db:
                self.s_conn = sqlite3.connect(f"file:{self.states_db}?mode=ro", uri=True)
            else:
                self.s_conn = sqlite3.connect(self.states_db)
        return self.s_conn
        
    def get_premise_json(self, premise_id: str) -> Optional[Dict]:
        cursor = self._get_p_conn().cursor()
        cursor.execute("SELECT json_data FROM premises WHERE id = ?", (premise_id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def get_state_json(self, state_id: str) -> Optional[Dict]:
        cursor = self._get_s_conn().cursor()
        cursor.execute("SELECT json_data FROM states WHERE id = ?", (state_id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def get_all_premise_ids(self) -> List[str]:
        cursor = self._get_p_conn().cursor()
        cursor.execute("SELECT id FROM premises")
        return [row[0] for row in cursor.fetchall()]

    def get_all_state_ids(self) -> List[str]:
        cursor = self._get_s_conn().cursor()
        cursor.execute("SELECT id FROM states")
        return [row[0] for row in cursor.fetchall()]


class LeanRetrievalDataset(Dataset):
    def __init__(self, 
                 premises_db: str,
                 states_db: str,
                 graph_processor: ExprGraphProcessor,
                 max_positives: int = 10,
                 preprocessed_dir: Optional[str] = None):
        super().__init__()
        self.processor = graph_processor
        self.max_positives = max_positives
        self.use_preprocessed = False
        
        # Check for preprocessed data
        if preprocessed_dir:
            prem_path = os.path.join(preprocessed_dir, "premises_processed.pt")
            state_path = os.path.join(preprocessed_dir, "states_processed.pt")
            if os.path.exists(prem_path) and os.path.exists(state_path):
                print(f"Loading preprocessed data from {preprocessed_dir}...")
                # weights_only=False is needed for PyG objects
                self.premise_cache = torch.load(prem_path, weights_only=False)
                self.state_cache = torch.load(state_path, weights_only=False)
                self.state_ids = list(self.state_cache.keys())
                self.use_preprocessed = True
                print(f"Dataset ready (Preprocessed): {len(self.state_ids)} states, {len(self.premise_cache)} premises.")
            else:
                print(f"Warning: Preprocessed files not found in {preprocessed_dir}. Falling back to SQLite.")
        
        if not self.use_preprocessed:
            self.lib = LeanLibrary(premises_db, states_db)
            self.state_ids = self.lib.get_all_state_ids()
            self.premise_cache = {}
            print(f"Dataset ready (SQLite): {len(self.state_ids)} states.")

    def len(self):
        return len(self.state_ids)

    def get(self, idx):
        state_id = self.state_ids[idx]
        
        if self.use_preprocessed:
            entry = self.state_cache[state_id]
            state_graph = entry['graph']
            all_pos_ids = entry['targets']
        else:
            entry = self.lib.get_state_json(state_id)
            if entry is None or not entry.get('target_premises'):
                return self.get(random.randint(0, len(self.state_ids)-1))
            state_graph = self.processor.process_json_graph(entry['graph'])
            all_pos_ids = entry['target_premises']
            
        if len(all_pos_ids) > self.max_positives:
            pos_ids = random.sample(all_pos_ids, self.max_positives)
        else:
            pos_ids = all_pos_ids
            
        pos_graphs = []
        final_pos_ids = []
        for pid in pos_ids:
            if pid in self.premise_cache:
                pos_graphs.append(self.premise_cache[pid])
                final_pos_ids.append(pid)
            elif not self.use_preprocessed:
                p_json = self.lib.get_premise_json(pid)
                if p_json:
                    p_graph = self.processor.process_json_graph(p_json['graph'])
                    self.premise_cache[pid] = p_graph
                    pos_graphs.append(p_graph)
                    final_pos_ids.append(pid)
        
        if not final_pos_ids:
            return self.get(random.randint(0, len(self.state_ids)-1))
            
        return state_graph, pos_graphs, final_pos_ids


def collate_fn(batch):
    states, pos_graphs_list, pos_ids_list = zip(*batch)
    
    # 1. Deduplicate premises in the batch
    unique_premises_dict = {}
    for pos_ids, pos_graphs in zip(pos_ids_list, pos_graphs_list):
        for pid, pgraph in zip(pos_ids, pos_graphs):
            unique_premises_dict[pid] = pgraph
            
    unique_premise_ids = list(unique_premises_dict.keys())
    id_to_idx = {pid: i for i, pid in enumerate(unique_premise_ids)}
    
    # 2. Build Batches
    batched_states = Batch.from_data_list(states)
    unique_premise_graphs = [unique_premises_dict[pid] for pid in unique_premise_ids]
    batched_premises = Batch.from_data_list(unique_premise_graphs)
    
    # 3. Build Masks
    num_states = len(states)
    num_unique_prems = len(unique_premise_ids)
    
    pos_mask = torch.zeros((num_states, num_unique_prems), dtype=torch.bool)
    cooccur_mask = torch.zeros((num_unique_prems, num_unique_prems), dtype=torch.bool)
    
    for i, pids in enumerate(pos_ids_list):
        p_indices = [id_to_idx[pid] for pid in pids if pid in id_to_idx]
        if not p_indices:
            continue
        pos_mask[i, p_indices] = True
        
        # Vectorized co-occurrence (Much faster than nested loops)
        if len(p_indices) > 1:
            p_t = torch.tensor(p_indices, dtype=torch.long)
            # Create a grid of indices
            grid_i = p_t.view(-1, 1).expand(-1, len(p_t)).reshape(-1)
            grid_j = p_t.view(1, -1).expand(len(p_t), -1).reshape(-1)
            # Remove self-co-occurrence (optional but usually preferred)
            mask = grid_i != grid_j
            cooccur_mask[grid_i[mask], grid_j[mask]] = True
                    
    return batched_states, batched_premises, pos_mask, cooccur_mask
