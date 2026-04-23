import json
import sqlite3
import torch
import random
import os
import glob
from typing import List, Dict, Tuple, Any, Optional
from torch_geometric.data import Dataset, HeteroData, Batch
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
            # Support read-only mode for Kaggle /kaggle/input
            if "/kaggle/input" in self.premises_db:
                self.p_conn = sqlite3.connect(f"file:{self.premises_db}?mode=ro", uri=True)
            else:
                self.p_conn = sqlite3.connect(self.premises_db)
        return self.p_conn

    def _get_s_conn(self):
        if self.s_conn is None:
            if not os.path.exists(self.states_db):
                raise FileNotFoundError(f"States database not found at {self.states_db}")
            # Support read-only mode for Kaggle /kaggle/input
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
                 max_positives: int = 10):
        super().__init__()
        self.lib = LeanLibrary(premises_db, states_db)
        self.processor = graph_processor
        self.max_positives = max_positives
        
        print("Initialising dataset from databases...")
        self.state_ids = self.lib.get_all_state_ids()
        self.all_premise_ids = self.lib.get_all_premise_ids()
        print(f"Dataset ready: {len(self.state_ids)} states, {len(self.all_premise_ids)} premises.")

    def len(self):
        return len(self.state_ids)

    def get(self, idx):
        state_id = self.state_ids[idx]
        entry = self.lib.get_state_json(state_id)
        if entry is None or not entry.get('target_premises'):
            return self.get(random.randint(0, len(self.state_ids)-1))
            
        all_pos_ids = entry['target_premises']
        # Sample or limit positives
        if len(all_pos_ids) > self.max_positives:
            pos_ids = random.sample(all_pos_ids, self.max_positives)
        else:
            pos_ids = all_pos_ids
            
        state_graph = self.processor.process_json_graph(entry['graph'])
        
        pos_graphs = []
        final_pos_ids = []
        for pid in pos_ids:
            p_json = self.lib.get_premise_json(pid)
            if p_json:
                pos_graphs.append(self.processor.process_json_graph(p_json['graph']))
                final_pos_ids.append(pid)
        
        if not final_pos_ids:
            return self.get(random.randint(0, len(self.state_ids)-1))
            
        return state_graph, pos_graphs, final_pos_ids

def collate_fn(batch):
    # Each item: (state_graph, pos_graphs, pos_ids)
    states, pos_graphs_list, pos_ids_list = zip(*batch)
    
    # 1. Deduplicate premises across the batch
    unique_premises_dict = {} # premise_id -> HeteroData
    for i in range(len(batch)):
        for pid, pgraph in zip(pos_ids_list[i], pos_graphs_list[i]):
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
    
    for i in range(num_states):
        p_indices = [id_to_idx[pid] for pid in pos_ids_list[i] if pid in id_to_idx]
        pos_mask[i, p_indices] = True
        
        # Co-occurrence: all pairs in pos_indices
        for idx_a in p_indices:
            for idx_b in p_indices:
                if idx_a != idx_b:
                    cooccur_mask[idx_a, idx_b] = True
                    
    return batched_states, batched_premises, pos_mask, cooccur_mask
