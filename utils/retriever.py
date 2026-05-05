import os
import sys
import torch
from torch_geometric.data import Batch

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.hgt_model import LeanHGT
from data.symbol_manager import SymbolManager

def get_full_metadata():
    node_types = ['expr', 'symbol', 'virtual']
    edge_types = []
    ast_base = ['has_fn', 'has_arg', 'has_type', 'has_body', 'has_value', 'has_expr']
    for rel in ast_base:
        edge_types.append(('expr', rel, 'expr'))
        edge_types.append(('expr', f'rev_{rel}', 'expr'))
    edge_types.append(('expr', 'is_instance_of', 'symbol'))
    edge_types.append(('symbol', 'rev_is_instance_of', 'expr'))
    edge_types.append(('expr', 'to_virtual', 'virtual'))
    edge_types.append(('virtual', 'from_virtual', 'expr'))
    edge_types.append(('symbol', 'sym_to_virtual', 'virtual'))
    edge_types.append(('virtual', 'sym_from_virtual', 'symbol'))
    return node_types, edge_types

class LeanRetriever:
    def __init__(self, 
                 model_path="checkpoints/hgt_epoch_29_val_loss_1.858.pt",
                 vocab_path="datatrain/symbol_vocab.json",
                 embeddings_path="datatrain/symbol_embeddings.pt",
                 precomputed_premises_path="datatrain/precomputed_50k/premises_dict.pt",
                 premise_embeddings_path="datatrain/precomputed_50k/premise_embeddings.pt",
                 device="cpu",
                 max_premises=None):
        self.device = torch.device(device)
        print(f"Loading Retriever on {self.device}...")
        
        # Load symbol manager
        self.symbol_manager = SymbolManager(vocab_path=vocab_path)
        if os.path.exists(embeddings_path):
            self.symbol_manager.load_embeddings(embeddings_path)
            
        # Setup model
        self.model = LeanHGT(
            metadata=get_full_metadata(),
            pretrained_symbol_embeddings=self.symbol_manager.embeddings if self.symbol_manager.embeddings is not None else None,
            hidden_channels=512,  # Default from evaluate.py
            out_channels=512,
            num_heads=8,
            num_layers=4
        ).to(self.device)
        
        # Load weights
        print(f"Loading checkpoint {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Try to load precomputed embeddings
        if os.path.exists(premise_embeddings_path):
            print(f"Loading precomputed premise matrix from {premise_embeddings_path}...")
            data = torch.load(premise_embeddings_path, map_location='cpu', weights_only=False)
            self.all_pids = data['pids']
            self.P_matrix = data['embeddings'].to(self.device)
            if max_premises is not None:
                self.all_pids = self.all_pids[:max_premises]
                self.P_matrix = self.P_matrix[:max_premises]
            print(f"Retriever initialized. Premise matrix shape: {self.P_matrix.shape}")
            return

        # Fallback: compute from graphs
        print("Loading precomputed premises graphs...")
        # Since this could be large, make sure to load to CPU first
        premises_dict = torch.load(precomputed_premises_path, map_location='cpu', weights_only=False)
        
        print("Computing premise matrix...")
        all_premise_embs = []
        self.all_pids = []
        
        # Process premises in chunks to avoid memory issues
        # Convert dictionary to list
        premises_list = list(premises_dict.items())
        if max_premises is not None:
            premises_list = premises_list[:max_premises]
        
        batch_size = 128
        
        with torch.no_grad():
            for i in range(0, len(premises_list), batch_size):
                batch = premises_list[i:i+batch_size]
                pids, graphs = zip(*batch)
                batch_graph = Batch.from_data_list(graphs).to(self.device)
                
                embs = self.model(batch_graph.x_dict, batch_graph.edge_index_dict)
                all_premise_embs.append(embs.cpu())
                self.all_pids.extend(pids)
                
        self.P_matrix = torch.cat(all_premise_embs, dim=0).to(self.device) # [num_premises, hidden_dim]
        print(f"Retriever initialized. Premise matrix shape: {self.P_matrix.shape}")

    def retrieve(self, state_heterodata, top_k=5):
        """
        Retrieves the top_k premises for a given state graph.
        """
        # Convert single graph to batch
        batch_graph = Batch.from_data_list([state_heterodata]).to(self.device)
        
        with torch.no_grad():
            state_embs = self.model(batch_graph.x_dict, batch_graph.edge_index_dict) # [1, hidden_dim]
            
            # Compute cosine similarity
            sim_scores = torch.matmul(state_embs, self.P_matrix.t()) # [1, num_premises]
            
            # Get top K
            scores, indices = torch.topk(sim_scores[0], k=top_k)
            
        results = []
        for i in range(top_k):
            idx = indices[i].item()
            score = scores[i].item()
            pid = self.all_pids[idx]
            results.append((pid, score))
            
        return results
