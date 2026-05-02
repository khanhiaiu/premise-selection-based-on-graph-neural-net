import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from typing import Dict, List, Tuple

class LeanHGT(nn.Module):
    """
    Heterogeneous Graph Transformer for Lean 4 Premise Retrieval.
    Architecture:
    - 4 HGT Layers
    - 8 Attention Heads
    - 768-dim Hidden State
    - Virtual Node Readout
    """
    def __init__(self, 
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 expr_in_channels: int = 47,
                 symbol_in_channels: int = 768,
                 hidden_channels: int = 768,
                 out_channels: int = 768,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 pretrained_symbol_embeddings: torch.Tensor = None):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # 1. Node Encoders
        # Expr nodes: raw features -> hidden
        self.expr_encoder = nn.Sequential(
            Linear(expr_in_channels, hidden_channels),
            nn.GELU(),
            Linear(hidden_channels, hidden_channels)
        )
        
        # Symbol nodes: Pretrained 768-dim -> 512-dim
        if pretrained_symbol_embeddings is not None:
            # We freeze or keep them learnable? The user said "chiếu từ 768 xuống 512". 
            # I'll use a Linear layer on top of provided embeddings.
            self.symbol_emb_base = nn.Embedding.from_pretrained(pretrained_symbol_embeddings, freeze=True)
            self.symbol_projector = nn.Sequential(
                Linear(symbol_in_channels, hidden_channels),
                nn.GELU()
            )
        else:
            # Fallback if no precomputed embeddings provided
            self.symbol_emb_base = nn.Embedding(1000, symbol_in_channels) # Dummy
            self.symbol_projector = nn.Sequential(
                Linear(symbol_in_channels, hidden_channels),
                nn.GELU()
            )
            
        # Virtual node: 1 learnable vector
        self.virtual_emb = nn.Parameter(torch.randn(1, hidden_channels) * 0.02)
        
        # 4. Node Type Embedding (expr: 0, symbol: 1, virtual: 2)
        self.node_type_emb = nn.Embedding(3, hidden_channels)
        
        # 2. HGT Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)
            
        # 3. Output Readout (Projection Head)
        self.projection_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            Linear(hidden_channels, out_channels)
        )
        
        # Symbol Residual
        self.symbol_residual = nn.Parameter(torch.zeros(hidden_channels))

    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        x_dict contains:
            'expr': [N_expr, 47] - Float features
            'symbol': [N_sym] - Long IDs
            'virtual': [N_graphs, 1] - Dummy for batch identification
        """
        # Node Initialization
        h_dict = {}
        
        # Expr
        h_dict['expr'] = self.expr_encoder(x_dict['expr']) + self.node_type_emb.weight[0]
        
        # Symbol
        symbol_ids = x_dict['symbol']
        base_h = self.symbol_emb_base(symbol_ids)
        h_dict['symbol'] = self.symbol_projector(base_h) + self.symbol_residual + self.node_type_emb.weight[1]
        
        # Virtual
        # x_dict['virtual'] is just a placeholder for batching. 
        # We broadcast the learned virtual_emb to all graphs in the batch.
        num_graphs = x_dict['virtual'].size(0)
        h_dict['virtual'] = (self.virtual_emb + self.node_type_emb.weight[2]).expand(num_graphs, -1)
        
        # HGT Convolution Layers
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.gelu(v) for k, v in h_dict.items()}
            
        # Readout: Extract virtual node embeddings and project
        # These represent the global state of each graph in the batch
        out = self.projection_head(h_dict['virtual'])
        
        # Normalization for contrastive learning
        out = F.normalize(out, p=2, dim=1)
        
        return out
