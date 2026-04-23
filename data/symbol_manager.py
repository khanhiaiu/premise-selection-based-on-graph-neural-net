import os
import json
import torch
import sys
import numpy as np
from typing import List, Dict, Optional

# Add Premise-Retrieval to path to import FlagModel
sys.path.append('/home/hahaha/project/Premise-Retrieval/test_finetune')

try:
    from flag_model import FlagModel
except ImportError:
    FlagModel = None

class SymbolManager:
    """
    Manages symbol vocabulary and precomputed embeddings using the 
    pretrained CFR model from Premise-Retrieval.
    """
    
    def __init__(self, vocab_path: str, model_path: Optional[str] = None):
        self.vocab_path = vocab_path
        self.model_path = model_path
        self.symbol_to_id = {}
        self.id_to_symbol = {}
        self.embeddings = None
        self.flag_model = None
        
        self.load_vocab()
        
    def load_vocab(self):
        """Load symbol vocabulary from JSON."""
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'r') as f:
                self.symbol_to_id = json.load(f)
            self.id_to_symbol = {v: k for k, v in self.symbol_to_id.items()}
            print(f"Loaded {len(self.symbol_to_id)} symbols from {self.vocab_path}")
        else:
            print(f"Vocab file {self.vocab_path} not found.")

    def _init_flag_model(self):
        """Initialize the pretrained FlagModel."""
        if self.flag_model is None and FlagModel is not None and self.model_path:
            print(f"Initializing FlagModel from {self.model_path}...")
            self.flag_model = FlagModel(
                self.model_path,
                model_type='encoder_only',
                pooling_method='mean',
                normalize_embeddings=True,
                use_fp16=torch.cuda.is_available()
            )
        elif FlagModel is None:
            print("Warning: FlagModel could not be imported. Check Premise-Retrieval path.")

    def generate_embeddings(self, output_path: str, batch_size: int = 512):
        """
        Generate 768-dim embeddings for all symbols in the vocab 
        using the pretrained encoder.
        """
        self._init_flag_model()
        if self.flag_model is None:
            raise RuntimeError("FlagModel is not initialized. Cannot generate embeddings.")
            
        symbols = [self.id_to_symbol[i] for i in range(len(self.id_to_symbol))]
        print(f"Encoding {len(symbols)} symbols...")
        
        # FlagModel.encode handles batching
        embeddings = self.flag_model.encode(symbols, batch_size=batch_size)
        
        # Save as torch tensor
        torch.save(torch.from_numpy(embeddings), output_path)
        self.embeddings = torch.from_numpy(embeddings)
        print(f"Saved embeddings to {output_path}")

    def load_embeddings(self, embedding_path: str):
        """Load precomputed embeddings from file."""
        if os.path.exists(embedding_path):
            self.embeddings = torch.load(embedding_path)
            print(f"Loaded embeddings from {embedding_path} (Shape: {self.embeddings.shape})")
        else:
            print(f"Embeddings file {embedding_path} not found.")

    def get_embedding_tensor(self) -> torch.Tensor:
        """Returns the full embedding tensor for nn.Embedding initialization."""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
        return self.embeddings

    def lookup_id(self, symbol_name: str) -> int:
        """Returns the ID of a symbol name, or UNK ID."""
        return self.symbol_to_id.get(symbol_name, self.symbol_to_id.get("<UNK>", 0))
