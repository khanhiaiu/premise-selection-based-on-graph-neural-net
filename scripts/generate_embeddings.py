import os
import sys
import torch
import argparse

# Add parent directory to path to import data modules
sys.path.append(os.getcwd())

from data.symbol_manager import SymbolManager

def main():
    parser = argparse.ArgumentParser(description="Generate symbol embeddings using pretrained CFR model.")
    parser.add_argument("--vocab_path", type=str, default="symbol_vocab.json", help="Path to symbol vocab JSON.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained FlagModel directory.")
    parser.add_argument("--output_path", type=str, default="symbol_embeddings.pt", help="Path to save precomputed embeddings.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for encoding.")
    args = parser.parse_args()

    # Initialize SymbolManager
    manager = SymbolManager(vocab_path=args.vocab_path, model_path=args.model_path)
    
    # Generate embeddings
    manager.generate_embeddings(output_path=args.output_path, batch_size=args.batch_size)
    
    print("Done!")

if __name__ == "__main__":
    main()
