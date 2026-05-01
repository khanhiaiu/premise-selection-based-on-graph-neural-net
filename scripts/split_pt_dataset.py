import os
import torch
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Split states_list.pt into train and validation sets.")
    parser.add_argument("--input_pt", type=str, required=True, help="Path to the precomputed states_list.pt")
    parser.add_argument("--out_train", type=str, default="states_list_train.pt", help="Path to output train pt")
    parser.add_argument("--out_val", type=str, default="states_list_val.pt", help="Path to output val pt")
    parser.add_argument("--val_size", type=int, default=2000, help="Number of samples for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    print(f"Loading {args.input_pt}...")
    states_list = torch.load(args.input_pt, weights_only=False)
    total_states = len(states_list)
    print(f"Loaded {total_states} states.")

    if total_states <= args.val_size:
        raise ValueError(f"Total states ({total_states}) is less than or equal to val_size ({args.val_size}). Cannot split.")

    # Shuffle with a fixed seed to ensure reproducibility
    rng = random.Random(args.seed)
    rng.shuffle(states_list)

    val_list = states_list[:args.val_size]
    train_list = states_list[args.val_size:]

    print(f"Splitting into {len(train_list)} train and {len(val_list)} val samples.")

    # Save validation set
    out_val_path = os.path.join(os.path.dirname(args.input_pt), args.out_val)
    torch.save(val_list, out_val_path)
    print(f"Saved validation set to {out_val_path}")

    # Save train set
    out_train_path = os.path.join(os.path.dirname(args.input_pt), args.out_train)
    torch.save(train_list, out_train_path)
    print(f"Saved train set to {out_train_path}")

    print("Done! You can now use these two separate files for training and validation.")

if __name__ == "__main__":
    main()
