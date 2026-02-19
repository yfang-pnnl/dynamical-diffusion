"""
Create train/val/test splits for your custom PFLOTRAN data.

This script creates index arrays for train/val/test splitting of your 300 realizations.
Default split: 192 train, 48 val, 60 test (80/16/20).

Usage:
    python create_data_splits.py \
      --output_dir /qfs/projects/dl_calibration/d3m045/test_dd/output \
      --n_realizations 300 \
      --train_ratio 0.8 \
      --val_ratio 0.16 \
      --seed 42
"""

import numpy as np
import argparse
import os


def create_splits(
    output_dir,
    n_realizations=300,
    train_ratio=0.80,
    val_ratio=0.16,
    test_ratio=None,
    seed=42,
):
    """
    Create train/val/test split indices.
    
    Parameters
    ----------
    output_dir : str
        Directory to save split indices
    n_realizations : int
        Total number of realizations
    train_ratio : float
        Fraction for training (default: 0.8)
    val_ratio : float
        Fraction for validation (default: 0.16)
    test_ratio : float or None
        Fraction for testing (if None, uses remainder)
    seed : int
        Random seed for reproducibility
    """
    
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    # Calculate split sizes
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio
    
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    n_train = int(np.round(n_realizations * train_ratio))
    n_val = int(np.round(n_realizations * val_ratio))
    n_test = n_realizations - n_train - n_val
    
    # Create random permutation of indices
    all_indices = np.arange(n_realizations)
    np.random.shuffle(all_indices)
    
    # Split
    train_idx = np.sort(all_indices[:n_train])
    val_idx = np.sort(all_indices[n_train:n_train + n_val])
    test_idx = np.sort(all_indices[n_train + n_val:])
    
    # Save
    np.save(os.path.join(output_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(output_dir, "val_idx.npy"), val_idx)
    np.save(os.path.join(output_dir, "test_idx.npy"), test_idx)
    
    print("Data splits created:")
    print(f"  Train: {len(train_idx)} ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} ({test_ratio*100:.1f}%)")
    print(f"  Seed: {seed}")
    print(f"\nSaved to {output_dir}:")
    print(f"  - train_idx.npy")
    print(f"  - val_idx.npy")
    print(f"  - test_idx.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for PFLOTRAN data"
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for split indices")
    parser.add_argument("--n_realizations", type=int, default=300,
                        help="Total number of realizations")
    parser.add_argument("--train_ratio", type=float, default=0.80,
                        help="Fraction for training")
    parser.add_argument("--val_ratio", type=float, default=0.16,
                        help="Fraction for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    create_splits(
        output_dir=args.output_dir,
        n_realizations=args.n_realizations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
