"""
Data inspection and debugging script for PFLOTRAN simulations.

Use this to understand your data before and after preparation.

Usage:
    python inspect_data.py --help
"""

import numpy as np
import argparse
import os
from pathlib import Path

# Set matplotlib backend for remote server compatibility
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def inspect_raw_files(conc_dir, n_files=5):
    """
    Inspect individual concentration files.
    
    Parameters
    ----------
    conc_dir : str
        Directory containing *.npy files
    n_files : int
        How many files to inspect
    """
    
    print("=" * 60)
    print("INSPECTING RAW CONCENTRATION FILES")
    print("=" * 60)
    
    conc_files = sorted([f for f in os.listdir(conc_dir) if f.endswith('.npy')])
    
    if not conc_files:
        print(f"ERROR: No .npy files found in {conc_dir}")
        return
    
    print(f"Found {len(conc_files)} concentration files")
    print()
    
    for i, fname in enumerate(conc_files[:n_files]):
        fpath = os.path.join(conc_dir, fname)
        try:
            conc = np.load(fpath)
            print(f"{i}: {fname}")
            print(f"   Shape: {conc.shape}")
            print(f"   Type: {conc.dtype}")
            print(f"   Range: [{conc.min():.6e}, {conc.max():.6e}]")
            print(f"   Mean: {conc.mean():.6e}, Std: {conc.std():.6e}")
            print(f"   NaNs: {np.isnan(conc).sum()}, Infs: {np.isinf(conc).sum()}")
            print()
        except Exception as e:
            print(f"ERROR loading {fname}: {e}")
            print()


def inspect_prepared_data(perm_path, poro_path, conc_path, show_samples=True):
    """
    Inspect prepared data matrices.
    
    Parameters
    ----------
    perm_path, poro_path, conc_path : str
        Paths to prepared .npy files
    show_samples : bool
        Whether to show sample statistics
    """
    
    print("=" * 60)
    print("INSPECTING PREPARED DATA")
    print("=" * 60)
    
    for name, path in [("Permeability", perm_path), 
                       ("Porosity", poro_path), 
                       ("Concentration", conc_path)]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            continue
        
        print(f"\n{name.upper()}")
        print("-" * 40)
        
        data = np.load(path)
        print(f"  Shape: {data.shape}")
        print(f"  Type: {data.dtype}")
        print(f"  Memory: {data.nbytes / 1e9:.2f} GB")
        print(f"  Range: [{data.min():.6e}, {data.max():.6e}]")
        print(f"  Mean: {data.mean():.6e}")
        print(f"  Std: {data.std():.6e}")
        print(f"  NaNs: {np.isnan(data).sum()}")
        print(f"  Infs: {np.isinf(data).sum()}")
        
        if show_samples and name == "Concentration":
            # Show some temporal statistics
            print(f"\n  Temporal Statistics:")
            mean_per_time = data.mean(axis=(0, 2, 3))  # Average over realization and space
            print(f"    First timestep: {mean_per_time[0]:.6e}")
            print(f"    Mid timestep: {mean_per_time[data.shape[1]//2]:.6e}")
            print(f"    Last timestep: {mean_per_time[-1]:.6e}")
            print(f"    Increasing? {mean_per_time[-1] > mean_per_time[0]}")


def show_sample_field(path, sample_idx=0, time_idx=None, title=""):
    """
    Visualize a single spatial field.
    
    Parameters
    ----------
    path : str
        Path to .npy file
    sample_idx : int
        Which realization to show
    time_idx : int or None
        Which timestep (for concentration)
    title : str
        Plot title
    """
    
    data = np.load(path)
    
    if len(data.shape) == 4:  # Concentration (N, T, H, W)
        if time_idx is None:
            time_idx = 0
        field = data[sample_idx, time_idx]
        title += f" (Realization {sample_idx}, Time {time_idx})"
    elif len(data.shape) == 3:  # Perm/Poro (N, H, W)
        field = data[sample_idx]
        title += f" (Realization {sample_idx})"
    else:
        print(f"Unexpected shape: {data.shape}")
        return
    
    plt.figure(figsize=(8, 6))
    plt.imshow(field, cmap='viridis', origin='upper')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('X (nx)')
    plt.ylabel('Z (nz)')
    plt.tight_layout()
    return plt


def compare_distributions(perm_path, poro_path, conc_path):
    """
    Compare value distributions across datasets.
    
    Parameters
    ----------
    perm_path, poro_path, conc_path : str
        Paths to prepared .npy files
    """
    
    print("=" * 60)
    print("DISTRIBUTION COMPARISON")
    print("=" * 60)
    
    perm = np.load(perm_path).flatten()
    poro = np.load(poro_path).flatten()
    conc = np.load(conc_path).flatten()
    conc_nonzero = conc[conc > 0]  # Exclude zeros for log analysis
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Permeability
    axes[0, 0].hist(perm, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Permeability Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Count')
    
    # Porosity
    axes[0, 1].hist(poro, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title('Porosity Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Count')
    
    # Concentration (linear)
    axes[1, 0].hist(conc, bins=100, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_title('Concentration Distribution (Linear)')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Count')
    
    # Concentration (log scale, non-zero values)
    if len(conc_nonzero) > 0:
        axes[1, 1].hist(np.log10(conc_nonzero), bins=50, edgecolor='black', alpha=0.7, color='red')
        axes[1, 1].set_title('Concentration Distribution (log10, non-zero)')
        axes[1, 1].set_xlabel('log10(Value)')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    return fig


def temporal_analysis(conc_path):
    """
    Analyze temporal evolution of concentration.
    
    Parameters
    ----------
    conc_path : str
        Path to concentration .npy file
    """
    
    print("=" * 60)
    print("TEMPORAL ANALYSIS")
    print("=" * 60)
    
    conc = np.load(conc_path)  # (N, T, H, W)
    
    # Time evolution
    mean_per_time = conc.mean(axis=(0, 2, 3))  # Average over realization and space
    max_per_time = conc.max(axis=(0, 2, 3))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mean_per_time, label='Mean', linewidth=2)
    plt.fill_between(range(len(mean_per_time)), 
                     conc.min(axis=(0, 2, 3)),
                     conc.max(axis=(0, 2, 3)),
                     alpha=0.3, label='[Min, Max]')
    plt.xlabel('Time (days)')
    plt.ylabel('Concentration')
    plt.title('Temporal Evolution (Ensemble Average)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Variance
    var_per_time = conc.var(axis=(0, 2, 3))
    
    plt.subplot(1, 2, 2)
    plt.plot(var_per_time, color='orange', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Variance')
    plt.title('Ensemble Variance Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect PFLOTRAN data before and after preparation"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Raw inspection
    raw_parser = subparsers.add_parser('raw', help='Inspect raw files in directory')
    raw_parser.add_argument('--conc_dir', required=True, help='Directory with concentration files')
    raw_parser.add_argument('--n_files', type=int, default=5, help='How many files to inspect')
    
    # Prepared data inspection
    prepared_parser = subparsers.add_parser('prepared', help='Inspect prepared data files')
    prepared_parser.add_argument('--perm', required=True, help='Path to perm.npy')
    prepared_parser.add_argument('--poro', required=True, help='Path to poro.npy')
    prepared_parser.add_argument('--conc', required=True, help='Path to conc.npy')
    
    # Distributions
    dist_parser = subparsers.add_parser('dist', help='Compare value distributions')
    dist_parser.add_argument('--perm', required=True, help='Path to perm.npy')
    dist_parser.add_argument('--poro', required=True, help='Path to poro.npy')
    dist_parser.add_argument('--conc', required=True, help='Path to conc.npy')
    dist_parser.add_argument('--savefig', type=str, default=None, help='Save figure to file')
    
    # Temporal analysis
    time_parser = subparsers.add_parser('temporal', help='Temporal analysis')
    time_parser.add_argument('--conc', required=True, help='Path to conc.npy')
    time_parser.add_argument('--savefig', type=str, default=None, help='Save figure to file')
    
    # Visualize sample
    sample_parser = subparsers.add_parser('sample', help='Visualize sample field')
    sample_parser.add_argument('--file', required=True, help='Path to .npy file')
    sample_parser.add_argument('--sample', type=int, default=0, help='Realization index')
    sample_parser.add_argument('--time', type=int, default=None, help='Timestep (for concentration)')
    sample_parser.add_argument('--title', type=str, default='', help='Plot title')
    sample_parser.add_argument('--savefig', type=str, default=None, help='Save figure to file')
    
    args = parser.parse_args()
    
    if args.command == 'raw':
        inspect_raw_files(args.conc_dir, args.n_files)
    
    elif args.command == 'prepared':
        inspect_prepared_data(args.perm, args.poro, args.conc)
    
    elif args.command == 'dist':
        fig = compare_distributions(args.perm, args.poro, args.conc)
        if args.savefig:
            fig.savefig(args.savefig, dpi=150, bbox_inches='tight')
            print(f"Saved to {args.savefig}")
        else:
            plt.show()
    
    elif args.command == 'temporal':
        fig = temporal_analysis(args.conc)
        if args.savefig:
            fig.savefig(args.savefig, dpi=150, bbox_inches='tight')
            print(f"Saved to {args.savefig}")
        else:
            plt.show()
    
    elif args.command == 'sample':
        plt_obj = show_sample_field(args.file, args.sample, args.time, args.title)
        if args.savefig:
            plt_obj.savefig(args.savefig, dpi=150, bbox_inches='tight')
            print(f"Saved to {args.savefig}")
        else:
            plt_obj.show()
    
    else:
        parser.print_help()
