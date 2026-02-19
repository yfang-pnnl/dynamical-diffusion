"""
Data preparation script for custom PFLOTRAN simulations.

This script helps you organize your simulation data into the format expected
by the dynamical diffusion model.

Requirements:
- Your *.npy files containing concentration data
- Permeability and porosity values (can be scalar per realization or spatial fields)
- 300 realizations with grid size (28, 40) and temporal dimension up to 240 days

Usage:
    python prepare_custom_data.py \
      --conc_dir /path/to/concentration/files \
      --perm_values /path/to/permeability.npy \
      --poro_values /path/to/porosity.npy \
      --output_dir /path/to/output \
      --n_realizations 300 \
      --n_timesteps 240
"""

import os
import numpy as np
import argparse
from pathlib import Path


def prepare_data(
    conc_dir,
    output_dir,
    n_realizations=300,
    n_timesteps=240,
    nx=28,
    nz=40,
    perm_values=None,
    poro_values=None,
):
    """
    Prepare your custom PFLOTRAN data for use with dynamical diffusion.
    
    Parameters
    ----------
    conc_dir : str
        Directory containing *.npy files with concentration data.
        Files should be named: conc_0.npy, conc_1.npy, ..., conc_299.npy
        Each file shape: (n_timesteps, nz, nx) 
        
    output_dir : str
        Where to save processed files (perm.npy, poro.npy, conc.npy)
        
    n_realizations : int
        Number of simulations (default: 300)
        
    n_timesteps : int
        Number of time steps per simulation (default: 240)
        
    nx, nz : int
        Grid dimensions (default: nx=28, nz=40)
        
    perm_values : str or None
        Path to permeability values:
        - If None: will create constant permeability field
        - If path to .npy: should be shape (n_realizations,) or (n_realizations, nz, nx)
        
    poro_values : str or None
        Path to porosity values (same format as perm_values)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize containers
    conc_data = np.zeros((n_realizations, n_timesteps, nz, nx), dtype=np.float32)
    
    # Load concentration data from individual files
    print("Loading concentration data...")
    for i in range(n_realizations):
        conc_file = os.path.join(conc_dir, f"conc_{i}.npy")
        if not os.path.exists(conc_file):
            # Try alternative naming schemes
            for pattern in [f"realization_{i}.npy", f"sim_{i}.npy", f"{i}.npy"]:
                conc_file = os.path.join(conc_dir, pattern)
                if os.path.exists(conc_file):
                    break
            else:
                raise FileNotFoundError(f"Could not find concentration file for realization {i}")
        
        conc = np.load(conc_file)
        
        # Handle different possible shapes
        if conc.ndim == 3:
            # Assume (n_timesteps, nz, nx)
            if conc.shape[0] != n_timesteps or conc.shape[1] != nz or conc.shape[2] != nx:
                raise ValueError(
                    f"File {conc_file} has shape {conc.shape}, "
                    f"expected ({n_timesteps}, {nz}, {nx})"
                )
            conc_data[i] = conc
        else:
            raise ValueError(f"Unexpected concentration shape: {conc.shape}")
        
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{n_realizations}")
    
    print("✓ Concentration data loaded")
    
    # Handle permeability
    print("Processing permeability...")
    if perm_values is not None and os.path.exists(perm_values):
        perm_raw = np.load(perm_values)
        if perm_raw.ndim == 1 and len(perm_raw) == n_realizations:
            # Scalar per realization -> expand to spatial field
            perm_data = np.zeros((n_realizations, nz, nx), dtype=np.float32)
            for i in range(n_realizations):
                perm_data[i] = perm_raw[i]
        elif perm_raw.ndim == 3 and perm_raw.shape == (n_realizations, nz, nx):
            perm_data = perm_raw.astype(np.float32)
        else:
            raise ValueError(f"Unexpected permeability shape: {perm_raw.shape}")
    else:
        # Create constant permeability field
        print("  Creating constant permeability field (value = 1.0)")
        perm_data = np.ones((n_realizations, nz, nx), dtype=np.float32)
    
    print("✓ Permeability ready")
    
    # Handle porosity
    print("Processing porosity...")
    if poro_values is not None and os.path.exists(poro_values):
        poro_raw = np.load(poro_values)
        if poro_raw.ndim == 1 and len(poro_raw) == n_realizations:
            # Scalar per realization -> expand to spatial field
            poro_data = np.zeros((n_realizations, nz, nx), dtype=np.float32)
            for i in range(n_realizations):
                poro_data[i] = poro_raw[i]
        elif poro_raw.ndim == 3 and poro_raw.shape == (n_realizations, nz, nx):
            poro_data = poro_raw.astype(np.float32)
        else:
            raise ValueError(f"Unexpected porosity shape: {poro_raw.shape}")
    else:
        # Create constant porosity field
        print("  Creating constant porosity field (value = 0.3)")
        poro_data = np.ones((n_realizations, nz, nx), dtype=np.float32) * 0.3
    
    print("✓ Porosity ready")
    
    # Save combined arrays
    print("Saving combined arrays...")
    np.save(os.path.join(output_dir, "conc.npy"), conc_data)
    np.save(os.path.join(output_dir, "perm.npy"), perm_data)
    np.save(os.path.join(output_dir, "poro.npy"), poro_data)
    
    print(f"✓ Data saved to {output_dir}")
    print(f"  - conc.npy: {conc_data.shape}")
    print(f"  - perm.npy: {perm_data.shape}")
    print(f"  - poro.npy: {poro_data.shape}")
    
    return conc_data, perm_data, poro_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare custom PFLOTRAN data for dynamical diffusion model"
    )
    parser.add_argument("--conc_dir", type=str, required=True,
                        help="Directory containing concentration *.npy files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--perm_values", type=str, default=None,
                        help="Path to permeability values (.npy)")
    parser.add_argument("--poro_values", type=str, default=None,
                        help="Path to porosity values (.npy)")
    parser.add_argument("--n_realizations", type=int, default=300,
                        help="Number of realizations")
    parser.add_argument("--n_timesteps", type=int, default=240,
                        help="Number of timesteps (days)")
    parser.add_argument("--nx", type=int, default=28,
                        help="Grid dimension x")
    parser.add_argument("--nz", type=int, default=40,
                        help="Grid dimension z")
    
    args = parser.parse_args()
    
    prepare_data(
        conc_dir=args.conc_dir,
        output_dir=args.output_dir,
        n_realizations=args.n_realizations,
        n_timesteps=args.n_timesteps,
        nx=args.nx,
        nz=args.nz,
        perm_values=args.perm_values,
        poro_values=args.poro_values,
    )
