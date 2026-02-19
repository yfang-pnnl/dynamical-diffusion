"""
Utility script to help with custom permeability and porosity data.

This script provides helper functions if you have perm/poro data in different formats
or need to normalize/transform them.

Examples:
    # Load permeability from PFLOTRANDatabase or custom format
    python utils_extract_properties.py --help

    # Or use as a module:
    from utils_extract_properties import create_spatial_field
    perm_field = create_spatial_field(perm_values, shape=(300, 40, 28))
"""

import numpy as np
import argparse
import os


def extract_from_hdf5(hdf5_path, dataset_names, output_dir):
    """
    Extract permeability/porosity from HDF5 file.
    
    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 file containing your data
    dataset_names : list of str
        Names of datasets in the HDF5 file (e.g., ['permeability', 'porosity'])
    output_dir : str
        Where to save extracted .npy files
    """
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed. Install with: pip install h5py")
        return

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as f:
        for name in dataset_names:
            if name in f:
                data = np.array(f[name], dtype=np.float32)
                output_path = os.path.join(output_dir, f"{name}.npy")
                np.save(output_path, data)
                print(f"Extracted {name}: shape {data.shape}")
            else:
                print(f"WARNING: '{name}' not found in {hdf5_path}")
                print(f"Available datasets: {list(f.keys())}")


def create_spatial_field(scalar_values, shape):
    """
    Create spatial field from scalar values per realization.
    
    Parameters
    ----------
    scalar_values : ndarray
        Shape (N,) with one scalar per realization
    shape : tuple
        Target shape (N, H, W)
    
    Returns
    -------
    ndarray
        Shape (N, H, W) with scalar repeated spatially
    """
    N, H, W = shape
    field = np.zeros((N, H, W), dtype=np.float32)
    for i in range(N):
        field[i] = scalar_values[i]
    return field


def normalize_properties(perm_values, poro_values=None, perm_range=(0.1, 10.0), poro_range=(0.1, 0.5)):
    """
    Normalize permeability and porosity to standard ranges.
    
    Parameters
    ----------
    perm_values : ndarray
        Raw permeability data
    poro_values : ndarray, optional
        Raw porosity data
    perm_range : tuple
        (min, max) for permeability
    poro_range : tuple
        (min, max) for porosity
    
    Returns
    -------
    perm_norm, poro_norm : ndarray
        Normalized arrays
    """
    # Permeability
    perm_min, perm_max = perm_values.min(), perm_values.max()
    perm_norm = perm_values.copy().astype(np.float32)
    if perm_max > perm_min:
        # Normalize to [0, 1]
        perm_norm = (perm_norm - perm_min) / (perm_max - perm_min)
        # Scale to target range
        perm_norm = perm_norm * (perm_range[1] - perm_range[0]) + perm_range[0]
    
    print(f"Permeability: {perm_values.min():.4e} -> {perm_norm.min():.4e}")
    print(f"             {perm_values.max():.4e} -> {perm_norm.max():.4e}")
    
    poro_norm = None
    if poro_values is not None:
        poro_min, poro_max = poro_values.min(), poro_values.max()
        poro_norm = poro_values.copy().astype(np.float32)
        if poro_max > poro_min:
            poro_norm = (poro_norm - poro_min) / (poro_max - poro_min)
            poro_norm = poro_norm * (poro_range[1] - poro_range[0]) + poro_range[0]
        
        print(f"Porosity:     {poro_values.min():.4f} -> {poro_norm.min():.4f}")
        print(f"             {poro_values.max():.4f} -> {poro_norm.max():.4f}")
    
    return perm_norm, poro_norm


def extract_from_txt_column(txt_path, cols=(0, 1), n_realizations=300):
    """
    Extract permeability and porosity from text file (e.g., CSV).
    
    Parameters
    ----------
    txt_path : str
        Path to text file (CSV, space-separated, etc.)
    cols : tuple
        Column indices (perm_col, poro_col)
    n_realizations : int
        Number of realizations expected
    
    Returns
    -------
    perm, poro : ndarray
        Shape (n_realizations,)
    """
    data = np.loadtxt(txt_path)
    if data.shape[0] != n_realizations:
        print(f"WARNING: File has {data.shape[0]} rows, expected {n_realizations}")
    
    perm = data[:, cols[0]].astype(np.float32)
    poro = data[:, cols[1]].astype(np.float32) if len(cols) > 1 else None
    
    return perm, poro


def validate_data(perm_path, poro_path, conc_path, expected_shape=(300, 40, 28)):
    """
    Validate that your prepared data matrices have correct shapes and values.
    
    Parameters
    ----------
    perm_path, poro_path, conc_path : str
        Paths to prepared .npy files
    expected_shape : tuple
        Expected shape for perm/poro (N, H, W)
    """
    expected_N, expected_H, expected_W = expected_shape
    
    perm = np.load(perm_path)
    poro = np.load(poro_path)
    conc = np.load(conc_path)
    
    print("Data Validation Results:")
    print("=" * 50)
    
    # Check shapes
    print(f"\nShapes:")
    print(f"  perm: {perm.shape} (expected: ({expected_N}, {expected_H}, {expected_W}))")
    print(f"  poro: {poro.shape} (expected: ({expected_N}, {expected_H}, {expected_W}))")
    print(f"  conc: {conc.shape} (expected: ({expected_N}, *, {expected_H}, {expected_W}))")
    
    # Check ranges
    print(f"\nValue Ranges:")
    print(f"  perm: [{perm.min():.4e}, {perm.max():.4e}]")
    print(f"  poro: [{poro.min():.6f}, {poro.max():.6f}]")
    print(f"  conc: [{conc.min():.6e}, {conc.max():.6e}]")
    
    # Check for NaNs/Infs
    print(f"\nNaN/Inf Check:")
    print(f"  perm NaNs: {np.isnan(perm).sum()}, Infs: {np.isinf(perm).sum()}")
    print(f"  poro NaNs: {np.isnan(poro).sum()}, Infs: {np.isinf(poro).sum()}")
    print(f"  conc NaNs: {np.isnan(conc).sum()}, Infs: {np.isinf(conc).sum()}")
    
    # Shape validation
    all_good = True
    if perm.shape != (expected_N, expected_H, expected_W):
        print(f"\n⚠ ERROR: perm shape mismatch!")
        all_good = False
    if poro.shape != (expected_N, expected_H, expected_W):
        print(f"⚠ ERROR: poro shape mismatch!")
        all_good = False
    if conc.shape[0] != expected_N or conc.shape[2] != expected_H or conc.shape[3] != expected_W:
        print(f"⚠ ERROR: conc shape mismatch!")
        all_good = False
    
    if all_good and np.isnan(perm).sum() == 0 and np.isnan(conc).sum() == 0:
        print(f"\n✓ All checks passed!")
    
    return perm, poro, conc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utilities for custom permeability/porosity data"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate prepared data')
    validate_parser.add_argument('--perm', required=True, help='Path to perm.npy')
    validate_parser.add_argument('--poro', required=True, help='Path to poro.npy')
    validate_parser.add_argument('--conc', required=True, help='Path to conc.npy')
    validate_parser.add_argument('--shape', nargs=3, type=int, default=[300, 40, 28],
                                 help='Expected shape (N H W)')
    
    # Extract from HDF5 command
    hdf5_parser = subparsers.add_parser('from_hdf5', help='Extract from HDF5 file')
    hdf5_parser.add_argument('--input', required=True, help='HDF5 file path')
    hdf5_parser.add_argument('--datasets', nargs='+', default=['permeability', 'porosity'],
                             help='Dataset names to extract')
    hdf5_parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        validate_data(args.perm, args.poro, args.conc, tuple(args.shape))
    elif args.command == 'from_hdf5':
        extract_from_hdf5(args.input, args.datasets, args.output)
    else:
        parser.print_help()
