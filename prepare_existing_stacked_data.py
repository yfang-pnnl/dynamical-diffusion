"""
Prepare data from existing stacked numpy arrays.

Your case:
- perm.npy: (1500, nz, nx) - use first 300
- poro.npy: (1500, nz, nx) - use first 300
- conc.npy: (300, 7300, nz, nx) - slice timesteps as needed
- material.npy: (nz, nx) or (n_members, nz, nx) - mask where 0 means skip

This script:
1. Loads your existing stacked files
2. Extracts first 300 realizations from perm/poro
3. Slices concentration to desired timesteps (e.g., first 240 days)
4. Applies material mask (sets masked regions to 0 or NaN)
5. Validates shapes and data quality
6. Saves processed subset for training

Usage:
    python prepare_existing_stacked_data.py \
      --perm_path /path/to/perm.npy \
      --poro_path /path/to/poro.npy \
      --conc_path /path/to/conc.npy \
      --output_dir /path/to/output \
      --n_members 300 \
      --n_timesteps 240 \
      --material_path /path/to/material.npy  # optional mask
"""

import os
import numpy as np
import argparse


def prepare_from_stacked(
    perm_path,
    poro_path,
    conc_path,
    output_dir,
    n_members=300,
    n_timesteps=240,
    timestep_start=0,
    material_path=None,
    mask_value=0.0,
    validate=True,
):
    """
    Prepare training data from existing stacked arrays.
    
    Parameters
    ----------
    perm_path : str
        Path to permeability array (N_total, nz, nx)
    poro_path : str
        Path to porosity array (N_total, nz, nx)
    conc_path : str
        Path to concentration array (N_conc, T_total, nz, nx)
    output_dir : str
        Where to save processed files
    n_members : int
        Number of realizations to use (default: 300)
    n_timesteps : int
        Number of timesteps to use (default: 240)
    timestep_start : int
        Starting timestep index (default: 0)
    material_path : str, optional
        Path to material mask array (nz, nx) or (N, nz, nx).
        Locations where mask==0 will be set to mask_value in output.
    mask_value : float
        Value to set for masked regions (default: 0.0)
    validate : bool
        Whether to validate data quality
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("LOADING STACKED DATA")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading permeability from: {perm_path}")
    perm_full = np.load(perm_path, mmap_mode='r')
    print(f"  Full shape: {perm_full.shape}")
    
    print(f"\nLoading porosity from: {poro_path}")
    poro_full = np.load(poro_path, mmap_mode='r')
    print(f"  Full shape: {poro_full.shape}")
    
    print(f"\nLoading concentration from: {conc_path}")
    conc_full = np.load(conc_path, mmap_mode='r')
    print(f"  Full shape: {conc_full.shape}")
    
    # Load material mask if provided
    material_mask = None
    if material_path is not None:
        print(f"\nLoading material mask from: {material_path}")
        material_mask = np.load(material_path)
        print(f"  Mask shape: {material_mask.shape}")
        print(f"  Mask will be applied: locations where mask==0 → {mask_value}")
        n_masked = (material_mask == 0).sum()
        n_total = material_mask.size
        print(f"  Masked locations: {n_masked}/{n_total} ({100*n_masked/n_total:.1f}%)")
    
    # Validate input shapes
    print("\n" + "=" * 70)
    print("VALIDATING INPUT DATA")
    print("=" * 70)
    
    if len(perm_full.shape) != 3:
        raise ValueError(f"Permeability should be 3D (N, nz, nx), got shape {perm_full.shape}")
    
    if len(poro_full.shape) != 3:
        raise ValueError(f"Porosity should be 3D (N, nz, nx), got shape {poro_full.shape}")
    
    if len(conc_full.shape) != 4:
        raise ValueError(f"Concentration should be 4D (N, T, nz, nx), got shape {conc_full.shape}")
    
    # Check spatial dimensions match
    if perm_full.shape[1:] != poro_full.shape[1:]:
        raise ValueError(f"Perm and poro spatial dimensions don't match: {perm_full.shape[1:]} vs {poro_full.shape[1:]}")
    
    if conc_full.shape[2:] != perm_full.shape[1:]:
        raise ValueError(f"Conc and perm spatial dimensions don't match: {conc_full.shape[2:]} vs {perm_full.shape[1:]}")
    
    # Check we have enough data
    if perm_full.shape[0] < n_members:
        raise ValueError(f"Perm has only {perm_full.shape[0]} members, need {n_members}")
    # Validate material mask if provided
    if material_mask is not None:
        if len(material_mask.shape) == 2:
            # Spatial mask (nz, nx) - same for all realizations
            if material_mask.shape != perm_full.shape[1:]:
                raise ValueError(f"Material mask shape {material_mask.shape} doesn't match spatial dimensions {perm_full.shape[1:]}")
            print(f"✓ Material mask is spatial (same for all realizations)")
        elif len(material_mask.shape) == 3:
            # Per-realization mask (N, nz, nx)
            if material_mask.shape[0] < n_members:
                raise ValueError(f"Material mask has only {material_mask.shape[0]} members, need {n_members}")
            if material_mask.shape[1:] != perm_full.shape[1:]:
                raise ValueError(f"Material mask spatial shape {material_mask.shape[1:]} doesn't match {perm_full.shape[1:]}")
            print(f"✓ Material mask is per-realization")
        else:
            raise ValueError(f"Material mask should be 2D (nz, nx) or 3D (N, nz, nx), got shape {material_mask.shape}")
    
    
    if poro_full.shape[0] < n_members:
        raise ValueError(f"Poro has only {poro_full.shape[0]} members, need {n_members}")
    
    if conc_full.shape[0] < n_members:
        raise ValueError(f"Conc has only {conc_full.shape[0]} members, need {n_members}")
    
    timestep_end = timestep_start + n_timesteps
    if conc_full.shape[1] < timestep_end:
        raise ValueError(f"Conc has only {conc_full.shape[1]} timesteps, need {timestep_end}")
    
    print(f"✓ All input shapes compatible")
    print(f"✓ Sufficient data available")
    
    # Extract subsets
    print("\n" + "=" * 70)
    print("EXTRACTING SUBSETS")
    print("=" * 70)
    
    print(f"\nExtracting first {n_members} realizations...")
    perm_subset = perm_full[:n_members].astype(np.float32)
    poro_subset = poro_full[:n_members].astype(np.float32)
    
    print(f"Extracting timesteps {timestep_start}:{timestep_end}...")
    conc_subset = conc_full[:n_members, timestep_start:timestep_end].astype(np.float32)
    
    print(f"\n✓ Subset shapes:")
    print(f"  perm: {perm_subset.shape}")
    print(f"  poro: {poro_subset.shape}")
    print(f"  conc: {conc_subset.shape}")
    
    # Apply material mask
    if material_mask is not None:
        print("\n" + "=" * 70)
        print("APPLYING MATERIAL MASK")
        print("=" * 70)
        
        if len(material_mask.shape) == 2:
            # Spatial mask - broadcast to all realizations and timesteps
            mask_subset = material_mask
            print(f"Using spatial mask (same for all realizations)")
        else:
            # Per-realization mask
            mask_subset = material_mask[:n_members]
            print(f"Using per-realization mask")
        
        # Apply mask to data
        print(f"Applying mask (setting masked regions to {mask_value})...")
        
        # Mask concentration (all timesteps)
        if len(mask_subset.shape) == 2:
            # Broadcast spatial mask: (nz, nx) -> (1, 1, nz, nx)
            mask_broad = mask_subset[None, None, :, :]
            conc_subset = np.where(mask_broad != 0, conc_subset, mask_value)
        else:
            # Per-realization mask: (N, nz, nx) -> (N, 1, nz, nx)
            mask_broad = mask_subset[:, None, :, :]
            conc_subset = np.where(mask_broad != 0, conc_subset, mask_value)
        
        # Ensure mask is also applied to perm/poro if masked regions are invalid
        # Typically we want conditioning to be valid everywhere, but if invalid, we zero them.
        # Here we assume minimal masking for perm/poro unless specified.
        
        # Optionally mask perm/poro (usually you want to keep these values even in masked regions)
        # Uncomment below if you want to mask perm/poro as well
        # if len(mask_subset.shape) == 2:
        #     perm_subset = np.where(mask_subset != 0, perm_subset, mask_value)
        #     poro_subset = np.where(mask_subset != 0, poro_subset, mask_value)
        # else:
        #     perm_subset = np.where(mask_subset != 0, perm_subset, mask_value)
        #     poro_subset = np.where(mask_subset != 0, poro_subset, mask_value)
        
        n_masked_per_realization = (mask_broad == 0).sum() // (conc_subset.shape[0] * conc_subset.shape[1])
        print(f"✓ Masked {n_masked_per_realization} locations per realization/timestep")
    
    # Validate data quality
    if validate:
        print("\n" + "=" * 70)
        print("VALIDATING DATA QUALITY")
        print("=" * 70)
        
        for name, data in [("Permeability", perm_subset), 
                           ("Porosity", poro_subset), 
                           ("Concentration", conc_subset)]:
            print(f"\n{name}:")
            print(f"  Range: [{data.min():.6e}, {data.max():.6e}]")
            print(f"  Mean: {data.mean():.6e}")
            print(f"  Std: {data.std():.6e}")
            
            n_nan = np.isnan(data).sum()
            n_inf = np.isinf(data).sum()
            n_neg = (data < 0).sum()
            
            print(f"  NaNs: {n_nan}")
            print(f"  Infs: {n_inf}")
            print(f"  Negative values: {n_neg}")
            
            if n_nan > 0:
                print(f"  ⚠ WARNING: Found {n_nan} NaN values!")
            if n_inf > 0:
                print(f"  ⚠ WARNING: Found {n_inf} Inf values!")
            
            # Check if data is constant (might indicate an issue)
            if data.std() < 1e-10:
                print(f"  ⚠ WARNING: Data appears to be constant!")
        
        print(f"\n{'✓ Data validation complete' if n_nan == 0 and n_inf == 0 else '⚠ Validation found issues (see above)'}")
    
    # Save processed data
    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)
    
    perm_out = os.path.join(output_dir, "perm.npy")
    poro_out = os.path.join(output_dir, "poro.npy")
    conc_out = os.path.join(output_dir, "conc.npy")
    
    print(f"\nSaving to {output_dir}")
    np.save(perm_out, perm_subset)
    print(f"  ✓ perm.npy: {perm_subset.shape} ({perm_subset.nbytes/1e6:.1f} MB)")
    
    np.save(poro_out, poro_subset)
    print(f"  ✓ poro.npy: {poro_subset.shape} ({poro_subset.nbytes/1e6:.1f} MB)")
    
    np.save(conc_out, conc_subset)
    print(f"  ✓ conc.npy: {conc_subset.shape} ({conc_subset.nbytes/1e6:.1f} MB)")
    
    # Save metadata
    metadata = {
        'n_members': n_members,
        'n_timesteps': n_timesteps,
        'timestep_start': timestep_start,
        'timestep_end': timestep_end,
        'spatial_shape': perm_subset.shape[1:],
        'material_mask_applied': material_path is not None,
        'mask_value': mask_value if material_path is not None else None,
        'source_files': {
            'perm': perm_path,
            'poro': poro_path,
            'conc': conc_path,
            'material': material_path if material_path is not None else 'None'
        }
    }
    
    metadata_out = os.path.join(output_dir, "metadata.txt")
    with open(metadata_out, 'w') as f:
        f.write("PROCESSED DATA METADATA\n")
        f.write("=" * 50 + "\n\n")
        for key, val in metadata.items():
            f.write(f"{key}: {val}\n")
    
    print(f"  ✓ metadata.txt")
    
    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"\nProcessed data saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Create data splits:")
    print(f"   python create_data_splits.py \\")
    print(f"     --output_dir {output_dir}/splits \\")
    print(f"     --n_realizations {n_members}")
    print(f"\n2. Update config file with these paths:")
    print(f"   perm_path: {perm_out}")
    print(f"   poro_path: {poro_out}")
    print(f"   conc_path: {conc_out}")
    print(f"   t_keep: {n_timesteps}")
    
    return perm_subset, poro_subset, conc_subset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data from existing stacked numpy arrays"
    )
    parser.add_argument("--perm_path", type=str, required=True,
                        help="Path to permeability .npy file")
    parser.add_argument("--poro_path", type=str, required=True,
                        help="Path to porosity .npy file")
    parser.add_argument("--conc_path", type=str, required=True,
                        help="Path to concentration .npy file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--n_members", type=int, default=300,
                        help="Number of realizations to extract (default: 300)")
    parser.add_argument("--n_timesteps", type=int, default=240,
                        help="Number of timesteps to extract (default: 240)")
    parser.add_argument("--timestep_start", type=int, default=0,
                        help="Starting timestep index (default: 0)")
    parser.add_argument("--material_path", type=str, default=None,
                        help="Path to material mask .npy file (optional)")
    parser.add_argument("--mask_value", type=float, default=0.0,
                        help="Value to set for masked regions (default: 0.0)")
    parser.add_argument("--no_validate", action='store_true',
                        help="Skip data quality validation")
    
    args = parser.parse_args()
    
    prepare_from_stacked(
        material_path=args.material_path,
        mask_value=args.mask_value,
        perm_path=args.perm_path,
        poro_path=args.poro_path,
        conc_path=args.conc_path,
        output_dir=args.output_dir,
        n_members=args.n_members,
        n_timesteps=args.n_timesteps,
        timestep_start=args.timestep_start,
        validate=not args.no_validate,
    )
