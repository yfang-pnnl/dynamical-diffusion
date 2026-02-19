#!/usr/bin/env python3
"""
Quick checkpoint validation - loads checkpoint and runs one batch to verify it works.

Usage:
    python core/quick_check_checkpoint.py <checkpoint_path>
"""

import sys
import os
import yaml
import torch
from dydiff.dydiff_pflotran import DynamicalLDMForPFLOTRANWithStaticCondition
from datasets.pflotran.pflotran_datamodule import PFLOTRANDataModule


def quick_check(checkpoint_path):
    """Quick sanity check that checkpoint loads and can run inference."""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Checking checkpoint: {checkpoint_path}")
    print(f"Size: {os.path.getsize(checkpoint_path) / 1e6:.1f} MB")
    
    # Load config
    config_path = "core/models/pflotran_identity.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Load model
        print("\n1. Loading model from checkpoint...")
        model_config = config['model']['params']
        model = DynamicalLDMForPFLOTRANWithStaticCondition.load_from_checkpoint(
            checkpoint_path,
            strict=False,
            **model_config
        )
        model.eval()
        model.cuda()
        print("   ✓ Model loaded successfully")
        
        # Setup data
        print("\n2. Loading validation data...")
        datamodule = PFLOTRANDataModule(
            perm_path=config['data']['perm_path'],
            poro_path=config['data']['poro_path'],
            conc_path=config['data']['conc_path'],
            train_idx_path=config['data']['train_idx_path'],
            val_idx_path=config['data']['val_idx_path'],
            test_idx_path=config['data']['test_idx_path'],
            input_length=config['data']['input_length'],
            pred_length=config['data']['pred_length'],
            total_length=config['data']['total_length'],
            t_keep=config['data']['t_keep'],
            dt=config['data']['dt'],
            batch_size=2,
            num_workers=4,
            shuffle_train=False
        )
        datamodule.setup()
        val_loader = datamodule.val_dataloader()
        print(f"   ✓ Loaded {len(val_loader.dataset)} validation samples")
        
        # Run one batch
        print("\n3. Running inference on one batch...")
        batch = next(iter(val_loader))
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
        
        with torch.no_grad():
            pred = model.sample(batch, ddim_steps=10, ddim_eta=0.0)
        
        print(f"   ✓ Prediction shape: {pred.shape}")
        print(f"   ✓ Ground truth shape: {batch['image'].shape}")
        
        # Check shapes match
        assert pred.shape == batch['image'].shape, "Shape mismatch!"
        
        print("\n✅ Checkpoint is valid and ready for evaluation!")
        print(f"\nTo run full evaluation:")
        print(f"  python core/eval_pflotran_checkpoint.py \\")
        print(f"    --config {config_path} \\")
        print(f"    --checkpoint {checkpoint_path} \\")
        print(f"    --num_samples 5")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core/quick_check_checkpoint.py <checkpoint_path>")
        print("\nExample:")
        print("  python core/quick_check_checkpoint.py \\")
        print("    logs/custom_pflotran/pflotran_identity/lightning_logs/version_515055/checkpoints/epoch=0-step=2000.ckpt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    success = quick_check(checkpoint_path)
    sys.exit(0 if success else 1)
