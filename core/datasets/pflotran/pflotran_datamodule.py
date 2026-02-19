import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.pflotran.pflotran import PFLOTRANDatasetConcStatic

class PFLOTRANDataModuleForDyDiff(pl.LightningDataModule):
    def __init__(self, cfg, batch_size=16, num_workers=8):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_idx = np.load(self.cfg.train_idx_path)
        val_idx   = np.load(self.cfg.val_idx_path)
        test_idx  = np.load(self.cfg.test_idx_path)

        common = dict(
            perm_path=self.cfg.perm_path,
            poro_path=self.cfg.poro_path,
            conc_path=self.cfg.conc_path,
            input_length=self.cfg.input_length,
            pred_length=self.cfg.pred_length,
            t_keep=self.cfg.t_keep,
            dt=getattr(self.cfg, "dt", 1),
            mmap=True,
        )

        self.train_dataset = PFLOTRANDatasetConcStatic(indices=train_idx, **common)
        self.val_dataset   = PFLOTRANDatasetConcStatic(indices=val_idx, **common)
        self.test_dataset  = PFLOTRANDatasetConcStatic(indices=test_idx, **common)

        print(f"Train windows: {len(self.train_dataset)}")
        print(f"Val windows:   {len(self.val_dataset)}")
        print(f"Test windows:  {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

