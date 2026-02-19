import numpy as np
import torch
from torch.utils.data import Dataset

class PFLOTRANDatasetConcStatic(Dataset):
    """
    conc is diffused/predicted (log1p space).
    perm/poro are static conditioning maps.
    Expects:
      perm.npy: (N,H,W)
      poro.npy: (N,H,W)
      conc.npy: (N,T,H,W) (we slice first t_keep)
    Returns dict:
      cond:  (H,W,P*1)   concentration history
      image: (H,W,S*1)   concentration future
      prev:  (H,W,P*1)   same as cond (DyDiff uses it)
      total: (H,W,(P+S)*1) full sequence (for optional debug)
      static:(2,H,W) perm/poro (NOT flattened)
    """
    def __init__(self, perm_path, poro_path, conc_path, indices,
                 input_length, pred_length=20, t_keep=365, dt=1,
                 mmap=True, clamp_min=0.0):
        self.input_length = int(input_length)
        self.pred_length = int(pred_length)
        self.total_length = self.input_length + self.pred_length
        self.t_keep = int(t_keep)
        self.dt = int(dt)
        self.clamp_min = float(clamp_min)

        mode = "r" if mmap else None
        self.perm = np.load(perm_path, mmap_mode=mode)   # (N,H,W)
        self.poro = np.load(poro_path, mmap_mode=mode)   # (N,H,W)
        self.conc = np.load(conc_path, mmap_mode=mode)   # (N,T,H,W)

        self.indices = np.asarray(indices, dtype=np.int64)

        T = min(self.conc.shape[1], self.t_keep)
        self.T = T
        self.T_eff = (T - 1) // self.dt + 1
        self.max_start = self.T_eff - self.total_length
        if self.max_start <= 0:
            raise ValueError(f"Not enough timesteps: T_eff={self.T_eff}, total_length={self.total_length}")

        # global window index (realization, start)
        self.sample_index = []
        for ridx in range(len(self.indices)):
            for s in range(self.max_start + 1):
                self.sample_index.append((ridx, s))

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        ridx, s = self.sample_index[idx]
        real_id = self.indices[ridx]

        start = s * self.dt
        perm = self.perm[real_id]                  # (H,W)
        poro = self.poro[real_id]                  # (H,W)
        # Optimize reading from mmap by slicing directly
        # t_idx is start + dt * [0, 1, ..., total_length-1]
        # Equivalent to slice: start : start + total_length*dt : dt
        stop = start + self.total_length * self.dt
        conc_seq = self.conc[real_id, start:stop:self.dt]  # (L,H,W)

        # ---- log1p transform (stabilizes sparse plumes)
        conc_seq = np.clip(conc_seq, self.clamp_min, None)
        conc_seq = np.log1p(conc_seq).astype(np.float32)

        # tensorize
        conc_seq = torch.from_numpy(conc_seq).float()  # (L,H,W)
        perm = torch.from_numpy(perm).float()          # (H,W)
        poro = torch.from_numpy(poro).float()          # (H,W)

        L, H, W = conc_seq.shape
        # (L,1,H,W)
        data = conc_seq[:, None, :, :]

        cond = data[:self.input_length]                               # (P,1,H,W)
        pred = data[self.input_length:self.total_length]              # (S,1,H,W)
        prev = data[:self.input_length]                               # (P,1,H,W)
        total = data[:self.total_length]                              # (P+S,1,H,W)

        def pack(x):
            # (T,1,H,W) -> (H,W,T*1)
            return x.reshape(-1, H, W).permute(1, 2, 0).contiguous()

        static = torch.stack([perm, poro], dim=0)  # (2,H,W)

        return {
            "cond": pack(cond),
            "image": pack(pred),
            "prev": pack(prev),
            "total": pack(total),
            "static": static,
        }

