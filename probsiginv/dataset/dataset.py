"""Dataset readers and thin tensor wrappers for saved path/signature artifacts."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from probsiginv.utils.contract import resolve_split_artifact_paths_for_aug


class PathSigDataset:
    """Load raw path/signature artifacts produced by build_dataset.py."""

    def __init__(self, path: str, *, preferred_aug_type: str | None = None):
        split_dir, ts_path, sig_path = resolve_split_artifact_paths_for_aug(
            path,
            preferred_aug_type=preferred_aug_type,
        )

        sig_data = np.load(sig_path, allow_pickle=True)
        self.sigs = sig_data["sigs"]
        self.sig_depth = int(sig_data["sig_depth"])
        if "sig_channels" in sig_data:
            self.sig_channels = int(sig_data["sig_channels"])
        else:
            import iisignature

            _n, _d = self.sigs.shape[1], self.sig_depth
            for _c in range(2, 20):
                if iisignature.logsiglength(_c, _d) == _n:
                    self.sig_channels = _c
                    break
            else:
                self.sig_channels = 2
        self.sig_type = str(sig_data["sig_type"]) if "sig_type" in sig_data else "logsig"
        self.aug_type = str(sig_data["aug_type"]) if "aug_type" in sig_data else "TA"
        self.split = str(sig_data["split"]) if "split" in sig_data else "full"

        ts_data = np.load(ts_path, allow_pickle=True)
        self.paths = ts_data["paths"]
        self.time_grid = ts_data["time_grid"]
        if "path_dim" in ts_data:
            self.path_dim = int(np.asarray(ts_data["path_dim"]).item())
        elif self.paths.ndim == 3:
            self.path_dim = int(self.paths.shape[-1])
        else:
            self.path_dim = 1
        if "process" in ts_data:
            self.process = str(ts_data["process"])
        else:
            self.process = split_dir.parents[1].name.split("_", 1)[0]

        def _optional_scalar_or_array(key: str):
            if key not in ts_data:
                return None
            raw = ts_data[key]
            return float(raw) if raw.ndim == 0 else raw

        self.theta = _optional_scalar_or_array("theta")
        self.mu = _optional_scalar_or_array("mu")
        self.sigma = _optional_scalar_or_array("sigma")
        self.hurst = _optional_scalar_or_array("hurst")

        self.S0 = float(ts_data["S0"])
        self.T_horizon = float(ts_data["T_horizon"])
        self.n_steps = int(ts_data["n_steps"])
        self.seed_data = int(ts_data["seed"])

    def __len__(self) -> int:
        return self.paths.shape[0]

    def __getitem__(self, idx):
        return self.paths[idx], self.sigs[idx]


class TensorDataset(Dataset):
    """Wrap preprocessed numpy path/sig arrays as a torch Dataset."""

    def __init__(self, paths: np.ndarray, sigs: np.ndarray):
        self.paths = torch.from_numpy(paths).float()
        self.sigs = torch.from_numpy(sigs).float()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx], self.sigs[idx]
