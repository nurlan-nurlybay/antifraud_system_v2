"""
PyTorch Dataset Definitions for Fraud Detection.

This module provides custom Dataset classes to handle 2D (MLP/VAE) and 3D (LSTM) 
preprocessed features stored in .npz format.
"""

import torch
import numpy as np
from torch.utils.data import Dataset

class FraudDataset(Dataset):
    """
    General-purpose Dataset for 2D tabular data (MLP/VAE paths).
    """
    def __init__(self, path: str):
        with np.load(path) as data:
            # Cast to float32 once at load time to save GPU cycles during training
            self.X = data['X'].astype(np.float32)
            self.y = data['y'].astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # returns (141,) and (1,)
        x_tensor = torch.from_numpy(self.X[idx])
        y_tensor = torch.from_numpy(np.array([self.y[idx]])) 
        return x_tensor, y_tensor

class SequenceFraudDataset(Dataset):
    """
    Specialized Dataset for 3D sequence data (LSTM path).
    Input Shape: [Samples, TimeSteps, Features] -> [N, 5, 141]
    """
    def __init__(self, path: str):
        with np.load(path) as data:
            self.X = data['X'].astype(np.float32)
            self.y = data['y'].astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # returns (5, 141) and (1,)
        x_tensor = torch.from_numpy(self.X[idx])
        y_tensor = torch.from_numpy(np.array([self.y[idx]]))
        return x_tensor, y_tensor