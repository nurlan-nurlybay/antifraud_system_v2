"""
PyTorch Dataset Definitions for Fraud Detection.

This module provides custom Dataset classes to handle 2D (MLP) and 3D (LSTM) 
preprocessed features stored in .npz format.
"""

from typing import Optional
import torch
import numpy as np
from torch.utils.data import Dataset

class FraudDataset(Dataset):
    """
    A general-purpose Dataset for 2D tabular data (MLP path).
    """
    def __init__(self, x_path: str):
        """
        Args:
            x_path (str): Path to the .npz file containing 'X' (features).
            y_path (str): Path to the .npz file containing 'y' (labels).
        """
        data = np.load(x_path)
        self.X = torch.from_numpy(data['X']).float()
        
        if 'y' in data:
            self.y = torch.from_numpy(data['y']).float().view(-1, 1)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_sample = self.X[idx]
        y_sample = self.y[idx] if self.y is not None else torch.tensor([-1.0])
        return x_sample, y_sample

class SequenceFraudDataset(Dataset):
    """
    A specialized Dataset for 3D sequence data (LSTM path).
    Input Shape: [Samples, TimeSteps, Features]
    """
    def __init__(self, npz_path: str):
        """
        Args:
            npz_path (str): Path to the compressed .npz file.
        """
        data = np.load(npz_path)
        # Convert to float32 tensors for the RTX 5060
        self.X = torch.from_numpy(data['X']).float()
        self.y = torch.from_numpy(data['y']).float().view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns a 3D sequence for a single user/UID
        return self.X[idx], self.y[idx]
