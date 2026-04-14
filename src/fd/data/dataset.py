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
    Accepts either a .npz file path or pre-sliced in-memory numpy arrays
    for Walk-Forward chronological CV.
    """
    def __init__(self, path: str | None = None, X: np.ndarray | None = None, y: np.ndarray | None = None):
        if X is not None and y is not None:
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
        elif path is not None:
            with np.load(path) as data:
                # Cast to float32 once at load time to save GPU cycles during training
                self.X = data['X'].astype(np.float32)
                self.y = data['y'].astype(np.float32)
        else:
            raise ValueError("Must provide either 'path' or both 'X' and 'y' arrays.")

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class SequenceFraudDataset(Dataset):
    """
    Specialized Dataset for 3D sequence data (LSTM path).
    Input Shape: [Samples, TimeSteps, Features] -> [N, 5, 143]
    Accepts either a .npz file path or pre-sliced in-memory numpy arrays.
    """
    def __init__(self, path: str | None = None, X: np.ndarray | None = None, y: np.ndarray | None = None):
        if X is not None and y is not None:
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
        elif path is not None:
            with np.load(path) as data:
                self.X = data['X'].astype(np.float32)
                self.y = data['y'].astype(np.float32)
        else:
            raise ValueError("Must provide either 'path' or both 'X' and 'y' arrays.")

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class FastTensorDataLoader:
    """
    A unified DataLoader-like object for VRAM-resident datasets. 
    Bypasses PyTorch's native CPU Multiprocessing logic fully.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 2048, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Immediate mapping to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move entire dataset to GPU once
        self.X = torch.tensor(X.astype(np.float32), device=device)
        self.y = torch.tensor(y.astype(np.float32), device=device).unsqueeze(1)
        self.dataset_len = self.X.shape[0]
        self.indices = None
        self.i = 0
        
    def __iter__(self):
        if self.shuffle:
            # Shuffle the entire dataset once per epoch on GPU
            perm = torch.randperm(self.dataset_len, device=self.X.device)
            self.X = self.X[perm]
            self.y = self.y[perm]
        self.i = 0
        return self
        
    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
            
        end = min(self.i + self.batch_size, self.dataset_len)
        # Slicing creates a 'view' on GPU memory (zero-copy)
        b_X = self.X[self.i:end]
        b_y = self.y[self.i:end]
            
        self.i = end
        return b_X, b_y

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


