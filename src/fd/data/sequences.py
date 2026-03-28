"""
DataLoader Factory for Fraud Detection.

This module initializes PyTorch DataLoaders for both 2D and 3D datasets.
It handles batching, shuffling, and optimizing memory transfer (pin_memory)
to keep the GPU saturated without bottlenecking the CPU.
"""

import torch
from torch.utils.data import DataLoader, Dataset

def create_dataloaders(
    train_ds: Dataset, 
    val_ds: Dataset, 
    batch_size: int = 1024, 
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader]:
    """
    Creates optimized PyTorch DataLoaders for training and validation.

    Args:
        train_ds (Dataset): The training dataset.
        val_ds (Dataset): The validation dataset.
        batch_size (int): Number of samples per batch. 1024 is standard for tabular.
        num_workers (int): Number of CPU subprocesses for data loading. 
                           (Rule of thumb: 4 is usually optimal for a 16GB/8-core machine).

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # Essential for training to prevent cyclical bias
        num_workers=num_workers,
        pin_memory=True,       # Dramatically speeds up CPU-to-GPU memory transfer
        drop_last=True         # Prevents batchnorm crash on small final batches
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,         # NEVER shuffle validation data (messes up OOF tracking)
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader

def create_test_loader(
    test_ds: Dataset, 
    batch_size: int = 1024, 
    num_workers: int = 4
) -> DataLoader:
    """Creates a DataLoader specifically for Out-Of-Fold (OOF) or Final Test predictions."""
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
