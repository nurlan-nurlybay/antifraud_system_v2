"""
DataLoader Factory for Fraud Detection.

This module initializes PyTorch DataLoaders for the 3-split Nested Walk-Forward CV.
It optimizes memory transfer (pin_memory) and parallelizes data fetching 
to maximize GPU utilization on the RTX 5060.
"""

from torch.utils.data import DataLoader, Dataset

def get_loaders(
    train_ds: Dataset, 
    val_ds: Dataset, 
    test_ds: Dataset, 
    batch_size: int = 1024, 
    num_workers: int = 8
) -> dict[str, DataLoader]:
    """
    Creates a dictionary of optimized DataLoaders for a single CV fold.

    Args:
        train_ds: Dataset for model weight updates.
        val_ds: Dataset for hyperparameter tuning (Optuna) and Early Stopping.
        test_ds: Dataset for generating Out-Of-Fold (OOF) predictions for the Stacker.
        batch_size: Samples per batch (1024 is optimal for 16GB RAM/RTX 5060).
        num_workers: CPU cores dedicated to pre-fetching data.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    
    # 1. Train Loader: The "Gym"
    # Essential: shuffle=True to break chronological bias within the fold.
    # Essential: drop_last=True to keep gradients stable (prevents tiny final batches).
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True  
    )

    # 2. Validation Loader: The "Judge"
    # Used by Optuna to decide if this 'Trial' is a winner.
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False
    )

    # 3. Test Loader: The "Source of Truth"
    # Used to create the unbiased predictions that will train the Meta-Model (Stacker).
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
