"""
PyTorch Training Engine.

This module handles the core training loops, evaluation, and Out-Of-Fold (OOF) 
prediction generation. It utilizes Automatic Mixed Precision (AMP) to maximize 
RTX 5060 Tensor Core performance.
"""

import torch
import torch.cuda.amp as amp  # Stable namespace for AMP components
import structlog
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

logger = structlog.get_logger(__name__)

class FraudTrainer:
    """
    Generic training engine for PyTorch models.
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        
        # Using the standard CUDA GradScaler
        self.scaler = amp.GradScaler()

    def train_epoch(self, loader: DataLoader) -> float:
        """Trains the model for one epoch and returns the average loss."""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Context for Tensor Core acceleration
            with amp.autocast():
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

            # Scale gradients and backpropagate
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Evaluates the model and returns (Average Loss, ROC-AUC Score)."""
        self.model.eval()
        total_loss = 0.0
        
        all_preds = []
        all_targets = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            with amp.autocast():
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

            total_loss += float(loss.item())
            
            # Convert logits to probabilities using Sigmoid for AUC calculation
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(loader)
        
        # Calculate AUC (Explicitly cast to float to satisfy type checkers)
        try:
            auc = float(roc_auc_score(all_targets, all_preds))
        except ValueError:
            auc = 0.5 # Fallback if only one class is present in the batch

        return avg_loss, auc

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        """
        Generates Out-Of-Fold (OOF) probabilities for the Stacker model.
        Returns a 1D numpy array of probabilities.
        """
        self.model.eval()
        all_preds = []

        for X_batch, _ in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            with amp.autocast():
                logits = self.model(X_batch)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)

        return np.array(all_preds, dtype=np.float32).flatten()
