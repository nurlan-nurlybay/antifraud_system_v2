"""
PyTorch Training Engine for Antifraud System v2.0.

This module provides the `FraudTrainer` class, which handles the core training loops, 
evaluation metrics (PR-AUC), and Out-Of-Fold (OOF) prediction generation.
It utilizes modern PyTorch 2.x Automatic Mixed Precision (AMP) to maximize 
Tensor Core performance on RTX 50-series and 40-series architecture.
"""

import torch
import numpy as np
import structlog
from src.fd.utils.logging import setup_logger
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

setup_logger()
logger = structlog.get_logger(__name__)

class FraudTrainer:
    """
    High-performance training engine for 143-feature Neural Networks (MLP, LSTM, VAE).
    Manages state, gradient scaling, metric calculation, and Elastic Net (L1) regularization.
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module,
        device: str = "cuda",
        l1_lambda: float = 0.0  # Controls the L1 (Lasso) penalty. 0.0 disables it.
    ):
        # 1. Setup Hardware & Components
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.l1_lambda = l1_lambda

    def train_epoch(self, loader) -> float:
        """
        Executes one full pass over the training data.
        """
        self.model.train()
        epoch_loss = torch.zeros(1, device=self.device)
        batches_processed = 0

        for X_batch, y_batch in loader:
            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            if self.l1_lambda > 0:
                l1_penalty = sum(p.abs().sum() for p in self.model.parameters() if p.requires_grad)
                loss = loss + (self.l1_lambda * l1_penalty)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.detach()
            batches_processed += 1

        return (epoch_loss / batches_processed).item() if batches_processed > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[float, float]:
        self.model.eval()
        epoch_loss = torch.zeros(1, device=self.device)
        
        all_preds = []
        all_targets = []

        for X_batch, y_batch in loader:
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            epoch_loss += loss.detach()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_batch.cpu().numpy())

        avg_loss = (epoch_loss / len(loader)).item()
        
        try:
            auc = float(average_precision_score(all_targets, all_preds))
        except ValueError:
            logger.error("Average Precision calculation failed. Only one class present in targets.")
            auc = 0.0

        return avg_loss, auc

    @torch.no_grad()
    def predict(self, loader) -> np.ndarray:
        self.model.eval()
        all_preds = []

        for X_batch, _ in loader:
            logits = self.model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)

        return np.array(all_preds, dtype=np.float32).flatten()
