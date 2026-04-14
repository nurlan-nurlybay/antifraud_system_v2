"""
PyTorch VAE Training Engine for Antifraud System v2.0.

Dedicated engine for unsupervised anomaly detection. Calculates 
reconstruction errors to serve as OOF anomaly scores for the Stacker.
"""

import torch
import torch.nn.functional as F
import numpy as np
import structlog
from src.fd.utils.logging import setup_logger
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

setup_logger()
logger = structlog.get_logger(__name__)

class VAETrainer:
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        device: str = "cuda",
        kld_weight: float = 0.01  # Tuned by Optuna (Beta-VAE)
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.kld_weight = kld_weight

    def loss_function(self, recon_x, x, mu, logvar):
        """Calculates MSE + KLD."""
        # Mean Squared Error across the 143 features
        recon_loss = F.mse_loss(recon_x, x, reduction='none').mean(dim=1)
        
        # KL Divergence formula
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # We return the batch mean for backprop, and the raw recon_loss for scoring
        total_batch_loss = (recon_loss + self.kld_weight * kld_loss).mean()
        return total_batch_loss, recon_loss

    def train_epoch(self, loader) -> float:
        self.model.train()
        epoch_loss = torch.zeros(1, device=self.device)
        batches_processed = 0

        for X_batch, _ in loader: 
            self.optimizer.zero_grad(set_to_none=True)

            recon_batch, mu, logvar = self.model(X_batch)
            loss, _ = self.loss_function(recon_batch, X_batch, mu, logvar)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.detach()
            batches_processed += 1

        return (epoch_loss / batches_processed).item() if batches_processed > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[float, float]:
        """Returns (Average VAE Loss, PR-AUC of the Anomaly Score)"""
        self.model.eval()
        epoch_loss = torch.zeros(1, device=self.device)
        
        all_anomaly_scores = []
        all_targets = []

        for X_batch, y_batch in loader:
            recon_batch, mu, logvar = self.model(X_batch)
            loss, recon_errors = self.loss_function(recon_batch, X_batch, mu, logvar)

            epoch_loss += loss.detach()
            
            all_anomaly_scores.extend(recon_errors.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        avg_loss = (epoch_loss / len(loader)).item()
        
        try:
            auc = float(average_precision_score(all_targets, all_anomaly_scores))
        except ValueError:
            logger.error("Average Precision calculation failed.")
            auc = 0.0

        return avg_loss, auc

    @torch.no_grad()
    def predict(self, loader) -> np.ndarray:
        """
        Generates the Unsupervised Anomaly Scores for the Stacker.
        """
        self.model.eval()
        all_anomaly_scores = []

        for X_batch, _ in loader:
            recon_batch, mu, logvar = self.model(X_batch)
            _, recon_errors = self.loss_function(recon_batch, X_batch, mu, logvar)
            
            all_anomaly_scores.extend(recon_errors.cpu().numpy())

        return np.array(all_anomaly_scores, dtype=np.float32).flatten()
