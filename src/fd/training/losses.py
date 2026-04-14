"""
Custom Loss Functions for Imbalanced Fraud Detection.

Provides Focal Loss for advanced gradient weighting and Weighted BCE 
for baseline comparisons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Balanced Focal Loss for handling extreme fraud imbalance.
    Formula: FL = -alpha_t * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Focal Loss for binary classification.

        Args:
            inputs (torch.Tensor): Logits predicted by the model, shape (N, 1) or (N,).
            targets (torch.Tensor): Binary targets, shape (N,) or (N, 1).

        Returns:
            torch.Tensor: The computed Focal Loss. Reduction is controlled by ``self.reduction``.
        """
        targets = targets.view(-1, 1).float()
        inputs = inputs.view(-1, 1)

        # Standard BCE per sample
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Probability of the true class
        pt = torch.exp(-ce_loss)
        
        # alpha_t = alpha for fraud (1), (1-alpha) for normal (0)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss

class WeightedBCE(nn.Module):
    """
    Standard Binary Cross Entropy with a fixed multiplier for the fraud class.
    """
    # Type hint to satisfy linters, actual value set in __init__
    pos_weight: torch.Tensor 

    def __init__(self, pos_weight: float = 32.0):
        super(WeightedBCE, self).__init__()
        # Buffer ensures the tensor moves to GPU automatically
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE from logits and binary targets.

        Args:
            inputs (torch.Tensor): Logits predicted by the model, shape (N, 1) or (N,).
            targets (torch.Tensor): Binary targets, shape (N,) or (N, 1).

        Returns:
            torch.Tensor: Weighted binary cross entropy loss.
        """
        return F.binary_cross_entropy_with_logits(
            inputs, 
            targets.view(-1, 1).float(), 
            pos_weight=self.pos_weight
        )



"""
Custom Loss Functions for Imbalanced Fraud Detection.
Default gamma = 2.0 for Focal Loss, alpha = 0.8 for class weighting.
We want to keep alpha < true imbalance (0.97) to avoid overfitting, 
gamma makes up for the rest of the imbalance handling.

Btw maybe it would make sense to define alpha and gamma in configs 
rather than as defaults in init.
"""
# from math import log

# class FocalLoss:
#     def __init__(self, a: float = 0.97, y: int = 2) -> None:
#         self._a = a
#         self._y = y

#     def loss(self, P, Y):  # P is an array of predictions for a batch, Y is the real values
#         total = 0
#         N = len(P)
#         for i, p in enumerate(P):
#             if Y[i] == 1:  # fraud
#                 l = self._a * (1-p)**self._y * log(p)
#             elif Y[i] == 0:  # normal 
#                 l = (1-self._a) * p**self._y * log(1-p)
#             else:
#                 raise RuntimeError("Class neither 1 nor 0.")
#             total += l

#         return -total/N
