"""
Sequence Neural Network Architectures for Antifraud System v2.0.

This module contains the 3D sequence model:
1. FraudLSTM: A multi-layer LSTM equipped with Additive (Bahdanau) Attention 
   to isolate the most suspicious transaction in a user's chronological window.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Additive Attention mechanism.
    Evaluates all timesteps and creates a weighted context vector.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # W and V are the learnable weight matrices that calculate the "importance" score
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # lstm_outputs shape: [Batch, SeqLen (5), HiddenDim]
        
        # 1. Calculate alignment scores
        u = torch.tanh(self.W(lstm_outputs))       # [Batch, SeqLen, HiddenDim]
        scores = self.V(u)                         # [Batch, SeqLen, 1]
        
        # 2. Normalize scores into probabilities (alphas) that sum to 1.0
        alphas = F.softmax(scores, dim=1)          # [Batch, SeqLen, 1]

        # 3. Multiply the original LSTM outputs by their importance weights
        # and sum them up to create the final Context Vector
        context = torch.sum(lstm_outputs * alphas, dim=1) # [Batch, HiddenDim]
        
        return context, alphas


class FraudLSTM(nn.Module):
    """
    LSTM network with temporal attention for chronological fraud sequences.
    """
    def __init__(
        self,
        input_dim: int = 143,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()
        
        # 1. The Sequence Engine
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Critical: tells PyTorch the batch is dimension 0
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Handle dimension doubling if bidirectional is ever turned to True
        self.direction_factor = 2 if bidirectional else 1
        attn_dim = hidden_dim * self.direction_factor

        # 2. The Temporal Attention Layer
        self.attention = BahdanauAttention(attn_dim)
        
        # 3. The Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape expects: [Batch, 5, 143]
        
        # lstm_out contains the hidden states for ALL 5 timesteps
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # The attention layer decides which of the 5 timesteps mattered the most
        context, alphas = self.attention(lstm_out)
        
        return self.classifier(context)
