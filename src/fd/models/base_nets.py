"""
Base Neural Network Architectures for Antifraud System v2.0.

This module contains the 2D tabular models:
1. FraudMLP: A standard deep feed-forward network for supervised classification.
2. FraudVAE: An unsupervised Variational Autoencoder for anomaly detection.
"""

import torch
import torch.nn as nn

class FraudMLP(nn.Module):
    """
    Standard Multi-Layer Perceptron for Fraud Detection.
    Uses BatchNorm and Dropout for robust learning on tabular data.
    """
    def __init__(
        self, 
        input_dim: int = 143, 
        hidden_dims: list[int] = [512, 256, 128], 
        dropout: float = 0.3,
        use_batchnorm: bool = True
    ):
        super().__init__()
        
        layers = []
        in_features = input_dim
        
        # Dynamically build the hidden layers based on the config
        for out_features in hidden_dims:
            layers.append(nn.Linear(in_features, out_features))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_features))
                
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            in_features = out_features
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final classification head (Output is 1 logit)
        # Note: No Sigmoid here! The Engine's Criterion applies it internally.
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)


class FraudVAE(nn.Module):
    """
    Unsupervised Variational Autoencoder.
    Learns the distribution of 'Normal' transactions to flag anomalies.
    """
    def __init__(
        self, 
        input_dim: int = 143, 
        hidden_dims: list[int] = [64, 32], 
        latent_dim: int = 16,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # 1. ENCODER
        encoder_layers = []
        in_features = input_dim
        for out_features in hidden_dims:
            encoder_layers.append(nn.Linear(in_features, out_features))
            encoder_layers.append(nn.BatchNorm1d(out_features))
            encoder_layers.append(nn.Mish())
            encoder_layers.append(nn.Dropout(dropout))
            in_features = out_features
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 2. LATENT SPACE (The Reparameterization Trick)
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)
        
        # 3. DECODER
        decoder_layers = []
        in_features = latent_dim
        # Reverse the hidden dimensions for the decoder
        for out_features in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_features, out_features))
            decoder_layers.append(nn.BatchNorm1d(out_features))
            decoder_layers.append(nn.Mish())
            decoder_layers.append(nn.Dropout(dropout))
            in_features = out_features
            
        # Final layer reconstructs back to 143
        decoder_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Samples from the latent distribution during training."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu # Skip sampling during inference (evaluation/predicting)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = torch.clamp(self.fc_logvar(encoded), min=-10.0, max=10.0)
        
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar
