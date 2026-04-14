"""
Shared Evaluation Utilities — Antifraud System v2.0

Provides cross-script utilities for loading meta-val/test partitions, 
executing model-specific predictions, and performing consistent 
VAE Log-Squash normalization.
"""

import numpy as np
import joblib
import yaml
import torch
import torch.nn.functional as F
import structlog
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from src.fd.data.dataset import FastTensorDataLoader
from src.fd.models.base_nets import FraudMLP, FraudVAE
from src.fd.models.pytorch.lstm import FraudLSTM
from src.fd.training.engine import FraudTrainer
from src.fd.training.vae_engine import VAETrainer
from pipelines.train_base_models import safe_load_weights

logger = structlog.get_logger(__name__)

ALL_MODELS = ["lightgbm", "xgboost", "catboost", "mlp", "vae", "lstm", "random_forest"]
TREE_MODELS = {"lightgbm", "xgboost", "catboost", "random_forest"}

def _predict_tree(model_name: str, X_val: np.ndarray) -> np.ndarray:
    model_path = f"models/base/{model_name}_final.joblib"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Missing tree model: {model_path}")
    model = joblib.load(model_path)
    return np.array(model.predict_proba(X_val))[:, 1]

def _predict_pytorch(model_name: str, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    with open(f"configs/models/{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    if model_name == "mlp":
        model = FraudMLP(
            input_dim=cfg["model"]["input_dim"],
            hidden_dims=cfg["model"].get("hidden_dims", [512, 256, 128]),
            dropout=cfg["model"].get("dropout", 0.3),
        )
    elif model_name == "vae":
        model = FraudVAE(
            input_dim=cfg["model"]["input_dim"],
            hidden_dims=cfg["model"].get("hidden_dims", [64, 32]),
            dropout=cfg["model"].get("dropout", 0.2),
        )
    elif model_name == "lstm":
        model = FraudLSTM(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim=cfg["model"].get("hidden_dim", 128),
            num_layers=cfg["model"].get("num_layers", 2),
            dropout=cfg["model"].get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown PyTorch model: {model_name}")

    # Use safe_load to handle torch.compile prefixes
    model_path = f"models/base/{model_name}_final.pt"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Missing PT model: {model_path}")
        
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    safe_load_weights(model, state_dict)
    model.eval()

    loader = FastTensorDataLoader(X_val, y_val, batch_size=16384, shuffle=False)

    if model_name == "vae":
        trainer = VAETrainer(model, torch.optim.AdamW(model.parameters(), lr=1e-3))
    else:
        trainer = FraudTrainer(model, torch.optim.AdamW(model.parameters(), lr=1e-3), torch.nn.BCEWithLogitsLoss())

    return trainer.predict(loader)

def load_meta_test_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Inferences the models on the meta-test partition (raw, unscaled)."""
    preds_dir = Path("data/predictions")
    preds_dir.mkdir(parents=True, exist_ok=True)
    
    val_columns = []
    y_target = None
    
    for model_name in ALL_MODELS:
        cache_path = preds_dir / f"{model_name}_meta_test.npy"
        
        d_type = "tree" if model_name in TREE_MODELS else model_name
        if model_name == "vae": 
            d_type = "mlp"
        
        data = np.load(f"data/processed/X_y_meta_test_{d_type}.npz")
        if y_target is None:
            y_target = data['y']
            
        if cache_path.exists():
            preds = np.load(cache_path)
            logger.info(f"Loaded cached test predictions", model=model_name)
        else:
            logger.info(f"Inferencing on meta_test...", model=model_name)
            if model_name in TREE_MODELS:
                preds = _predict_tree(model_name, data['X'])
            else:
                preds = _predict_pytorch(model_name, data['X'], data['y'])
            np.save(cache_path, preds)
            
        val_columns.append(preds)
        
    return np.column_stack(val_columns), y_target

def load_meta_train_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Loads OOF predictions (raw, unscaled)."""
    oof_columns = []
    for model in ALL_MODELS:
        oof_path = Path(f"data/predictions/{model}_oof.npy")
        if not oof_path.exists():
            raise FileNotFoundError(f"Missing OOF for {model}. Run train_base_models.py first.")
        oof_columns.append(np.load(oof_path))
        
    n_rows = oof_columns[0].shape[0]
    dev_tree_path = "data/processed/X_y_dev_tree.npz"
    if not Path(dev_tree_path).exists():
        raise FileNotFoundError(f"Missing dev tree data: {dev_tree_path}")
        
    dev_data = np.load(dev_tree_path)
    y_full = dev_data["y"]
    y_meta_train = y_full[-n_rows:]
    
    return np.column_stack(oof_columns), y_meta_train

def normalize_meta_vae(X_matrix: np.ndarray, scaler: MinMaxScaler = None, fit: bool = False) -> tuple[np.ndarray, MinMaxScaler]:
    """Applies Log-Squash (log1p) and MinMax Scaling to the VAE column."""
    vae_idx = ALL_MODELS.index("vae")
    vae_log = np.log1p(X_matrix[:, vae_idx]).reshape(-1, 1)
    
    if fit:
        scaler = MinMaxScaler()
        X_matrix[:, vae_idx] = scaler.fit_transform(vae_log).flatten()
    else:
        if scaler is None:
            raise ValueError("Must provide a fitted scaler if not fitting.")
        X_matrix[:, vae_idx] = scaler.transform(vae_log).flatten()
        
    return X_matrix, scaler
