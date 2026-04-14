"""
Consolidated Base Model Tuning & Training Pipeline.
Antifraud System v2.0

Performs walk-forward temporal cross-validation with Optuna hyperparameter
tuning, then generates OOF predictions and trains final models. Supports
all 7 base models (4 trees + MLP + VAE + LSTM).

Usage:
    PYTHONPATH=. python pipelines/train_base_models.py --models lightgbm mlp
    PYTHONPATH=. python pipelines/train_base_models.py  # runs all 7 models
    PYTHONPATH=. python pipelines/train_base_models.py --skip-tune --models lstm
"""

import argparse
import sys
import os
import yaml
import joblib
import structlog
import logging
import warnings
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import torch
import time
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

import mlflow
import mlflow.sklearn
import mlflow.pytorch

from src.fd.utils.config import load_config
from src.fd.utils.logging import setup_logger
from src.fd.data.dataset import FraudDataset, SequenceFraudDataset, FastTensorDataLoader
from src.fd.training.engine import FraudTrainer
from src.fd.training.vae_engine import VAETrainer
from src.fd.training.losses import FocalLoss
from src.fd.models.base_nets import FraudMLP, FraudVAE
from src.fd.models.pytorch.lstm import FraudLSTM
from src.fd.models.sklearn.wrappers import TreeWrapper

# Logger setup will happen inside the per-model loop.
logger = structlog.get_logger(__name__)

ALL_MODELS = ["lightgbm", "xgboost", "catboost", "mlp", "vae", "lstm", "random_forest"]
TREE_MODELS = {"lightgbm", "xgboost", "catboost", "random_forest"}

# ----------------------------------------------------------------------------
# Walk-Forward Splits (must be identical everywhere)
# ----------------------------------------------------------------------------
def get_walk_forward_splits(total_length: int):
    """
    Assuming the data is 90% of the total, we split it into 9 chunks.
    Fold 1: Train 0-40%, Val 40-50%, Test 50-60%
    Fold 4: Train 0-70%, Val 70-80%, Test 80-90%+remainder
    """
    chunk_size = total_length // 9
    folds = []
    for train_chunks in [4, 5, 6, 7]:
        train_end = train_chunks * chunk_size
        val_end = train_end + chunk_size
        test_end = val_end + chunk_size
        if train_chunks == 7:
            test_end = total_length
        folds.append((0, train_end, val_end, test_end))
    return folds

def safe_load_weights(model: torch.nn.Module, state_dict: dict):
    """
    Safely loads a state_dict into a model, handling the '_orig_mod.' prefix 
    added by torch.compile() if necessary.
    """
    model_keys = list(model.state_dict().keys())
    has_orig_mod_model = any(k.startswith("_orig_mod.") for k in model_keys)
    has_orig_mod_dict = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    
    final_state_dict = {}
    if has_orig_mod_model and not has_orig_mod_dict:
        for k, v in state_dict.items():
            final_state_dict["_orig_mod." + k] = v
    elif not has_orig_mod_model and has_orig_mod_dict:
        for k, v in state_dict.items():
            final_state_dict[k.replace("_orig_mod.", "")] = v
    else:
        final_state_dict = state_dict
        
    model.load_state_dict(final_state_dict)

# ----------------------------------------------------------------------------
# Configuration Updaters
# ----------------------------------------------------------------------------
def _reconstruct_hidden_dims(best_params: dict) -> list[int] | None:
    """Reconstructs hidden_dims list from Optuna's n_units_l0..lN keys."""
    n_layers = best_params.get('n_layers')
    if n_layers is not None:
        return [best_params[f'n_units_l{i}'] for i in range(n_layers) if f'n_units_l{i}' in best_params]
        
    vae_n_layers = best_params.get('vae_n_layers')
    if vae_n_layers is not None:
        return [best_params[f'vae_l{i}'] for i in range(vae_n_layers) if f'vae_l{i}' in best_params]
        
    return None

def update_yaml_config(model: str, best_params: dict):
    """Writes Optuna's best params back into the correct YAML config file."""
    config_path = "configs/models/trees.yaml" if model in TREE_MODELS else f"configs/models/{model}.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if model in TREE_MODELS:
        for k, v in best_params.items():
            # CatBoost uses 'depth' not 'max_depth'
            cfg[model]['params'][k] = round(v, 6) if isinstance(v, float) else v
    else:
        loss_keys = {"alpha", "gamma"}
        train_keys = {"lr": "learning_rate", "l1_lambda": "l1_lambda", "kld_weight": "kld_weight", "weight_decay": "weight_decay"}
        model_keys = {"dropout", "hidden_dim", "num_layers", "latent_dim"}
        
        for k, v in best_params.items():
            if k in loss_keys:
                cfg.setdefault('loss_params', {}).setdefault('focal', {})[k] = round(v, 4)
            elif k in train_keys:
                cfg['training'][train_keys[k]] = round(v, 8) if isinstance(v, float) else v
            elif k in model_keys:
                cfg.setdefault('model', {})[k] = round(v, 4) if isinstance(v, float) else v
                
        hidden_dims = _reconstruct_hidden_dims(best_params)
        if hidden_dims is not None:
            cfg.setdefault('model', {})['hidden_dims'] = hidden_dims

    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    
    logger.info("config_updated", model=model, path=config_path, params=best_params)

# ----------------------------------------------------------------------------
# Model Builders
# ----------------------------------------------------------------------------
def build_pytorch_model(model_name: str, cfg: dict):
    """Instantiates a PyTorch model from its YAML config."""
    if model_name == "mlp":
        return FraudMLP(
            input_dim=cfg["model"]["input_dim"],
            hidden_dims=cfg["model"]["hidden_dims"],
            dropout=cfg["model"]["dropout"],
        )
    elif model_name == "vae":
        return FraudVAE(
            input_dim=cfg["model"]["input_dim"],
            hidden_dims=cfg["model"]["hidden_dims"],
            dropout=cfg["model"]["dropout"],
        )
    elif model_name == "lstm":
        return FraudLSTM(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            dropout=cfg["model"]["dropout"],
        )
    raise ValueError(f"Unknown model: {model_name}")

def build_loss(model_name: str, cfg: dict) -> torch.nn.Module:
    """Builds the correct loss function from config, including FocalLoss."""
    if model_name == "vae":
        return torch.nn.MSELoss()  # placeholder — VAETrainer has its own loss
    
    loss_cfg = cfg.get("loss_params", {})
    loss_type = loss_cfg.get("type", "focal")
    
    if loss_type == "focal":
        focal = loss_cfg.get("focal", {})
        return FocalLoss(
            alpha=focal.get("alpha", 0.5),
            gamma=focal.get("gamma", 2.0),
        )
    else:
        pos_weight = loss_cfg.get("wbce", {}).get("pos_weight", 32.0)
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

def build_trainer(model_name: str, pt_model, optimizer, cfg: dict):
    """Builds FraudTrainer or VAETrainer with all config params injected."""
    if model_name == "vae":
        kld_weight = cfg["training"].get("kld_weight", 0.01)
        return VAETrainer(pt_model, optimizer, kld_weight=kld_weight)
    else:
        criterion = build_loss(model_name, cfg)
        l1_lambda = cfg["training"].get("l1_lambda", 0.0)
        return FraudTrainer(pt_model, optimizer, criterion, l1_lambda=l1_lambda)

# ----------------------------------------------------------------------------
# Optuna Tuning
# ----------------------------------------------------------------------------
def run_tuning(model_name: str, X: np.ndarray, y: np.ndarray, folds: list) -> dict:
    """Walk-Forward CV Optuna Tuning with proper loss functions and early stopping."""
    logger.info("optuna_tuning_started", model=model_name)
    
    n_trials = 120 if model_name in TREE_MODELS else 300
    if model_name in ["vae", "lstm"]:
        n_trials = 100
    if model_name == "random_forest":
        n_trials = 30
    
    # Tuning epochs: enough for signal, not full training
    tune_epochs = 15 if model_name != "vae" else 10

    def objective(trial):
        if model_name in TREE_MODELS:
            # --- Tree hyperparameters ---
            params = {}
            if model_name == "lightgbm":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
                }
            elif model_name == "xgboost":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                }
            elif model_name == "catboost":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                }
            elif model_name == "random_forest":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "max_features": trial.suggest_float("max_features", 0.3, 1.0),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                }
            
            fold_aucs = []
            for t_start, t_end, v_end, test_end in folds:
                wrapper = TreeWrapper(model_type=model_name, params=params)
                wrapper.fit(X[t_start:t_end], y[t_start:t_end], X[t_end:v_end], y[t_end:v_end])
                preds = wrapper.predict_proba(X[t_end:v_end])
                fold_aucs.append(average_precision_score(y[t_end:v_end], preds))
            return float(np.mean(fold_aucs))
            
        else:
            # --- PyTorch hyperparameters ---
            params = {
                "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            }
            # MLP gets L1 and Focal Loss. VAE/LSTM do NOT get L1.
            if model_name == "mlp":
                params["alpha"] = trial.suggest_float("alpha", 0.25, 0.95)
                params["gamma"] = trial.suggest_float("gamma", 1.0, 3.0)
                params["l1_lambda"] = trial.suggest_float("l1_lambda", 1e-7, 1e-4, log=True)
                
                n_layers = trial.suggest_int("n_layers", 1, 5)
                for i in range(n_layers):
                    params[f"n_units_l{i}"] = trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128, 256, 512, 1024, 2048])
                    
            # LSTM gets Focal Loss, but NO L1.
            if model_name == "lstm":
                params["alpha"] = trial.suggest_float("alpha", 0.25, 0.95)
                params["gamma"] = trial.suggest_float("gamma", 1.0, 3.0)
                
                params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
                params["num_layers"] = trial.suggest_int("num_layers", 2, 3)

            # VAE and LSTM get Tuned L2 (Weight Decay)
            if model_name in ["vae", "lstm"]:
                params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
                
            if model_name == "vae":
                params["kld_weight"] = trial.suggest_float("kld_weight", 1e-4, 1e-2, log=True)
                params["latent_dim"] = trial.suggest_categorical("latent_dim", [8, 16, 32])
                
                n_layers = trial.suggest_int("vae_n_layers", 1, 2)
                for i in range(n_layers):
                    params[f"vae_l{i}"] = trial.suggest_categorical(f"vae_l{i}", [128, 256, 512])
            
            # Build a temporary config dict for this trial
            cfg_file = f"configs/models/{model_name}.yaml"
            cfg = load_config(cfg_file)
            
            # Inject trial params into a working copy
            if "alpha" in params:
                cfg.setdefault("loss_params", {}).setdefault("focal", {})["alpha"] = params["alpha"]
                cfg["loss_params"]["focal"]["gamma"] = params["gamma"]
            cfg["training"]["l1_lambda"] = params.get("l1_lambda", 0.0)
            cfg["training"]["kld_weight"] = params.get("kld_weight", cfg["training"].get("kld_weight", 0.01))
            if "weight_decay" in params:
                cfg["training"]["weight_decay"] = params["weight_decay"]
            cfg["model"]["dropout"] = params["dropout"]
            if "latent_dim" in params:
                cfg["model"]["latent_dim"] = params["latent_dim"]
            
            hidden_dims = _reconstruct_hidden_dims(params)
            if hidden_dims is not None:
                cfg["model"]["hidden_dims"] = hidden_dims
            if "hidden_dim" in params:
                cfg["model"]["hidden_dim"] = params["hidden_dim"]
            if "num_layers" in params:
                cfg["model"]["num_layers"] = params["num_layers"]
            
            DatasetClass = SequenceFraudDataset if model_name == "lstm" else FraudDataset
            
            fold_aucs = []
            for t_start, t_end, v_end, test_end in folds:
                X_train_fold, y_train_fold = X[t_start:t_end], y[t_start:t_end]
                
                if model_name == "vae":
                    # Pre-filter normal transactions once per fold to save thousands of batch masks
                    normal_mask = (y_train_fold.flatten() == 0)
                    X_train_final = X_train_fold[normal_mask]
                    y_train_final = y_train_fold[normal_mask]
                else:
                    X_train_final, y_train_final = X_train_fold, y_train_fold

                # Massive batch size (16k) minimizes Python loop overhead
                train_dl = FastTensorDataLoader(X_train_final, y_train_final, batch_size=16384, shuffle=True)
                val_dl = FastTensorDataLoader(X[t_end:v_end], y[t_end:v_end], batch_size=16384, shuffle=False)
                
                pt_model = build_pytorch_model(model_name, cfg)
                # Fuses kernels for maximum throughput on 5090
                try:
                    pt_model = torch.compile(pt_model)
                    logger.info("Model compiled successfully")
                except Exception:
                    logger.warning("torch.compile failed, continuing with eager mode")
                
                optimizer = torch.optim.AdamW(pt_model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 1e-5))
                trainer = build_trainer(model_name, pt_model, optimizer, cfg)
                
                # Train for a short horizon to get a tuning signal
                for _ in range(tune_epochs):
                    trainer.train_epoch(train_dl)
                
                if model_name == "vae":
                    # VAE: evaluate anomaly detection quality on validation
                    val_preds = trainer.predict(val_dl)
                    fold_aucs.append(average_precision_score(y[t_end:v_end], val_preds))
                else:
                    preds = trainer.predict(val_dl)
                    fold_aucs.append(average_precision_score(y[t_end:v_end], preds))
                    
            return float(np.mean(fold_aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info("optuna_tuning_complete", best_auc=study.best_value, params=study.best_params)
    print(f"Params: {study.best_params}")  # For backward compat with parse_best_params
    return study.best_params

# ----------------------------------------------------------------------------
# OOF Generation & Final Training
# ----------------------------------------------------------------------------
def generate_oof_and_train(model_name: str, X: np.ndarray, y: np.ndarray, folds: list) -> float:
    """Generates Walk-Forward OOF predictions with early stopping and trains the final model."""
    logger.info("training_started", model=model_name)
    
    cfg_file = "trees.yaml" if model_name in TREE_MODELS else f"{model_name}.yaml"
    cfg = load_config(f"configs/models/{cfg_file}")
    
    # Collect OOF predictions from test folds only
    all_oof_preds = []
    all_oof_labels = []
    
    for fold_idx, (t_start, t_end, v_end, test_end) in enumerate(folds):
        logger.info(f"  Fold {fold_idx+1}/{len(folds)}: train=[0:{t_end}] val=[{t_end}:{v_end}] test=[{v_end}:{test_end}]")
        
        if model_name in TREE_MODELS:
            params = cfg[model_name]["params"]
            wrapper = TreeWrapper(model_type=model_name, params=params)
            wrapper.fit(X[t_start:t_end], y[t_start:t_end], X[t_end:v_end], y[t_end:v_end])
            fold_preds = wrapper.predict_proba(X[v_end:test_end])
        else:
            DatasetClass = SequenceFraudDataset if model_name == "lstm" else FraudDataset
            
            X_train_fold, y_train_fold = X[t_start:t_end], y[t_start:t_end]
            if model_name == "vae":
                normal_mask = (y_train_fold.flatten() == 0)
                X_train_final, y_train_final = X_train_fold[normal_mask], y_train_fold[normal_mask]
            else:
                X_train_final, y_train_final = X_train_fold, y_train_fold

            train_dl = FastTensorDataLoader(X_train_final, y_train_final,
                                            batch_size=16384, shuffle=True)
            val_dl = FastTensorDataLoader(X[t_end:v_end], y[t_end:v_end],
                                          batch_size=16384, shuffle=False)
            test_dl = FastTensorDataLoader(X[v_end:test_end], y[v_end:test_end],
                                           batch_size=16384, shuffle=False)
            
            pt_model = build_pytorch_model(model_name, cfg)
            try:
                pt_model = torch.compile(pt_model)
            except Exception:
                pass

            optimizer = torch.optim.AdamW(pt_model.parameters(), lr=cfg["training"]["learning_rate"],
                                          weight_decay=cfg["training"].get("weight_decay", 1e-5))
            trainer = build_trainer(model_name, pt_model, optimizer, cfg)
            
            # Train with early stopping
            patience = cfg["training"].get("early_stopping_patience", 10)
            best_val_auc = -1.0
            epochs_no_improve = 0
            best_state = None
            
            for epoch in range(cfg["training"]["epochs"]):
                train_loss = trainer.train_epoch(train_dl)
                
                if model_name == "vae":
                    val_preds = trainer.predict(val_dl)
                    val_auc = average_precision_score(y[t_end:v_end], val_preds)
                else:
                    _, val_auc = trainer.evaluate(val_dl)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    epochs_no_improve = 0
                    best_state = {k: v.clone() for k, v in pt_model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience:
                    logger.info(f"    Early stop at epoch {epoch+1}, best val PR-AUC: {best_val_auc:.4f}")
                    break
            
            # Restore best weights
            if best_state is not None:
                safe_load_weights(pt_model, best_state)
            
            fold_preds = trainer.predict(test_dl)
        
        all_oof_preds.append(fold_preds)
        all_oof_labels.append(y[v_end:test_end])
    
    # Concatenate OOF predictions from all folds
    oof_preds = np.concatenate(all_oof_preds)
    oof_labels = np.concatenate(all_oof_labels)
    oof_auc = average_precision_score(oof_labels, oof_preds)
    
    logger.info(f"OOF PR-AUC: {oof_auc:.4f} ({len(oof_preds)} predictions)")
    
    # Save OOF
    oof_dir = Path("data/predictions")
    oof_dir.mkdir(parents=True, exist_ok=True)
    np.save(oof_dir / f"{model_name}_oof.npy", oof_preds)

    # --- Train final model on ALL data ---
    logger.info("training_final_model", model=model_name)
    model_dir = Path("models/base")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if model_name in TREE_MODELS:
        params = cfg[model_name]["params"]
        final = TreeWrapper(model_type=model_name, params=params)
        final.fit(X, y)
        model_path = model_dir / f"{model_name}_final.joblib"
        joblib.dump(final.get_raw_model(), model_path)
    else:
        if model_name == "vae":
            normal_mask = (y.flatten() == 0)
            X_final, y_final = X[normal_mask], y[normal_mask]
        else:
            X_final, y_final = X, y

        final_dl = FastTensorDataLoader(X_final, y_final, batch_size=16384, shuffle=True)
        
        pt_model = build_pytorch_model(model_name, cfg)
        try:
            pt_model = torch.compile(pt_model)
        except Exception:
            pass

        optimizer = torch.optim.AdamW(pt_model.parameters(), lr=cfg["training"]["learning_rate"],
                                      weight_decay=cfg["training"].get("weight_decay", 1e-5))
        trainer = build_trainer(model_name, pt_model, optimizer, cfg)
        
        for epoch in tqdm(range(cfg["training"]["epochs"]), desc=f"Final {model_name.upper()}"):
            trainer.train_epoch(final_dl)
        
        model_path = model_dir / f"{model_name}_final.pt"
        torch.save(pt_model.state_dict(), model_path)
    
    logger.info("training_complete", model=model_name, oof_pr_auc=oof_auc, model_path=str(model_path))
    return float(oof_auc)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Baseline Auto-Tuner & Trainer v2.0")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--skip-tune", action="store_true", help="Skip Optuna tuning (use existing config)")
    args = parser.parse_args()
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Antifraud_v2_Base_Models")
    
    results = {}
    
    for model_name in args.models:
        log_file = f"logs/{model_name}_training.log"
        setup_logger(log_file=log_file, terminal_level=logging.ERROR, file_level=logging.WARNING)
        
        # Hard print to stdout to announce model without relying on the ERROR logger
        print(f"\n{'='*60}\n  MODEL: {model_name.upper()}\n{'='*60}")
        
        logger.info("model_started", model=model_name)
        
        try:
            data_type = "tree" if model_name in TREE_MODELS else model_name
            if model_name == "vae":
                data_type = "mlp"
                
            data = np.load(f"data/processed/X_y_dev_{data_type}.npz")
            X, y = data['X'], data['y']
            folds = get_walk_forward_splits(len(X))
            
            if not args.skip_tune:
                best_params = run_tuning(model_name, X, y, folds)
                update_yaml_config(model_name, best_params)
            
            with mlflow.start_run(run_name=f"{model_name}_final"):
                val_auc = generate_oof_and_train(model_name, X, y, folds)
                mlflow.log_metric("oof_pr_auc", val_auc)
            
            results[model_name] = f"SUCCESS (PR-AUC: {val_auc:.4f})"
            logger.info(f"[{model_name}] ✅ Complete")
            
            # Cloud instances do not need thermal throttles, proceed immediately
                
        except Exception as e:
            results[model_name] = f"FAILED: {e}"
            logger.error(f"[{model_name}] ❌ {e}", exc_info=True)
    
    # Summary
    logger.info(f"\n{'='*60}\nPIPELINE SUMMARY\n{'='*60}")
    for model, status in results.items():
        icon = "✅" if "SUCCESS" in status else "❌"
        logger.info(f"  {icon} {model.upper():15s} → {status}")

if __name__ == "__main__":
    main()
