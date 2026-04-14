"""
Ablation Study Pipeline — Antifraud System v2.0

Evaluates individual baseline models and performs 127 combinatorial subsets
of the Meta-Stacker, retraining Logistic Regressions to measure ensemble synergy.
"""

import itertools
import numpy as np
import pandas as pd
import joblib
import yaml
import torch
import structlog
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src.fd.data.dataset import FastTensorDataLoader
from src.fd.models.base_nets import FraudMLP, FraudVAE
from src.fd.models.pytorch.lstm import FraudLSTM
from src.fd.training.engine import FraudTrainer
from src.fd.training.vae_engine import VAETrainer
from src.fd.utils.logging import setup_logger
from src.fd.utils.evaluation import (
    ALL_MODELS, 
    TREE_MODELS,
    load_meta_test_matrices, 
    load_meta_train_matrices, 
    normalize_meta_vae
)
from pipelines.train_base_models import safe_load_weights

setup_logger()
logger = structlog.get_logger(__name__)

# Logic moved to src.fd.utils.evaluation


def run_ablation():
    logger.info("Initialization beginning...")
    
    X_test_meta, y_test = load_meta_test_matrices()
    X_train_meta, y_train = load_meta_train_matrices()
    
    # Apply Log-Squash Scaling Consistently
    X_train_meta, scaler = normalize_meta_vae(X_train_meta, fit=True)
    X_test_meta, _ = normalize_meta_vae(X_test_meta, scaler=scaler, fit=False)
    
    logger.info(f"Meta-Train Matrix (Log-Scaled): {X_train_meta.shape}")
    logger.info(f"Meta-Test Matrix (Log-Scaled): {X_test_meta.shape}")
    
    # Load optimized Logistics Regression variables
    try:
        with open("configs/models/meta_model.yaml", "r") as f:
            meta_cfg = yaml.safe_load(f)["meta_model"]["best_params"]
    except FileNotFoundError:
        logger.warning("meta_model.yaml missing. Falling back to default L2 C=1.0")
        meta_cfg = {"C": 1.0, "penalty": "l2"}
        
    results = []
    
    # PART A (Baselines) moved to evaluate.py

    # -----------------------------------------------------------------
    # Part B: Combinatorial Ablation
    # -----------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("PART B: Iterating 127 Combinatorial Stacker Meta-Models")
    logger.info("=" * 50)
    
    for k in range(1, len(ALL_MODELS) + 1):
        for combo in itertools.combinations(range(len(ALL_MODELS)), k):
            combo = list(combo)
            combo_names = "+".join([ALL_MODELS[i] for i in combo])
            
            X_tr_sub = X_train_meta[:, combo]
            X_te_sub = X_test_meta[:, combo]
            
            lr = LogisticRegression(
                C=meta_cfg["C"], 
                penalty=meta_cfg["penalty"], 
                solver="saga" if meta_cfg["penalty"] == "elasticnet" else "lbfgs", 
                l1_ratio=meta_cfg.get("l1_ratio", None),
                max_iter=1000, 
                random_state=42
            )
            lr.fit(X_tr_sub, y_train)
            
            preds = lr.predict_proba(X_te_sub)[:, 1]
            pr = average_precision_score(y_test, preds)
            roc = roc_auc_score(y_test, preds)
            
            stage = "Ensemble" if k > 1 else "Lone Meta-Model"
            results.append({
                "stage": stage,
                "combo_size": k,
                "models": combo_names,
                "pr_auc": pr,
                "roc_auc": roc
            })
            
    df_res = pd.DataFrame(results)
    
    # Save Report
    out_dir = Path("reports/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "ablation_results.csv", index=False)
    
    # Display Top 10
    top = df_res[df_res['combo_size'] > 1].sort_values("pr_auc", ascending=False).head(10)
    
    logger.info("=" * 50)
    logger.info("Ablation Study Complete!")
    logger.info("Top 10 Meta-Ensembles (PR-AUC Selected):")
    print("\n" + top.to_string(index=False) + "\n")


if __name__ == "__main__":
    run_ablation()
