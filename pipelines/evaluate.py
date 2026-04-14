"""
Model Evaluation Script — Antifraud System v2.0

Evaluates all 7 base models and the final Meta-Stacker on the held-out 
meta-test partition. Prints a clean performance comparison table.
"""

import numpy as np
import joblib
import yaml
import structlog
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

from src.fd.utils.evaluation import (
    load_meta_test_matrices, 
    normalize_meta_vae, 
    ALL_MODELS
)
from src.fd.utils.logging import setup_logger

setup_logger()
logger = structlog.get_logger(__name__)

def run_evaluation():
    logger.info("=" * 60)
    logger.info("  ANTIFRAUD v2.0 — PERFORMANCE EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    # 1. Load Data
    X_test_meta, y_test = load_meta_test_matrices()
    
    # 2. Normalize VAE (Load scaler from preprocessors)
    vae_scaler_path = Path("models/preprocessors/vae_scaler.joblib")
    if not vae_scaler_path.exists():
        logger.warning("VAE scaler missing. VAE scores might be unscaled.")
    else:
        scaler = joblib.load(vae_scaler_path)
        X_test_meta, _ = normalize_meta_vae(X_test_meta, scaler=scaler, fit=False)
        
    # 3. Load Meta-Stacker
    meta_path = Path("models/meta/logreg_stacker.joblib")
    stacker = None
    if meta_path.exists():
        stacker = joblib.load(meta_path)
    else:
        logger.warning("Meta-Stacker model not found at models/meta/logreg_stacker.joblib")

    results = []
    
    # Evaluate Base Models
    for idx, model_name in enumerate(ALL_MODELS):
        preds = X_test_meta[:, idx]
        pr = average_precision_score(y_test, preds)
        roc = roc_auc_score(y_test, preds)
        results.append((model_name, pr, roc))
        
    # Evaluate Meta-Stacker
    if stacker:
        meta_preds = stacker.predict_proba(X_test_meta)[:, 1]
        pr = average_precision_score(y_test, meta_preds)
        roc = roc_auc_score(y_test, meta_preds)
        results.append(("META_STACKER", pr, roc))
        
    # 4. Print Table and Save to File
    out_dir = Path("reports/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "summary.txt"
    
    with open(report_path, "w") as f_out:
        header = "=" * 65
        title = f"{'Model Name':25s} | {'PR-AUC':10s} | {'ROC-AUC':10s}"
        divider = "-" * 65
        
        # Build Table String
        table_lines = [header, title, divider]
        
        # Sort by PR-AUC (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        for name, pr, roc in results:
            hl = ">> " if name == "META_STACKER" else "   "
            line = f"{hl}{name:22s} | {pr:8.4f}   | {roc:8.4f}"
            table_lines.append(line)
            
        table_lines.append(header)
        table_output = "\n".join(table_lines)
        
        # Print and Save
        print("\n" + table_output + "\n")
        f_out.write(table_output)
        
    logger.info(f"Evaluation report saved to {report_path}")
    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    run_evaluation()
