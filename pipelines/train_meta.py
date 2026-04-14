"""
Meta-Model (Stacker) Training Pipeline — Antifraud System v2.0.

Loads OOF predictions from all 7 base models (all now transaction-level and
perfectly 1:1 aligned), runs inference on the held-out Meta-Validation set,
tunes a Logistic Regression stacker via Optuna, and saves the final meta-model
with full artifact export.

Usage:
    PYTHONPATH=. python pipelines/train_meta.py
"""

import joblib
import yaml
import torch
import numpy as np
import optuna
import mlflow
import mlflow.sklearn as mlflow_sklearn
import structlog
import logging
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src.fd.data.dataset import FraudDataset, SequenceFraudDataset
from src.fd.models.base_nets import FraudMLP, FraudVAE
from src.fd.models.pytorch.lstm import FraudLSTM
from src.fd.training.engine import FraudTrainer
from src.fd.training.vae_engine import VAETrainer

# --- Logger Setup ---
from src.fd.utils.logging import setup_logger
import logging
setup_logger(log_file="logs/meta_training.log", terminal_level=logging.ERROR, file_level=logging.WARNING)
logger = structlog.get_logger("train_meta")

# --- Constants ---
ALL_MODELS = ["lightgbm", "xgboost", "catboost", "mlp", "vae", "lstm", "random_forest"]
TREE_MODELS = {"lightgbm", "xgboost", "catboost", "random_forest"}

PATHS = {
    "dev_tree":  "data/processed/X_y_dev_tree.npz",
}


# =====================================================================
#  STEP 1: Load OOF predictions → X_meta_train, y_meta_train
# =====================================================================
def load_oof_predictions() -> tuple[np.ndarray, np.ndarray]:
    """
    Loads all 7 OOF arrays and stacks horizontally.
    All models are now transaction-level and perfectly aligned.
    """
    logger.info("=" * 50)
    logger.info("STEP 1: Loading OOF predictions")
    logger.info("=" * 50)

    oof_columns = []
    for model in ALL_MODELS:
        oof = np.load(f"data/predictions/{model}_oof.npy")
        logger.info(f"  {model:15s}: {oof.shape}, [{oof.min():.4f}, {oof.max():.4f}]")
        oof_columns.append(oof)

    # Verify all OOFs have the same length
    n_rows = oof_columns[0].shape[0]
    for model, arr in zip(ALL_MODELS, oof_columns):
        assert arr.shape[0] == n_rows, f"{model} OOF has {arr.shape[0]} rows, expected {n_rows}"

    X_oof = np.column_stack(oof_columns)

    # Labels: last N rows of dev y (OOF covers the test folds)
    dev_data = np.load(PATHS["dev_tree"])
    y_full = dev_data["y"]
    y_oof = y_full[-n_rows:]

    logger.info(f"X_oof: {X_oof.shape}")
    logger.info(f"y_oof: {y_oof.shape}, fraud_rate={y_oof.mean():.4f}")

    return X_oof, y_oof


# DELETED: Base model inference on Meta-Val (Now deferred to independent ablation scripts)

# =====================================================================
#  STEP 3: Optuna Tuning (Logistic Regression)
# =====================================================================
def tune_stacker(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Tunes LogisticRegression hyperparameters via Optuna on PR-AUC."""
    logger.info("=" * 50)
    logger.info(f"STEP 3: Optuna Stacker Tuning ({n_trials} trials)")
    logger.info("=" * 50)

    def objective(trial):
        C = trial.suggest_float("C", 1e-6, 1e2, log=True)
        penalty = trial.suggest_categorical("penalty", ["l2", "elasticnet"])

        if penalty == "elasticnet":
            solver = "saga"
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        else:
            solver = "lbfgs"
            l1_ratio = None

        lr = LogisticRegression(
            C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio,
            max_iter=2000, random_state=42,
        )
        lr.fit(X_train, y_train)
        preds = lr.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, preds))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")

    return study.best_params


# =====================================================================
#  STEP 4: Final Training, Coefficients, Artifacts, MLflow
# =====================================================================
def train_and_export(
    best_params: dict,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
):
    """Trains final stacker, prints coefficients, saves model + YAML config, logs to MLflow."""
    logger.info("=" * 50)
    logger.info("STEP 4: Final Stacker Training & Export")
    logger.info("=" * 50)

    # Combine train + val for final model
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    logger.info(f"Combined training set: {X_combined.shape}")

    # Build final model
    penalty = best_params["penalty"]
    solver = "saga" if penalty == "elasticnet" else "lbfgs"
    l1_ratio = best_params.get("l1_ratio", None)

    final_model = LogisticRegression(
        C=best_params["C"], penalty=penalty, solver=solver, l1_ratio=l1_ratio,
        max_iter=2000, random_state=42,
    )
    final_model.fit(X_combined, y_combined)

    # Evaluate on val (for reporting)
    val_preds = final_model.predict_proba(X_val)[:, 1]
    val_auc = float(average_precision_score(y_val, val_preds))

    # --- Print Coefficients ---
    coefs = final_model.coef_[0]
    intercept = float(final_model.intercept_[0])

    logger.info("")
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║     META-STACKER LEARNED COEFFICIENTS        ║")
    logger.info("╠══════════════════════════════════════════════╣")
    for name, coef in zip(ALL_MODELS, coefs):
        bar = "█" * int(abs(coef) * 3)
        sign = "+" if coef >= 0 else "-"
        logger.info(f"║  {name:15s}  {sign}{abs(coef):7.4f}  {bar:20s}║")
    logger.info(f"║  {'intercept':15s}  {intercept:+8.4f}                     ║")
    logger.info(f"╠══════════════════════════════════════════════╣")
    logger.info(f"║  Meta-Val PR-AUC:  {val_auc:.4f}                     ║")
    logger.info(f"╚══════════════════════════════════════════════╝")

    # --- Save Model ---
    models_dir = Path("models/meta")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "logreg_stacker.joblib"
    joblib.dump(final_model, model_path)
    logger.info(f"Model saved to {model_path}")

    # --- Save YAML Config ---
    coef_map = {name: round(float(c), 6) for name, c in zip(ALL_MODELS, coefs)}
    meta_config = {
        "meta_model": {
            "type": "LogisticRegression",
            "best_params": {k: round(v, 6) if isinstance(v, float) else v for k, v in best_params.items()},
            "base_model_order": ALL_MODELS,
            "coefficients": coef_map,
            "intercept": round(intercept, 6),
            "meta_val_pr_auc": round(val_auc, 4),
        }
    }

    config_path = Path("configs/models/meta_model.yaml")
    with open(config_path, "w") as f:
        yaml.dump(meta_config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved to {config_path}")

    # --- MLflow ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Antifraud_v2_Meta_Stacker")

    with mlflow.start_run(run_name="meta_stacker"):
        mlflow.log_params(best_params)
        mlflow.log_metric("meta_val_pr_auc", val_auc)
        mlflow.log_metric("n_base_models", len(ALL_MODELS))
        for name, coef in zip(ALL_MODELS, coefs):
            mlflow.log_metric(f"coef_{name}", float(coef))
        mlflow_sklearn.log_model(final_model, artifact_path="meta_stacker")
        mlflow.log_artifact(str(config_path))

    logger.info("MLflow run logged ✅")

    return final_model, val_auc


# =====================================================================
#  MAIN
# =====================================================================
def main():
    logger.info("=" * 60)
    logger.info("  ANTIFRAUD v2.0 — META-STACKER TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1
    X_oof, y_oof = load_oof_predictions()

    # Step 2: Chronological Split (75% Train, 25% Val inside the OOF block limits)
    split_idx = int(len(X_oof) * 0.75)
    X_meta_train, X_meta_val = X_oof[:split_idx], X_oof[split_idx:]
    y_meta_train, y_meta_val = y_oof[:split_idx], y_oof[split_idx:]

    logger.info(f"OOF Validation Split: Train={X_meta_train.shape[0]}, Val={X_meta_val.shape[0]}")

    # Step 2b: VAE Log-Squash + MinMax Scaling (fit on train only to avoid leakage)
    vae_idx = ALL_MODELS.index("vae")
    logger.info("Applying VAE Log-Squash + MinMax Scaling (fit on train only)...")
    scaler = MinMaxScaler()
    X_meta_train[:, vae_idx] = scaler.fit_transform(
        np.log1p(X_meta_train[:, vae_idx]).reshape(-1, 1)
    ).flatten()
    X_meta_val[:, vae_idx] = scaler.transform(
        np.log1p(X_meta_val[:, vae_idx]).reshape(-1, 1)
    ).flatten()

    # Save scaler for inference
    prep_dir = Path("models/preprocessors")
    prep_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, prep_dir / "vae_scaler.joblib")
    logger.info(f"VAE scaler saved to {prep_dir / 'vae_scaler.joblib'}")

    # Step 3
    best_params = tune_stacker(X_meta_train, y_meta_train, X_meta_val, y_meta_val)

    # Step 4
    final_model, val_auc = train_and_export(
        best_params, X_meta_train, y_meta_train, X_meta_val, y_meta_val,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  PIPELINE COMPLETE — Meta-Val PR-AUC: {val_auc:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
