"""
Test Inference Pipeline — Antifraud System v2.0

This script loads the preprocessed Kaggle 'test' features, executes inference 
across all 7 base models, and orchestrates the final probabilities through 
the Meta-Stacker, generating a perfectly formatted `submission.csv`.
"""

import pandas as pd
import numpy as np
import joblib
import structlog
import torch
import torch.nn.functional as F
from pathlib import Path

from src.fd.utils.config import load_config
from src.fd.data.dataset import FastTensorDataLoader
from pipelines.train_base_models import build_pytorch_model, safe_load_weights

logger = structlog.get_logger(__name__)

ALL_MODELS = ["lightgbm", "xgboost", "catboost", "mlp", "vae", "lstm", "random_forest"]
TREE_MODELS = {"lightgbm", "xgboost", "catboost", "random_forest"}

def run_inference(cfg_path: str):
    logger.info("Initializing Kaggle Test Inference Workflow")
    
    cfg = load_config(cfg_path)
    processed_dir = Path(cfg['paths']['processed_dir'])
    preds_dir = Path("data/predictions")
    preds_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Preprocessed Test Arrays
    logger.info("Loading preprocessed test data matrices...")
    X_test_dict = {
        "tree": np.load(processed_dir / "X_test_tree.npy"),
        "mlp": np.load(processed_dir / "X_test_mlp.npy"),
        "vae": np.load(processed_dir / "X_test_mlp.npy"), # VAE uses MLP features natively
        "lstm": np.load(processed_dir / "X_test_lstm.npy")
    }
    
    # Number of test samples
    N = len(X_test_dict["tree"])
    
    # Meta Stacker Input Matrix
    X_meta = np.zeros((N, len(ALL_MODELS)), dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------------------------------------
    # 2. RUN BASE MODELS
    # -------------------------------------------------------------
    for idx, model_name in enumerate(ALL_MODELS):
        logger.info("Making test predictions", model=model_name)
        
        preds_path = preds_dir / f"test_preds_{model_name}.npy"
        
        # Check if already predicted (caching to save time)
        if preds_path.exists():
            fold_preds = np.load(preds_path)
            X_meta[:, idx] = fold_preds
            logger.info("Loaded cached predictions", model=model_name)
            continue
            
        if model_name in TREE_MODELS:
            model_path = f"models/base/{model_name}_final.joblib"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Missing {model_path}. Train models first.")
            model = joblib.load(model_path)
            
            # Predict
            fold_preds = model.predict_proba(X_test_dict["tree"])[:, 1]
            
        else:
            model_path = f"models/base/{model_name}_final.pt"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Missing {model_path}. Train models first.")
                
            # Load PyTorch Config and Weights
            model_cfg = load_config(f"configs/models/{model_name}.yaml")
            pt_model = build_pytorch_model(model_name, model_cfg)
            # Safe loading handles the _orig_mod. prefix from torch.compile
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            safe_load_weights(pt_model, state_dict)
            pt_model = pt_model.to(device)
            pt_model.eval()
            
            # Predict using FastTensorDataLoader for maximum hardware bandwidth
            dummy_y = np.zeros(N)
            loader = FastTensorDataLoader(X_test_dict[model_name], dummy_y, batch_size=16384, shuffle=False)
            
            all_preds = []
            with torch.no_grad():
                for batch_X, _ in loader:
                    out = pt_model(batch_X)
                    
                    if model_name == "vae":
                        recon_batch, _, _ = out
                        # Calculate MSE on GPU and move to CPU ONLY once
                        recon_err = F.mse_loss(recon_batch, batch_X, reduction='none').mean(dim=1)
                        all_preds.extend(recon_err.cpu().numpy())
                    else:
                        probs = torch.sigmoid(out).squeeze(-1)
                        all_preds.extend(probs.cpu().numpy())
            
            fold_preds = np.array(all_preds, dtype=np.float32)
            
        # Save output and log to Meta matrix
        np.save(preds_path, fold_preds)
        X_meta[:, idx] = fold_preds

    # -------------------------------------------------------------
    # 3. RUN META-STACKER
    # -------------------------------------------------------------
    logger.info("Executing Meta-Stacker Super-Learner via Logistic Regression")
    meta_path = "models/meta/logreg_stacker.joblib"
    if not Path(meta_path).exists():
        raise FileNotFoundError("Missing meta-stacker. Run train_meta.py first.")
        
    stacker = joblib.load(meta_path)
    
    # CRITICAL: Apply Log-Squash + MinMax Scaling to VAE test data before stacking
    vae_scaler_path = Path("models/preprocessors/vae_scaler.joblib")
    if not vae_scaler_path.exists():
        logger.error("Missing VAE scaler! Run train_meta.py first.")
        return
        
    scaler = joblib.load(vae_scaler_path)
    vae_idx = ALL_MODELS.index("vae")
    
    vae_log = np.log1p(X_meta[:, vae_idx])
    X_meta[:, vae_idx] = scaler.transform(vae_log.reshape(-1, 1)).flatten()
    
    final_preds = stacker.predict_proba(X_meta)[:, 1]
    
    # -------------------------------------------------------------
    # 4. GENERATE KAGGLE SUBMISSION
    # -------------------------------------------------------------
    logger.info("Formatting submission payload...")
    sub_path = Path("data/raw/sample_submission.csv")
    if not sub_path.exists():
        logger.error("Missing sample_submission.csv! Cannot map TransactionIDs.")
        return
        
    sub = pd.read_csv(sub_path)
    
    if len(sub) != len(final_preds):
        logger.error("Dimension mismatch!", sub_length=len(sub), pred_length=len(final_preds))
        return
        
    sub['isFraud'] = final_preds
    sub.to_csv("submission.csv", index=False)
    
    logger.info("Inference Complete!", submission_file="submission.csv")

if __name__ == "__main__":
    run_inference("configs/data_prep.yaml")
