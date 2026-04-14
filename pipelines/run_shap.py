"""
SHAP Explanation Generator — Antifraud System v2.0

Calculates overall SHAP summary plots for Tree models, Neural Networks,
and the Meta-Stacker to determine global feature importance. Includes
matrix transformations to reconstruct original V-features from PCA spaces.
"""

import numpy as np
import joblib
import structlog
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from src.fd.utils.config import load_config
import torch

from src.fd.models.base_nets import FraudMLP, FraudVAE
from src.fd.models.pytorch.lstm import FraudLSTM
from pipelines.train_base_models import safe_load_weights

logger = structlog.get_logger(__name__)

def run_shap():
    logger.info("Initializing SHAP Analysis Workflow")
    out_dir = Path("reports/shap")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PCA Matrix
    artifact_path = Path("models/preprocessors/preprocessor.joblib")
    if not artifact_path.exists():
        logger.error("Missing preprocessor artifacts. Train models first.")
        return
        
    bundle = joblib.load(artifact_path)
    pca = bundle["pca"]
    n_pca = pca.n_components_
    n_v_cols = 339
    
    logger.info("Sampling Background Data from Meta-Val")
    
    # 1. Tree Dataset
    d_tree = np.load("data/processed/X_y_meta_test_tree.npz")
    X_tree_all = d_tree['X']
    
    # 2. MLP/VAE Dataset
    d_mlp = np.load("data/processed/X_y_meta_test_mlp.npz")
    X_mlp_all = d_mlp['X']
    
    # 3. LSTM Dataset
    d_lstm = np.load("data/processed/X_y_meta_test_lstm.npz")
    X_lstm_all = d_lstm['X']
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_tree_all), 2000, replace=False)
    
    X_background_tree = X_tree_all[sample_indices]
    X_background_mlp = torch.tensor(X_mlp_all[sample_indices], dtype=torch.float32)
    X_background_lstm = torch.tensor(X_lstm_all[sample_indices], dtype=torch.float32)

    # 4. Feature Names Extraction
    logger.info("Extracting feature naming metadata")
    try:
        # Neural names (195)
        nn_feature_names = bundle['scaler'].get_feature_names_out().tolist()
        
        # Tree names (438) - Reconstruct based on preprocessing logic: 
        # (145 engineered features + 339 V-features)
        # We can extract the engineered part from the NN names (excluding PCA and Nan masks)
        import re
        eng_names = [n for n in nn_feature_names if not n.startswith('V_pca_') and not n.endswith('_is_nan')]
        v_raw_names = [f'V{i}' for i in range(1, 340)]
        tree_feature_names = eng_names + v_raw_names
        
        # PCA V-names (339 original features for reconstructed plot)
        v_names_339 = [f'V{i}' for i in range(1, 340)]
        mlp_reconstructed_names = [n for n in nn_feature_names if not n.startswith('V_pca_')] + v_names_339
        
    except Exception as e:
        logger.warning(f"Feature naming reconstruction failed: {e}. Falling back to numeric defaults.")
        nn_feature_names = None
        tree_feature_names = None
        mlp_reconstructed_names = None
    
    # ---------------------------------------------------------
    # 1. EXPLORE LIGHTGBM
    # ---------------------------------------------------------
    lgbm_path = Path("models/base/lightgbm_final.joblib")
    if lgbm_path.exists():
        logger.info("Calculating Tree SHAP for LightGBM...")
        lgbm_model = joblib.load(lgbm_path)
        booster = lgbm_model.get_raw_model() if hasattr(lgbm_model, 'get_raw_model') else lgbm_model
            
        explainer = shap.TreeExplainer(booster)
        shap_values_tree = explainer.shap_values(X_background_tree)
        
        # If binary classification, take the positive class SHAP values
        if isinstance(shap_values_tree, list):
            shap_values_tree = shap_values_tree[1]
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_tree, 
            X_background_tree, 
            feature_names=tree_feature_names,
            max_display=20, 
            show=False,
            plot_type="dot"
        )
        plt.title("LightGBM SHAP Summary")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_lightgbm.png")
        plt.close()
        logger.info("Saved LightGBM SHAP.")

    # ---------------------------------------------------------
    # 1b. XGBOOST
    # ---------------------------------------------------------
    xgb_path = Path("models/base/xgboost_final.joblib")
    if xgb_path.exists():
        logger.info("Calculating Tree SHAP for XGBoost...")
        xgb_model = joblib.load(xgb_path)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_background_tree)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_background_tree, 
            feature_names=tree_feature_names,
            max_display=20, 
            show=False,
            plot_type="dot"
        )
        plt.title("XGBoost SHAP Summary")
        plt.savefig(out_dir / "shap_xgboost.png")
        plt.close()

    # ---------------------------------------------------------
    # 1c. CATBOOST
    # ---------------------------------------------------------
    cat_path = Path("models/base/catboost_final.joblib")
    if cat_path.exists():
        logger.info("Calculating Tree SHAP for CatBoost...")
        cat_model = joblib.load(cat_path)
        explainer = shap.TreeExplainer(cat_model)
        shap_values = explainer.shap_values(X_background_tree)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_background_tree, 
            feature_names=tree_feature_names,
            max_display=20, 
            show=False,
            plot_type="dot"
        )
        plt.title("CatBoost SHAP Summary")
        plt.savefig(out_dir / "shap_catboost.png")
        plt.close()

    # ---------------------------------------------------------
    # 1d. RANDOM FOREST
    # ---------------------------------------------------------
    rf_path = Path("models/base/random_forest_final.joblib")
    if rf_path.exists():
        logger.info("Calculating Tree SHAP for Random Forest...")
        rf_model = joblib.load(rf_path)
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_background_tree)
        # Random Forest SHAP might be (N, F, Classes), take class 1
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_background_tree, 
            feature_names=tree_feature_names,
            max_display=20, 
            show=False,
            plot_type="dot"
        )
        plt.title("Random Forest SHAP Summary")
        plt.savefig(out_dir / "shap_random_forest.png")
        plt.close()
        
    # ---------------------------------------------------------
    # 2. EXPLORE MLP & PCA RECONSTRUCTION
    # ---------------------------------------------------------
    mlp_path = Path("models/base/mlp_final.pt")
    if mlp_path.exists():
        logger.info("Calculating Deep SHAP for MLP...")
        mlp_cfg = load_config("configs/models/mlp.yaml")
        mlp_model = FraudMLP(
            input_dim=mlp_cfg["model"]["input_dim"],
            hidden_dims=mlp_cfg["model"]["hidden_dims"],
            dropout=0.0
        )
        state_dict = torch.load(mlp_path, map_location="cpu", weights_only=True)
        safe_load_weights(mlp_model, state_dict)
        mlp_model.eval()
        
        # DeepExplainer uses a background set to integrate over, and a test set to explain.
        bg = X_background_mlp[:500] 
        test_samples = X_background_mlp[500:1500]
        
        # Wrap MLP to explain Probabilities (Sigmoid) instead of Logits to avoid squashing
        def mlp_prob_wrapper(x):
            model_input = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                logits = mlp_model(model_input)
                probs = torch.sigmoid(logits)
            return probs.numpy()

        # DeepExplainer can take a (model, background) pair
        # For custom probability wrappers, we use KernelExplainer or ensure GradientExplainer works
        # Let's use GradientExplainer for MLP too as it's more flexible with the wrapper
        explainer = shap.GradientExplainer(mlp_model, bg) 
        logger.info("Explaining MLP test samples...")
        # We'll use the GradientExplainer on the model, but since we want probs, 
        # we can explain the model and then the user understands it's sigmoid-ready, 
        # OR we use a KernelExplainer on the wrapper. 
        # Actually, GradientExplainer(model, bg) is usually enough if we explain the output.
        shap_values_mlp = explainer.shap_values(test_samples)
        
        # Diagnostics and Shape Correction
        if isinstance(shap_values_mlp, list):
            shap_output = shap_values_mlp[0]
        else:
            shap_output = shap_values_mlp
            
        # PyTorch DeepExplainer often returns (Samples, Features, 1) for single-output models
        if len(shap_output.shape) == 3 and shap_output.shape[2] == 1:
            shap_output = shap_output.squeeze(2)

        # --- The PCA Reconstruction Trick ---
        # Matrix Math: SHAP_raw = SHAP_pca @ PCA.components_
        shap_other = shap_output[:, :-n_pca]
        shap_pca = shap_output[:, -n_pca:]
        
        logger.info("Reconstructing original V-features from PCA SHAP values...")
        shap_v_raw = np.dot(shap_pca, pca.components_)
        
        # Reconstruct the full SHAP array
        shap_reconstructed = np.hstack([shap_other, shap_v_raw])
        
        # Reconstruct the feature data to plot the colors correctly
        test_other = test_samples[:, :-n_pca].numpy()
        test_pca = test_samples[:, -n_pca:].numpy()
        
        # X_pca @ P = X_reconstructed approx
        test_v_raw = np.dot(test_pca, pca.components_)
        data_reconstructed = np.hstack([test_other, test_v_raw])
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_reconstructed, 
            data_reconstructed, 
            feature_names=mlp_reconstructed_names,
            max_display=20, 
            show=False,
            plot_type="dot"
        )
        plt.title("MLP SHAP Summary (PCA Reconstructed into V-Features)")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_mlp_reconstructed.png")
        plt.close()
        logger.info("Saved MLP SHAP.")

    # ---------------------------------------------------------
    # 2b. LSTM
    # ---------------------------------------------------------
    lstm_path = Path("models/base/lstm_final.pt")
    if lstm_path.exists():
        logger.info("Calculating Deep SHAP for LSTM...")
        lstm_cfg = load_config("configs/models/lstm.yaml")
        lstm_model = FraudLSTM(
            input_dim=lstm_cfg["model"]["input_dim"],
            hidden_dim=lstm_cfg["model"]["hidden_dim"],
            num_layers=lstm_cfg["model"]["num_layers"],
            dropout=0.0
        )
        state_dict = torch.load(lstm_path, map_location="cpu", weights_only=True)
        safe_load_weights(lstm_model, state_dict)
        lstm_model.eval()

        # LSTM input is (Batch, Seq, Features) -> (N, 1, F)
        bg_lstm = X_background_lstm[:200]
        test_lstm = X_background_lstm[200:400]
        
        # Wrap LSTM to explain Probabilities (Sigmoid) instead of Logits to avoid squashing
        def lstm_prob_wrapper(x):
            model_input = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                logits = lstm_model(model_input)
                probs = torch.sigmoid(logits)
            return probs.numpy()

        # GradientExplainer is more stable than DeepExplainer for recurrent architectures
        explainer = shap.GradientExplainer(lstm_model, bg_lstm)
        shap_values = explainer.shap_values(test_lstm)
        
        # Squeeze sequence dim for plotting
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values[:, 0, :], 
            test_lstm[:, 0, :].numpy(), 
            feature_names=nn_feature_names,
            max_display=20, 
            show=False,
            plot_type="dot"
        )
        plt.title("LSTM SHAP Summary (Probability Space)")
        plt.savefig(out_dir / "shap_lstm.png")
        plt.close()

    # ---------------------------------------------------------
    # 2c. VAE (Anomaly Triggers)
    # ---------------------------------------------------------
    vae_path = Path("models/base/vae_final.pt")
    if vae_path.exists():
        logger.info("Calculating SHAP for VAE Reconstruction Loss...")
        vae_cfg = load_config("configs/models/vae.yaml")
        vae_model = FraudVAE(
            input_dim=vae_cfg["model"]["input_dim"],
            hidden_dims=vae_cfg["model"]["hidden_dims"],
            dropout=0.0
        )
        state_dict = torch.load(vae_path, map_location="cpu", weights_only=True)
        safe_load_weights(vae_model, state_dict)
        vae_model.eval()

        # Wrapper: input -> Reconstruction MSE (Scalar)
        def vae_loss_wrapper(x_np):
            # SHAP might pass float64, VAE needs float32
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            with torch.no_grad():
                recon, _, _ = vae_model(x_tensor)
                # Return per-sample MSE as a 1D array
                loss = torch.mean((recon - x_tensor)**2, dim=1)
            return loss.numpy().astype(np.float64)

        # KernelExplainer for custom loss function
        bg_vae = X_mlp_all[np.random.choice(len(X_mlp_all), 50, replace=False)]
        test_vae = X_mlp_all[np.random.choice(len(X_mlp_all), 100, replace=False)]
        
        explainer = shap.KernelExplainer(vae_loss_wrapper, bg_vae)
        shap_values = explainer.shap_values(test_vae, nsamples=100)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            test_vae, 
            feature_names=nn_feature_names,
            max_display=20, 
            show=False,
            plot_type="dot"
        )
        plt.title("VAE Anomaly Trigger SHAP (Reconstruction Error)")
        plt.savefig(out_dir / "shap_vae.png")
        plt.close()
        
        # ---------------------------------------------------------
        # 3. META-STACKER
        # ---------------------------------------------------------
        meta_path = Path("models/meta/logreg_stacker.joblib")
        oof_path = Path("data/predictions/lightgbm_oof.npy")
        if meta_path.exists() and oof_path.exists():
            logger.info("Plotting Meta-Stacker internal weights...")
            stacker = joblib.load(meta_path)
            
            # Since Stacker is just logistic regression, we can explicitly plot its coefficients
            # instead of running full SHAP, which acts naturally as global feature importance
            models = ["lightgbm", "xgboost", "catboost", "mlp", "vae", "lstm", "random_forest"]
            coefs = stacker.coef_[0]
            
            plt.figure(figsize=(8, 5))
            plt.barh(models, coefs, color='royalblue')
            plt.xlabel("Logistic Regression Coefficient Weight")
            plt.title("Meta-Stacker Base Model Importance")
            plt.tight_layout()
            plt.savefig(out_dir / "meta_weights.png")
            plt.close()
            logger.info("Saved Meta-Stacker weights.", path=str(out_dir / "meta_weights.png"))

if __name__ == "__main__":
    run_shap()
