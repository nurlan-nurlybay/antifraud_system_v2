# 🛡️ Antifraud System v2.0 — Super-Learner Ensemble

Advanced transaction-level fraud detection system built for the [IEEE-CIS Fraud Detection Challenge](https://www.kaggle.com/competitions/ieee-fraud-detection). This version implements a high-throughput, VRAM-resident ensemble of heterogeneous models, orchestrated by a Logistic Regression meta-stacker.

## 🚀 Performance Benchmarks

The system achieves robust generalization through combinatorial synergy, optimized specifically for **PR-AUC** to maximize real-world commercial viability over standard ROC-AUC metrics.

* **🏆 Kaggle Public Leaderboard:** `0.923659` ROC-AUC
* **🏆 Kaggle Private Leaderboard:** `0.899386` ROC-AUC

### Base Models vs. Meta-Stacker Evaluation

| Model Name | PR-AUC | ROC-AUC |
| :--- | :--- | :--- |
| **>> META_STACKER** | **0.6062** | **0.9240** |
| lightgbm | 0.5939 | 0.9203 |
| xgboost | 0.5709 | 0.9012 |
| catboost | 0.5483 | 0.9203 |
| random_forest | 0.5003 | 0.9033 |
| lstm | 0.4865 | 0.8738 |
| mlp | 0.4795 | 0.8625 |
| vae | 0.1088 | 0.7559 |

## 🏗️ Model Architecture

### Base Model Stack
* **Tree Ensemble**: LightGBM, XGBoost, CatBoost, and Random Forest.
* **Deep Learning (PyTorch)**:
    * **MLP**: Multi-Layer Perceptron optimized with sparse architecture search.
    * **LSTM**: Sequential transaction processing for temporal velocity patterns.
    * **VAE**: Unsupervised anomaly detection via reconstruction error.

### Meta-Stacker
* **Logistic Regression**: A constrained linear combiner trained on Out-Of-Fold (OOF) predictions to learn the optimal weights for each base model.

## 🧠 Key Techniques & Feature Engineering

* **Principal Component Analysis (PCA)**: Reduced 339 highly collinear V-features into 50 Principal Components, dramatically cutting noise while preserving 87.7% of cumulative variance.
* **Expanding Mean Target Encoder**: Encodes high-cardinality categoricals strictly on past data (chronologically sorted) to prevent future-data leakage.
* **Velocity & Rolling Features**: Engineered behavioral trackers per user (hashed from card/address/email combinations), including `time_dist_last` and rolling 5-transaction Amount means/stds.
* **Cyclical Time Encoding**: Sine and cosine transformations applied to `TransactionDT` to map 24-hour and 7-day cyclical continuity.
* **Signal of Absence**: Explicit binary NaN-masks injected prior to median imputation to capture the predictive power of missing data.
* **Sequential LSTM Windows**: Causal, zero-padded 5-step transaction history windows built per user.
* **Walk-Forward Cross-Validation**: Strict chronological validation splits to mimic real-world deployment, reducing estimation bias and generating leak-free OOF predictions for the stacker.
* **Focal BCE Loss**: Mathematically stabilizes PyTorch model gradients against the extreme class imbalance (~3.5% fraud rate).

## 🛠️ Usage Instructions

The pipeline is fully automated and modular, optimized for high-throughput execution on NVIDIA RTX hardware.

### 1. Data Preparation & Analysis
Generate insights and process the raw IEEE-CIS datasets into model-ready `.npz` arrays:
```bash
# Generate PCA variance reports and feature distribution visualizations
PYTHONPATH=. python pipelines/analyze_data.py

# Process raw training/validation data (generates Base/MLP/LSTM arrays)
PYTHONPATH=. python pipelines/run_data_prep.py

# Apply the fitted preprocessor to the dedicated blind test set
PYTHONPATH=. python pipelines/run_test_prep.py
```

### 2. Training Cycle
Train the base models and sequentially build the meta-stacker:
```bash
# Train all base models and run hyperparameter tuning
PYTHONPATH=. python pipelines/train_base_models.py

# Train the Logistic Regression Meta-Stacker on OOF predictions
PYTHONPATH=. python pipelines/train_meta.py
```

### 3. Evaluation & Ablation
Score the models and calculate their marginal contributions:
```bash
# Verify model PR-AUC/ROC-AUC scores locally
PYTHONPATH=. python pipelines/evaluate.py

# Run the 127-combination combinatorial ablation study
PYTHONPATH=. python pipelines/run_ablation.py
```

### 4. Inference & Interpretability
```bash
# Generate submission.csv for Kaggle
PYTHONPATH=. python pipelines/run_inference.py

# Generate Tree SHAP and Deep SHAP summary plots in reports/shap/
PYTHONPATH=. python pipelines/run_shap.py
```

## 📁 Repository Structure
* **`pipelines/`**: Core execution scripts.
  * `analyze_data.py`: PCA and feature visualization.
  * `run_data_prep.py` / `run_test_prep.py`: Data transformation and array building.
  * `train_base_models.py` / `train_meta.py`: Optuna tuning and model training.
  * `evaluate.py` / `run_ablation.py`: Scoring and meta-stacker subset analysis.
  * `run_inference.py` / `run_shap.py`: Kaggle submissions and interpretability.
* **`src/fd/models/`**: Architecture definitions for PyTorch and Tree models.
* **`src/fd/training/`**: High-performance engines (FastTensorDataLoader).
* **`configs/`**: Hyperparameters and metadata coefficients.
* **`reports/`**: Performance summaries, PCA visualizations, and ablation CSVs.
