"""
Data Analysis & PCA Scree Plot Generator — Antifraud System v2.0

Uses the ACTUAL preprocessing pipeline (ExpandingMeanEncoder, cyclical time,
rolling velocity, NaN masks, StandardScaler) to prepare the V-features
BEFORE running PCA. This gives an honest variance decomposition.

Usage:
    PYTHONPATH=. python pipelines/analyze_data.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.fd.utils.config import load_config
from src.fd.data.preprocessing import DataPreprocessor
from src.fd.utils.memory import reduce_mem_usage, clear_memory

def run_analysis(cfg_path: str = "configs/data_prep.yaml"):
    cfg = load_config(cfg_path)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. LOAD & PREPROCESS (same path as run_data_prep.py — no PCA applied)
    # =========================================================================
    print("Loading raw data...")
    train_trans = reduce_mem_usage(pd.read_csv(cfg['paths']['train_transaction']))
    train_id = reduce_mem_usage(pd.read_csv(cfg['paths']['train_identity']))
    df = pd.merge(train_trans, train_id, on=cfg['data']['id_col'], how='left')
    df = reduce_mem_usage(df, verbose=False)
    del train_trans, train_id
    clear_memory()
    
    print(f"Dataset loaded: {df.shape}")

    # Run our preprocessor's clean_base_data (creates Uid, sorts chronologically,
    # adds velocity, rolling amount features, imputes categoricals)
    pp = DataPreprocessor(cfg)
    df = pp.clean_base_data(df)
    
    # Run the neural base transforms WITHOUT PCA:
    # NaN masks, median imputation, cyclical time, target encoding, log transforms
    target_col = cfg['data']['target_col']
    ignore_cols = pp.ignore_cols
    
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols]
    pp.numerical_medians = df[num_cols].median().to_dict()
    pp.nan_mask_columns = [c for c in num_cols if df[c].isnull().any()]
    
    # Add NaN masks
    for col in pp.nan_mask_columns:
        df[col + '_is_nan'] = df[col].isnull().astype(np.float32)
    
    # Fill NaNs
    for col in num_cols:
        df[col] = df[col].fillna(pp.numerical_medians.get(col, 0))
    
    # Cyclical time
    df = pp._apply_cyclical_time(df)
    
    # Expanding Target Encoding
    df = pp._apply_target_encoding(df, is_train=True)
    
    # Log transforms
    if cfg['features']['log_transform_amount']:
        df[cfg['data']['amount_col']] = np.log1p(df[cfg['data']['amount_col']])
        df['time_dist_last'] = np.log1p(df['time_dist_last'].clip(lower=0))

    # =========================================================================
    # 2. SCALED PCA ANALYSIS ON V-FEATURES
    # =========================================================================
    import re
    v_cols = [c for c in df.columns if re.match(r'^V\d+$', c)]
    n_v = len(v_cols)
    print(f"\nFound {n_v} V-features for PCA analysis")
    
    V_data = df[v_cols].values.astype(np.float32)
    
    # THIS IS THE FIX: Scale BEFORE PCA
    scaler = StandardScaler()
    V_scaled = scaler.fit_transform(V_data)
    del V_data
    clear_memory()
    
    print("Running PCA on SCALED V-features...")
    pca = PCA(random_state=42)
    pca.fit(V_scaled)
    
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    # Find thresholds
    n_95 = int(np.searchsorted(cum_var, 0.95)) + 1
    n_99 = int(np.searchsorted(cum_var, 0.99)) + 1
    
    print(f"\n{'='*50}")
    print(f"PCA RESULTS (SCALED V-features)")
    print(f"{'='*50}")
    print(f"Components for 95% variance: {n_95}")
    print(f"Components for 99% variance: {n_99}")
    print(f"\nTop 100 components:")
    print(f"{'PC':>4} | {'Individual %':>14} | {'Cumulative %':>14}")
    print(f"{'-'*4}-+-{'-'*14}-+-{'-'*14}")
    for i in range(min(100, len(exp_var))):
        print(f"{i+1:4d} | {exp_var[i]*100:13.2f}% | {cum_var[i]*100:13.2f}%")

    # =========================================================================
    # 3. SCREE PLOT
    # =========================================================================
    n_show = min(100, len(exp_var))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('V-Features PCA Variance Analysis (StandardScaler Applied)', fontsize=14, fontweight='bold')
    
    # Left: Top 30 components
    ax1.bar(range(1, n_show+1), exp_var[:n_show]*100, alpha=0.5, color='steelblue', label='Individual')
    ax1.step(range(1, n_show+1), cum_var[:n_show]*100, where='mid', color='crimson', linewidth=2, label='Cumulative')
    ax1.axhline(y=95, color='green', linestyle='--', alpha=0.7, label=f'95% → {n_95} PCs')
    ax1.axhline(y=99, color='blue', linestyle=':', alpha=0.7, label=f'99% → {n_99} PCs')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title(f'Top {n_show} Components')
    ax1.legend(loc='center right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Full cumulative curve
    ax2.plot(range(1, len(cum_var)+1), cum_var*100, color='crimson', linewidth=2)
    ax2.axhline(y=95, color='green', linestyle='--', alpha=0.7, label=f'95% @ {n_95} PCs')
    ax2.axhline(y=99, color='blue', linestyle=':', alpha=0.7, label=f'99% @ {n_99} PCs')
    ax2.axvline(x=n_95, color='green', linestyle='--', alpha=0.3)
    ax2.axvline(x=n_99, color='blue', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance (%)')
    ax2.set_title(f'Full Curve ({n_v} V-features)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = reports_dir / "v_features_pca_variance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nScree plot saved to {plot_path}")
    
    # =========================================================================
    # 4. WRITE REPORT
    # =========================================================================
    report_path = reports_dir / "data_analysis.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ANTIFRAUD v2.0 — DATA ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Dataset: {df.shape[0]:,} transactions, {df.shape[1]} columns\n")
        f.write(f"Fraud rate: {df[target_col].mean():.4f} ({int(df[target_col].sum()):,} fraudulent)\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("V-FEATURES PCA (SCALED)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total V-features: {n_v}\n")
        f.write(f"Components for 95% variance: {n_95}\n")
        f.write(f"Components for 99% variance: {n_99}\n\n")
        
        f.write(f"{'PC':>4} | {'Individual':>12} | {'Cumulative':>12}\n")
        f.write(f"{'-'*4}-+-{'-'*12}-+-{'-'*12}\n")
        for i in range(min(100, len(exp_var))):
            f.write(f"{i+1:4d} | {exp_var[i]*100:11.2f}% | {cum_var[i]*100:11.2f}%\n")
        
        f.write(f"\n{'='*60}\n")
        f.write("RECOMMENDATION\n")
        f.write(f"{'='*60}\n")
        
        if n_95 <= 2:
            f.write("⚠️  WARNING: Only 2 PCs capturing 95% suggests scaling may\n")
            f.write("    still be failing or V-features are extremely redundant.\n")
        else:
            f.write(f"Set v_features_pca_dims: {n_99} in data_prep.yaml\n")
            f.write(f"This captures 99% variance in {n_99} components (down from {n_v}).\n")
            f.write(f"\nAlternative: {n_95} components for 95% (more aggressive compression).\n")
    
    print(f"Report saved to {report_path}")
    print(f"\n🔑 KEY RESULT: You need {n_99} PCs for 99% variance (was 2 — that was unscaled garbage)")

if __name__ == "__main__":
    run_analysis()
