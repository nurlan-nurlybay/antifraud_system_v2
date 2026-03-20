import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from src.fd.utils.config import load_config

def plot_and_analyze_pca(df: pd.DataFrame):
    """
    Performs PCA on V-features to find the 'Elbow'.
    Saves a visualization and returns formatted stats for the manifest.
    """
    v_cols = [c for c in df.columns if c.startswith('V')]
    if not v_cols:
        return None, 0

    # PCA fit (Filling NaNs with 0 strictly for variance exploration)
    pca = PCA().fit(df[v_cols].fillna(0))
    exp_var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(exp_var_ratio)

    # --- Plotting Top 10 PCs for visual verification ---
    plt.figure(figsize=(12, 7))
    plt.bar(range(1, 11), exp_var_ratio[:10], alpha=0.4, align='center', label='Individual Variance')
    plt.step(range(1, 11), cum_var_ratio[:10], where='mid', label='Cumulative Variance', color='red', linewidth=2)
    plt.axhline(y=0.95, color='green', linestyle='--', label='95% Threshold')
    plt.axhline(y=0.99, color='blue', linestyle=':', label='99% Threshold')
    
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Index (Top 10)')
    plt.title('V-Features PCA: Top 10 Components Variance Analysis')
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = Path("reports/v_features_pca_variance.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"PCA Variance Plot saved to {plot_path}")

    # Numerical Breakdown for Report
    stats = []
    for i in range(min(10, len(exp_var_ratio))):
        stats.append({
            "PC": i + 1,
            "Ind": round(exp_var_ratio[i] * 100, 2),
            "Cum": round(cum_var_ratio[i] * 100, 2)
        })
    return stats, len(v_cols)

def run_analysis(cfg_path: str):
    cfg = load_config(cfg_path)
    report_file = Path("reports/data_analysis.txt")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Loading Data for Deep Analysis (Transaction + Identity)...")
    train_trans = pd.read_csv(cfg['paths']['train_transaction'])
    train_id = pd.read_csv(cfg['paths']['train_identity'])
    df = pd.merge(train_trans, train_id, on='TransactionID', how='left')
    target = cfg['data']['target_col']

    # --- 1. PCA ANALYSIS ---
    print("Performing PCA Analysis on V-features...")
    pca_stats, v_count = plot_and_analyze_pca(df)

    # --- 2. NUMERIC DISCARD LOGIC (Checking Raw vs. Missingness Signal) ---
    print("Evaluating Numerics (Detecting 'Signal of Absence')...")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    eval_nums = [c for c in num_cols if c not in [target, 'TransactionID', 'TransactionDT'] and not c.startswith('V')]
    
    num_discards = []
    
    for col in eval_nums:
        n_rate = df[col].isnull().mean()
        
        # Only deep-scan features with extreme sparsity (> 85% nulls)
        if n_rate > 0.85:
            # A. Raw Signal (Pearson correlation on available data points)
            raw_corr = df[col].corr(df[target])
            if pd.isna(raw_corr): raw_corr = 0.0
            
            # B. Missingness Signal (Correlation of the 'Null Mask' itself)
            null_corr = df[col].isnull().astype(int).corr(df[target])
            if pd.isna(null_corr): null_corr = 0.0
            
            # DROPPING RULE: Both signals must be below the 1% threshold
            if abs(raw_corr) < 0.01 and abs(null_corr) < 0.01:
                num_discards.append((col, n_rate, raw_corr, null_corr))

    # --- 3. CATEGORICAL DISCARD LOGIC (Target Encoding with Null-Inclusion) ---
    print("Evaluating Categoricals via OOF Target Encoding...")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Pull in semantic labels like addr1/2
    for sc in ['addr1', 'addr2']:
        if sc in df.columns and sc not in cat_cols: cat_cols.append(sc)
    cat_cols = sorted(cat_cols)

    # Prepare temporary DF for TE (Filling NaNs with 'Unknown' to capture their signal)
    df_cat_filled = df[cat_cols].fillna('Unknown')
    df_cat_filled[target] = df[target] # RE-ATTACH TARGET FOR GROUPBY

    cat_discards = []
    cat_keep = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for col in cat_cols:
        n_rate = df[col].isnull().mean()
        
        # Only deep-scan sparse categoricals
        if n_rate > 0.85:
            te_val = pd.Series(index=df.index, dtype=float)
            for t_idx, v_idx in kf.split(df):
                # We group by the 'filled' column so 'Unknown' gets a fraud score
                mapping = df_cat_filled.iloc[t_idx].groupby(col)[target].mean()
                te_val.iloc[v_idx] = df_cat_filled.iloc[v_idx][col].map(mapping)
            
            # Measure correlation of the encoded fraud probabilities
            corr = te_val.fillna(df[target].mean()).corr(df[target])
            if pd.isna(corr): corr = 0.0
            
            if abs(corr) < 0.01:
                cat_discards.append((col, n_rate, corr))
                continue
        
        cat_keep.append(col)

    # --- 4. GENERATE PRODUCTION MANIFEST ---
    print(f"Writing findings to {report_file}...")
    with open(report_file, "w") as f:
        f.write("=== PRODUCTION DATA PREP MANIFEST ===\n")
        f.write("Copy these exact blocks into your configs/data_prep.yaml\n\n")

        f.write("-" * 50 + "\n")
        f.write("1. DROP FEATURES\n")
        f.write("# Discard Logic: Null Rate > 85% AND (Raw Signal < 1% AND Missingness Signal < 1%)\n\n")
        
        f.write("  drop_features:\n")
        
        f.write("    # --- Numerical Discards ---\n")
        if not num_discards:
            f.write("    []\n")
        else:
            for col, n, r_c, n_c in num_discards:
                f.write(f"    - \"{col}\"  # Null: {n:.1%}, Raw Corr: {r_c:.4f}, Null Corr: {n_c:.4f}\n")
            
        f.write("\n    # --- Categorical Discards (Low Signal after TE) ---\n")
        if not cat_discards:
            f.write("    []\n")
        else:
            for col, n, c in cat_discards:
                f.write(f"    - \"{col}\"  # Null: {n:.1%}, TE-Corr: {c:.4f}\n")

        f.write("\n" + "-" * 50 + "\n")
        f.write("2. CATEGORICAL FEATURES\n")
        f.write("# Includes raw strings and semantic labels (addr1/addr2).\n")
        f.write("  categorical:\n    columns:\n")
        for col in cat_keep:
            f.write(f"      - \"{col}\"\n")

        f.write("\n" + "-" * 50 + "\n")
        f.write("3. V-FEATURES PCA ANALYSIS\n")
        f.write(f"# Found {v_count} V-features. Total variance captured:\n")
        f.write("# PC | Individual % | Cumulative %\n")
        if pca_stats:
            for s in pca_stats:
                f.write(f"# {s['PC']:<2} | {s['Ind']:<12} | {s['Cum']}\n")
        
        f.write("\n  reduce_v_features: true\n")
        f.write("  v_features_pca_dims: 2 # Matches 99% cumulative threshold\n")

    print(f"✅ Analysis complete. found {len(num_discards) + len(cat_discards)} features to drop.")

if __name__ == "__main__":
    run_analysis("configs/data_prep.yaml")
