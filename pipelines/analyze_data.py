import pandas as pd
import numpy as np
from pathlib import Path
from src.fd.utils.config import load_config

def run_analysis(cfg_path: str):
    cfg = load_config(cfg_path)
    report_file = Path("reports/data_analysis.txt")
    report_file.parent.mkdir(exist_ok=True)

    print("Loading Data...")
    train_trans = pd.read_csv(cfg['paths']['train_transaction'])
    train_id = pd.read_csv(cfg['paths']['train_identity'])
    df = pd.merge(train_trans, train_id, on='TransactionID', how='left')

    null_rates = df.isnull().mean() * 100
    uniques = df.nunique()
    
    # Identify Categoricals (Object strings)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Numeric correlation only
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'isFraud' in num_cols: num_cols.remove('isFraud')
    if 'TransactionID' in num_cols: num_cols.remove('TransactionID')
        
    correlations = df[num_cols + ['isFraud']].corr()['isFraud'].abs()

    print("Building Report...")
    data = []
    for col in df.columns:
        if col in ['isFraud', 'TransactionID']: continue
        
        n_rate = null_rates[col]
        u_count = uniques[col]
        is_cat = col in cat_cols
        
        if is_cat:
            corr_str = "cat"
            discard_str = "N/A"
        else:
            corr_val = correlations.get(col, 0.0)
            if np.isnan(corr_val): corr_val = 0.0
            corr_str = f"{corr_val:.4f}"
            
            # Discard logic: strictly for numeric features missing > 85% with low signal
            discard_str = "YES" if (n_rate > 85.0 and corr_val < 0.01) else "NO"
            
        data.append({
            "Feature": col,
            "Unique": u_count,
            "Null_%": round(n_rate, 2),
            "Correlation": corr_str,
            "Discard": discard_str,
            "Is_Cat": is_cat
        })

    analysis_df = pd.DataFrame(data)

    # Table 1: 0% Missing (UID Candidates & Base Features)
    t1 = analysis_df[analysis_df['Null_%'] == 0.0].drop(columns=['Is_Cat'])
    
    # Table 2: All True Categoricals
    t2 = analysis_df[analysis_df['Is_Cat'] == True].drop(columns=['Is_Cat'])
    
    # Table 3: Top Missing (>85%) to evaluate Discards
    t3 = analysis_df[analysis_df['Null_%'] > 85.0].drop(columns=['Is_Cat']).sort_values(by="Null_%", ascending=False)

    print("Saving Report...")
    with open(report_file, "w") as f:
        f.write("=== TABLE 1: 0% MISSING FEATURES (UID Candidates) ===\n")
        f.write(t1.to_string(index=False))
        
        f.write("\n\n=== TABLE 2: ALL CATEGORICAL FEATURES (Strings) ===\n")
        f.write(t2.to_string(index=False))
        
        f.write("\n\n=== TABLE 3: HIGH MISSING (>85%) & DISCARD FLAG ===\n")
        f.write(t3.to_string(index=False))

    print(f"Done. Check {report_file}")

if __name__ == "__main__":
    run_analysis("configs/data_prep.yaml")
