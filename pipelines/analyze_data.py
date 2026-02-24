import pandas as pd
import numpy as np
from pathlib import Path
from src.fd.utils.config import load_config

def run_analysis(cfg_path: str):
    cfg = load_config(cfg_path)
    report_file = Path("reports/data_analysis.txt")
    report_file.parent.mkdir(exist_ok=True)

    print("[1/3] Loading Data...")
    train_trans = pd.read_csv(cfg['paths']['train_transaction'])
    train_id = pd.read_csv(cfg['paths']['train_identity'])
    df = pd.merge(train_trans, train_id, on='TransactionID', how='left')

    # Calculate global metrics
    null_rates = df.isnull().mean()
    # Numeric correlation only for this step
    correlations = df.corr(numeric_only=True)['isFraud'].abs()

    print("[2/3] Analyzing Feature Signal...")
    analysis_rows = []
    
    for col in df.columns:
        if col == 'isFraud' or col == 'TransactionID': continue
        
        n_rate = null_rates[col]
        # Correlation might be NaN for non-numeric or all-null cols
        corr = correlations.get(col, 0.0)
        if np.isnan(corr): corr = 0.0
        
        uniques = df[col].nunique()
        
        # LOGIC: Discard if Nulls > 90% AND Correlation is very low (< 0.01)
        # We keep high-null features if they show even a tiny signal
        discard = n_rate > 0.90 and corr < 0.01
        
        analysis_rows.append({
            "Feature": col,
            "Unique": uniques,
            "Null_%": round(n_rate * 100, 2),
            "Correlation": round(corr, 4),
            "Discard": "YES" if discard else "NO"
        })

    analysis_df = pd.DataFrame(analysis_rows)

    print("[3/3] Saving Report...")
    with open(report_file, "w") as f:
        f.write("=== MISSING DATA & CORRELATION ANALYSIS ===\n\n")
        
        f.write("TOP 30 FEATURES WITH MOST MISSING DATA:\n")
        # Sort by Null % to see the worst offenders
        high_nulls = analysis_df.sort_values(by="Null_%", ascending=False).head(30)
        f.write(high_nulls.to_string(index=False))
        
        f.write("\n\n--------------------------------------------------\n")
        
        f.write("CATEGORICAL FEATURES (Correlation is N/A):\n")
        # Show categoricals separately since Pearson correlation doesn't work on strings
        cats = analysis_df[analysis_df['Correlation'] == 0.0].sort_values(by="Null_%", ascending=False).head(20)
        f.write(cats.to_string(index=False))

    print(f"Analysis complete. See {report_file}")

if __name__ == "__main__":
    run_analysis("configs/data_prep.yaml")
