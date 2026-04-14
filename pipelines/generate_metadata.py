"""
Dataset Metadata Generator — Antifraud System v2.0

Inspects all 6 processed .npz files and generates a comprehensive
metadata document describing shapes, feature counts, preprocessing
pipelines, and encoding strategies.

Usage:
    PYTHONPATH=. python pipelines/generate_metadata.py
"""

import numpy as np
from pathlib import Path

def main():
    processed = Path("data/processed")
    output = processed / "README.md"

    files = {
        "dev_tree":     processed / "X_y_dev_tree.npz",
        "dev_mlp":      processed / "X_y_dev_mlp.npz",
        "dev_lstm":     processed / "X_y_dev_lstm.npz",
        "meta_val_tree": processed / "X_y_meta_val_tree.npz",
        "meta_val_mlp": processed / "X_y_meta_val_mlp.npz",
        "meta_val_lstm": processed / "X_y_meta_val_lstm.npz",
    }

    # Load shapes
    shapes = {}
    for name, path in files.items():
        if path.exists():
            d = np.load(path)
            x, y = d['X'], d['y']
            shapes[name] = {
                "x_shape": x.shape,
                "y_shape": y.shape,
                "fraud": int(y.sum()),
                "rate": float(y.mean()),
                "dtype": str(x.dtype),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
            }

    with open(output, "w") as f:
        f.write("# Processed Dataset Manifest — Antifraud System v2.0\n\n")
        f.write("**Source**: IEEE-CIS Fraud Detection (590,540 raw transactions)\n\n")
        f.write("**Split**: Chronological 90%/10% (Dev / Meta-Validation)\n\n")
        f.write("**Preprocessor Artifact**: `models/preprocessors/preprocessor.joblib`\n\n")
        
        # ===================================================================
        # FILE TABLE
        # ===================================================================
        f.write("---\n\n")
        f.write("## Dataset Files\n\n")
        f.write("| File | X Shape | y Shape | Fraud | Rate | dtype | Size |\n")
        f.write("|------|---------|---------|-------|------|-------|------|\n")
        for name, s in shapes.items():
            fname = f"X_y_{name}.npz"
            f.write(f"| `{fname}` | `{s['x_shape']}` | `{s['y_shape']}` | "
                    f"{s['fraud']:,} | {s['rate']:.2%} | {s['dtype']} | {s['size_mb']} MB |\n")
        
        # ===================================================================
        # SHARED PREPROCESSING
        # ===================================================================
        f.write("\n---\n\n")
        f.write("## Shared Preprocessing (All Datasets)\n\n")
        f.write("These steps are applied identically to Dev and Meta-Val data via "
                "`DataPreprocessor.clean_base_data()`:\n\n")
        f.write("1. **Merge**: Transaction table LEFT JOIN Identity table on `TransactionID`\n")
        f.write("2. **User ID Creation**: `Uid` = hash of `[card1, card2, card3, card4, card5, card6, addr1, P_emaildomain]`\n")
        f.write("3. **Chronological Sort**: All rows sorted by `TransactionDT` (ascending). "
                "This is critical for the Expanding Mean Target Encoder.\n")
        f.write("4. **Velocity Feature**: `time_dist_last` = time since user's previous transaction "
                "(`groupby(Uid).diff()` on `TransactionDT`). First transaction per user → NaN.\n")
        f.write("5. **Rolling Amount Features**:\n")
        f.write("   - `amt_rolling_mean`: Rolling 5-transaction mean of `TransactionAmt` per user\n")
        f.write("   - `amt_rolling_std`: Rolling 5-transaction standard deviation per user. "
                "Detects sudden spending anomalies.\n")
        f.write("6. **Categorical Imputation**: All string columns filled with `\"Unknown\"`\n\n")
        
        f.write("### Expanding Mean Target Encoder (Zero Temporal Leakage)\n\n")
        f.write("Replaces each categorical column with a numeric `_te` suffix column.\n\n")
        f.write("For transaction *i* with category value *c*:\n\n")
        f.write("```\n")
        f.write("encoding_i = (cumulative_fraud_sum_of_c_before_i + m * global_mean) / "
                "(cumulative_count_of_c_before_i + m)\n")
        f.write("```\n\n")
        f.write("- **m** = 10 (Bayesian smoothing strength)\n")
        f.write("- **First occurrence** of any category → falls back to `global_mean` (the Bayesian prior)\n")
        f.write("- **Val/Test sets** are encoded using the *final* cumulative averages from the Dev set (no refit)\n")
        f.write("- The original categorical string columns are **dropped** after encoding\n\n")
        
        f.write("### Cyclical Time Features\n\n")
        f.write("- `hour_sin`, `hour_cos`: Hour of day (24h cycle) from `TransactionDT`\n")
        f.write("- `day_sin`, `day_cos`: Day of week (7-day cycle) from `TransactionDT`\n")
        f.write("- Captures that 11:59 PM ↔ 12:01 AM are adjacent, not 24h apart\n\n")
        
        # ===================================================================
        # TREE FEATURES
        # ===================================================================
        f.write("---\n\n")
        f.write("## Tree Features (`X_y_*_tree.npz`)\n\n")
        f.write("Used by: **LightGBM, XGBoost, CatBoost, Random Forest**\n\n")
        
        tree_shape = shapes.get("dev_tree", {}).get("x_shape", ("?",))
        f.write(f"**Dimensions**: {tree_shape[1] if len(tree_shape) > 1 else '?'} features\n\n")
        
        f.write("### Preprocessing Pipeline\n\n")
        f.write("1. Cyclical time encoding (hour + day)\n")
        f.write("2. Expanding Mean Target Encoding on all categoricals\n")
        f.write("3. **NaN Imputation**: All remaining numeric NaNs filled with `-999`\n")
        f.write("   - Trees can learn splits on -999 → the missingness pattern becomes a feature itself\n")
        f.write("4. **NO PCA**: V1–V339 kept as raw features (339 columns)\n")
        f.write("5. **NO StandardScaler**: Trees are scale-invariant\n\n")
        
        f.write("### Feature Composition\n\n")
        f.write("| Group | Count | Description |\n")
        f.write("|-------|-------|-------------|\n")
        f.write("| V-features (raw) | 339 | Vesta engineered features, kept uncompressed |\n")
        f.write("| C-features | 14 | Counting features (C1–C14) |\n")
        f.write("| D-features | 15 | Time delta features (D1–D15) |\n")
        f.write("| Target Encoded | ~20 | Categorical columns → numeric TE values |\n")
        f.write("| Cyclical Time | 4 | hour_sin, hour_cos, day_sin, day_cos |\n")
        f.write("| Velocity | 3 | time_dist_last, amt_rolling_mean, amt_rolling_std |\n")
        f.write("| Other numeric | ~43 | TransactionAmt, dist1/2, id_* numeric, etc. |\n\n")
        
        # ===================================================================
        # MLP/VAE FEATURES
        # ===================================================================
        f.write("---\n\n")
        f.write("## Neural Features (`X_y_*_mlp.npz`)\n\n")
        f.write("Used by: **MLP, VAE**\n\n")
        
        mlp_shape = shapes.get("dev_mlp", {}).get("x_shape", ("?",))
        f.write(f"**Dimensions**: {mlp_shape[1] if len(mlp_shape) > 1 else '?'} features\n\n")
        
        f.write("### Preprocessing Pipeline\n\n")
        f.write("1. **NaN Mask Injection**: For every column with missing values, a binary `{col}_is_nan` "
                "column is added (1 = missing, 0 = present). This preserves the 'Signal of Absence'.\n")
        f.write("2. **Median Imputation**: All numeric NaNs filled with the column median from the Dev set.\n")
        f.write("3. Cyclical time encoding (hour + day)\n")
        f.write("4. Expanding Mean Target Encoding on all categoricals\n")
        f.write("5. **Log Transforms**: `log1p(TransactionAmt)` and `log1p(time_dist_last)` to compress "
                "heavy-tailed distributions\n")
        f.write("6. **V-Feature Scaling + PCA**: 339 raw V-features → StandardScaler → PCA → "
                "**50 principal components** (87.7% cumulative variance)\n")
        f.write("7. **Global StandardScaler**: All remaining numeric features scaled to mean=0, var=1\n\n")
        
        f.write("### Key Differences from Tree Features\n\n")
        f.write("| Aspect | Trees | Neural (MLP/VAE) |\n")
        f.write("|--------|-------|------------------|\n")
        f.write("| V-features | 339 raw columns | 50 PCA components |\n")
        f.write("| NaN handling | -999 sentinel | Median + binary mask |\n")
        f.write("| Scaling | None | StandardScaler (mean=0, var=1) |\n")
        f.write("| Amount/Velocity | Raw | log1p transformed |\n\n")
        
        # ===================================================================
        # LSTM FEATURES
        # ===================================================================
        f.write("---\n\n")
        f.write("## LSTM Sequence Features (`X_y_*_lstm.npz`)\n\n")
        f.write("Used by: **FraudLSTM**\n\n")
        
        lstm_shape = shapes.get("dev_lstm", {}).get("x_shape", ("?",))
        if len(lstm_shape) == 3:
            f.write(f"**Dimensions**: `({lstm_shape[0]}, {lstm_shape[1]}, {lstm_shape[2]})` = "
                    f"(transactions, seq_len, features)\n\n")
        
        f.write("### Preprocessing Pipeline\n\n")
        f.write("Same as MLP/VAE (steps 1–7 above), then:\n\n")
        f.write("8. **Transaction-Centric Rolling Windows**: For every transaction at index *i*:\n")
        f.write("   - Identify the user (`Uid`)\n")
        f.write("   - Retrieve up to `seq_len=5` of that user's past transactions (including *i*)\n")
        f.write("   - **Causal zero-padding**: If user has < 5 transactions, pad at the BEGINNING\n")
        f.write("   - Example (user's 2nd transaction): `[0, 0, 0, T1, T2]`\n\n")
        
        f.write("### Key Properties\n\n")
        f.write("- **1:1 alignment**: Exactly the same number of rows as Tree/MLP datasets (no user-level collapse)\n")
        f.write("- **No future leakage**: Transaction *i*'s window only contains transactions ≤ *i*\n")
        f.write("- **Every transaction gets a prediction**: Even a user's first-ever swipe (fully zero-padded)\n")
        f.write("- **Same feature space as MLP**: Each timestep has the same 50 PCA + engineered features\n\n")
        
        # ===================================================================
        # SAVED ARTIFACTS
        # ===================================================================
        f.write("---\n\n")
        f.write("## Saved Preprocessing Artifacts\n\n")
        f.write("`models/preprocessors/preprocessor.joblib` contains:\n\n")
        f.write("| Object | Purpose |\n")
        f.write("|--------|--------|\n")
        f.write("| `scaler` (StandardScaler) | Global feature scaling (mean=0, var=1) |\n")
        f.write("| `v_scaler` (StandardScaler) | V-feature scaling applied BEFORE PCA |\n")
        f.write("| `pca` (PCA, random_state=42) | 339 V-features → 50 principal components |\n")
        f.write("| `te_encoder` (ExpandingMeanEncoder) | Final cumulative target encoding mappings |\n")
        f.write("| `numerical_medians` (dict) | Per-column medians for NaN imputation |\n")
        f.write("| `nan_mask_columns` (list) | Columns that require NaN binary masks |\n")

    print(f"Metadata written to {output}")

if __name__ == "__main__":
    main()
