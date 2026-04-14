# Processed Dataset Manifest — Antifraud System v2.0

**Source**: IEEE-CIS Fraud Detection (590,540 raw transactions)

**Split**: Chronological 90%/10% (Dev / Meta-Validation)

**Preprocessor Artifact**: `models/preprocessors/preprocessor.joblib`

---

## Dataset Files

| File | X Shape | y Shape | Fraud | Rate | dtype | Size |
|------|---------|---------|-------|------|-------|------|
| `X_y_dev_tree.npz` | `(531486, 438)` | `(531486,)` | 18,450 | 3.47% | float32 | 101.4 MB |
| `X_y_dev_mlp.npz` | `(531486, 195)` | `(531486,)` | 18,450 | 3.47% | float32 | 143.0 MB |
| `X_y_dev_lstm.npz` | `(531486, 5, 195)` | `(531486,)` | 18,450 | 3.47% | float32 | 666.3 MB |
| `X_y_meta_val_tree.npz` | `(59054, 438)` | `(59054,)` | 2,213 | 3.75% | float32 | 7.6 MB |
| `X_y_meta_val_mlp.npz` | `(59054, 195)` | `(59054,)` | 2,213 | 3.75% | float32 | 11.5 MB |
| `X_y_meta_val_lstm.npz` | `(59054, 5, 195)` | `(59054,)` | 2,213 | 3.75% | float32 | 39.1 MB |

---

## Shared Preprocessing (All Datasets)

These steps are applied identically to Dev and Meta-Val data via `DataPreprocessor.clean_base_data()`:

1. **Merge**: Transaction table LEFT JOIN Identity table on `TransactionID`
2. **User ID Creation**: `Uid` = hash of `[card1, card2, card3, card4, card5, card6, addr1, P_emaildomain]`
3. **Chronological Sort**: All rows sorted by `TransactionDT` (ascending). This is critical for the Expanding Mean Target Encoder.
4. **Velocity Feature**: `time_dist_last` = time since user's previous transaction (`groupby(Uid).diff()` on `TransactionDT`). First transaction per user → NaN.
5. **Rolling Amount Features**:
   - `amt_rolling_mean`: Rolling 5-transaction mean of `TransactionAmt` per user
   - `amt_rolling_std`: Rolling 5-transaction standard deviation per user. Detects sudden spending anomalies.
6. **Categorical Imputation**: All string columns filled with `"Unknown"`

### Expanding Mean Target Encoder (Zero Temporal Leakage)

Replaces each categorical column with a numeric `_te` suffix column.

For transaction *i* with category value *c*:

```
encoding_i = (cumulative_fraud_sum_of_c_before_i + m * global_mean) / (cumulative_count_of_c_before_i + m)
```

- **m** = 10 (Bayesian smoothing strength)
- **First occurrence** of any category → falls back to `global_mean` (the Bayesian prior)
- **Val/Test sets** are encoded using the *final* cumulative averages from the Dev set (no refit)
- The original categorical string columns are **dropped** after encoding

### Cyclical Time Features

- `hour_sin`, `hour_cos`: Hour of day (24h cycle) from `TransactionDT`
- `day_sin`, `day_cos`: Day of week (7-day cycle) from `TransactionDT`
- Captures that 11:59 PM ↔ 12:01 AM are adjacent, not 24h apart

---

## Tree Features (`X_y_*_tree.npz`)

Used by: **LightGBM, XGBoost, CatBoost, Random Forest**

**Dimensions**: 438 features

### Preprocessing Pipeline

1. Cyclical time encoding (hour + day)
2. Expanding Mean Target Encoding on all categoricals
3. **NaN Imputation**: All remaining numeric NaNs filled with `-999`
   - Trees can learn splits on -999 → the missingness pattern becomes a feature itself
4. **NO PCA**: V1–V339 kept as raw features (339 columns)
5. **NO StandardScaler**: Trees are scale-invariant

### Feature Composition

| Group | Count | Description |
|-------|-------|-------------|
| V-features (raw) | 339 | Vesta engineered features, kept uncompressed |
| C-features | 14 | Counting features (C1–C14) |
| D-features | 15 | Time delta features (D1–D15) |
| Target Encoded | ~20 | Categorical columns → numeric TE values |
| Cyclical Time | 4 | hour_sin, hour_cos, day_sin, day_cos |
| Velocity | 3 | time_dist_last, amt_rolling_mean, amt_rolling_std |
| Other numeric | ~43 | TransactionAmt, dist1/2, id_* numeric, etc. |

---

## Neural Features (`X_y_*_mlp.npz`)

Used by: **MLP, VAE**

**Dimensions**: 195 features

### Preprocessing Pipeline

1. **NaN Mask Injection**: For every column with missing values, a binary `{col}_is_nan` column is added (1 = missing, 0 = present). This preserves the 'Signal of Absence'.
2. **Median Imputation**: All numeric NaNs filled with the column median from the Dev set.
3. Cyclical time encoding (hour + day)
4. Expanding Mean Target Encoding on all categoricals
5. **Log Transforms**: `log1p(TransactionAmt)` and `log1p(time_dist_last)` to compress heavy-tailed distributions
6. **V-Feature Scaling + PCA**: 339 raw V-features → StandardScaler → PCA → **50 principal components** (87.7% cumulative variance)
7. **Global StandardScaler**: All remaining numeric features scaled to mean=0, var=1

### Key Differences from Tree Features

| Aspect | Trees | Neural (MLP/VAE) |
|--------|-------|------------------|
| V-features | 339 raw columns | 50 PCA components |
| NaN handling | -999 sentinel | Median + binary mask |
| Scaling | None | StandardScaler (mean=0, var=1) |
| Amount/Velocity | Raw | log1p transformed |

---

## LSTM Sequence Features (`X_y_*_lstm.npz`)

Used by: **FraudLSTM**

**Dimensions**: `(531486, 5, 195)` = (transactions, seq_len, features)

### Preprocessing Pipeline

Same as MLP/VAE (steps 1–7 above), then:

8. **Transaction-Centric Rolling Windows**: For every transaction at index *i*:
   - Identify the user (`Uid`)
   - Retrieve up to `seq_len=5` of that user's past transactions (including *i*)
   - **Causal zero-padding**: If user has < 5 transactions, pad at the BEGINNING
   - Example (user's 2nd transaction): `[0, 0, 0, T1, T2]`

### Key Properties

- **1:1 alignment**: Exactly the same number of rows as Tree/MLP datasets (no user-level collapse)
- **No future leakage**: Transaction *i*'s window only contains transactions ≤ *i*
- **Every transaction gets a prediction**: Even a user's first-ever swipe (fully zero-padded)
- **Same feature space as MLP**: Each timestep has the same 50 PCA + engineered features

---

## Saved Preprocessing Artifacts

`models/preprocessors/preprocessor.joblib` contains:

| Object | Purpose |
|--------|--------|
| `scaler` (StandardScaler) | Global feature scaling (mean=0, var=1) |
| `v_scaler` (StandardScaler) | V-feature scaling applied BEFORE PCA |
| `pca` (PCA, random_state=42) | 339 V-features → 50 principal components |
| `te_encoder` (ExpandingMeanEncoder) | Final cumulative target encoding mappings |
| `numerical_medians` (dict) | Per-column medians for NaN imputation |
| `nan_mask_columns` (list) | Columns that require NaN binary masks |
