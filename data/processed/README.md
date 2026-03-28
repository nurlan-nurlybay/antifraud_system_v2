# Stage A: Data Preprocessing & Hybrid Feature Engineering
**Project:** Antifraud System v2.2  
**Dataset:** IEEE-CIS Fraud Detection (Transaction & Identity)

## 1. Feature Engineering: Added & Dropped Dimensions
The system transforms the raw 434-column IEEE-CIS dataset into optimized tensors by explicitly adding behavioral signals and removing high-cardinality or redundant noise.

### 1.1 Engineered Features (Added)
* **Uid (User Identifier):** Synthesized by concatenating `card1-6`, `addr1`, and `P_emaildomain`. This feature is used for chronological sorting and LSTM grouping before being moved to metadata.
* **Cyclical Time (hour_sin, hour_cos):** Derived from `TransactionDT` to map the 86,400-second day into a circular wave, capturing 24-hour periodic fraud patterns.
* **Target Encodings (_te):** All 33 categorical columns (e.g., `ProductCD`, `DeviceInfo`) were transformed into scalar likelihoods using 5-fold Bayesian smoothed target encoding.
* **Binary Missingness Masks (_is_nan):** [Neural/LSTM Path only] 42+ new binary columns added to flag the "Signal of Absence" for critical numerical features like `D`-columns and `dist1`.
* **Principal Components (V_pca_0, V_pca_1):** [Neural/LSTM Path only] Two linear projections capturing >99% variance of the 339 V-features.

### 1.2 Feature Selection (Dropped)
* **Metadata Discards:** `TransactionID`, `TransactionDT`, and `Uid` are removed from the feature matrix $X$ to prevent the model from memorizing specific IDs or timestamps.
* **Raw Categoricals:** All original string/object columns are dropped after being replaced by their respective Target Encoded scalar versions.
* **V-Feature Redundancy:** [Neural/LSTM Path only] The original 339 V-features are entirely discarded and replaced by the 2 PCA components.
* **Target Leakage:** `isFraud` is extracted and saved as the target vector $y$.

---

## 2. Model-Specific Transformation Paths
To maximize the mathematical strengths of different model archetypes, the pipeline bifurcates into four distinct normalization and transformation strategies.

### 2.1 Universal Base Logic (All Paths)
All transaction amounts are normalized using a Log-Transform ($\log(1+x)$) to handle the extreme positive skew and the presence of "Whale" transactions. Categorical encoding is standardized across all versions to ensure feature parity in the Meta-Stacker.

### 2.2 Path A: Neural Transformation (MLP & VAE)
* **Normalization:** Employs Z-score Standard Scaling (Mean 0, Std 1) for all 141 features. This ensures numerical stability for backpropagation and prevents the "Exploding Gradient" problem.
* **Dimensionality:** Reduced to 141 features. This includes the 2 PCA V-components and the 42+ missingness masks required to preserve signal in the presence of medians.

### 2.3 Path B: Tree-Based Transformation (XGBoost / CatBoost)
* **Imputation:** Utilizes "Spatial Splitting" by filling all numerical NaNs with **-999**. This creates a distinct territory in the decision tree manifold for missing data.
* **Dimensionality:** Retains the high-density 433-feature footprint. Decision trees natively handle the collinearity of the 339 V-features, making PCA unnecessary and potentially harmful to information gain.

### 2.4 Path C: Sequence Transformation (LSTM)
* **Reshaping:** Data is transformed into Rank-3 Tensors $[N, 5, 141]$. 
* **Temporal Padding:** Individual user histories are truncated or pre-padded with zeros to reach a fixed depth of 5 timesteps.
* **Feature Set:** Mirrors the MLP path (141 features) to ensure that behavioral shifts across the 5 steps are captured with standardized gradients.

---

## 3. The "Signal of Absence" (NaN Strategy)
The system addresses the "Median Signal Killer" problem—where filling missing values with the median makes suspicious, data-poor transactions appear "average."
* **The Strategy:** For tree models, the -999 constant allows for explicit binary splits on missingness. For neural models, the supplementary binary masks (`_is_nan`) force the network to weight the "fact of missingness" separately from the "imputed value," preserving the behavioral intent behind hidden data fields.

---

## 4. Artifact Structure
The preprocessing pipeline outputs the following compressed `.npz` files, split into a **90% Dev Set** and a **10% Meta-Validation Set**.

| Artifact | Purpose | Depth | Feature Width | Fraud Rate |
| :--- | :--- | :--- | :--- | :--- |
| `X_y_dev_tree.npz` | Tree Training | 1 (2D) | 433 | 3.47% |
| `X_y_dev_mlp.npz` | Neural Training | 1 (2D) | 141 | 3.47% |
| `X_y_dev_lstm.npz` | Sequence Training | 5 (3D) | 141 | 2.69%* |

*\*Note: The LSTM fraud rate is lower because labels are assigned based only on the latest transaction in a user's 5-step sequence.*