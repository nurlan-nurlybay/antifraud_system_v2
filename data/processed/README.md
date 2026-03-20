# Stage A: Data Preprocessing & Hybrid Feature Engineering
**Project:** Antifraud System v2.2  
**Dataset:** IEEE-CIS Fraud Detection (Transaction & Identity)

## 1. The Validation Strategy: The "Wall of Time"
Unlike standard tabular datasets, financial transactions are strictly **non-i.i.d.** (Independent and Identically Distributed). Standard K-Fold or **Nested K-Fold CV** were rejected for this stage.
* **The Problem with K-Fold:** In this dataset, multiple transactions often belong to the same user in a short window. Randomly shuffling these into training and validation folds leads to "Data Leakage," where the model "remembers" a specific user's behavior rather than learning general fraud patterns.
* **The Solution:** We implemented a strict **Chronological Split (80/10/10)**. 
    * **Dev Set (80%):** Used for L0 model training.
    * **Meta-Train (10%):** Used to generate Out-Of-Fold (OOF) predictions for the Stacker.
    * **Meta-Val (10%):** The final, untouched "future" data to verify system performance.

---

## 2. Dimensionality Reduction: PCA vs. The Alternatives
The dataset contains **339 V-features** (V1-V339) which are highly collinear and redundant.
* **Why PCA?** We utilized Principal Component Analysis to compress these 339 dimensions into just **2-5 components**, capturing over **99% of the variance**. 
* **Why not UMAP or t-SNE?** While UMAP and t-SNE are excellent for visualization, they are non-linear and computationally expensive. More importantly, they do not provide a stable linear projection that can be easily applied to new, unseen data during inference without significant risk of manifold shift. PCA provides a stable, repeatable linear transformation optimized for the MLP and LSTM paths.



---

## 3. The "Signal of Absence" (NaN Strategy)
Initial analysis revealed a critical "Median Signal Killer." Standard imputation (filling NaNs with the median) was inadvertently sanitizing the data, making fraudsters look like "average" customers.
* **The Discovery:** Many features in fraud detection are **MNAR** (Missing Not At Random). The absence of data (e.g., a missing proxy IP or device ID) is often a stronger fraud indicator than the data itself.
* **The Solution:** * **Tree-Based Path:** Imputed with an out-of-bounds constant (**-999**). This allows decision trees to create a clean split between "Data Present" and "Data Missing."
    * **Neural Path (MLP/LSTM):** Imputed with the **Median** to maintain gradient stability, but supplemented with a **Binary Mask** (`_is_nan` column). This forces the Neural Network to recognize the difference between a "real" average value and an "imputed" one.

---

## 4. Normalization & Transformation Techniques
To ensure all model archetypes can converge efficiently, the following mathematical transforms were applied:
* **Logarithmic Transform:** $TransactionAmt$ was transformed using $\log(1+x)$ to handle heavy-tailed skewness and reduce the influence of extreme outliers.
* **Cyclical Time Encoding:** The $TransactionDT$ (seconds from start) was mapped to a 24-hour cycle using Sine and Cosine transforms:
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
  This allows the model to understand that 23:59 and 00:01 are mathematically adjacent.
* **Standard Scaling:** All numerical inputs for the Neural paths were Z-score normalized to a mean of 0 and standard deviation of 1.

---

## 5. Artifact Structure
The preprocessing pipeline outputs three distinct `.npz` files for each split to minimize memory overhead during training:

| Artifact | Purpose | Imputation | Dimensionality |
| :--- | :--- | :--- | :--- |
| `_tree.npz` | XGBoost/CatBoost | -999 | Raw (Target Encoded) |
| `_mlp.npz` | Neural Base | Median + Mask | PCA Compressed |
| `_lstm.npz` | Sequence Model | Median + Mask | 3D Tensors $[N, 5, F]$ |

---