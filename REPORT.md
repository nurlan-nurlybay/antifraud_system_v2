# 📑 REPORT.md: Antifraud System v2.0 — Comprehensive Engineering Architecture

## 1. Executive Summary

This document details the architectural decisions, mathematical foundations, and MLOps engineering behind the IEEE-CIS Fraud Detection "Super-Learner" ensemble. The primary objective of this system is to deploy a VRAM-resident, high-throughput model resilient to real-world concept drift.

### The PR-AUC Imperative & System Performance
In real-world fraud detection, ROC-AUC can be a vanity metric because it inherently rewards the correct identification of the 96.5% of legitimate transactions. The true measure of a commercial antifraud system is **PR-AUC (Precision-Recall Area Under Curve)**, which focuses exclusively on the minority fraud class.

* **Ensemble PR-AUC (Strict Temporal Holdout):** `0.6062`
* **Ensemble ROC-AUC (Kaggle Private):** `0.899386`

While recent academic benchmarks frequently report PR-AUCs exceeding 0.90 on this dataset, those results are predominantly achieved via Random Cross-Validation—a leaky strategy that allows models to "look into the future" to learn retroactive fraud schemes. By enforcing a strict chronological validation split, our `0.6062` PR-AUC represents an honest, robust baseline capable of generalizing against forward-moving fraud velocity without temporal leakage.

---

## 2. Data Engineering & The "Signal of Absence"

The preprocessing pipeline handles 531,486 training rows dynamically, splitting logic between tree-based architectures and gradient-descent neural networks.

### Temporal Ordering & The Expanding Mean Target Encoder
All data is strictly sorted chronologically by `TransactionDT`. Categorical features are encoded using an Expanding Mean Target Encoder with Bayesian smoothing to absolutely prevent future-data leakage:

$$E_i = \frac{\sum_{j < i} y_j + m \cdot \mu}{N_{j < i} + m}$$

Where $m=10$ provides stabilization against rare, early-sequence categories.

### Feature Engineering Highlights
* **Velocity Tracking:** The `time_dist_last` feature calculates the delta since a user's previous transaction (hashed via `Uid`). Rolling 5-transaction means and standard deviations of `TransactionAmt` flag sudden spending bursts.
* **Cyclical Time Encoding:** 24-hour and 7-day cyclical continuity is preserved using trigonometric mappings:
    * $Hour_{sin} = \sin(2\pi \cdot \frac{hour}{24})$
    * $Hour_{cos} = \cos(2\pi \cdot \frac{hour}{24})$
* **The Signal of Absence:** For neural networks (MLP/VAE/LSTM), numeric missing values are median-imputed, but a simultaneous `{col}_is_nan` binary mask is injected. This ensures the network retains the exact structural topology of the missingness—which is often a primary indicator of synthetic fraud identities.
* **Sequential Padding:** The LSTM ingests causal, zero-padded 5-step rolling windows `[0, 0, 0, T_{i-1}, T_i]`, maintaining a perfect 1:1 row alignment with the static models.

---

## 3. Dimensionality Reduction: The Mathematical Necessity of PCA

This section details the precise rationale for selecting **Principal Component Analysis (PCA)** over other dimensionality reduction techniques to process the 339 highly-correlated V-features.

### A. The "Flat Gradient" Problem & Noise Reduction
V-features are heavily redundant and highly correlated (similar to having `price_per_meter` and `price_per_foot` as separate variables). Passing 339 raw features into a Neural Network causes catastrophic gradient dilution.
* **Fragmented Weights:** There are infinite combinations for an algorithm to split the "true signal" weight across perfectly correlated inputs. Network weights dilute so intensely they can drop beneath computational precision limits.
* **Overfitting to Glitches:** If $V_1$ and $V_2$ correlate at 99.9%, the remaining 0.1% variance is almost exclusively sensor noise. A loss optimizer will exploit this noise to cheat the training loss lower, failing on real-world transactions.
* **The Diluted Gradient:** If the network predicts off by 10 units, PCA (using 2 components) issues a large, clear derivative command. Without PCA, 100 redundant features share the signal, severely diluting the gradient (e.g., 0.1 units). The PC loss landscape forms a steep **V-shape** allowing rapid descent, whereas redundant feature space forms a wide, shallow **U-shape**.

### B. Comparing Algorithms
* **PCA (Unsupervised):** Operates blind to the target, maximizing variance. It elegantly compresses the 339 dimensions into 50 Principal Components (capturing 87.7% variance), scrubbing redundancy while retaining topology. It operates blazingly fast with an instantaneous `.transform()` method for live data out-of-sample mapping.
* **LDA (Supervised):** Hunts for linear separation. Because fraud detection is a binary problem, LDA violently squashes the complex 339-dimensional manifold into exactly **one single integer (a 1D straight line)**. This destroys the critical non-linear synergy between V-features. It is highly prone to weaponizing noise to create the illusion of perfect linear separability, which shatters in production.
* **MDS / t-SNE / UMAP:** While excellent for visual "Hero Slides" on small data samples, $O(N^2)$ scaling and the lack of native, instantaneous `.transform()` mechanisms make them impossible to deploy in a live transaction pipeline.

---

## 4. Defeating Temporal Leakage: Cross-Validation Strategy

Standard nested K-fold Cross Validation is actively harmful for sequential transaction data and was explicitly rejected in favor of strict **Walk-Forward CV**.

### Why K-Fold Fails for Fraud Detection
Frauds come in trends or schemes that are effective for a certain period before they get patched (e.g., Type A in January, Type B in February). 
In a 10-fold CV setup, the models validate on data while training on both *past* and *future* neighboring data.
* **Trend Bias Overfitting:** When training models (e.g., folds 4, 5, 6), 6 out of the 9 training folds are contextually biased (3 months prior, 3 months subsequent). These models overfit to temporary fraud trends rather than learning rigorous, universal fraud characteristics.
* If we quantify the bias: 4 folds are 6/9 biased, 2 are 5/9, 2 are 4/9, and 2 are 3/9. The total bias averages **54%**, and picking the "top 3" outer models guarantees selecting the models with a maximum **67% contextual bias**.

### The Walk-Forward CV Solution
We utilize an expanding chronological window: Train on $0 \to i$, validate on $i+1$, and test on $i+2$.
* **Reduced Bias:** The later models have significantly less trend bias (average bias drops to roughly **42%**, and the best models sit at **33%** bias). 
* This approach organically forces the algorithm to adapt to concept drift, ensuring hyperparameter selection rewards universal stability over temporal memorization. The final Stacker weights are dynamically penalized or boosted based on this rigorous chronological robustness.

---

## 5. The Imbalance Engine: Focal BCE Loss

The dataset exhibits a 97% normal to 3% fraud ratio (a 32x imbalance). To stabilize PyTorch gradient descent, we utilize a customized Focal Binary Cross-Entropy Loss.

### $\alpha$: Class Weighting Tradeoffs
Standard weighting multiplies the rare class loss by 32 to achieve mathematical balance. 
* **The Flaw:** Fraud is inherently noisy. Forcing 3% of the data to carry the exact same gradient weight as 97% of the data causes the network to violently overfit to the fraud noise.
* **The Solution:** We utilize a dampened $\alpha < ratio$. For example, an effective 4x multiplier ($\alpha = 0.8$ for fraud, $0.3$ for normal) ensures the network cares about the rare cases without becoming fanatical.

### $\gamma$: Focal-Like Scaling (The Two-Stage Learner)
Focal scaling dynamically adjusts the loss based on prediction confidence:
* Fraud loss scaled by: $(1 - p)^\gamma$
* Normal loss scaled by: $p^\gamma$

Setting $\gamma = 2$ fundamentally alters training dynamics:
* *Bad fraud prediction ($p=0.1$):* $(1 - 0.1)^2 = 0.81$ (preserves 81% of the loss).
* *Good normal prediction ($p=0.1$):* $0.1^2 = 0.01$ (nukes 99% of the loss).

**The Two-Stage Training Result:** Because total loss is dominated by the normal class, **Stage A** forces the model to quickly learn how to identify normals. Once normal predictions become competent ($p \approx 0.2$), their loss mathematically shrinks as $x^2$. As the gradient for normal data flatlines, the optimizer smoothly transitions into **Stage B**, focusing its remaining momentum exclusively on the high-loss fraud data.

$$L_{FBCE} = -\alpha (1-p)^\gamma y \log(p) - (1-\alpha) p^\gamma (1-y) \log(1-p)$$

---

## 6. PyTorch Regularization: The "Triple Threat" Architecture

To maximize the diversity of the Meta-Stacker, we applied specific, divergent regularization strategies to the PyTorch models, creating a "Triple Threat" ensemble.

### The MLP (Fixed L2, Tuned L1): The Sparse Architect
We instantiated a massive 5-layer, 2048-width deep learning space.
* **L2 (Weight Decay):** Locked at a safe floor ($10^{-5}$) by AdamW to prevent gradient explosion.
* **L1 (Sparsity):** Tuned dynamically by Optuna.
* **The Result:** Optuna selected an extraordinarily small `32-unit` width architecture. The L1 penalty created a massive "penalty debt" for the 4.1 million connections in the 2048-width layers. It forced the model to prune itself, proving that the 50 PCA components provided such high-quality signal density that massive non-linear capacity was unnecessary. 

### The LSTM (Fixed L2, ZERO L1): The Smooth Temporalist
LSTMs rely on recurrent weights connecting time steps.
* **The Risk of L1:** Applying L1 to an LSTM causes "Dead History." If L1 zeroes out a connection to keep the model sparse, that connection is dead for the entire 5-step sequence, destroying the velocity signal.
* **The Solution:** We removed L1 entirely from the LSTM and relied solely on tuned L2 Ridge regularization. This allows the connections to stay alive but smooth, maintaining a faint memory of $T_{i-4}$ while focusing heavily on $T_i$.

---

## 7. The VAE Post-Mortem & FP16 Explosion Fix

The Variational Autoencoder (VAE) was designed for unsupervised anomaly detection via reconstruction error. While it ultimately contributed marginally less to the linear Meta-Stacker, the engineering required to stabilize it was significant.

### Resolving the FP16 `NaN` Explosion
During Automatic Mixed Precision (AMP) training, model activations are calculated in `float16` (maximum ceiling: `65504.0`).
Because the `kld_weight` was dynamically tuned and L1 was removed, the unconstrained latent layer `logvar` spiked to `11.1`. The KLD penalty calculates `logvar.exp()`, resulting in $e^{11.1} = 66171.0$, immediately breaching the `float16` ceiling and poisoning the GPU weights with `NaN`.

**The Architectural Fix:**
A strict gradient clamp was injected directly into the forward path:
```python
logvar = torch.clamp(self.fc_logvar(encoded), min=-10.0, max=10.0)
```
* **Why 10.0 and not 5.0?** `torch.clamp()` simultaneously zeroes out the gradient for values hitting the ceiling. If capped too tightly at `5.0`, natural initialization swings hit the wall, the gradient goes flat (`0.0`), and the optimizer becomes permanently blind, unable to pull the parameter back down. Capping at `10.0` provides a massive 22,000 variance safety buffer, allowing early wild swings to generate the steep downward gradients required to stabilize.

### Root Causes of Underperformance
1.  **The 10-Epoch Blindfold:** VAEs fight a two-front war (MSE vs. KLD). Given only 10 epochs during tuning, Optuna could not accurately judge convergence.
2.  **KLD Collapse:** The `beta` penalty was tuned too low ($0.0001$), turning the VAE into a lazy identity-mapper rather than a strict Gaussian bottleneck.

---

## 8. Meta-Stacker Orchestration & Ablation

The final orchestration layer combines the OOF probabilities of the 7 base models utilizing Logistic Regression.

### Stacker Bugs & Resolution
Initially, the stacker catastrophically underperformed the individual base models due to two structural mathematical conflicts:
1.  **The Scale Mismatch:** The VAE output raw, unbounded MSE errors (e.g., $0.5$ to $450.0$), while trees output standard $[0, 1]$ probabilities. L2 regularization penalized the massive VAE values exponentially, forcing its learned coefficient to near-zero. **Fix:** We applied a `log1p` squash followed by a `MinMaxScaler` to safely bind the anomalies to a $[0, 1]$ interval without destroying outlier distancing.
2.  **The Double-Balance Trap:** Initializing the stacker with `class_weight="balanced"` forced a second 14x penalty multiplier on top of already-calibrated base probabilities. This broke the optimizer, driving the $C$ parameter to the absolute floor ($0.000137$). **Fix:** Removed the class weight, allowing the stacker to learn raw combinatorial trust.

### The 3-Tiered Holdout & 127-Combination Ablation
To prevent the LogReg from overfitting its own hyperparameters, we mapped a strict **80 / 10 / 10** chronological split in-memory (Train Base / Tune Meta / Blind Test).

Because setting a model's prediction vector to `0.0` alters the LogReg intercept shift and breaks the underlying math, the ablation study was conducted rigorously. We wrote a combinatorics script to sequentially generate and retrain the Logistic Regression across all **127 possible subsets** ($\sum \binom{7}{k}$).

**Results:** The final 6-model ensemble (excluding the VAE) provided a mathematically proven **+1.2% PR-AUC boost** over the single best base model (LightGBM), yielding the final `0.6062` validation score.