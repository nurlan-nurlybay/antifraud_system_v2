# 🧠 SHAP Interpretability Report: Antifraud System v2.0

This report provides a technical breakdown of the feature importance and model logic for the full v2.0 ensemble. These insights were generated using SHapley Additive exPlanations (SHAP) to ensure transparency in both our Gradient Boosted Trees and Deep Learning models.

---

## 🏗️ Ensemble Logic (Meta-Weights)

Refer to **`meta_weights.png`** in this directory.

### Findings:
- **Primary Anchors**: The stacker relies most heavily on **LightGBM** and **CatBoost**. These models provide the most stable "fraud signal."
- **Noise Correction (Random Forest)**: You will notice a **negative coefficient** for Random Forest. 
  - *Reason*: RF is highly redundant with LightGBM. In a linear meta-model, a negative weight on a less-accurate but highly-correlated model allows the stacker to "subtract" out consistent errors or biases, essentially refining the higher-quality LightGBM prediction.
- **Deep Learning Synergy**: While individual neural network weights (MLP, LSTM) are lower, they contribute unique "temporal" and "anomaly" signals that improved the final ensemble's PR-AUC by +1.2%.

---

## 🌲 Tree Model Insights (LightGBM / XGBoost / CatBoost)

Refer to **`shap_lightgbm.png`**, **`shap_xgboost.png`**, and **`shap_catboost.png`**.

### How to read these "Beeswarm" plots:
- **Horizontal Axis (SHAP Value)**: How much a feature shifted the fraud probability. **Right of 0** = Increases Risk; **Left of 0** = Decreases Risk.
- **Color**: **Red** = High feature value; **Blue** = Low feature value.

### Key Drivers across Trees:
1. **`TransactionAmt` (Red on Right)**: High transaction amounts are a top-3 universal indicator of fraud risk.
2. **`card1` / `addr1` (Low Values = Risk)**: Specific card ranges and zip code clusters (blue dots) show significant risk spikes, likely representing targeted BIN/region attacks.
3. **`C13` (Low Values = Risk)**: This counting feature often captures account age/velocity; newer accounts or low-activity paths are seen as more suspicious.
4. **`amt_rolling_std` (Red on Right)**: High volatility in a user's recent spending history is a massive fraud trigger.

---

## 🧠 Deep Learning & Anomaly Patterns (MLP / LSTM / VAE)

### 1. Neural Networks (MLP & LSTM)
Refer to **`shap_mlp_reconstructed.png`** and **`shap_lstm.png`**.

- **Probability Space**: These plots are normalized to the `[0, 1]` range.
- **Temporal Insight (LSTM)**: The LSTM prioritizes the sequence of `TransactionAmt` and `hour_sin`. It is uniquely capable of spotting "burst" activity at night (nocturnal fraud patterns) that static tree models might miss.
- **V-Feature Importance**: Our MLP reconstruction shows that specific PCA components (capturing Vesta engineered interactions) are the primary drivers for the neural stack.

### 2. Anomaly Triggers (VAE Reconstruction Error)
Refer to **`shap_vae.png`**.

- **What it interprets**: This plot shows what makes a transaction "weird" according to the VAE.
- **Top Driver**: `amt_rolling_std`. The VAE is highly sensitive to behavioral shifts. If a user's spend amount suddenly varies wildly compared to their last 5 swipes, the VAE flags it as a high-reconstruction-error event.

---

## 🏁 Summary for Stakeholders
The v2.0 system is not a "black box." It identifies fraud based on **Value Velocity** (rolling spending patterns), **Identity Clusters** (card/address signatures), and **Temporal Bursts** (sequence modeling).

*Report Generated: 2026-04-15*
