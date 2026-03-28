# Architectural Decision Document: Dimensionality Reduction

This document details the precise rationale for selecting **Principal Component Analysis (PCA)** over other dimensionality reduction techniques to process the 339 highly-correlated V-features in our IEEE-CIS Fraud Detection architecture. It ensures all technical, mathematical, and production-level constraints are addressed.

---

## 1. The Mathematical Necessity: Why Not Feed Raw Features to the Model?
*Why perform dimensionality reduction at all? Why not just impute nulls, target encode, and pass all 339 raw capabilities directly into the Neural Network?*

### A. Computational Expense
The difference in Neural Network scale between processing 339 raw nodes versus a compressed subset (e.g., 2 Principal Components capturing >99% variance) is massive, drastically accelerating both training cycles and live inference latency while retaining the underlying mathematical structure.

### B. Noise Reduction (Avoiding Corrupt Math & Overfitting)
V-features are heavily redundant and highly correlated (similar to having `price_per_meter` and `price_per_foot` as separate variables).
- **Fragmented Weights:** There are infinite combinations for an algorithm to split the "true signal" weight across perfectly correlated inputs (e.g., `(1, 0)`, `(0, 1)`, `(-100, 101)`, `(0.5, 0.5)`). As a result, network weights can dilute so intensely that they drop beneath computational precision limits, recording as a literal error of `0` and corrupting the system's math. The model might also waste countless epochs attempting to balance these redundant columns.
- **Overfitting to Glitches:** If $V_1$ and $V_2$ correlate at 99.9%, the remaining 0.1% variance is almost exclusively sensor noise or data-collection errors. Given sufficient time, a loss optimizer will exploit this noise to cheat the training loss lower. The resulting model appears flawless on paper but fails miserably on new, real-world transactions.

### C. The "Flat Gradient" Problem
Even if the total loss is identical, extreme redundancy catastrophically alters the **individual feature derivatives (gradients)**—how much responsibility each weight carries for an error.
- **The Diluted Gradient:** If the network predicts off by a value of 10 regarding a specific signal:
  - *With PCA:* The signal is densely captured by just 2 Principal Components. The optimizer computes the derivative and issues a large, clear command: *"Move this specific PC weight by 10 units."* 
  - *Without PCA (100 Redundant Features):* 100 features share the exact same signal. Calculating the partial derivative for a single feature ($V_1$) while holding the other 99 steady tells the optimizer that moving $V_1$ barely impacts total loss. The gradient is severely diluted (e.g., 0.1 units).
- **The Optimizer Update:** Optimizers like Adam or SGD update parameters based on these individual gradients. The compressed PCA model takes a giant leap toward the correct answer. The raw redundant model takes 100 minuscule baby steps.
- **Terrain Topology:** Plotted mathematically, the PC loss landscape forms a steep **V-shape** allowing rapid descent. Redundant feature space forms a wide, shallow **U-shape**. The optimizer crawls massive horizontal distances (weight changes) for barely any vertical payback (loss reduction).
- **The Learning Rate Trap:** With a normal learning rate, the model practically halts in this "flat" landscape. Radically increasing the learning rate to forcibly speed up the descent will mathematically overshoot and destroy the learned weights of *non-redundant* features.

---

## 2. Comparing Dimensionality Reduction Algorithms

### PCA (Unsupervised — "The Compressor")
PCA operates completely blind to targets (`isFraud`), focusing mathematically on maximizing feature **variance**.
- **Mechanics:** "Squashes" 339 redundant columns into dense components (1 up to as many distinct features as exists) containing maximum information.
- **Analogy:** Flattening a 3D pancake into a 2D circle. The core visual footprint is retained, while the irrelevant minimal "thickness" (noise) is discarded.
- **Production Fit:** Blazing fast, excellent at noise reduction, and possesses an instantaneous `.transform()` method for live data out-of-sample mapping. 
- **Requirement Constraint:** Highly sensitive to feature discrepancies; scaling (`StandardScaler`) is strictly required beforehand.

### LDA (Supervised —  "The Over-Fitter")
LDA uses target labels (`isFraud`). It hunts for the exact linear direction that **maximizes the gap** between class means. Its reputation is built entirely on providing "perfect linear separability." However, in modern, high-dimensional fraud detection, this strength is active liability.

#### Why LDA is Not a Top Choice:

**1. The Destructive 1D Squeeze (Squashing a Sphere into a Line)**
  - *The Math Issue:* LDA component generation is mathematically capped at `(Number of Classes - 1)`. Because fraud detection is a binary problem (Fraud vs. Legit), LDA forces all 339 engineered V-features into exactly **one single integer (a 1D straight line)**. 
  - *The Visual Analogy:* Imagine the complex relationships in your data form a 3D sphere. By utilizing 2 PCs that capture 99% of the variance, PCA gracefully compresses that sphere into a 2D circle, retaining its core physical geometry. LDA, however, takes that exact same 3D sphere and violently squashes it into a 1D piece of string. 
  - *Losing the Non-Linear Signal:* Fraud operates fundamentally through dense, non-linear interactions (e.g., "V15 is suspicious *only if* V302 is simultaneously exceptionally low"). By flattening 339 features onto a single continuous axis, LDA completely destroys the internal synergy between those V-features. The Neural Network (LSTM/MLP) never gets the chance to learn those critical non-linear signals because LDA permanently erased the topological "texture" at the extraction layer.

**2. The Illusion of Linear Separability (Weaponizing Noise)**
  - The IEEE-CIS dataset is heavily imbalanced, incredibly sparse, and packed with underlying sensor noise. Because LDA is *supervised*, it essentially "cheats" by looking at the answers during training. 
  - Given 339 features, LDA will happily construct a bizarre, highly artificial mathematical combination that perfectly splits the classes in your *training data* by simply memorizing nulls and noise. This creates a dangerous illusion of perfect linear separability on paper. 
  - Because it hyper-optimizes its rigid 1D boundary by memorizing *past* signals (yesterday's exact fraud dimensions), it behaves extremely fragilely in production. When fraudsters inevitably adapt and real-world data drifts, LDA's strict 1D boundary shatters rendering the model thoroughly blind. Its biggest strength—linear separability—becomes entirely irrelevant on live data.

**3. Broken Core Statistical Assumptions**
  - *The Math Issue:* LDA fundamentally relies on two extremely strict mathematical assumptions: **Multivariate Normality** (that features are perfectly bell-curve distributed) and **Equal Covariance Matrices** (that the "spread" and correlations of the features are exactly the same across all classes).
  - *The Reality Check:* In fraud detection, both assumptions are violently violated. The data is extremely skewed, and fraudsters act erratically—their behavior carries a chaotic, massive variance compared to normal, highly predictable legitimate users. 
  - *Why it matters:* Because LDA inherently restricts itself to assuming the statistical "spread" is identical between Fraud and Legit profiles, it draws a structurally sub-optimal boundary on real transaction data where the true class variances differ drastically.

#### Why 2 PCA Components Win the Architecture
We isolated the V-features down to **just 2 Principal Components** because they mathematically capture **>99% of the dataset’s entire total variance** (PC1 captured ~93%, PC2 captured ~6%). PCA inherently ignores the `isFraud` label, making it immune to overfitting class noise. By providing the Neural Network with a 2D mathematical plane rather than LDA’s 1D line, PCA grants the LSTM the vital topological geometry it requires to confidently map out localized, non-linear fraud "islands" while gracefully surviving long-term production data drift.


### MDS & PCoA (The Matrix Mappers)
Unlike PCA/LDA which analyze columns (variables), MDS (*Multidimensional Scaling*) and PCoA (*Principal Coordinate Analysis / Classical Scaling*) analyze the **Dissimilarity Matrix** (the row-to-row observational distances).
- **MDS Utility:** Mathematically preserves high-dimensional distances on a low-dimensional plane. Excellent for seeing if 100 "Known Fraudsters" naturally cluster given their behavioral footprints.
- **PCoA Utility:** If utilizing Euclidean distance, PCoA delivers the precise outputs of PCA. However, PCoA thrives when using non-Euclidean formats (e.g., Jaccard, Bray-Curtis). Because fraud data is inherently sparse, using metrics that ignore "double zeros" (where features for two accounts are null) can expose groupings that PCA natively skips.
- **The Fatal Flaws for core modeling:** 
  1. *Computationally Impossible:* These algorithms scale at $O(N^2)$. Executing them across 500k rows of IEEE-CIS data will literally crash standard hardware. They must be reserved for narrow samples (~5,000 rows).
  2. *No Live Integration:* Neither tool enables out-of-sample mapping (`transform()`). When processing brand new live transactions, you cannot just project them; you must rerun the entire $O(N^2)$ algorithm over the combined dataset—a production nightmare.

### t-SNE vs. UMAP (The Visual Artists)
- **t-SNE:** The "high-definition camera". It reconstructs localized neighborhoods by calculating high-dimensional neighbor probabilities and mapping them locally in 2D/3D. It exposes incredible non-linear, hidden "islands" invisible to linear algorithms.
- **The Reality Check:** Do not use t-SNE for core feature extraction to an LSTM. It carries massive computational gravity ($O(N^2)$ or $O(N \log N)$), lacks a `transform()` method (zero mapping utility for new data), and mathematically, macro distances and cluster sizes mean nothing (only internal cluster connectivity is reliable).
- **UMAP (The Modern Rival):** Provides t-SNE's beautiful topological clusters but operates noticeably faster, preserves a more truthful global structure, and importantly supports new-data transformations natively.

### Summary Comparison Tables

**Table 1: Training Models (PCA vs. LDA)**
| **Feature** | **PCA** | **LDA** |
| --- | --- | --- |
| **Logic Type** | Unsupervised (No labels) | Supervised (Uses `isFraud` label) |
| **Core Goal** | Maximize Variance | Maximize Class Separation |
| **Component Limit** | As many as original features | Restricted to (Number of Classes - 1) |
| **Primary UseCase**| General Noise Reduction / Compression | Classification Feature Extraction |
| **Sensitivity** | Data Scaling (Requires StandardScaler) | Data Scaling & Outlier Sensitivity |

**Table 2: Algorithm Mechanics Overview**
| **Algorithm** | **Input Type** | **Calculative Logic** | **Best Used For** |
| --- | --- | --- | --- |
| **PCA** | Raw Features | Maximize Variance (Linear) | Noise Reduction for V-Features |
| **LDA** | Features + Labels | Maximize Separation (Linear)| Classification Signal Check |
| **MDS** | Distance Matrix | Preserved Matrix Distances | Visualizing Clusters (Small Samples)|
| **PCoA** | Distance Matrix | PCA on Calculated Distances | Mapping Non-Euclidean Relationships |

**Table 3: Compression vs. Topology**
| **Feature** | **PCA** | **t-SNE** | **UMAP** |
| --- | --- | --- | --- |
| **Mathematical Logic**| Linear Array Variance | Non-Linear (Local Neighbors) | Non-Linear (Topological) |
| **Processing Speed** | Blazing Fast | Very Slow | Fast |
| **Used For Training?**| **Yes** (Model Compression) | **No** (Visualization Exclusively) | **Maybe** (Feature Extraction) |
| **Maps New Data?** | Yes | No | Yes |

---

## 3. Final Architecture Verdict & Senior Recommendations

For a robust, production-grade fraud system capable of impressing at a professional level:

1. **Core Pipeline (Group 1 & 3):** Exclusively deploy **PCA** onto the V-features. It bypasses the flat-gradient issue, economically scrubs variance redundancies, constructs a preserved 2D topological plane for the NN to navigate (capturing >99% variance in just 2 components), and scales effortlessly to live production architectures via rapid `.transform()` functions.
2. **Exploratory Tools:** Utilize **LDA** early merely as validation to verify if data subsets are linearly separable (fraud rarely is, hence the necessity of the LSTM pipeline).
3. **The "Hero Slide":** Reserve **MDS** and **t-SNE** for your final presentation. Apply them exclusively against distinct, computationally-safe data fragments (~5,000 row samples). They operate dynamically as high-quality visualization assets enabling you to showcase visually separate fraud clusters explicitly to professors and stakeholders (e.g., *"Look how cleanly my features partitioned the fraud cluster"*).