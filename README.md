# DA5401 – Assignment 7: Multi-Class Model Selection using ROC and Precision–Recall Curves  

**Name:** Bavge Deepak Rajkumar  
**Roll Number:** NA22B031  

---

## Overview  

This assignment focuses on **multi-class model selection** using **Receiver Operating Characteristic (ROC)** and **Precision–Recall (PRC)** analysis.  
The objective is to evaluate multiple classification models on the **Landsat Satellite dataset** and determine the best-performing classifier by comparing **threshold-independent metrics** like **AUC (Area Under the Curve)** and **Average Precision (AP)** rather than relying solely on accuracy.  

This exercise emphasizes how ROC and PRC offer deeper insights into **ranking quality**, **class separability**, and **trade-offs between precision and recall**, especially under class imbalance.  

---

## Dataset Used  

- **Source:** [UCI Landsat Satellite Dataset (Statlog)](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite)  
- **Files Used:**  
  - `sat.trn` – Training set (4435 samples)  
  - `sat.tst` – Test set (2000 samples)  
- **Features:** 36 spectral attributes (4 bands × 9 pixels in a 3×3 neighborhood)  
- **Classes:** 6 land-cover types (1, 2, 3, 4, 5, 7)  
- **Nature:** Multi-class classification with moderate imbalance  

Each record corresponds to the **central pixel** in a 3×3 patch, labeled as one of six terrain types such as red soil, cotton crop, grey soil, etc.  

---

## Assignment Breakdown  

### **Part A – Data Preparation and Baseline Evaluation**

#### Q1. Data Loading and Standardization  
- Loaded both train (`sat.trn`) and test (`sat.tst`) datasets.  
- Verified the shape (4435 × 36 for training, 2000 × 36 for testing).  
- Standardized features using **StandardScaler** (fit on training data only).  

#### Q2. Train–Test Split Confirmation  
- Used the provided split as per assignment instructions.  
- No cross-validation or reshuffling was performed.  

#### Q3. Model Training  
Trained six classifiers using scikit-learn:  
1. K-Nearest Neighbors (KNN)  
2. Decision Tree  
3. Dummy Classifier (Prior)  
4. Logistic Regression  
5. Gaussian Naive Bayes  
6. Support Vector Classifier (SVC, with `probability=True`)  

#### Q4. Baseline Metrics  
- Evaluated **Accuracy** and **Weighted F1-score** on the **test set**.  
- **KNN** and **SVC** showed the highest performance (~90%), while the Dummy model served as a valid low baseline (~23%).  

---

### **Part B – ROC Analysis for Model Selection**

#### Q1. Multi-Class ROC Explanation  
- Explained how **One-vs-Rest (OvR)** decomposition is used for ROC in multi-class settings.  
- Computed **per-class ROC curves** and **macro-/weighted-average AUCs**.  

#### Q2. ROC Curve Plotting  
- Plotted macro-averaged ROC curves for all six models on the test set.  
- **SVC** achieved the highest **macro AUC (0.985)**, followed closely by **KNN (0.978)** and **Logistic Regression (0.975)**.  
- **Dummy (Prior)** remained at random baseline (AUC = 0.5).  

#### Q3. ROC Interpretation  
- AUC < 0.5 (as seen in DummyPrior) indicates **inverted ranking**, where the model systematically misorders classes.  
- Models with AUC close to 1.0 separate classes cleanly across thresholds.  
- SVC was identified as the **best threshold-free classifier** under ROC evaluation.  

---

### **Part C – Precision–Recall Curve (PRC) Analysis**

#### Q1. Why PRC is Important  
- PRC is more informative than ROC under **class imbalance**, as it focuses on **precision (TP / (TP + FP))** rather than FPR.  
- One-vs-Rest PRCs were computed to assess how models maintain precision as recall increases.  

#### Q2. PRC Plotting  
- Plotted **macro-averaged PRC curves** for all models on the **test set**.  
- **KNN** achieved the highest **macro Average Precision (AP ≈ 0.922)**, slightly better than **SVC (0.917)**.  

#### Q3. PRC Interpretation  
- **KNN** sustained high precision even at large recall values, indicating strong positive ranking.  
- **DummyPrior** dropped sharply due to random scoring — as recall increased, false positives overwhelmed true positives, collapsing precision.  
- This illustrated how poor models lose discriminative power under relaxed thresholds.  

---

### **Part D – Final Recommendation**

| Model | Weighted F1 | Macro ROC-AUC | Macro PRC-AP | Verdict |
|:------|:------------:|:--------------:|:--------------:|:---------|
| **SVC** | 0.892 | **0.985** | 0.917 | Excellent generalization |
| **KNN** | **0.904** | 0.978 | **0.922** | Strong balance across metrics |
| Logistic Regression | 0.830 | 0.975 | 0.871 | Reliable linear baseline |
| GaussianNB | 0.804 | 0.955 | 0.810 | Decent but oversimplified |
| Decision Tree | 0.855 | 0.903 | 0.743 | Moderate, slightly unstable |
| DummyPrior | 0.086 | 0.500 | 0.167 | Random baseline |

- **Best Model:** **SVC** – best overall threshold-independent performance (highest ROC-AUC).  
- **Close Second:** **KNN** – slightly better precision–recall behavior at practical thresholds.  
- **Inference:** Both nonlinear models outperform linear and tree-based baselines; Dummy serves as sanity check.

---

### **🌟 Brownie Points Section**

- Added **Random Forest** and **XGBoost** as advanced ensemble models.  
- Also tested an intentionally poor **Flipped Logistic Regression** to illustrate AUC < 0.5.  

| Model | Macro ROC-AUC | Macro PRC-AP | Observation |
|:------|:--------------:|:-------------:|:-------------|
| **XGBoost** | **0.991** | **0.953** | Best overall performance |
| **Random Forest** | 0.990 | 0.950 | Very strong and stable |
| **Flipped LogReg** | 0.024 | 0.090 | Worse than random (inverted ranking) |

**Summary:**  
Ensemble models outperformed all previous classifiers.  
**XGBoost** achieved near-perfect separation across classes, while the flipped model validated that AUC < 0.5 signifies reversed ranking.  
These experiments reinforced how ROC and PRC capture *ranking quality* rather than simple label accuracy.

---

## How to Run

1. Place `sat.trn` and `sat.tst` in the same folder as the notebook.  
2. Open `model_selection.ipynb` in Jupyter Notebook.  
3. Run all cells in order.  
4. The notebook automatically generates:
   - Baseline metrics (Accuracy, F1)
   - ROC & PRC plots
   - AUC and AP tables
   - Brownie Points analysis with ensemble models  

**Dependencies:**  
`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`

---

## Key Learnings

- ROC-AUC and PRC-AP are **threshold-independent** measures that provide a more complete view of classifier performance.  
- **SVC** and **KNN** performed best among traditional models.  
- Ensemble methods like **Random Forest** and **XGBoost** further improved class separation and precision.  
- **PRC** is more informative than **ROC** in imbalanced multi-class setups.  
- Models with AUC < 0.5 represent **inverted discrimination** — useful for diagnosing fundamentally flawed models.  

---

## Final Recommendation

**Preferred Model:**  
→ **Support Vector Classifier (SVC)**  

**Why:**  
- Achieved the **highest macro ROC-AUC** (0.985).  
- Balanced trade-off between precision and recall.  
- Robust generalization across classes.  

> **Alternative:** KNN performs almost equally well and may be preferred for interpretability and lower computational cost.

**Extended Insight:**  
For real-world deployment, **XGBoost** would be the most practical choice — it offers the best AUC/AP combination, handles nonlinearity efficiently, and scales well for larger datasets.
