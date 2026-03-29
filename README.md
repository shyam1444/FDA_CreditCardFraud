# Automated Credit Card Analytics Framework

## Overview
A comprehensive, two-phase machine learning pipeline governing the entire credit card life cycle: Pre-Issuance Approval Classification (via Orange Data Mining) and Post-Issuance Fraud Detection Analytics (via custom Python Streamlit architecture).

# Dataset Link: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data

---

## Phase 1: Pre-Issuance Approval Classification
**Objective**: Automate customer credit card approval decisions using strictly demographic and financial topologies.

### Technical Implementation
* **Core Workspace**: `CC_Credit_Card_Approval.ows`
* **Execution Environment**: Orange Data Mining Tool
* **Methodology**: 
  - Ingests structured customer applicant profiles (e.g., `Age`, `Debt`, `Income`, `CreditScore`, `PriorDefault`).
  - Utilizes visual data pipeline mechanics to map normalized categorical attributes directly to a binary `ApprovalStatus` target constraint.
  - Implements out-of-the-box classification algorithms (Standard Logistic Regression & Tree Ensembles) visually.
  - Computes native mathematical evaluation procedures (AUC, Classification Accuracy, F1 Scores) exclusively through local widget cross-validation mapping without continuous code debugging.

---

## Phase 2: Post-Issuance Fraud Detection System
**Objective**: Monitor and classify live post-issuance transactions to algorithmically flag anomalies and actively prevent fiscal loss.

### Technical Implementation
* **Dataset Initialization**: Ingests anonymized PCA-transformed transaction matrices (`V1-V28`) and unscaled `Amount` variables loaded via `creditcard_balanced.csv`.
* **Descriptive Framework**: Computes highly memory-optimized `MiniBatchKMeans` algorithms (`k=3`) for instantaneous spatial profiling of standard normative transaction arrays natively.
* **Diagnostic Architecture**: Employs `shap.TreeExplainer` engines to map the marginal Game-Theoretic Shapley values strictly to the XGBoost predictor arrays, programmatically sourcing the exact mathematical feature triggering any anomaly detection.
* **Predictive Machinery**: Dispatches an `XGBClassifier` running a dynamically calibrated `scale_pos_weight` matrix to heavily penalize mathematical False Negatives against rare fraud vectors. Monitors model health via immediate Confusion Mapping and Precision-Recall isolation.
* **Prescriptive Optimization**: Integrates an isolated, bespoke Tabular Q-Learning Reinforcement Agent routing XGBoost probability states mathematically to absolute business actions (`Approve`, `Review`, `Deny`). Extrapolates ideal financial cut-off bounds utilizing epsilon-greedy Bellman equation derivations targeted explicitly at diminishing long-term loss.

### Usage Protocol
Execute the Fraud Analytics Command Center locally via Windows/Linux terminal:
```bash
python -m streamlit run app.py
```
