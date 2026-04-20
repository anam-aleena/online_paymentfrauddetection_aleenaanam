# Online Payment Fraud Detection — ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)
[![Azure](https://img.shields.io/badge/Microsoft_Azure-Deployed-0078D4?logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

> **Detect fraudulent online transactions in real time — before they cost the business.**  
> An end-to-end ML pipeline trained on financial transaction data, deployed as a live prediction API on Microsoft Azure.

---

## Problem Statement

Online payment fraud costs businesses globally over **$48 billion per year**. Traditional rule-based systems miss sophisticated fraud patterns and generate excessive false positives — blocking legitimate customers. This project builds a machine learning system that learns fraud patterns from historical data and flags high-risk transactions in real time with high precision and minimal false negatives.

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.91 | 0.87 | 0.83 | 0.85 | 0.94 |
| Random Forest | 0.96 | 0.93 | 0.91 | 0.92 | 0.97 |
| **XGBoost** ✅ | **0.98** | **0.96** | **0.94** | **0.95** | **0.99** |

> XGBoost selected as the final model — highest ROC-AUC with best recall (minimizing missed fraud = missed losses).

---

## Project Structure

```
online-payment-fraud-detection/
│
├── notebooks/
│   └── fraud_detection_pipeline.ipynb   # Full EDA + training + evaluation
│
├── reports/
│   ├── confusion_matrix.png             # Model evaluation heatmap
│   ├── feature_importance.png           # Top fraud predictors
│   ├── roc_curve.png                    # ROC-AUC comparison
│   └── class_distribution.png          # Target variable analysis
│
├── requirements.txt
└── README.md
```

---

## ML Pipeline

```
Raw Transaction Data
    → Exploratory Data Analysis (EDA)
    → Data Cleaning & Preprocessing
    → Feature Engineering
    → Class Imbalance Handling
    → Model Training (LR / RF / XGBoost)
    → Cross-Validation & Evaluation
    → Best Model Selection
    → Deployment on Microsoft Azure
```

---

## Dataset Features

| Feature | Type | Description |
|---|---|---|
| `step` | Numeric | Time step of transaction (1 step = 1 hour) |
| `type` | Categorical | Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER) |
| `amount` | Numeric | Transaction amount in local currency |
| `nameOrig` | String | Customer initiating the transaction |
| `oldbalanceOrg` | Numeric | Sender balance before transaction |
| `newbalanceOrig` | Numeric | Sender balance after transaction |
| `nameDest` | String | Recipient of the transaction |
| `oldbalanceDest` | Numeric | Recipient balance before transaction |
| `newbalanceDest` | Numeric | Recipient balance after transaction |
| `isFraud` | Binary | **Target — 1 = Fraud, 0 = Legitimate** |

---

## Key Findings from EDA

- **Only 0.13% of transactions are fraudulent** — severe class imbalance handled with SMOTE + class weights
- **TRANSFER and CASH-OUT** transaction types account for 100% of fraud cases
- **Large amount transactions** (> 200,000) are disproportionately fraudulent
- **Balance discrepancies** (oldBalance - newBalance ≠ amount) are the strongest fraud signal
- **Feature engineering** — added `balanceOrigDiff` and `balanceDestDiff` as engineered features, significantly boosting model performance

---

## Techniques Used

- **Exploratory Data Analysis** — distribution plots, correlation heatmap, fraud pattern analysis
- **Data Preprocessing** — label encoding, feature scaling (StandardScaler), outlier analysis
- **Class Imbalance** — SMOTE oversampling + XGBoost `scale_pos_weight`
- **Feature Engineering** — balance difference features derived from raw columns
- **Models** — Logistic Regression (baseline), Random Forest, XGBoost (final)
- **Evaluation** — Confusion matrix, ROC-AUC, Precision-Recall curve, F1 Score
- **Deployment** — REST API on Microsoft Azure (Microsoft Elevate × AICTE internship)

---

## Quick Start

```bash
git clone https://github.com/anam-aleena/online_paymentfrauddetection_aleenaanam.git
cd online_paymentfrauddetection_aleenaanam
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection_pipeline.ipynb
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Imbalance | imbalanced-learn (SMOTE) |
| Deployment | Microsoft Azure |
| Environment | Jupyter Notebook |

---

## Business Impact

A model with **94% recall** means that out of every 100 fraudulent transactions, 94 are caught automatically. At an average fraud loss of $500 per transaction, this system prevents approximately **$470 in losses per 100 transactions** — translating to millions in savings at scale for financial institutions.

---

## Author

**Aleena Anam** — AI/ML Engineer & Data Scientist  
📧 anamaleena0@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/aleena-anam-2056a4368) | [GitHub](https://github.com/anam-aleena)

*Built during Microsoft Elevate × AICTE × Edunet Foundation ML Internship (Jan–Feb 2026)*

---

## License

MIT License — free to use, modify, and distribute.
