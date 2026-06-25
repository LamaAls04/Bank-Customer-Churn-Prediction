# 🏦 Vanguard Client Retention Platform

An enterprise-grade **Predictive AI Analytics Dashboard** built with **Streamlit** to analyze, diagnose, and forecast banking customer churn risk. The system evaluates real-time parameter vectors against a highly optimized **Random Forest Classifier** to deliver explainable retention strategies.

---

## 🎨 Application Interface & Workspaces

### 1. Client Risk Audit Workspace
This view provides an interactive diagnostic engine for individual retail accounts. It computes localized customer parameters, generates an automated risk metric gauge, outputs confidence percentages, and delivers customized strategic countermeasures based on the risk profile.

#### ⚙️ Account Parameter Filters (Main Interface)
![Account Parameter Filters](screenshots/Customer_parameters.png)

#### 🟢 Low-Risk Customer Profile Evaluation
![Low-Risk Assessment](screenshots/low_risk_customer.png)

#### 🔴 High-Risk Customer Profile Evaluation
![High-Risk Assessment](screenshots/high_risk_customer.png)
---

### 2. Portfolio Distribution Analysis Workspace
An executive macro-level overview displaying cross-segment trends, metrics, and regional data variations across active accounts.
![Portfolio Analytics Overview](screenshots/portfolio_analysis.png)

---

### 3. Model Diagnostic Panel
A transparent core metrics control desk displaying predictive diagnostics, pipeline optimization states, and feature importance matrices.
![Model Quality Diagnostics](screenshots/model_diagnostics.png)

---

## 📊 Dataset Specifications
The pipeline monitors client records across approximately 10,000 base system profiles with the following data attributes:

* **CreditScore:** Credit rating index matrix.
* **Geography:** Primary branch region location (France, Spain, Germany).
* **Gender:** Customer gender identity classification.
* **Age:** Client biological age matrix.
* **Tenure:** Total active chronological years with the banking institution.
* **Balance:** Real-time ledger capital ledger tracking.
* **NumOfProducts:** Total subscribed active bank services or products.
* **HasCrCard:** Active corporate credit cardholder flag indicator.
* **IsActiveMember:** Active account system interaction over trailing 30 days.
* **EstimatedSalary:** Calculated annual income capability model indicator.
* **Exited:** Attrition classification target label (*1 = Churned, 0 = Active/Retained*).

---

## ⚙️ Machine Learning Pipeline Architecture
1.  **Preprocessing & Feature Engineering:** Automated One-Hot Encoding for categorical inputs, standard variable scaling vector adjustment via `StandardScaler`.
2.  **Imbalance Handling:** Implemented Synthetic Minority Over-sampling Technique (**SMOTE**) to correct structural class distribution variance.
3.  **Model Selection & Benchmarking:** Evaluated architectural capabilities across Logistic Regression, SVC, KNN, Decision Tree, Gradient Boosting, and Random Forest models.
4.  **Production Algorithm Core:** **Random Forest Classifier** achieved optimal precision metrics with an accuracy of **~86.8%**. Production assets are deployed globally via binary serialization anchors (`model.pkl`).

