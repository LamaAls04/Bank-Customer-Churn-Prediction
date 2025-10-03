# 🏦 Bank Customer Churn Prediction

This project is a **Machine Learning web app** built with **Streamlit** to predict whether a bank customer is likely to exit (churn) or stay.  
It is based on the **Bank Churn Modelling dataset** and uses a trained **Random Forest Classifier**.

---

## 📊 Dataset
The dataset contains information about 10,000 bank customers:

- **CreditScore**
- **Geography**
- **Gender**
- **Age**
- **Tenure**
- **Balance**
- **NumOfProducts**
- **HasCrCard**
- **IsActiveMember**
- **EstimatedSalary**
- **Exited** (1 = churned, 0 = stayed)

---

## ⚙️ Model Training
- Preprocessing: One-Hot Encoding, Scaling, SMOTE (to balance classes).  
- Algorithms Tested: Logistic Regression, SVC, KNN, Decision Tree, Random Forest, Gradient Boosting.  
- **Best Model**: Random Forest with ~87% accuracy.  
- Final model saved as `model.pkl`.

---

## 🌐 Streamlit App
The app allows users to:
1. Input customer details (age, balance, tenure, etc.).
2. Get a prediction whether the customer will churn.  

### Run locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
