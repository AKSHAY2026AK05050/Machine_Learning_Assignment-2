# Bank Marketing Term Deposit Prediction

## a. Problem Statement
Predict whether a bank client will subscribe to a term deposit (`yes` or `no`) based on client demographics, account information, and previous marketing campaign contacts.

## b. Dataset Description
- Source: UCI Bank Marketing Dataset
- Instances: 45,211
- Features: 16 input features + 1 target
- Target: binary (`yes` / `no`)
- No missing values
- Imbalanced classes (~11.7% yes)

## c. Github Repository Link
https://github.com/AKSHAY2026AK05050/Machine_Learning_Assignment-2

## d. Models Used

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.8458   | 0.9079| 0.4184    | 0.8147 | 0.5529| 0.5094|
| Decision Tree       | 0.8262   | 0.8581| 0.3865    | 0.8270 | 0.5268| 0.4845|
| KNN                 | 0.8965   | 0.8368| 0.5962    | 0.3573 | 0.4468| 0.4094|
| Naive Bayes         | 0.8639   | 0.8088| 0.4282    | 0.4877 | 0.4560| 0.3797|
| Support Vector Machine | 0.8999 | 0.9077| 0.6393    | 0.3318 | 0.4368| 0.4132|

### Observations
| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Best overall balance. Highest F1 and MCC, excellent AUC and strong Recall. |
| Decision Tree       | Highest Recall but lower precision and F1 than Logistic Regression. |
| KNN                 | High accuracy but poor Recall on the minority class. |
| Naive Bayes         | Moderate performance; weakest AUC and MCC. |
| Support Vector Machine | Highest Accuracy and Precision but low Recall. |
| **Overall Winner**  | **Logistic Regression** |

## Files
- `app.py` – Streamlit application
- `requirements.txt`
- `test_data.csv` – Test set for evaluation
- `model/` – Saved trained models and preprocessor