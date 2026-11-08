import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")
st.title("Bank Marketing - Term Deposit Prediction")
st.write("Upload the test CSV, select a model, and view evaluation metrics.")

# Load models and preprocessor
@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
        "Decision Tree": joblib.load("model/decision_tree.joblib"),
        "KNN": joblib.load("model/knn.joblib"),
        "Naive Bayes": joblib.load("model/naive_bayes.joblib"),
        "SVM": joblib.load("model/svm.joblib"),
    }
    preprocessor = joblib.load("model/preprocessor.joblib")
    return models, preprocessor

models, preprocessor = load_artifacts()

# File uploader
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    if "y" not in test_df.columns:
        st.error("The uploaded file must contain the target column 'y'.")
    else:
        X_raw = test_df.drop("y", axis=1)
        y_true = test_df["y"].map({"no": 0, "yes": 1}) if test_df["y"].dtype == object else test_df["y"]
        
        # Preprocess
        X_processed = preprocessor.transform(X_raw)
        
        # Model selection
        model_name = st.selectbox("Select Model", list(models.keys()))
        model = models[model_name]
        
        # Predictions
        y_pred = model.predict(X_processed)
        y_proba = model.predict_proba(X_processed)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        st.subheader("Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col1.metric("Recall", f"{rec:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        # Classification Report (neat table)
        st.subheader("Classification Report")
        report_dict = classification_report(y_true, y_pred, target_names=["No", "Yes"], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)
else:
    st.info("Please upload the test_data.csv file to evaluate the models.")