import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# Load saved models, scaler, label encoder
# ===============================
scaler = load("scaler.joblib")
rf_model = load("random_forest_model.joblib")
lr_model = load("logistic_regression_model.joblib")
svm_model = load("svm_model.joblib")
label_encoder = load("label_encoder.joblib")

# Mapping for model selection
models = {
    "Random Forest": rf_model,
    "Logistic Regression": lr_model,
    "SVM (RBF)": svm_model
}

st.title("💻 DDoS Detection ML App")
st.write("Upload a CSV dataset, select a model, and predict DDoS attacks.")

# ===============================
# Upload CSV
# ===============================
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # ===============================
    # Model selection
    # ===============================
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[selected_model_name]

    # ===============================
    # Preprocessing input data
    # ===============================
    if "Label" in df.columns:
        y_true = df["Label"]
        X_input = df.drop("Label", axis=1)
        y_true_encoded = label_encoder.transform(y_true)
    else:
        X_input = df.copy()
        y_true_encoded = None

    # Handle Inf/NaN and clip extreme values
    X_input = X_input.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in X_input.select_dtypes(include=[np.number]).columns:
        lower = X_input[col].quantile(0.001)
        upper = X_input[col].quantile(0.999)
        X_input[col] = X_input[col].clip(lower, upper)

    # Scale features
    X_scaled = scaler.transform(X_input)

    # ===============================
    # Make predictions
    # ===============================
    y_pred_encoded = model.predict(X_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    st.subheader("Predictions")
    st.dataframe(pd.DataFrame({"Prediction": y_pred}))

    # ===============================
    # Show metrics
    # ===============================
    if y_true_encoded is not None:
        st.subheader("Model Metrics")
        acc = accuracy_score(y_true_encoded, y_pred_encoded)
        st.write(f"Accuracy: {acc:.4f}")

        labels_all = list(range(len(label_encoder.classes_)))
        report = classification_report(
            y_true_encoded, y_pred_encoded,
            labels=labels_all,
            target_names=label_encoder.classes_,
            zero_division=0
        )
        st.text("Classification Report:\n" + report)

        cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=labels_all)
        st.text("Confusion Matrix:\n" + str(cm))
