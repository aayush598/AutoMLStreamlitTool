# app.py

import streamlit as st
import os
from src import trainer, tester
from src.constants import MODEL_DIR, PREDICTIONS_DIR, PLOTS_DIR, CLASSIFICATION_MODELS

st.set_page_config(page_title="AutoML CSV Trainer", layout="wide")
st.title("🤖 AutoML CSV Trainer & Tester")
st.markdown("Upload a CSV file, train models automatically, and get predictions instantly!")

# Ensure output directories exist
for directory in [MODEL_DIR, PREDICTIONS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Sidebar for navigation
mode = st.sidebar.radio("Select Mode", ["Train Model", "Test Model"])

# =============================
# 🔧 TRAINING MODE
# =============================
if mode == "Train Model":
    st.header("📥 Upload CSV for Training")

    train_file = st.file_uploader("Upload CSV File", type=["csv"], key="train_csv")

    if train_file is not None:
        st.success("✅ CSV File Uploaded Successfully")

        if st.button("🚀 Train Model Automatically"):
            with st.spinner("Training in progress..."):
                output = trainer.train_model_from_csv(train_file)

            st.success("🎉 Model trained successfully!")

            # Show metrics
            st.subheader("📈 Evaluation Metrics")
            st.json(output["metrics"])

            # Show plots
            st.subheader("🧾 Confusion Matrix")
            if output["confusion_plot"]:
                st.image(output["confusion_plot"])

            if output["feature_plot"]:
                st.subheader("🌟 Feature Importance")
                st.image(output["feature_plot"])

            # Download model
            st.subheader("💾 Download Trained Model")
            with open(output["model_path"], "rb") as f:
                st.download_button(
                    label="Download Model (.pkl)",
                    data=f,
                    file_name=os.path.basename(output["model_path"]),
                    mime="application/octet-stream"
                )

# =============================
# ✅ TESTING MODE
# =============================
elif mode == "Test Model":
    st.header("🧪 Upload CSV for Testing")
    test_file = st.file_uploader("Upload CSV File", type=["csv"], key="test_csv")

    st.subheader("📁 Upload Trained Model (.pkl)")
    model_file = st.file_uploader("Upload Trained Model", type=["pkl"], key="model_pkl")

    target_col = st.text_input("Optional: Enter target column name (if available in test CSV)", "")

    if test_file and model_file:
        if st.button("🔍 Run Prediction & Evaluate"):
            with st.spinner("Testing model on new data..."):
                result = tester.test_model_on_csv(
                    test_file,
                    model_path=model_file,
                    target_column=target_col.strip() if target_col else None
                )

            st.success("✅ Testing Completed")

            # Download predictions
            st.subheader("📥 Download Predictions")
            with open(result["predictions_csv"], "rb") as f:
                st.download_button(
                    label="Download Predictions (.csv)",
                    data=f,
                    file_name=os.path.basename(result["predictions_csv"]),
                    mime="text/csv"
                )

            # Show metrics
            if result["metrics"]:
                st.subheader("📊 Evaluation Metrics")
                st.json(result["metrics"])

            # Show confusion plot
            if result["confusion_plot"]:
                st.subheader("📌 Confusion Matrix")
                st.image(result["confusion_plot"])
