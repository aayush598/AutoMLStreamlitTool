# src/utils.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# ================================
# File Operations
# ================================

def save_model(model, filename, folder="models"):
    """
    Save the trained model to disk using joblib.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    joblib.dump(model, path)
    return path


def load_model(filepath):
    """
    Load a saved model from disk using joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)


def save_dataframe(df, filename, folder="outputs/predictions"):
    """
    Save a DataFrame to CSV.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    return path


def load_csv(file):
    """
    Load a CSV file from Streamlit uploader or file path.
    """
    if isinstance(file, str):
        return pd.read_csv(file)
    else:
        return pd.read_csv(file)

# ================================
# Data Preprocessing
# ================================

def handle_missing_values(df, strategy="drop"):
    """
    Handle missing values in the dataset.
    Currently only supports 'drop' strategy.
    """
    if strategy == "drop":
        return df.dropna()
    else:
        raise NotImplementedError("Only 'drop' strategy is currently supported.")


def encode_categorical_columns(df):
    """
    Encode all categorical columns using LabelEncoder.
    Returns modified DataFrame and a dictionary of encoders.
    """
    label_encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders


def scale_numeric_columns(df, exclude_columns=[]):
    """
    Scale numeric columns using StandardScaler.
    Returns scaled DataFrame and scaler object.
    """
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def split_features_target(df, target_column):
    """
    Split the dataset into features (X) and target (y).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# ================================
# Utility Functions
# ================================

def get_unique_filename(base_name, extension=".pkl"):
    """
    Generate a unique filename using a timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"


def get_classification_metrics(y_true, y_pred):
    """
    Return a dictionary of common classification metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted"), 4),
        "recall": round(recall_score(y_true, y_pred, average="weighted"), 4),
        "f1_score": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def is_csv(filename):
    """
    Check if the uploaded file is a CSV.
    """
    return filename.lower().endswith(".csv")


def infer_target_column(df):
    """
    Attempt to infer the target column based on name heuristics.
    """
    candidates = ['target', 'label', 'class', 'output', 'y']
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return df.columns[-1]  # Default to last column if no match

