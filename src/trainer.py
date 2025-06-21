# src/trainer.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from src import utils
from src.visualizer import plot_confusion_matrix, plot_feature_importance


# ================================
# Supported Models Dictionary
# ================================
SUPPORTED_MODELS = {
    "Random Forest": RandomForestClassifier,
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Support Vector Machine": SVC,
}


# ================================
# Main Training Function
# ================================
def train_model_from_csv(csv_file, target_column=None, model_name="Random Forest"):
    """
    Full pipeline: Load -> Preprocess -> Train -> Evaluate -> Save

    Returns:
        dict: metrics, model_path, plot paths, and label encoders
    """

    # Load and clean the data
    df = utils.load_csv(csv_file)
    df = utils.handle_missing_values(df)

    # Auto-infer target column if not provided
    if not target_column:
        target_column = utils.infer_target_column(df)

    # Encode categorical features
    df_encoded, label_encoders = utils.encode_categorical_columns(df)

    # Split data
    X, y = utils.split_features_target(df_encoded, target_column)
    X, scaler = utils.scale_numeric_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Select model
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    model_class = SUPPORTED_MODELS[model_name]
    model = model_class()

    # Train model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    metrics = utils.get_classification_metrics(y_test, y_pred)

    # Save model
    model_filename = utils.get_unique_filename(model_name.replace(" ", "_"))
    model_path = utils.save_model(model, model_filename)

    # Save plots
    plots_dir = "outputs/plots"
    os.makedirs(plots_dir, exist_ok=True)

    confusion_plot_path = os.path.join(plots_dir, f"{model_filename}_confusion.png")
    plot_confusion_matrix(y_test, y_pred, save_path=confusion_plot_path)

    feature_plot_path = None
    if hasattr(model, "feature_importances_"):
        feature_plot_path = os.path.join(plots_dir, f"{model_filename}_features.png")
        plot_feature_importance(model, X.columns, save_path=feature_plot_path)

    # Return results
    return {
        "metrics": metrics,
        "model_path": model_path,
        "confusion_plot": confusion_plot_path,
        "feature_plot": feature_plot_path,
        "target_column": target_column,
        "label_encoders": label_encoders,
        "scaler": scaler,
    }
