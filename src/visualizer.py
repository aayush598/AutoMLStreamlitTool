# src/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================
# Confusion Matrix Plotter
# ============================
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot and save confusion matrix.

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true)))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


# ============================
# Feature Importance Plotter
# ============================
def plot_feature_importance(model, feature_names, save_path=None, top_n=20):
    """
    Plot and save feature importance for tree-based models.

    Args:
        model: Trained model (must have `feature_importances_`)
        feature_names (list): Feature names
        save_path (str): File path to save plot
        top_n (int): Number of top features to show
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support feature importance.")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()
