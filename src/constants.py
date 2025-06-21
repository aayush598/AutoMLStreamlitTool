# src/constants.py

import os

# ===============================
# Directory Constants
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "outputs", "predictions")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

# ===============================
# Default Values
# ===============================
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# ===============================
# Supported Models
# ===============================
CLASSIFICATION_MODELS = {
    "Logistic Regression": "logistic",
    "Random Forest": "random_forest",
    "Decision Tree": "decision_tree",
    "Support Vector Machine": "svm",
    "K-Nearest Neighbors": "knn"
}

# Mapping for actual sklearn classes (used in trainer)
SKLEARN_MODEL_MAPPING = {
    "logistic": "sklearn.linear_model.LogisticRegression",
    "random_forest": "sklearn.ensemble.RandomForestClassifier",
    "decision_tree": "sklearn.tree.DecisionTreeClassifier",
    "svm": "sklearn.svm.SVC",
    "knn": "sklearn.neighbors.KNeighborsClassifier"
}

# ===============================
# Other Constants
# ===============================
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
ALLOWED_FILE_EXTENSIONS = [".csv"]
