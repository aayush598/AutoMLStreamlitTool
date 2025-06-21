# test/test_tester.py

import os
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src import trainer, tester

def setup_module(module):
    """Train and save a model for testing."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Train and save model
    model, _ = trainer.train_model(train_df, "target")
    joblib.dump(model, "outputs/test_model.joblib")

    # Save test file
    test_df.to_csv("outputs/test_data.csv", index=False)

def teardown_module(module):
    """Clean up after tests"""
    if os.path.exists("outputs/test_model.joblib"):
        os.remove("outputs/test_model.joblib")
    if os.path.exists("outputs/test_data.csv"):
        os.remove("outputs/test_data.csv")

def test_test_model_returns_expected_metrics():
    model_path = "outputs/test_model.joblib"
    test_csv = "outputs/test_data.csv"

    # Load test data
    df = pd.read_csv(test_csv)

    # Run test_model function
    metrics = tester.test_model(model_path, df, "target")

    # Check that metrics are returned and valid
    expected_keys = {"accuracy", "precision", "recall", "f1_score"}
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert expected_keys.issubset(metrics.keys()), "Missing keys in metrics"

    # Check all metrics are floats
    for key in expected_keys:
        assert isinstance(metrics[key], float), f"{key} should be a float"

    # Check all values are between 0 and 1
    for value in metrics.values():
        assert 0.0 <= value <= 1.0, f"Metric {value} out of range [0,1]"

