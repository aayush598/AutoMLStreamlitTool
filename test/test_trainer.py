# test/test_trainer.py

import os
import shutil
import pandas as pd
from src import trainer
from sklearn.datasets import load_iris

def setup_module(module):
    """Create test_outputs directory before tests run"""
    if not os.path.exists("outputs"):
        os.mkdir("outputs")

def teardown_module(module):
    """Clean up after tests"""
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")

def test_train_model_returns_expected_outputs():
    # Load a sample dataset
    iris = load_iris(as_frame=True)
    df = iris.frame
    target_col = 'target'

    # Run training
    model, metrics = trainer.train_model(df, target_col)

    # Assert model and metrics exist
    assert model is not None, "Model is None"
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    
    # Check keys in metrics
    expected_keys = {'accuracy', 'precision', 'recall', 'f1_score'}
    assert expected_keys.issubset(metrics.keys()), f"Missing keys in metrics: {metrics.keys()}"

    # Check if model was saved
    saved_model_path = os.path.join("outputs", "trained_model.joblib")
    assert os.path.exists(saved_model_path), "Trained model file not saved"

    # Check metrics values are floats
    for key in expected_keys:
        assert isinstance(metrics[key], float), f"{key} should be a float"

