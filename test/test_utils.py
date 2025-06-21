# test/test_utils.py

import pandas as pd
import numpy as np
from src.utils import preprocess_data
from sklearn.datasets import load_iris

def test_preprocess_data_output_format():
    # Load sample iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    target_col = 'target'

    X_scaled, y = preprocess_data(df, target_col)

    # Check outputs are correct type
    assert isinstance(X_scaled, np.ndarray), "X should be a NumPy array"
    assert isinstance(y, pd.Series), "y should be a pandas Series"

    # Shapes match
    assert X_scaled.shape[0] == y.shape[0], "X and y should have same number of samples"
    assert X_scaled.shape[1] == df.drop(columns=[target_col]).shape[1], "Feature count mismatch"

def test_preprocess_data_handles_missing_values():
    # Create a small dummy DataFrame with missing values
    data = {
        'feature1': [1, 2, np.nan, 4],
        'feature2': ['A', 'B', 'B', np.nan],
        'target': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Preprocess
    X_scaled, y = preprocess_data(df, 'target')

    # Assert missing values handled (no NaNs)
    assert not np.isnan(X_scaled).any(), "Preprocessed features contain NaNs"
    assert not y.isnull().any(), "Target values contain NaNs"

def test_preprocess_data_encodes_categorical_features():
    # Include categorical data
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['red', 'blue', 'green'],
        'target': [0, 1, 1]
    })

    X_scaled, y = preprocess_data(df, 'target')

    # After one-hot encoding, feature count should be > original 1 numeric + 1 categorical
    assert X_scaled.shape[1] > 2, "Categorical feature not encoded properly"

def test_preprocess_data_invalid_target_column():
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })

    try:
        preprocess_data(df, 'nonexistent')
        assert False, "Expected KeyError for invalid target column"
    except KeyError:
        pass
