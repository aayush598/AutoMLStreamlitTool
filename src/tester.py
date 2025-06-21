# src/tester.py

import os
import pandas as pd
from src import utils
from src.visualizer import plot_confusion_matrix


# ================================
# Main Testing Function
# ================================
def test_model_on_csv(csv_file, model_path, target_column=None):
    """
    Load a trained model and run predictions on new data.

    Args:
        csv_file: path or uploaded file
        model_path: trained model .pkl path
        target_column: (optional) column name if true labels are present

    Returns:
        dict: predictions, metrics (optional), confusion matrix path (optional)
    """

    # Load model
    model = utils.load_model(model_path)

    # Load and preprocess CSV
    df = utils.load_csv(csv_file)
    df = utils.handle_missing_values(df)

    # Inference target if available
    has_target = False
    if target_column and target_column in df.columns:
        has_target = True
    elif not target_column:
        inferred = utils.infer_target_column(df)
        if inferred in df.columns:
            target_column = inferred
            has_target = True

    if has_target:
        X = df.drop(columns=[target_column])
        y_true = df[target_column]
    else:
        X = df
        y_true = None

    # Encode and scale
    df_encoded, _ = utils.encode_categorical_columns(X)
    X_scaled, _ = utils.scale_numeric_columns(df_encoded)

    # Predict
    predictions = model.predict(X_scaled)

    # Save predictions
    pred_df = df.copy()
    pred_df["Prediction"] = predictions
    pred_filename = utils.get_unique_filename("predictions", ".csv")
    pred_path = utils.save_dataframe(pred_df, pred_filename)

    # Evaluate (if true labels available)
    metrics = None
    confusion_plot_path = None
    if has_target:
        metrics = utils.get_classification_metrics(y_true, predictions)

        plots_dir = "outputs/plots"
        os.makedirs(plots_dir, exist_ok=True)
        confusion_plot_path = os.path.join(plots_dir, f"{pred_filename}_confusion.png")
        plot_confusion_matrix(y_true, predictions, save_path=confusion_plot_path)

    # Return results
    return {
        "predictions_csv": pred_path,
        "metrics": metrics,
        "confusion_plot": confusion_plot_path,
    }
