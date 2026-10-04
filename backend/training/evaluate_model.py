from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from prepare_data import prepare_dataset


# ============================================================
# Paths
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


MODEL_PATH = ARTIFACTS_DIR / "cineview_gru.keras"
HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"


# ============================================================
# Load Model
# ============================================================

def load_trained_model():
    """
    Load the previously trained CineView GRU model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    print("Loading trained CineView GRU model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    return model


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(model, x_test, y_test):
    """
    Generate predictions and calculate classification metrics.
    """

    print("\nGenerating predictions on test data...")

    probabilities = model.predict(
        x_test,
        batch_size=64,
        verbose=1
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"],
        zero_division=0
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CINEVIEW GRU PERFORMANCE REPORT")
    print("=" * 60)

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    print("=" * 60)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": matrix.tolist()
    }

    metrics_path = EVALUATION_DIR / "model_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nMetrics saved to:")
    print(metrics_path)

    return probabilities, predictions, metrics


# ============================================================
# Plot Training History
# ============================================================

def plot_training_history():
    """
    Generate accuracy and loss plots from training history.
    """

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_PATH}"
        )

    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = range(1, len(history["accuracy"]) + 1)

    # --------------------------------------------------------
    # Accuracy graph
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["accuracy"],
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )

    plt.title("CineView GRU Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(True, alpha=0.3)

    accuracy_path = EVALUATION_DIR / "accuracy_curve.png"

    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=150)
    plt.close()

    # --------------------------------------------------------
    # Loss graph
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["loss"],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.title("CineView GRU Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(True, alpha=0.3)

    loss_path = EVALUATION_DIR / "loss_curve.png"

    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    print("\nTraining graphs saved:")
    print(accuracy_path)
    print(loss_path)


# ============================================================
# Plot Confusion Matrix
# ============================================================

def plot_confusion_matrix(matrix):
    """
    Generate a visual confusion matrix.
    """

    plt.figure(figsize=(6, 5))

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title("CineView GRU Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(2)

    plt.xticks(
        tick_marks,
        ["Negative", "Positive"]
    )

    plt.yticks(
        tick_marks,
        ["Negative", "Positive"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    # Display values inside cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    matrix_path = EVALUATION_DIR / "confusion_matrix.png"

    plt.savefig(
        matrix_path,
        dpi=150
    )

    plt.close()

    print(f"Confusion matrix saved to:")
    print(matrix_path)


# ============================================================
# Main
# ============================================================

def main():
    """
    Complete model evaluation pipeline.
    """

    print("Preparing IMDb test dataset...")

    (
        _,
        _,
        _,
        _,
        x_test,
        y_test
    ) = prepare_dataset()

    model = load_trained_model()

    _, _, metrics = evaluate_model(
        model,
        x_test,
        y_test
    )

    plot_training_history()

    confusion = np.array(
        metrics["confusion_matrix"]
    )

    plot_confusion_matrix(confusion)

    print("\nCineView model evaluation completed successfully.")


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()