from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from prepare_data import (
    MAX_FEATURES,
    MAX_LENGTH,
    prepare_dataset,
)


# ============================================================
# Configuration
# ============================================================

EMBEDDING_DIM = 128
GRU_UNITS = 64
DENSE_UNITS = 32

BATCH_SIZE = 64
EPOCHS = 5

LEARNING_RATE = 0.001


# ============================================================
# Paths
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Build GRU Model
# ============================================================

def build_gru_model():
    """
    Build and compile the CineView GRU sentiment classifier.
    """

    model = Sequential(
        [
            Embedding(
                input_dim=MAX_FEATURES,
                output_dim=EMBEDDING_DIM,
                mask_zero=True
            ),

            GRU(
                GRU_UNITS
            ),

            Dense(
                DENSE_UNITS,
                activation="relu"
            ),

            Dropout(
                0.30
            ),

            Dense(
                1,
                activation="sigmoid"
            )
        ],
        name="CineView_GRU"
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Train Model
# ============================================================

def train_model():
    """
    Prepare IMDb data, train the GRU model,
    evaluate it and save the trained model.
    """

    print("\nPreparing IMDb dataset...")

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    ) = prepare_dataset()

    print("\nBuilding GRU model...")

    model = build_gru_model()

    print("\nModel architecture:")
    model.summary()

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting GRU training...")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nEvaluating model on test data...")

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("CINEVIEW GRU MODEL RESULTS")
    print("=" * 60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("=" * 60)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = ARTIFACTS_DIR / "cineview_gru.keras"

    model.save(model_path)

    print(f"\nModel saved to:")
    print(model_path)

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = ARTIFACTS_DIR / "training_history.json"

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=4)

    print(f"Training history saved to:")
    print(history_path)

    return model, history


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    train_model()