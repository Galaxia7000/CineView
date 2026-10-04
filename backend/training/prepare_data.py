from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# Configuration
# ============================================================

MAX_FEATURES = 10_000   # Top 10,000 most frequent words
MAX_LENGTH = 500        # Maximum review length
VALIDATION_SIZE = 0.20  # 20% of training data for validation
RANDOM_STATE = 42


# ============================================================
# Paths
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset Loading
# ============================================================

def load_imdb_data():
    """
    Load the IMDb movie review dataset using Keras.

    Returns:
        x_train: Training review sequences
        y_train: Training labels
        x_test: Testing review sequences
        y_test: Testing labels
    """

    print("Loading IMDb dataset...")

    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=MAX_FEATURES
    )

    print("IMDb dataset loaded successfully.")

    return x_train, y_train, x_test, y_test


def save_imdb_word_index():
    """
    Save IMDb's original word-to-index mapping so that
    user-entered reviews can use the same vocabulary later.
    """

    word_index = imdb.get_word_index()

    word_index_path = ARTIFACTS_DIR / "imdb_word_index.json"

    with word_index_path.open("w", encoding="utf-8") as file:
        json.dump(word_index, file)

    print(f"IMDb word index saved to:")
    print(word_index_path)

# ============================================================
# Sequence Padding
# ============================================================

def pad_review_sequences(x_train, x_test):
    """
    Pad all review sequences to MAX_LENGTH.
    """

    print(f"\nPadding reviews to {MAX_LENGTH} tokens...")

    x_train = pad_sequences(
        x_train,
        maxlen=MAX_LENGTH,
        padding="pre",
        truncating="pre"
    )

    x_test = pad_sequences(
        x_test,
        maxlen=MAX_LENGTH,
        padding="pre",
        truncating="pre"
    )

    print("Padding completed.")

    return x_train, x_test


# ============================================================
# Validation Split
# ============================================================

def create_validation_split(x_train, y_train):
    """
    Split the original training dataset into training
    and validation sets while preserving class proportions.
    """

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    return x_train, x_val, y_train, y_val


# ============================================================
# Dataset Information
# ============================================================

def print_dataset_summary(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Display useful information about the processed dataset.
    """

    print("\n" + "=" * 60)
    print("CINEVIEW IMDb DATASET SUMMARY")
    print("=" * 60)

    print(f"Training samples:   {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Testing samples:    {len(x_test)}")

    print(f"\nInput shape:")
    print(f"  Training:   {x_train.shape}")
    print(f"  Validation: {x_val.shape}")
    print(f"  Testing:    {x_test.shape}")

    print("\nSentiment labels:")
    print("  0 = Negative")
    print("  1 = Positive")

    print("\nTraining distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_train, counts_train):
        sentiment = "Negative" if label == 0 else "Positive"
        print(f"  {sentiment}: {count}")

    print("\nValidation distribution:")
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    for label, count in zip(unique_val, counts_val):
        sentiment = "Negative" if label == 0 else "Positive"
        print(f"  {sentiment}: {count}")

    print("\nTesting distribution:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for label, count in zip(unique_test, counts_test):
        sentiment = "Negative" if label == 0 else "Positive"
        print(f"  {sentiment}: {count}")

    print("=" * 60)


# ============================================================
# Save Configuration
# ============================================================

def save_preprocessing_config():
    """
    Save preprocessing settings so the exact same configuration
    can be reused during prediction.
    """

    config = {
        "max_features": MAX_FEATURES,
        "max_length": MAX_LENGTH,
        "padding": "pre",
        "truncating": "pre",
        "labels": {
            "0": "Negative",
            "1": "Positive"
        },
        "validation_size": VALIDATION_SIZE,
        "random_state": RANDOM_STATE
    }

    config_path = ARTIFACTS_DIR / "preprocessing_config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print(f"\nPreprocessing configuration saved to:")
    print(config_path)


# ============================================================
# Main Pipeline
# ============================================================

def prepare_dataset():
    """
    Complete IMDb preprocessing pipeline.
    """

    # Step 1: Load dataset
    x_train, y_train, x_test, y_test = load_imdb_data()
    save_imdb_word_index()
    # Step 2: Pad sequences
    x_train, x_test = pad_review_sequences(
        x_train,
        x_test
    )

    # Step 3: Create validation set
    x_train, x_val, y_train, y_val = create_validation_split(
        x_train,
        y_train
    )

    # Step 4: Display dataset information
    print_dataset_summary(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    )

    # Step 5: Save configuration
    save_preprocessing_config()

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    )


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    prepare_dataset()