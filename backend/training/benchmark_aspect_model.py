from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Lowerated/deberta-v3-lm6"

ASPECTS = [
    "Cinematography",
    "Direction",
    "Story",
    "Characters",
    "Production Design",
    "Unique Concept",
    "Emotions",
]

DATASET_NAME = "Lowerated/imdb-reviews-rated"

SAMPLE_SIZE = 200
RANDOM_SEED = 42
BATCH_SIZE = 8


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load Model
# ============================================================

def load_aspect_model():
    """
    Load the movie-specific LM6 aspect model.
    """

    print("=" * 70)
    print("CINEVIEW ASPECT MODEL BENCHMARK")
    print("=" * 70)

    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    print("\nLoading tokenizer...")

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    print("\nLoading DeBERTa model...")

    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return tokenizer, model


# ============================================================
# Prediction
# ============================================================

def predict_reviews(
    reviews: list[str],
    tokenizer,
    model,
) -> np.ndarray:
    """
    Predict seven continuous aspect scores.

    The LM6 model outputs one score per aspect.
    """

    predictions = []

    for start in range(
        0,
        len(reviews),
        BATCH_SIZE,
    ):
        batch = reviews[
            start:start + BATCH_SIZE
        ]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        batch_predictions = (
            outputs.logits
            .detach()
            .cpu()
            .numpy()
        )

        predictions.append(
            batch_predictions
        )

    return np.concatenate(
        predictions,
        axis=0,
    )


# ============================================================
# Challenge Review Suite
# ============================================================

def run_challenge_suite(
    tokenizer,
    model,
):
    """
    Test the model on carefully selected movie-review
    sentences that contain mixed aspect sentiment.
    """

    reviews = [
        (
            "The acting was phenomenal, but the screenplay "
            "was painfully weak and the visuals were breathtaking."
        ),
        (
            "The acting and dialogue were excellent. "
            "The story was boring, but the visuals were stunning."
        ),
        (
            "The story was excellent, although the direction "
            "felt amateurish and the cinematography was incredible."
        ),
        (
            "The performances were weak and the plot was "
            "predictable, but the cinematography was gorgeous."
        ),
        (
            "A beautifully directed film with memorable "
            "characters and an original concept."
        ),
        (
            "The movie looked fantastic, but the characters "
            "felt empty and the story dragged."
        ),
    ]

    print("\n")
    print("=" * 70)
    print("CHALLENGE REVIEW SUITE")
    print("=" * 70)

    predictions = predict_reviews(
        reviews,
        tokenizer,
        model,
    )

    for index, (
        review,
        prediction,
    ) in enumerate(
        zip(reviews, predictions),
        start=1,
    ):
        print(f"\nREVIEW {index}")
        print("-" * 70)
        print(review)

        print("\nAspect scores:")

        for aspect, score in zip(
            ASPECTS,
            prediction,
        ):
            sentiment = score_to_sentiment(
                score
            )

            print(
                f"  {aspect:<20} "
                f"{score:>7.3f}   "
                f"{sentiment}"
            )


# ============================================================
# Score Interpretation
# ============================================================

def score_to_sentiment(
    score: float,
) -> str:
    """
    Convert the continuous LM6 score to a readable
    qualitative label.

    These thresholds are for presentation only.
    The raw score remains the authoritative model output.
    """

    if score >= 0.50:
        return "Positive"

    if score <= -0.50:
        return "Negative"

    return "Neutral"


# ============================================================
# Dataset Loading
# ============================================================

def load_benchmark_dataset():
    """
    Load the movie aspect dataset from Hugging Face.
    """

    print("\n")
    print("=" * 70)
    print("LOADING LABELED IMDb ASPECT DATASET")
    print("=" * 70)

    dataset = load_dataset(
        DATASET_NAME
    )

    print(
        f"\nAvailable splits: "
        f"{list(dataset.keys())}"
    )

    # Select the first available split.
    split_name = list(
        dataset.keys()
    )[0]

    data = dataset[split_name]

    print(
        f"Using split: {split_name}"
    )

    print(
        f"Total rows: {len(data)}"
    )

    return data


# ============================================================
# Dataset Schema
# ============================================================

def identify_columns(data):
    """
    Identify the review and aspect columns.
    """

    columns = data.column_names

    review_column = None

    for candidate in [
        "Review",
        "review",
        "text",
        "Text",
    ]:
        if candidate in columns:
            review_column = candidate
            break

    if review_column is None:
        raise ValueError(
            "Unable to identify review column. "
            f"Available columns: {columns}"
        )

    missing_aspects = [
        aspect
        for aspect in ASPECTS
        if aspect not in columns
    ]

    if missing_aspects:
        raise ValueError(
            "Missing expected aspect columns: "
            f"{missing_aspects}"
        )

    return review_column


# ============================================================
# Dataset Benchmark
# ============================================================

def run_dataset_benchmark(
    data,
    review_column: str,
    tokenizer,
    model,
):
    """
    Evaluate the model against a reproducible sample
    from the labeled dataset.
    """

    print("\n")
    print("=" * 70)
    print("DATASET BENCHMARK")
    print("=" * 70)

    sample_size = min(
        SAMPLE_SIZE,
        len(data),
    )

    sample = data.shuffle(
        seed=RANDOM_SEED
    ).select(
        range(sample_size)
    )

    reviews = sample[
        review_column
    ]

    true_values = np.array(
        [
            [
                float(row[aspect])
                for aspect in ASPECTS
            ]
            for row in sample
        ],
        dtype=np.float32,
    )

    predicted_values = predict_reviews(
        reviews,
        tokenizer,
        model,
    )

    print(
        f"\nBenchmark samples: "
        f"{sample_size}"
    )

    print(
        "\nPer-aspect results:"
    )

    all_mse = []
    all_mae = []

    for index, aspect in enumerate(
        ASPECTS
    ):
        actual = true_values[
            :, index
        ]

        predicted = predicted_values[
            :, index
        ]

        mse = mean_squared_error(
            actual,
            predicted,
        )

        mae = mean_absolute_error(
            actual,
            predicted,
        )

        all_mse.append(mse)
        all_mae.append(mae)

        print(
            f"  {aspect:<20} "
            f"MSE: {mse:.4f}   "
            f"MAE: {mae:.4f}"
        )

    overall_mse = mean_squared_error(
        true_values,
        predicted_values,
    )

    overall_mae = mean_absolute_error(
        true_values,
        predicted_values,
    )

    print("\nOverall:")
    print(
        f"  MSE: {overall_mse:.4f}"
    )

    print(
        f"  MAE: {overall_mae:.4f}"
    )

    # --------------------------------------------------------
    # Sign accuracy
    # --------------------------------------------------------

    actual_sign = np.sign(
        true_values
    )

    predicted_sign = np.sign(
        predicted_values
    )

    sign_accuracy = (
        actual_sign == predicted_sign
    ).mean()

    print(
        f"  Sign accuracy: "
        f"{sign_accuracy * 100:.2f}%"
    )

    return {
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "sign_accuracy": sign_accuracy,
        "per_aspect_mse": dict(
            zip(
                ASPECTS,
                all_mse,
            )
        ),
        "per_aspect_mae": dict(
            zip(
                ASPECTS,
                all_mae,
            )
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    tokenizer, model = load_aspect_model()

    # First: controlled CineView tests.
    run_challenge_suite(
        tokenizer,
        model,
    )

    # Second: real labeled data.
    data = load_benchmark_dataset()

    review_column = identify_columns(
        data
    )

    print(
        f"\nReview column: "
        f"{review_column}"
    )

    run_dataset_benchmark(
        data,
        review_column,
        tokenizer,
        model,
    )

    print("\n")
    print("=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()