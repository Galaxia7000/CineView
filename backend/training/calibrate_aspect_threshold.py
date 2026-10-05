from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_ID = "Lowerated/deberta-v3-lm6"
DATASET_ID = "Lowerated/imdb-reviews-rated"

ASPECT_COLUMNS = (
    "Cinematography",
    "Direction",
    "Story",
    "Characters",
    "Production Design",
    "Unique Concept",
    "Emotions",
)

REVIEW_COLUMN = "review"

# Deterministic calibration sample.
SAMPLE_SIZE = 1000
RANDOM_SEED = 42

BATCH_SIZE = 8
MAX_LENGTH = 512

# Candidate neutral-band thresholds.
THRESHOLDS = np.round(
    np.arange(0.00, 0.41, 0.01),
    2,
)


@dataclass
class CalibrationResult:
    threshold: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    positive_f1: float
    negative_f1: float
    neutral_f1: float


def label_from_target(value: float) -> int:
    """
    Convert the dataset's continuous target into a class:

        -1 -> Negative
         0 -> Neutral
        +1 -> Positive

    Intermediate values use their sign.
    """

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def label_from_prediction(
    value: float,
    threshold: float,
) -> int:
    """
    Convert an LM6 continuous prediction into:

        -1 = Negative
         0 = Neutral
        +1 = Positive
    """

    if value >= threshold:
        return 1

    if value <= -threshold:
        return -1

    return 0


def evaluate_threshold(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> CalibrationResult:
    """
    Evaluate one threshold across all seven aspects.
    """

    predicted_classes = np.array(
        [
            label_from_prediction(
                prediction,
                threshold,
            )
            for prediction in predictions
        ],
        dtype=np.int8,
    )

    target_classes = np.array(
        [
            label_from_target(target)
            for target in targets
        ],
        dtype=np.int8,
    )

    accuracy = accuracy_score(
        target_classes,
        predicted_classes,
    )

    balanced_accuracy = balanced_accuracy_score(
        target_classes,
        predicted_classes,
    )

    macro_f1 = f1_score(
        target_classes,
        predicted_classes,
        labels=[-1, 0, 1],
        average="macro",
        zero_division=0,
    )

    class_f1 = f1_score(
        target_classes,
        predicted_classes,
        labels=[-1, 0, 1],
        average=None,
        zero_division=0,
    )

    return CalibrationResult(
        threshold=threshold,
        accuracy=float(accuracy),
        balanced_accuracy=float(balanced_accuracy),
        macro_f1=float(macro_f1),
        negative_f1=float(class_f1[0]),
        neutral_f1=float(class_f1[1]),
        positive_f1=float(class_f1[2]),
    )


def main() -> None:
    print("=" * 70)
    print("CINEVIEW ASPECT THRESHOLD CALIBRATION")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Model:   {MODEL_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Device:  {device}")
    print()

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=False,
    )

    print("Tokenizer loaded.")
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    model.to(device)
    model.eval()

    if model.config.num_labels != len(ASPECT_COLUMNS):
        raise RuntimeError(
            "Unexpected number of model outputs: "
            f"{model.config.num_labels}. "
            f"Expected {len(ASPECT_COLUMNS)}."
        )

    print("Model loaded successfully.")
    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading labeled IMDb aspect dataset...")

    dataset = load_dataset(
        DATASET_ID,
        split="train",
    )

    print(f"Dataset rows: {len(dataset)}")
    print()

    required_columns = {
        REVIEW_COLUMN,
        *ASPECT_COLUMNS,
    }

    missing_columns = required_columns - set(
        dataset.column_names
    )

    if missing_columns:
        raise RuntimeError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Deterministic sample
    # --------------------------------------------------------

    sample = dataset.shuffle(
        seed=RANDOM_SEED
    ).select(
        range(
            min(
                SAMPLE_SIZE,
                len(dataset),
            )
        )
    )

    print(
        f"Calibration samples: {len(sample)}"
    )
    print()

    # --------------------------------------------------------
    # Batched model inference
    # --------------------------------------------------------

    print("Running model predictions...")

    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    reviews = sample[REVIEW_COLUMN]

    for start in range(
        0,
        len(reviews),
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            len(reviews),
        )

        batch_reviews = reviews[start:end]

        inputs = tokenizer(
            batch_reviews,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

        batch_predictions = (
            outputs.logits
            .detach()
            .cpu()
            .numpy()
        )

        all_predictions.append(
            batch_predictions
        )

        batch_targets = np.array(
            [
                [
                    float(row[aspect])
                    for aspect in ASPECT_COLUMNS
                ]
                for row in sample.select(
                    range(start, end)
                )
            ],
            dtype=np.float32,
        )

        all_targets.append(
            batch_targets
        )

        completed = end
        print(
            f"\rProcessed {completed}/{len(reviews)}",
            end="",
            flush=True,
        )

    print()
    print()

    predictions = np.concatenate(
        all_predictions,
        axis=0,
    )

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    if predictions.shape != targets.shape:
        raise RuntimeError(
            "Prediction/target shape mismatch: "
            f"{predictions.shape} vs "
            f"{targets.shape}"
        )

    # --------------------------------------------------------
    # Flatten all seven aspects.
    #
    # Every review contributes seven independently scored
    # aspect examples to threshold calibration.
    # --------------------------------------------------------

    flat_predictions = predictions.reshape(-1)
    flat_targets = targets.reshape(-1)

    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)

    results: list[CalibrationResult] = []

    for threshold in THRESHOLDS:
        result = evaluate_threshold(
            flat_predictions,
            flat_targets,
            float(threshold),
        )

        results.append(result)

    print()
    print(
        "Threshold   Accuracy   Balanced Acc   Macro F1   "
        "Neg F1   Neutral F1   Pos F1"
    )
    print("-" * 70)

    for result in results:
        print(
            f"{result.threshold:8.2f}   "
            f"{result.accuracy:8.4f}   "
            f"{result.balanced_accuracy:13.4f}   "
            f"{result.macro_f1:8.4f}   "
            f"{result.negative_f1:7.4f}   "
            f"{result.neutral_f1:10.4f}   "
            f"{result.positive_f1:7.4f}"
        )

    # --------------------------------------------------------
    # Select threshold.
    #
    # Primary objective:
    #   highest macro F1
    #
    # Tie breaker:
    #   highest balanced accuracy
    #
    # Final tie breaker:
    #   highest neutral F1
    # --------------------------------------------------------

    best = max(
        results,
        key=lambda result: (
            result.macro_f1,
            result.balanced_accuracy,
            result.neutral_f1,
        ),
    )

    print()
    print("=" * 70)
    print("BEST THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold:          {best.threshold:.2f}"
    )
    print(
        f"Accuracy:           {best.accuracy:.4f}"
    )
    print(
        f"Balanced accuracy:  {best.balanced_accuracy:.4f}"
    )
    print(
        f"Macro F1:            {best.macro_f1:.4f}"
    )
    print(
        f"Negative F1:         {best.negative_f1:.4f}"
    )
    print(
        f"Neutral F1:          {best.neutral_f1:.4f}"
    )
    print(
        f"Positive F1:         {best.positive_f1:.4f}"
    )

    # --------------------------------------------------------
    # Overall classification report using best threshold
    # --------------------------------------------------------

    predicted_classes = np.array(
        [
            label_from_prediction(
                prediction,
                best.threshold,
            )
            for prediction in flat_predictions
        ],
        dtype=np.int8,
    )

    target_classes = np.array(
        [
            label_from_target(target)
            for target in flat_targets
        ],
        dtype=np.int8,
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            target_classes,
            predicted_classes,
            labels=[-1, 0, 1],
            target_names=[
                "Negative",
                "Neutral",
                "Positive",
            ],
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Per-aspect best threshold analysis
    # --------------------------------------------------------

    print("=" * 70)
    print("PER-ASPECT BEST THRESHOLDS")
    print("=" * 70)

    for aspect_index, aspect_name in enumerate(
        ASPECT_COLUMNS
    ):
        aspect_predictions = predictions[
            :, aspect_index
        ]

        aspect_targets = targets[
            :, aspect_index
        ]

        aspect_results = [
            evaluate_threshold(
                aspect_predictions,
                aspect_targets,
                float(threshold),
            )
            for threshold in THRESHOLDS
        ]

        aspect_best = max(
            aspect_results,
            key=lambda result: (
                result.macro_f1,
                result.balanced_accuracy,
                result.neutral_f1,
            ),
        )

        print(
            f"{aspect_name:20s} "
            f"threshold={aspect_best.threshold:.2f} "
            f"macro_f1={aspect_best.macro_f1:.4f} "
            f"balanced_acc={aspect_best.balanced_accuracy:.4f}"
        )

    print()
    print("=" * 70)
    print("CALIBRATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()