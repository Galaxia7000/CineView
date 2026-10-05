from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


DATASET_ID = "Lowerated/imdb-reviews-rated"

REVIEW_COLUMN = "review"

ASPECT_COLUMNS = (
    "Cinematography",
    "Direction",
    "Story",
    "Characters",
    "Production Design",
    "Unique Concept",
    "Emotions",
)

# Continuous dataset scores with absolute value below this
# are treated as neutral/no-strong-aspect-signal.
#
# This is ONLY used to construct presence-training labels.
# It is not the final CineView presence threshold.
PRESENCE_LABEL_THRESHOLD = 0.10

VALIDATION_SIZE = 0.20
RANDOM_SEED = 42

# Word-level TF-IDF is deliberately used here because the job
# is aspect PRESENCE, not semantic sentiment scoring.
#
# Unigrams + bigrams capture phrases such as:
# "visual effects", "production design", "character development",
# "unique concept", etc.
MAX_FEATURES = 80_000

VECTORIZER_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": MAX_FEATURES,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "lowercase": True,
}

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "aspect_presence.joblib"


def main() -> None:
    print("=" * 70)
    print("CINEVIEW ASPECT PRESENCE MODEL")
    print("=" * 70)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(f"Dataset: {DATASET_ID}")
    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_ID,
        split="train",
    )

    print(
        f"Dataset rows: {len(dataset)}"
    )

    required_columns = {
        REVIEW_COLUMN,
        *ASPECT_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(dataset.column_names)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required dataset columns: "
            f"{sorted(missing_columns)}"
        )

    reviews = np.array(
        dataset[REVIEW_COLUMN],
        dtype=object,
    )

    # --------------------------------------------------------
    # Construct binary aspect-presence labels
    # --------------------------------------------------------

    aspect_targets = []

    for aspect in ASPECT_COLUMNS:
        values = np.asarray(
            dataset[aspect],
            dtype=np.float32,
        )

        present = (
            np.abs(values)
            >= PRESENCE_LABEL_THRESHOLD
        ).astype(np.int8)

        aspect_targets.append(present)

        positive_count = int(
            present.sum()
        )

        negative_count = (
            len(present)
            - positive_count
        )

        positive_ratio = (
            positive_count / len(present)
        )

        print()
        print(
            f"{aspect:20s} "
            f"present={positive_count:6d} "
            f"absent={negative_count:6d} "
            f"presence_rate={positive_ratio:.3f}"
        )

    targets = np.column_stack(
        aspect_targets
    )

    # --------------------------------------------------------
    # Train / validation split
    #
    # We stratify approximately using the number of active
    # aspects so the validation set does not become
    # accidentally biased toward reviews with many aspects.
    # --------------------------------------------------------

    aspect_count = targets.sum(axis=1)

    train_indices, validation_indices = (
        train_test_split(
            np.arange(len(reviews)),
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_SEED,
            stratify=aspect_count,
        )
    )

    train_reviews = reviews[train_indices]
    validation_reviews = reviews[
        validation_indices
    ]

    train_targets = targets[
        train_indices
    ]

    validation_targets = targets[
        validation_indices
    ]

    print()
    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        f"Training reviews:   {len(train_reviews)}"
    )
    print(
        f"Validation reviews: {len(validation_reviews)}"
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING TF-IDF FEATURES")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        **VECTORIZER_CONFIG
    )

    X_train = vectorizer.fit_transform(
        train_reviews
    )

    X_validation = vectorizer.transform(
        validation_reviews
    )

    print(
        f"Vocabulary/features: {X_train.shape[1]}"
    )
    print(
        f"Training matrix:     {X_train.shape}"
    )
    print(
        f"Validation matrix:   {X_validation.shape}"
    )

    # --------------------------------------------------------
    # Multi-label logistic regression
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING ASPECT PRESENCE CLASSIFIER")
    print("=" * 70)

    base_classifier = LogisticRegression(
        C=4.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )

    classifier = OneVsRestClassifier(
        base_classifier,
        n_jobs=-1,
    )

    classifier.fit(
        X_train,
        train_targets,
    )

    print(
        "Presence classifier trained successfully."
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    validation_probabilities = (
        classifier.predict_proba(
            X_validation
        )
    )

    # Initial 0.50 decision threshold.
    #
    # IMPORTANT:
    # This is NOT the final production threshold.
    # We will calibrate the thresholds separately.
    validation_predictions = (
        validation_probabilities >= 0.50
    ).astype(np.int8)

    # --------------------------------------------------------
    # Per-aspect metrics
    # --------------------------------------------------------

    for index, aspect in enumerate(
        ASPECT_COLUMNS
    ):
        y_true = validation_targets[
            :, index
        ]

        y_pred = validation_predictions[
            :, index
        ]

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        print()
        print(
            f"{aspect}"
        )
        print(
            f"  Precision: {precision:.4f}"
        )
        print(
            f"  Recall:    {recall:.4f}"
        )
        print(
            f"  F1:        {f1:.4f}"
        )

    # --------------------------------------------------------
    # Overall multi-label F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        validation_targets,
        validation_predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        validation_targets,
        validation_predictions,
        average="micro",
        zero_division=0,
    )

    print()
    print("=" * 70)
    print("OVERALL PRESENCE PERFORMANCE")
    print("=" * 70)

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print(
        f"Micro F1: {micro_f1:.4f}"
    )

    # --------------------------------------------------------
    # Full classification report
    # --------------------------------------------------------

    print()
    print(
        classification_report(
            validation_targets,
            validation_predictions,
            target_names=ASPECT_COLUMNS,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    artifact = {
        "model_type": "tfidf_ovr_logistic_regression",
        "dataset": DATASET_ID,
        "random_seed": RANDOM_SEED,
        "presence_label_threshold": (
            PRESENCE_LABEL_THRESHOLD
        ),
        "aspects": ASPECT_COLUMNS,
        "vectorizer": vectorizer,
        "classifier": classifier,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Path: {MODEL_PATH}"
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()