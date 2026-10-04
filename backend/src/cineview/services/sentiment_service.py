from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SentimentService:
    """
    CineView's core GRU sentiment analysis service.

    Responsibilities:
    - Load the trained GRU model
    - Load IMDb vocabulary/configuration
    - Convert raw review text into IMDb-compatible sequences
    - Generate sentiment predictions
    """

    def __init__(self) -> None:
        backend_dir = Path(__file__).resolve().parents[3]
        artifacts_dir = backend_dir / "artifacts"

        self.model_path = artifacts_dir / "cineview_gru.keras"
        self.word_index_path = artifacts_dir / "imdb_word_index.json"
        self.config_path = artifacts_dir / "preprocessing_config.json"

        self._validate_artifacts()

        self.config = self._load_config()
        self.word_index = self._load_word_index()

        print("Loading CineView GRU model...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("CineView GRU model loaded successfully.")

    # ========================================================
    # Artifact Loading
    # ========================================================

    def _validate_artifacts(self) -> None:
        """
        Ensure all required model artifacts exist.
        """

        required_files = [
            self.model_path,
            self.word_index_path,
            self.config_path,
        ]

        missing_files = [
            str(path)
            for path in required_files
            if not path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "Missing CineView model artifacts:\n"
                + "\n".join(missing_files)
            )

    def _load_config(self) -> dict[str, Any]:
        """
        Load preprocessing configuration.
        """

        with self.config_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    def _load_word_index(self) -> dict[str, int]:
        """
        Load the IMDb word-to-index mapping.
        """

        with self.word_index_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    # ========================================================
    # Text Preprocessing
    # ========================================================

    def _clean_text(self, text: str) -> str:
        """
        Normalize user-entered review text.
        """

        text = text.lower()

        # Preserve letters, numbers and apostrophes.
        text = re.sub(
            r"[^a-z0-9']+",
            " ",
            text
        )

        return text.strip()

    def _text_to_sequence(self, text: str) -> list[int]:
        """
        Convert raw review text into an IMDb-compatible
        integer sequence.

        IMDb reserves:
            0 = padding
            1 = start token
            2 = unknown word
        """

        cleaned_text = self._clean_text(text)

        words = cleaned_text.split()

        max_features = int(
            self.config["max_features"]
        )

        sequence = [1]  # IMDb start token

        for word in words:
            word_id = self.word_index.get(word)

            if word_id is None:
                # Unknown word
                sequence.append(2)
                continue

            # IMDb uses an index offset of 3.
            actual_index = word_id + 3

            # Keep only words within our vocabulary limit.
            if actual_index >= max_features:
                sequence.append(2)
            else:
                sequence.append(actual_index)

        return sequence

    def _prepare_text(self, text: str) -> np.ndarray:
        """
        Convert a review into the exact padded shape
        expected by the GRU model.
        """

        max_length = int(
            self.config["max_length"]
        )

        sequence = self._text_to_sequence(text)

        padded = pad_sequences(
            [sequence],
            maxlen=max_length,
            padding=self.config["padding"],
            truncating=self.config["truncating"],
        )

        return padded

    # ========================================================
    # Prediction
    # ========================================================

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Analyze a movie review and return sentiment
        and model confidence.
        """

        if not text or not text.strip():
            raise ValueError(
                "Review text cannot be empty."
            )

        model_input = self._prepare_text(text)

        probability = float(
            self.model.predict(
                model_input,
                verbose=0
            )[0][0]
        )

        if probability >= 0.5:
            sentiment = "Positive"
            confidence = probability
        else:
            sentiment = "Negative"
            confidence = 1 - probability

        return {
            "sentiment": sentiment,
            "confidence": round(confidence * 100, 2),
            "positive_probability": round(
                probability * 100,
                2
            ),
            "negative_probability": round(
                (1 - probability) * 100,
                2
            ),
        }