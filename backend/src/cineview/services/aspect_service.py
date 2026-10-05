from __future__ import annotations

import re
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class AspectService:
    """
    CineView movie aspect intelligence.

    Uses Lowerated/deberta-v3-lm6 to predict continuous sentiment
    scores for seven movie-related aspects.

    A conservative explicit aspect-presence gate is applied first.
    LM6 is then used only for aspects that are actually referenced
    in the review.

    Overall review sentiment remains handled separately by the
    CineView GRU model.

    IMPORTANT:
        LM6 outputs regression-style aspect scores, not probabilities.
        The raw model prediction is preserved separately from the
        bounded CineView score.
    """

    MODEL_ID = "Lowerated/deberta-v3-lm6"

    # The documented LM6 aspect ordering.
    ASPECT_COLUMNS = (
        "Cinematography",
        "Direction",
        "Story",
        "Characters",
        "Production Design",
        "Unique Concept",
        "Emotions",
    )

    MODEL_MAX_LENGTH = 512

    # Calibrated against the labeled IMDb aspect dataset.
    #
    # score >= +0.24 -> Positive
    # score <= -0.24 -> Negative
    # otherwise       -> Neutral
    NEUTRAL_THRESHOLD = 0.24

    # ========================================================
    # EXPLICIT ASPECT-PRESENCE VOCABULARY
    #
    # These terms are ONLY used to determine whether an aspect
    # is explicitly discussed.
    #
    # They do NOT determine sentiment.
    # Sentiment continues to come entirely from LM6.
    # ========================================================

    ASPECT_TERMS = {
        "Cinematography": (
            "cinematography",
            "camera work",
            "camera",
            "cameras",
            "shot",
            "shots",
            "framing",
            "composition",
            "lighting",
            "lens",
            "lenses",
            "color grading",
            "colour grading",
            "visual effects",
            "visual effect",
            "visuals",
            "vfx",
            "cgi",
            "photography",
            "visual treat",
            "movie looked",
            "film looked",
        ),

        "Direction": (
            "direction",
            "director",
            "directing",
            "directed",
            "directorial",
            "filmmaker",
            "filmmaking",
            "helmed",
        ),

        "Story": (
            "story",
            "plot",
            "narrative",
            "screenplay",
            "script",
            "writing",
            "writer",
            "storyline",
            "ending",
            "dialogue",
            "dialog",
            "lines",
            "pacing",
            "pace",
        ),

        "Characters": (
            "character",
            "characters",
            "acting",
            "actor",
            "actors",
            "actress",
            "actresses",
            "performance",
            "performances",
            "cast",
            "protagonist",
            "antagonist",
            "portrayal",
            "role",
            "roles",
        ),

        "Production Design": (
            "production design",
            "production designer",
            "set design",
            "set designs",
            "sets",
            "interior",
            "interiors",
            "costume",
            "costumes",
            "wardrobe",
            "props",
            "locations",
        ),

        "Unique Concept": (
            "concept",
            "premise",
            "original concept",
            "original premise",
            "unique concept",
            "originality",
            "uniqueness",
            "derivative",
            "unoriginal",
            "innovative",
            "innovation",
            "inventive",
            "fresh idea",
            "fresh concept",
        ),

        "Emotions": (
            "emotion",
            "emotions",
            "emotional",
            "emotionally",
            "heartbreaking",
            "heartbroken",
            "heartwarming",
            "heartfelt",
            "devastating",
            "devastated",
            "i felt",
            "feel nothing",
            "felt nothing",
            "made me feel",
            "felt emotional",
        ),
    }

    def __init__(self) -> None:
        """
        Load the LM6 tokenizer and model once.
        """

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("=" * 70)
        print("CINEVIEW ASPECT INTELLIGENCE")
        print("=" * 70)
        print(f"Model: {self.MODEL_ID}")
        print(f"Device: {self.device}")
        print()

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            use_fast=False,
        )

        print("Tokenizer loaded.")
        print()

        # ----------------------------------------------------
        # LM6 model
        # ----------------------------------------------------

        print("Loading DeBERTa aspect model...")

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_ID
            )
        )

        self.model.to(self.device)
        self.model.eval()

        # ----------------------------------------------------
        # Validate checkpoint configuration
        # ----------------------------------------------------

        self._validate_model_configuration()

        print("Aspect model loaded successfully.")
        print()

    # ========================================================
    # MODEL VALIDATION
    # ========================================================

    def _validate_model_configuration(self) -> None:
        """
        Validate the downloaded checkpoint before inference.

        CineView expects exactly seven aspect outputs.
        """

        expected_outputs = len(
            self.ASPECT_COLUMNS
        )

        actual_outputs = getattr(
            self.model.config,
            "num_labels",
            None,
        )

        if actual_outputs != expected_outputs:
            raise RuntimeError(
                "Incompatible aspect model configuration: "
                f"expected {expected_outputs} outputs, "
                f"received {actual_outputs}."
            )

    # ========================================================
    # ASPECT PRESENCE
    # ========================================================

    def _detect_mentioned_aspects(
        self,
        review: str,
    ) -> set[str]:
        """
        Detect aspects that are explicitly referenced.

        This is intentionally conservative.

        IMPORTANT:
            This method determines only whether an aspect is
            referenced. It does NOT determine sentiment.

            LM6 remains responsible for the sentiment score.
        """

        normalized = " ".join(
            review.lower().split()
        )

        mentioned: set[str] = set()

        for aspect_name, terms in self.ASPECT_TERMS.items():
            for term in terms:
                pattern = (
                    rf"(?<!\w)"
                    rf"{re.escape(term)}"
                    rf"(?!\w)"
                )

                if re.search(
                    pattern,
                    normalized,
                ):
                    mentioned.add(aspect_name)
                    break

        return mentioned

    # ========================================================
    # SENTIMENT LABEL
    # ========================================================

    def _sentiment_from_score(
        self,
        score: float,
    ) -> str:
        """
        Convert the continuous LM6 score into a CineView label.
        """

        if score >= self.NEUTRAL_THRESHOLD:
            return "Positive"

        if score <= -self.NEUTRAL_THRESHOLD:
            return "Negative"

        return "Neutral"

    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def analyze(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Analyze a movie review across the seven CineView aspects.

        Pipeline:

            review
              ↓
            explicit aspect presence gate
              ↓
            LM6 continuous aspect scoring
              ↓
            Positive / Neutral / Negative
        """

        if not isinstance(text, str):
            raise TypeError(
                "Aspect analysis input must be a string."
            )

        review = text.strip()

        if not review:
            return []

        # ----------------------------------------------------
        # Detect aspects actually discussed in the review.
        # ----------------------------------------------------

        mentioned_aspects = (
            self._detect_mentioned_aspects(
                review
            )
        )

        # No supported aspect reference.
        if not mentioned_aspects:
            return []

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        inputs = self.tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            max_length=self.MODEL_MAX_LENGTH,
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # LM6 inference
        # ----------------------------------------------------

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits

        # Expected shape:
        #
        #   [batch_size=1, seven_aspects]
        #
        if logits.ndim != 2:
            raise RuntimeError(
                "Unexpected aspect model output shape: "
                f"{tuple(logits.shape)}."
            )

        if logits.shape[0] != 1:
            raise RuntimeError(
                "Aspect service expected exactly one review."
            )

        predictions = (
            logits[0]
            .detach()
            .cpu()
            .tolist()
        )

        expected_outputs = len(
            self.ASPECT_COLUMNS
        )

        if len(predictions) != expected_outputs:
            raise RuntimeError(
                "Unexpected number of aspect predictions: "
                f"expected {expected_outputs}, "
                f"received {len(predictions)}."
            )

        # ----------------------------------------------------
        # Build results
        # ----------------------------------------------------

        results: list[dict[str, Any]] = []

        for aspect_name, raw_prediction in zip(
            self.ASPECT_COLUMNS,
            predictions,
        ):
            # ------------------------------------------------
            # Presence gate:
            #
            # Do not expose an LM6 prediction for an aspect
            # that the review does not explicitly discuss.
            # ------------------------------------------------

            if aspect_name not in mentioned_aspects:
                continue

            raw_score = float(
                raw_prediction
            )

            # LM6 is trained around the [-1, +1] rating scale,
            # but regression outputs can occasionally exceed
            # those boundaries slightly.
            bounded_score = max(
                -1.0,
                min(
                    1.0,
                    raw_score,
                ),
            )

            sentiment = (
                self._sentiment_from_score(
                    bounded_score
                )
            )

            # This is signal strength only.
            #
            # It must NOT be interpreted as a calibrated
            # probability or model confidence.
            signal_strength = round(
                abs(bounded_score) * 100,
                2,
            )

            results.append(
                {
                    "name": aspect_name,
                    "sentiment": sentiment,
                    "score": round(
                        bounded_score,
                        4,
                    ),
                    "raw_score": round(
                        raw_score,
                        4,
                    ),
                    "confidence": signal_strength,
                    "evidence": [],
                    "probabilities": {},
                }
            )

        return results