from __future__ import annotations

from typing import Any

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)


class AspectService:
    """
    Semantic Aspect-Based Sentiment Analysis.

    Stage 1:
        Extract aspect terms from the review.

    Stage 2:
        Classify sentiment for each (review, aspect) pair.

    Overall review sentiment continues to come from the
    custom CineView GRU model.
    """

    ASPECT_MODEL_ID = (
        "yangheng/deberta-v3-base-end2end-absa"
    )

    SENTIMENT_MODEL_ID = (
        "yangheng/deberta-v3-base-absa-v1.1"
    )

    # ========================================================
    # CineView aspect categories
    # ========================================================

    ASPECT_CATEGORIES = {
        "Acting": {
            "acting",
            "performance",
            "performances",
            "actor",
            "actors",
            "actress",
            "actresses",
            "cast",
        },
        "Story": {
            "story",
            "plot",
            "narrative",
            "storyline",
            "ending",
        },
        "Screenplay": {
            "screenplay",
            "script",
            "writing",
            "writer",
        },
        "Direction": {
            "direction",
            "director",
            "directing",
        },
        "Cinematography": {
            "cinematography",
            "camera work",
            "camera",
        },
        "Visuals": {
            "visual",
            "visuals",
            "visual effect",
            "visual effects",
            "effects",
            "special effects",
            "cgi",
            "vfx",
            "animation",
            "graphics",
        },
        "Music": {
            "music",
            "soundtrack",
            "score",
            "song",
            "songs",
        },
        "Sound": {
            "sound",
            "sound design",
            "audio",
        },
        "Characters": {
            "character",
            "characters",
            "protagonist",
            "antagonist",
        },
        "Dialogue": {
            "dialogue",
            "dialog",
            "lines",
        },
        "Pacing": {
            "pacing",
            "pace",
        },
    }

    def __init__(self) -> None:
        """
        Load both ABSA models once.
        """

        print(
            "Loading CineView semantic ABSA models..."
        )

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.SENTIMENT_MODEL_ID,
            use_fast=False,
        )

        # ----------------------------------------------------
        # Aspect extraction model
        # ----------------------------------------------------

        aspect_model = (
            AutoModelForTokenClassification.from_pretrained(
                self.ASPECT_MODEL_ID
            )
        )

        self.aspect_extractor = pipeline(
            "token-classification",
            model=aspect_model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )

        # ----------------------------------------------------
        # Aspect sentiment model
        # ----------------------------------------------------

        sentiment_model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.SENTIMENT_MODEL_ID
            )
        )

        self.sentiment_classifier = pipeline(
            "text-classification",
            model=sentiment_model,
            tokenizer=self.tokenizer,
            device=-1,
            top_k=None,
        )

        print(
            "CineView semantic ABSA models loaded successfully."
        )

    # ========================================================
    # Aspect Normalization
    # ========================================================

    def _normalize_aspect_name(
        self,
        aspect_text: str,
    ) -> str:
        """
        Map extracted aspect phrases to CineView categories.
        """

        normalized = " ".join(
            aspect_text.lower().split()
        )

        for category, terms in self.ASPECT_CATEGORIES.items():
            if normalized in terms:
                return category

        for category, terms in self.ASPECT_CATEGORIES.items():
            for term in terms:
                if (
                    term in normalized
                    or normalized in term
                ):
                    return category

        return aspect_text.strip().title()

    # ========================================================
    # Extract Aspects
    # ========================================================

    def _extract_aspects(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Extract semantic aspect spans.

        IMPORTANT:
        The token-classification pipeline is called directly
        because the installed Transformers version does not
        accept truncation/max_length through __call__().
        """

        entities = self.aspect_extractor(text)

        aspects: list[dict[str, Any]] = []

        for entity in entities:
            label = str(
                entity.get("entity_group")
                or entity.get("entity")
                or ""
            )

            # Only keep actual aspect entities.
            if (
                "ASP" not in label.upper()
                and "ASPECT" not in label.upper()
            ):
                continue

            aspect = str(
                entity.get("word", "")
            ).strip()

            if not aspect:
                continue

            start = entity.get("start")
            end = entity.get("end")

            if start is None or end is None:
                continue

            aspects.append(
                {
                    "aspect": aspect,
                    "start": int(start),
                    "end": int(end),
                    "extractor_label": label,
                    "extractor_confidence": round(
                        float(
                            entity.get(
                                "score",
                                0.0
                            )
                        )
                        * 100,
                        2,
                    ),
                }
            )

        return aspects

    # ========================================================
    # Aspect Sentiment
    # ========================================================

    def _classify_aspect(
        self,
        text: str,
        aspect: str,
    ) -> dict[str, Any]:
        """
        Run semantic sentiment classification on the
        (review, aspect) pair.
        """

        result = self.sentiment_classifier(
            {
                "text": text,
                "text_pair": aspect,
            }
        )

        # With top_k=None, Transformers can return:
        # [
        #   [
        #     {"label": "...", "score": ...},
        #     ...
        #   ]
        # ]
        #
        # Normalize either nested or flat output.

        if not result:
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "probabilities": {},
            }

        if (
            isinstance(result, list)
            and result
            and isinstance(result[0], list)
        ):
            scores = result[0]
        else:
            scores = result

        probability_map: dict[str, float] = {}

        for item in scores:
            if not isinstance(item, dict):
                continue

            label = str(
                item.get("label", "")
            ).strip()

            score = float(
                item.get("score", 0.0)
            )

            probability_map[label] = round(
                score * 100,
                2,
            )

        if not probability_map:
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "probabilities": {},
            }

        best_label = max(
            probability_map,
            key=probability_map.get,
        )

        confidence = probability_map[
            best_label
        ]

        normalized_label = (
            best_label.lower()
        )

        if "positive" in normalized_label:
            sentiment = "Positive"

        elif "negative" in normalized_label:
            sentiment = "Negative"

        else:
            sentiment = "Neutral"

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probability_map,
        }

    # ========================================================
    # Signed Score
    # ========================================================

    def _signed_score(
        self,
        sentiment: str,
        confidence: float,
    ) -> int:
        """
        Convert model confidence into a signed UI score.
        """

        if sentiment == "Positive":
            return round(confidence)

        if sentiment == "Negative":
            return round(-confidence)

        return 0

    # ========================================================
    # Main Analysis
    # ========================================================

    def analyze(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic two-stage ABSA.
        """

        if not text or not text.strip():
            return []

        extracted_aspects = self._extract_aspects(
            text
        )

        if not extracted_aspects:
            return []

        results: dict[str, dict[str, Any]] = {}

        for item in extracted_aspects:
            raw_aspect = item["aspect"]

            category = self._normalize_aspect_name(
                raw_aspect
            )

            classification = self._classify_aspect(
                text,
                raw_aspect,
            )

            sentiment = classification[
                "sentiment"
            ]

            confidence = classification[
                "confidence"
            ]

            score = self._signed_score(
                sentiment,
                confidence,
            )

            if category not in results:
                results[category] = {
                    "name": category,
                    "sentiment": sentiment,
                    "score": score,
                    "confidence": confidence,
                    "evidence": [raw_aspect],
                    "probabilities": classification[
                        "probabilities"
                    ],
                }

            else:
                existing = results[category]

                existing["evidence"].append(
                    raw_aspect
                )

                # Keep the strongest mention.
                if confidence > existing[
                    "confidence"
                ]:
                    existing["sentiment"] = sentiment
                    existing["score"] = score
                    existing["confidence"] = (
                        confidence
                    )
                    existing["probabilities"] = (
                        classification[
                            "probabilities"
                        ]
                    )

        final_results = []

        for result in results.values():
            final_results.append(
                {
                    "name": result["name"],
                    "sentiment": result["sentiment"],
                    "score": result["score"],
                    "confidence": result[
                        "confidence"
                    ],
                    "evidence": list(
                        dict.fromkeys(
                            result["evidence"]
                        )
                    )[:4],
                    "probabilities": result[
                        "probabilities"
                    ],
                }
            )

        return final_results