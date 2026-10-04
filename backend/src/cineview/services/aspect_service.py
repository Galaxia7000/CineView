import re
from typing import Any


class AspectService:
    """
    Rule-based cinema aspect extraction and sentiment association.

    This service does NOT replace the GRU.
    The GRU handles overall review sentiment.

    This module identifies movie-related aspects and looks for
    nearby sentiment-bearing language.
    """

    ASPECTS = {
        "Acting": [
            "acting",
            "performance",
            "performances",
            "actor",
            "actors",
            "actress",
            "actresses",
            "cast",
        ],
        "Story": [
            "story",
            "plot",
            "narrative",
        ],
        "Screenplay": [
            "screenplay",
            "script",
            "writing",
        ],
        "Direction": [
            "direction",
            "director",
            "directing",
        ],
        "Cinematography": [
            "cinematography",
            "camera work",
            "camera",
        ],
        "Visuals": [
            "visuals",
            "visual",
            "effects",
            "cgi",
            "animation",
            "vfx",
        ],
        "Music": [
            "music",
            "soundtrack",
            "score",
            "songs",
            "song",
        ],
        "Sound": [
            "sound",
            "sound design",
            "audio",
        ],
        "Characters": [
            "character",
            "characters",
            "protagonist",
            "antagonist",
        ],
        "Dialogue": [
            "dialogue",
            "dialog",
            "lines",
        ],
        "Pacing": [
            "pacing",
            "pace",
            "slow",
            "fast",
        ],
    }

    POSITIVE_WORDS = {
        "amazing": 2,
        "awesome": 2,
        "beautiful": 2,
        "best": 3,
        "brilliant": 3,
        "captivating": 3,
        "compelling": 2,
        "delightful": 2,
        "engaging": 2,
        "enjoyable": 2,
        "excellent": 3,
        "fantastic": 3,
        "good": 1,
        "great": 2,
        "hilarious": 2,
        "impressive": 2,
        "incredible": 3,
        "love": 2,
        "loved": 2,
        "masterpiece": 3,
        "memorable": 2,
        "moving": 2,
        "outstanding": 3,
        "perfect": 3,
        "phenomenal": 3,
        "powerful": 2,
        "remarkable": 3,
        "satisfying": 2,
        "spectacular": 3,
        "strong": 2,
        "stunning": 3,
        "superb": 3,
        "terrific": 3,
        "wonderful": 3,
    }

    NEGATIVE_WORDS = {
        "awful": -3,
        "bad": -1,
        "bland": -2,
        "boring": -3,
        "confusing": -2,
        "cringe": -2,
        "disappointing": -3,
        "dull": -2,
        "forgettable": -2,
        "frustrating": -2,
        "hate": -2,
        "hated": -2,
        "lifeless": -3,
        "mediocre": -2,
        "messy": -2,
        "poor": -2,
        "pointless": -3,
        "predictable": -2,
        "ridiculous": -2,
        "rough": -1,
        "shallow": -2,
        "slow": -2,
        "terrible": -3,
        "tedious": -3,
        "uninspired": -2,
        "weak": -2,
        "worse": -2,
        "worst": -3,
    }

    NEGATIONS = {
        "not",
        "never",
        "no",
        "hardly",
        "barely",
        "isn't",
        "wasn't",
        "weren't",
        "don't",
        "didn't",
        "doesn't",
        "can't",
        "couldn't",
    }

    INTENSIFIERS = {
        "very": 1.5,
        "really": 1.5,
        "incredibly": 1.7,
        "absolutely": 1.7,
        "extremely": 1.8,
        "truly": 1.4,
        "deeply": 1.4,
    }

    def _normalize_text(self, text: str) -> str:
        """
        Normalize review text while preserving apostrophes.
        """

        text = text.lower()

        return re.sub(
            r"[^a-z0-9'\s.!?,;-]+",
            " ",
            text,
        )

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        """
        Split review into reasonably clean sentences.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _contains_term(
        self,
        sentence: str,
        term: str,
    ) -> bool:
        """
        Check whether an aspect term appears as a whole word
        or phrase.
        """

        pattern = rf"\b{re.escape(term)}\b"

        return re.search(
            pattern,
            sentence,
        ) is not None

    def _find_aspect_terms(
        self,
        sentence: str,
    ) -> list[tuple[str, str]]:
        """
        Find all known aspect categories mentioned in a sentence.
        """

        matches: list[tuple[str, str]] = []

        for aspect, terms in self.ASPECTS.items():
            for term in terms:
                if self._contains_term(sentence, term):
                    matches.append((aspect, term))

        return matches

    def _score_tokens(
        self,
        sentence: str,
    ) -> tuple[float, list[str]]:
        """
        Calculate sentiment from lexical evidence in a sentence.

        This score is deliberately separate from the GRU's model
        probability because this is the aspect-analysis layer.
        """

        tokens = re.findall(
            r"[a-z]+(?:'[a-z]+)?",
            sentence.lower(),
        )

        score = 0.0
        evidence: list[str] = []

        for index, token in enumerate(tokens):
            base_score = (
                self.POSITIVE_WORDS.get(token)
                or self.NEGATIVE_WORDS.get(token)
            )

            if base_score is None:
                continue

            multiplier = 1.0

            if index > 0:
                previous = tokens[index - 1]

                if previous in self.INTENSIFIERS:
                    multiplier = self.INTENSIFIERS[previous]

            adjusted_score = base_score * multiplier

            # Handle simple negation within the previous three tokens.
            start = max(0, index - 3)

            if any(
                candidate in self.NEGATIONS
                for candidate in tokens[start:index]
            ):
                adjusted_score *= -1

            score += adjusted_score
            evidence.append(token)

        return score, evidence

    def _aspect_score(
        self,
        sentence: str,
        aspect_term: str,
    ) -> tuple[float, list[str]]:
        """
        Score sentiment around an aspect mention.

        We focus on the sentence containing the aspect, then
        restrict sentiment evidence to a local context window
        around the aspect when possible.
        """

        tokens = sentence.split()

        term_tokens = aspect_term.split()

        normalized_tokens = [
            re.sub(r"[^a-z']", "", token.lower())
            for token in tokens
        ]

        normalized_term_tokens = [
            token.lower()
            for token in term_tokens
        ]

        aspect_index = -1

        for index in range(
            len(normalized_tokens)
            - len(normalized_term_tokens)
            + 1
        ):
            if (
                normalized_tokens[index:index + len(normalized_term_tokens)]
                == normalized_term_tokens
            ):
                aspect_index = index
                break

        if aspect_index == -1:
            return self._score_tokens(sentence)

        window_start = max(
            0,
            aspect_index - 7,
        )

        window_end = min(
            len(tokens),
            aspect_index + len(term_tokens) + 8,
        )

        local_sentence = " ".join(
            tokens[window_start:window_end]
        )

        return self._score_tokens(local_sentence)

    def _classify_score(
        self,
        score: float,
    ) -> str:
        """
        Convert lexical score into an aspect sentiment.
        """

        if score > 0.5:
            return "Positive"

        if score < -0.5:
            return "Negative"

        return "Neutral"

    def _build_insight(
        self,
        aspect: str,
        score: float,
        evidence: list[str],
    ) -> dict[str, Any]:
        """
        Create a frontend-ready aspect result.
        """

        sentiment = self._classify_score(score)

        # Keep the score bounded so the UI can use it directly.
        display_score = max(
            -100,
            min(
                100,
                round(score * 20),
            ),
        )

        return {
            "name": aspect,
            "sentiment": sentiment,
            "score": display_score,
            "evidence": list(dict.fromkeys(evidence))[:4],
        }

    def analyze(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Extract movie-related aspects and associate sentiment
        evidence with each detected aspect.
        """

        if not text or not text.strip():
            return []

        normalized_text = self._normalize_text(text)

        sentences = self._split_sentences(
            normalized_text
        )

        results: dict[str, dict[str, Any]] = {}

        for sentence in sentences:
            aspect_matches = self._find_aspect_terms(
                sentence
            )

            for aspect, term in aspect_matches:
                score, evidence = self._aspect_score(
                    sentence,
                    term,
                )

                if aspect not in results:
                    results[aspect] = {
                        "score": 0.0,
                        "evidence": [],
                    }

                results[aspect]["score"] += score
                results[aspect]["evidence"].extend(
                    evidence
                )

        insights = []

        for aspect, values in results.items():
            insights.append(
                self._build_insight(
                    aspect=aspect,
                    score=values["score"],
                    evidence=values["evidence"],
                )
            )

        return insights