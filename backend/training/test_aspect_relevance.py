from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_ID = "cross-encoder/nli-deberta-v3-small"

ASPECT_HYPOTHESES = {
    "Cinematography":
        "This review discusses the cinematography, camera work, visuals, or visual photography of the movie.",

    "Direction":
        "This review discusses the direction, directing, or work of the movie's director.",

    "Story":
        "This review discusses the story, plot, screenplay, writing, narrative, or ending of the movie.",

    "Characters":
        "This review discusses the characters, actors, acting, performances, or cast of the movie.",

    "Production Design":
        "This review discusses the production design, sets, locations, costumes, or visual environments of the movie.",

    "Unique Concept":
        "This review discusses the originality, uniqueness, premise, or concept of the movie.",

    "Emotions":
        "This review discusses the emotional impact or feelings produced by the movie.",
}


# The same independent suite used for Module 8 validation.
TEST_CASES = [
    (
        "The cinematography was stunning and the camera work was gorgeous.",
        {"Cinematography"},
    ),
    (
        "The cinematography was dull and the shots looked poorly composed.",
        {"Cinematography"},
    ),
    (
        "The direction was brilliant, but the story was painfully weak.",
        {"Direction", "Story"},
    ),
    (
        "The director handled every scene clumsily and the direction felt amateurish.",
        {"Direction"},
    ),
    (
        "The screenplay was clever, tightly written, and consistently engaging.",
        {"Story"},
    ),
    (
        "The plot dragged endlessly and the story became predictable.",
        {"Story"},
    ),
    (
        "The characters were richly written and surprisingly memorable.",
        {"Characters"},
    ),
    (
        "The characters felt flat, empty, and impossible to care about.",
        {"Characters"},
    ),
    (
        "The production design was exquisite, especially the period interiors.",
        {"Production Design"},
    ),
    (
        "The sets looked cheap and the production design was disappointing.",
        {"Production Design"},
    ),
    (
        "The premise was wonderfully original and unlike anything I had seen before.",
        {"Unique Concept"},
    ),
    (
        "The concept sounded promising but turned out to be painfully derivative.",
        {"Unique Concept"},
    ),
    (
        "The film was deeply moving and left me emotionally devastated.",
        {"Emotions"},
    ),
    (
        "I felt absolutely nothing while watching it.",
        {"Emotions"},
    ),
    (
        "The movie looked fantastic, but its story dragged.",
        {"Cinematography", "Story"},
    ),
    (
        "The acting was excellent and the characters were compelling, although the cinematography was mediocre.",
        {"Characters", "Cinematography"},
    ),
    (
        "The direction was poor, but the cinematography was spectacular.",
        {"Direction", "Cinematography"},
    ),
    (
        "The story was not bad, but it was not particularly memorable either.",
        {"Story"},
    ),
    (
        "The characters were not completely convincing, but several performances were excellent.",
        {"Characters"},
    ),
    (
        "An original concept carried by beautiful production design and a wonderful emotional core.",
        {"Unique Concept", "Production Design", "Emotions"},
    ),
    (
        "The plot was awful, the characters were lifeless, and the direction was embarrassing.",
        {"Story", "Characters", "Direction"},
    ),
    (
        "Beautiful cinematography and excellent direction made this a visual treat.",
        {"Cinematography", "Direction"},
    ),
    (
        "The sets were magnificent, but the concept was completely unoriginal.",
        {"Production Design", "Unique Concept"},
    ),
    (
        "The ending was satisfying and the narrative remained gripping throughout.",
        {"Story"},
    ),
    (
        "The movie is visually impressive but emotionally hollow.",
        {"Cinematography", "Emotions"},
    ),
]


def get_label_scores(
    logits: torch.Tensor,
) -> dict[str, float]:
    """
    Return softmax probabilities for contradiction,
    entailment, and neutral.
    """

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )[0]

    return {
        "contradiction": float(probabilities[0]),
        "entailment": float(probabilities[1]),
        "neutral": float(probabilities[2]),
    }


def main() -> None:
    print("=" * 70)
    print("CINEVIEW SEMANTIC ASPECT RELEVANCE TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Model:  {MODEL_ID}")
    print(f"Device: {device}")
    print()

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=False,
    )

    print("Tokenizer loaded.")
    print()

    print("Loading NLI model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    model.to(device)
    model.eval()

    print("NLI model loaded successfully.")
    print()

    # --------------------------------------------------------
    # Test threshold sweep
    # --------------------------------------------------------

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    results_by_threshold: dict[
        float,
        dict[str, Any],
    ] = {}

    # Store all model scores first so we don't repeatedly
    # reload/run the NLI model for every threshold.
    all_scores = []

    print("=" * 70)
    print("RUNNING SEMANTIC RELEVANCE PREDICTIONS")
    print("=" * 70)

    for index, (review, expected_aspects) in enumerate(
        TEST_CASES,
        start=1,
    ):
        texts = []
        hypotheses = []
        aspect_names = []

        for aspect, hypothesis in ASPECT_HYPOTHESES.items():
            texts.append(review)
            hypotheses.append(hypothesis)
            aspect_names.append(aspect)

        inputs = tokenizer(
            texts,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

        review_scores = {}

        for aspect, logits in zip(
            aspect_names,
            outputs.logits,
        ):
            scores = get_label_scores(
                logits.unsqueeze(0)
            )

            review_scores[aspect] = scores

        all_scores.append(
            {
                "review": review,
                "expected": expected_aspects,
                "scores": review_scores,
            }
        )

        print(
            f"\rProcessed {index}/{len(TEST_CASES)}",
            end="",
            flush=True,
        )

    print()
    print()

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)

    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        review_passes = 0

        total_expected = 0
        total_predicted = 0

        for item in all_scores:
            expected = item["expected"]
            predicted = set()

            for aspect, scores in item["scores"].items():
                if scores["entailment"] >= threshold:
                    predicted.add(aspect)

                expected_present = (
                    aspect in expected
                )
                predicted_present = (
                    aspect in predicted
                )

                if (
                    expected_present
                    and predicted_present
                ):
                    tp += 1
                elif (
                    not expected_present
                    and predicted_present
                ):
                    fp += 1
                elif (
                    expected_present
                    and not predicted_present
                ):
                    fn += 1
                else:
                    tn += 1

            total_expected += len(expected)
            total_predicted += len(predicted)

            if predicted == expected:
                review_passes += 1

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        accuracy = (
            (tp + tn)
            / (tp + tn + fp + fn)
        )

        review_accuracy = (
            review_passes
            / len(TEST_CASES)
        )

        results_by_threshold[threshold] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "review_accuracy": review_accuracy,
        }

        print(
            f"Threshold={threshold:.2f}  "
            f"Accuracy={accuracy:.4f}  "
            f"Precision={precision:.4f}  "
            f"Recall={recall:.4f}  "
            f"F1={f1:.4f}  "
            f"Review Accuracy={review_accuracy:.4f}"
        )

    # --------------------------------------------------------
    # Select best threshold
    # --------------------------------------------------------

    best_threshold = max(
        results_by_threshold,
        key=lambda threshold: (
            results_by_threshold[threshold]["f1"],
            results_by_threshold[threshold]["precision"],
            results_by_threshold[threshold]["review_accuracy"],
        ),
    )

    best = results_by_threshold[
        best_threshold
    ]

    print()
    print("=" * 70)
    print("BEST RELEVANCE THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold:       {best_threshold:.2f}"
    )
    print(
        f"Accuracy:        {best['accuracy']:.4f}"
    )
    print(
        f"Precision:       {best['precision']:.4f}"
    )
    print(
        f"Recall:          {best['recall']:.4f}"
    )
    print(
        f"F1:              {best['f1']:.4f}"
    )
    print(
        f"Review accuracy: {best['review_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Show detailed predictions for best threshold
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    for index, item in enumerate(
        all_scores,
        start=1,
    ):
        predicted = {
            aspect
            for aspect, scores
            in item["scores"].items()
            if scores["entailment"]
            >= best_threshold
        }

        status = (
            "PASS"
            if predicted == item["expected"]
            else "FAIL"
        )

        print()
        print(
            f"[{status}] REVIEW {index}"
        )
        print(item["review"])
        print(
            f"  Expected: {sorted(item['expected'])}"
        )
        print(
            f"  Predicted: {sorted(predicted)}"
        )

        if status == "FAIL":
            for aspect, scores in item["scores"].items():
                print(
                    f"    {aspect:20s} "
                    f"entailment="
                    f"{scores['entailment']:.3f}"
                )

    print()
    print("=" * 70)
    print("RELEVANCE TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()