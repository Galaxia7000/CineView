from __future__ import annotations

from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score

from cineview.services.aspect_service import AspectService


ASPECTS = (
    "Cinematography",
    "Direction",
    "Story",
    "Characters",
    "Production Design",
    "Unique Concept",
    "Emotions",
)


# ------------------------------------------------------------
# Independent CineView validation set
#
# Expected labels are based on the explicit meaning of each
# review, NOT on LM6 predictions.
#
# An aspect that is not discussed is labeled Neutral.
# ------------------------------------------------------------

TEST_CASES = [
    (
        "The cinematography was stunning and the camera work was gorgeous.",
        {
            "Cinematography": "Positive",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The cinematography was dull and the shots looked poorly composed.",
        {
            "Cinematography": "Negative",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The direction was brilliant, but the story was painfully weak.",
        {
            "Cinematography": "Neutral",
            "Direction": "Positive",
            "Story": "Negative",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The director handled every scene clumsily and the direction felt amateurish.",
        {
            "Cinematography": "Neutral",
            "Direction": "Negative",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The screenplay was clever, tightly written, and consistently engaging.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Positive",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The plot dragged endlessly and the story became predictable.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Negative",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The characters were richly written and surprisingly memorable.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Positive",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The characters felt flat, empty, and impossible to care about.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Negative",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The production design was exquisite, especially the period interiors.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Positive",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The sets looked cheap and the production design was disappointing.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Negative",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The premise was wonderfully original and unlike anything I had seen before.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Positive",
            "Emotions": "Neutral",
        },
    ),
    (
        "The concept sounded promising but turned out to be painfully derivative.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Negative",
            "Emotions": "Neutral",
        },
    ),
    (
        "The film was deeply moving and left me emotionally devastated.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Positive",
        },
    ),
    (
        "I felt absolutely nothing while watching it.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Negative",
        },
    ),
    (
        "The movie looked fantastic, but its story dragged.",
        {
            "Cinematography": "Positive",
            "Direction": "Neutral",
            "Story": "Negative",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The acting was excellent and the characters were compelling, although the cinematography was mediocre.",
        {
            "Cinematography": "Negative",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Positive",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The direction was poor, but the cinematography was spectacular.",
        {
            "Cinematography": "Positive",
            "Direction": "Negative",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The story was not bad, but it was not particularly memorable either.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The characters were not completely convincing, but several performances were excellent.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "An original concept carried by beautiful production design and a wonderful emotional core.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Positive",
            "Unique Concept": "Positive",
            "Emotions": "Positive",
        },
    ),
    (
        "The plot was awful, the characters were lifeless, and the direction was embarrassing.",
        {
            "Cinematography": "Neutral",
            "Direction": "Negative",
            "Story": "Negative",
            "Characters": "Negative",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "Beautiful cinematography and excellent direction made this a visual treat.",
        {
         "Cinematography": "Positive",
         "Direction": "Positive",
         "Story": "Neutral",
         "Characters": "Neutral",
         "Production Design": "Neutral",
         "Unique Concept": "Neutral",
         "Emotions": "Neutral",
        },
    ),
    (
        "The sets were magnificent, but the concept was completely unoriginal.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Positive",
            "Unique Concept": "Negative",
            "Emotions": "Neutral",
        },
    ),
    (
        "The ending was satisfying and the narrative remained gripping throughout.",
        {
            "Cinematography": "Neutral",
            "Direction": "Neutral",
            "Story": "Positive",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Neutral",
        },
    ),
    (
        "The movie is visually impressive but emotionally hollow.",
        {
            "Cinematography": "Positive",
            "Direction": "Neutral",
            "Story": "Neutral",
            "Characters": "Neutral",
            "Production Design": "Neutral",
            "Unique Concept": "Neutral",
            "Emotions": "Negative",
        },
    ),
]


def main() -> None:
    print("=" * 70)
    print("CINEVIEW INDEPENDENT ASPECT VALIDATION")
    print("=" * 70)
    print(f"Test reviews: {len(TEST_CASES)}")
    print()

    service = AspectService()

    all_expected: list[str] = []
    all_predicted: list[str] = []

    per_aspect_expected: dict[str, list[str]] = defaultdict(list)
    per_aspect_predicted: dict[str, list[str]] = defaultdict(list)

    failures = []

    print("=" * 70)
    print("REVIEW RESULTS")
    print("=" * 70)

    for index, (review, expected) in enumerate(
        TEST_CASES,
        start=1,
    ):
        results = service.analyze(review)

        predicted_map = {
            item["name"]: item["sentiment"]
            for item in results
        }

        review_correct = True

        for aspect in ASPECTS:
            expected_label = expected[aspect]
            predicted_label = predicted_map.get(
                aspect,
                "Neutral",
            )

            all_expected.append(expected_label)
            all_predicted.append(predicted_label)

            per_aspect_expected[aspect].append(
                expected_label
            )
            per_aspect_predicted[aspect].append(
                predicted_label
            )

            if expected_label != predicted_label:
                review_correct = False

        status = "PASS" if review_correct else "FAIL"

        print()
        print(
            f"[{status}] REVIEW {index}"
        )
        print(review)

        if not review_correct:
            failures.append(
                (
                    index,
                    review,
                    expected,
                    predicted_map,
                )
            )

            for aspect in ASPECTS:
                if (
                    expected[aspect]
                    != predicted_map.get(
                        aspect,
                        "Missing",
                    )
                ):
                    print(
                        f"  {aspect:20s} "
                        f"expected={expected[aspect]:8s} "
                        f"predicted={predicted_map.get(aspect, 'Missing')}"
                    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERALL VALIDATION")
    print("=" * 70)

    accuracy = accuracy_score(
        all_expected,
        all_predicted,
    )

    macro_f1 = f1_score(
        all_expected,
        all_predicted,
        labels=[
            "Negative",
            "Neutral",
            "Positive",
        ],
        average="macro",
        zero_division=0,
    )

    print(
        f"Label accuracy: {accuracy:.4f}"
    )
    print(
        f"Macro F1:        {macro_f1:.4f}"
    )
    print(
        f"Review accuracy: "
        f"{(len(TEST_CASES) - len(failures)) / len(TEST_CASES):.4f}"
    )

    # --------------------------------------------------------
    # Per-aspect metrics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PER-ASPECT VALIDATION")
    print("=" * 70)

    for aspect in ASPECTS:
        accuracy = accuracy_score(
            per_aspect_expected[aspect],
            per_aspect_predicted[aspect],
        )

        macro_f1 = f1_score(
            per_aspect_expected[aspect],
            per_aspect_predicted[aspect],
            labels=[
                "Negative",
                "Neutral",
                "Positive",
            ],
            average="macro",
            zero_division=0,
        )

        print(
            f"{aspect:20s} "
            f"accuracy={accuracy:.4f} "
            f"macro_f1={macro_f1:.4f}"
        )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if not failures:
        print(
            "PASS: Every independent challenge review "
            "matched the expected aspect labels."
        )
    else:
        print(
            f"FAIL: {len(failures)} of "
            f"{len(TEST_CASES)} reviews contained "
            "at least one aspect mismatch."
        )

        print()
        print(
            "These mismatches must be inspected before "
            "Module 8 is considered production-ready."
        )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()