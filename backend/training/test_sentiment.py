from cineview.services.sentiment_service import SentimentService


def main():
    service = SentimentService()

    reviews = [
        "I absolutely loved this movie. The acting was brilliant and the story was incredibly engaging.",
        "This movie was painfully boring. The story was weak and the acting felt terrible.",
        "The cinematography was beautiful and the performances were excellent.",
        "I regret watching this. It was slow, predictable, and completely disappointing.",
    ]

    print("\n" + "=" * 60)
    print("CINEVIEW SENTIMENT ENGINE TEST")
    print("=" * 60)

    for review in reviews:
        result = service.analyze(review)

        print("\nReview:")
        print(review)

        print(f"\nSentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}%")
        print(
            f"Positive probability: "
            f"{result['positive_probability']}%"
        )
        print(
            f"Negative probability: "
            f"{result['negative_probability']}%"
        )

        print("-" * 60)


if __name__ == "__main__":
    main()