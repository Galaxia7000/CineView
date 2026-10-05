from typing import Literal

from pydantic import BaseModel, Field
class AspectInsight(BaseModel):
    """
    Continuous aspect-level sentiment result produced by LM6.
    """

    name: str

    sentiment: Literal[
        "Positive",
        "Negative",
        "Neutral",
    ]

    score: float

    raw_score: float

    confidence: float

    evidence: list[str] = Field(
        default_factory=list
    )

    probabilities: dict[str, float] = Field(
        default_factory=dict
    )

class ReviewRequest(BaseModel):
    """
    Request body for movie review analysis.
    """

    review: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Movie review text to analyze.",
    )


class SentimentResult(BaseModel):
    """
    Overall sentiment prediction returned by the GRU.
    """

    sentiment: Literal[
        "Positive",
        "Negative",
    ]
    confidence: float
    positive_probability: float
    negative_probability: float
    aspects: list[AspectInsight] = Field(
        default_factory=list
    )


class ReviewAnalysisResponse(BaseModel):
    """
    Complete CineView review analysis response.
    """

    review: str
    result: SentimentResult