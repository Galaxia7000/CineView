from fastapi import APIRouter, HTTPException

from cineview.schemas.analysis import (
    ReviewAnalysisResponse,
    ReviewRequest,
)
from cineview.services.aspect_service import AspectService
from cineview.services.sentiment_service import SentimentService


router = APIRouter(
    prefix="/api/v1",
    tags=["Analysis"],
)


sentiment_service = SentimentService()
aspect_service = AspectService()


@router.get("/health")
def health_check():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "service": "CineView API",
        "model": "GRU",
    }


@router.post(
    "/analyze",
    response_model=ReviewAnalysisResponse,
)
def analyze_review(
    request: ReviewRequest,
):
    """
    Perform overall sentiment and aspect analysis.
    """

    try:
        sentiment_result = sentiment_service.analyze(
            request.review
        )

        aspect_results = aspect_service.analyze(
            request.review
        )

        sentiment_result["aspects"] = aspect_results

        return {
            "review": request.review,
            "result": sentiment_result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(
            f"CineView analysis error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to analyze the review.",
        ) from error