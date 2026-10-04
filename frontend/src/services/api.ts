import type { ReviewAnalysisResponse } from "../types/analysis";

const API_BASE_URL = import.meta.env.VITE_API_URL || "";

export async function analyzeReview(
  review: string
): Promise<ReviewAnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      review,
    }),
  });

  if (!response.ok) {
    let message = "Unable to analyze the review.";

    try {
      const errorData = await response.json();

      if (typeof errorData.detail === "string") {
        message = errorData.detail;
      }
    } catch {
      // Use the default error message when the response
      // is not valid JSON.
    }

    throw new Error(message);
  }

  return (await response.json()) as ReviewAnalysisResponse;
}