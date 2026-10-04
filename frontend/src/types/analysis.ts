export interface AspectInsight {
  name: string;
  sentiment: "Positive" | "Negative" | "Neutral";
  score: number;
  evidence: string[];
}

export interface ReviewAnalysisResult {
  sentiment: "Positive" | "Negative";
  confidence: number;
  positive_probability: number;
  negative_probability: number;
  aspects: AspectInsight[];
}

export interface ReviewAnalysisResponse {
  review: string;
  result: ReviewAnalysisResult;
}