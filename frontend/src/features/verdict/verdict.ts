import type { ReviewAnalysisResult } from "../../types/analysis";

export interface VerdictData {
  sentimentIndex: number;
  toneLabel: string;
  signalLabel: string;
  signalLevel: "high" | "moderate" | "low";
  description: string;
}

export function calculateVerdict(
  analysis: ReviewAnalysisResult
): VerdictData {
  const sentimentIndex = Math.round(
    analysis.positive_probability -
      analysis.negative_probability
  );

  let toneLabel: string;

  if (sentimentIndex >= 80) {
    toneLabel = "Strongly Positive";
  } else if (sentimentIndex >= 40) {
    toneLabel = "Positive";
  } else if (sentimentIndex > 0) {
    toneLabel = "Leaning Positive";
  } else if (sentimentIndex <= -80) {
    toneLabel = "Strongly Negative";
  } else if (sentimentIndex <= -40) {
    toneLabel = "Negative";
  } else {
    toneLabel = "Leaning Negative";
  }

  let signalLabel: string;
  let signalLevel: "high" | "moderate" | "low";

  if (analysis.confidence >= 85) {
    signalLabel = "High Signal";
    signalLevel = "high";
  } else if (analysis.confidence >= 65) {
    signalLabel = "Moderate Signal";
    signalLevel = "moderate";
  } else {
    signalLabel = "Low Signal";
    signalLevel = "low";
  }

  const direction =
    sentimentIndex >= 0
      ? "positive"
      : "negative";

  const strength =
    Math.abs(sentimentIndex) >= 80
      ? "strongly"
      : Math.abs(sentimentIndex) >= 40
        ? "clearly"
        : "slightly";

  const description =
    `The review carries a ${strength} ${direction} ` +
    `sentiment signal according to the CineView GRU model.`;

  return {
    sentimentIndex,
    toneLabel,
    signalLabel,
    signalLevel,
    description,
  };
}