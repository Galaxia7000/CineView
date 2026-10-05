import type { ReviewAnalysisResult } from "../../types/analysis";

export interface VerdictData {
  reviewScore: number;
  scoreDisplay: string;

  sentimentIndex: number;

  toneLabel: string;

  signalLabel: string;
  signalLevel: "high" | "moderate" | "low";

  strongestPositiveAspect: string | null;
  strongestNegativeAspect: string | null;

  positiveAspectCount: number;
  negativeAspectCount: number;
  analyzedAspectCount: number;

  description: string;
}

function clamp(
  value: number,
  minimum: number,
  maximum: number
): number {
  return Math.min(
    Math.max(value, minimum),
    maximum
  );
}

export function calculateVerdict(
  analysis: ReviewAnalysisResult
): VerdictData {
  // --------------------------------------------------------
  // OVERALL SENTIMENT INDEX
  // --------------------------------------------------------

  const sentimentIndex = Math.round(
    analysis.positive_probability -
      analysis.negative_probability
  );

  // --------------------------------------------------------
  // CINEVIEW REVIEW SCORE
  //
  // This is a sentiment-derived score, NOT a claim about
  // objective movie quality.
  //
  // Positive probability:
  //   0%   -> 0.0 / 10
  //   50%  -> 5.0 / 10
  //   100% -> 10.0 / 10
  // --------------------------------------------------------

  const reviewScore = Number(
    clamp(
      analysis.positive_probability / 10,
      0,
      10
    ).toFixed(1)
  );

  const scoreDisplay =
    `${reviewScore.toFixed(1)} / 10`;

  // --------------------------------------------------------
  // CINEMATIC TONE
  // --------------------------------------------------------

  let toneLabel: string;

  if (sentimentIndex >= 80) {
    toneLabel = "Strongly Positive";
  } else if (sentimentIndex >= 40) {
    toneLabel = "Positive";
  } else if (sentimentIndex >= 10) {
    toneLabel = "Leaning Positive";
  } else if (sentimentIndex > -10) {
    toneLabel = "Mixed";
  } else if (sentimentIndex > -40) {
    toneLabel = "Leaning Negative";
  } else if (sentimentIndex > -80) {
    toneLabel = "Negative";
  } else {
    toneLabel = "Strongly Negative";
  }

  // --------------------------------------------------------
  // MODEL SIGNAL
  // --------------------------------------------------------

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

  // --------------------------------------------------------
  // ASPECT SUMMARY
  // --------------------------------------------------------

  const aspects = analysis.aspects ?? [];

  const positiveAspects = aspects.filter(
    (aspect) =>
      aspect.sentiment === "Positive"
  );

  const negativeAspects = aspects.filter(
    (aspect) =>
      aspect.sentiment === "Negative"
  );

  const strongestPositive =
    positiveAspects.length > 0
      ? [...positiveAspects].sort(
          (a, b) => b.score - a.score
        )[0]
      : null;

  const strongestNegative =
    negativeAspects.length > 0
      ? [...negativeAspects].sort(
          (a, b) => a.score - b.score
        )[0]
      : null;

  // --------------------------------------------------------
  // INTERPRETATION
  // --------------------------------------------------------

  let description: string;

  if (toneLabel === "Mixed") {
    description =
      "The review shows a balanced sentiment signal, " +
      "with both positive and negative elements.";
  } else {
    const direction =
      sentimentIndex > 0
        ? "positive"
        : "negative";

    const strength =
      Math.abs(sentimentIndex) >= 80
        ? "strong"
        : Math.abs(sentimentIndex) >= 40
          ? "clear"
          : "slight";

    description =
      `The review carries a ${strength} ${direction} ` +
      `sentiment signal according to the CineView GRU model.`;
  }

  return {
    reviewScore,
    scoreDisplay,

    sentimentIndex,

    toneLabel,

    signalLabel,
    signalLevel,

    strongestPositiveAspect:
      strongestPositive?.name ?? null,

    strongestNegativeAspect:
      strongestNegative?.name ?? null,

    positiveAspectCount:
      positiveAspects.length,

    negativeAspectCount:
      negativeAspects.length,

    analyzedAspectCount:
      aspects.length,

    description,
  };
}