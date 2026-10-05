import type { ReviewAnalysisResult } from "../../types/analysis";
import { calculateVerdict } from "./verdict";

interface VerdictPanelProps {
  analysis: ReviewAnalysisResult;
}

function VerdictPanel({
  analysis,
}: VerdictPanelProps) {
  const verdict = calculateVerdict(analysis);

  const polarityPosition = clamp(
    ((verdict.sentimentIndex + 100) / 200) * 100,
    0,
    100
  );

  return (
    <section className="verdict-panel">
      <div className="verdict-panel__header">
        <div>
          <span className="panel-overline">
            THE CINEVIEW VERDICT
          </span>

          <h2>
            {verdict.toneLabel.toUpperCase()}
          </h2>
        </div>

        <div className="verdict-panel__signal">
          <span
            className={`signal-indicator signal-indicator--${verdict.signalLevel}`}
          />

          <div>
            <strong>
              {verdict.signalLabel}
            </strong>

            <span>
              MODEL SIGNAL
            </span>
          </div>
        </div>
      </div>

      <div className="verdict-panel__divider" />

      <div className="verdict-panel__body">
        {/* ------------------------------------------------
            REVIEW SCORE
        ------------------------------------------------- */}

        <div className="verdict-score">
          <span className="verdict-score__label">
            CINEVIEW REVIEW SCORE
          </span>

          <strong
            className={
              verdict.reviewScore >= 5
                ? "verdict-score--positive"
                : "verdict-score--negative"
            }
          >
            {verdict.reviewScore.toFixed(1)}
          </strong>

          <span className="verdict-score__range">
            OUT OF 10
          </span>
        </div>

        {/* ------------------------------------------------
            SENTIMENT INDEX
        ------------------------------------------------- */}

        <div className="verdict-score">
          <span className="verdict-score__label">
            SENTIMENT INDEX
          </span>

          <strong
            className={
              verdict.sentimentIndex >= 0
                ? "verdict-score--positive"
                : "verdict-score--negative"
            }
          >
            {verdict.sentimentIndex > 0
              ? "+"
              : ""}
            {verdict.sentimentIndex}
          </strong>

          <span className="verdict-score__range">
            -100 TO +100
          </span>
        </div>

        {/* ------------------------------------------------
            POLARITY METER
        ------------------------------------------------- */}

        <div className="polarity-meter">
          <div className="polarity-meter__labels">
            <span>NEGATIVE</span>
            <span>BALANCED</span>
            <span>POSITIVE</span>
          </div>

          <div className="polarity-meter__track">
            <div className="polarity-meter__center" />

            <div
              className="polarity-meter__marker"
              style={{
                left: `${polarityPosition}%`,
              }}
            />
          </div>

          <div className="polarity-meter__ticks">
            <span>-100</span>
            <span>-50</span>
            <span>0</span>
            <span>+50</span>
            <span>+100</span>
          </div>
        </div>
      </div>

      {/* --------------------------------------------------
          GRU BREAKDOWN
      --------------------------------------------------- */}

      <div className="verdict-breakdown">
        <div className="verdict-breakdown__item">
          <span>POSITIVE SIGNAL</span>

          <strong>
            {analysis.positive_probability.toFixed(2)}%
          </strong>
        </div>

        <div className="verdict-breakdown__item">
          <span>NEGATIVE SIGNAL</span>

          <strong>
            {analysis.negative_probability.toFixed(2)}%
          </strong>
        </div>

        <div className="verdict-breakdown__item">
          <span>GRU CONFIDENCE</span>

          <strong>
            {analysis.confidence.toFixed(2)}%
          </strong>
        </div>
      </div>

      {/* --------------------------------------------------
          ASPECT SNAPSHOT
      --------------------------------------------------- */}

      {verdict.analyzedAspectCount > 0 && (
        <div className="verdict-breakdown">
          <div className="verdict-breakdown__item">
            <span>POSITIVE ASPECTS</span>

            <strong>
              {verdict.positiveAspectCount}
            </strong>
          </div>

          <div className="verdict-breakdown__item">
            <span>NEGATIVE ASPECTS</span>

            <strong>
              {verdict.negativeAspectCount}
            </strong>
          </div>

          <div className="verdict-breakdown__item">
            <span>ANALYZED ASPECTS</span>

            <strong>
              {verdict.analyzedAspectCount}
            </strong>
          </div>
        </div>
      )}

      {/* --------------------------------------------------
          INTERPRETATION
      --------------------------------------------------- */}

      <div className="verdict-panel__description">
        <span>INTERPRETATION</span>

        <p>
          {verdict.description}
        </p>

        <small
          style={{
            display: "block",
            marginTop: "0.55rem",
            opacity: 0.65,
            fontSize: "0.72rem",
            lineHeight: 1.5,
          }}
        >
          The review score reflects the GRU sentiment
          prediction and is not an objective rating of
          overall movie quality.
        </small>
      </div>

      {/* --------------------------------------------------
          STRONGEST ASPECTS
      --------------------------------------------------- */}

      {(verdict.strongestPositiveAspect ||
        verdict.strongestNegativeAspect) && (
        <div className="verdict-breakdown">
          {verdict.strongestPositiveAspect && (
            <div className="verdict-breakdown__item">
              <span>STRONGEST POSITIVE</span>

              <strong>
                {verdict.strongestPositiveAspect}
              </strong>
            </div>
          )}

          {verdict.strongestNegativeAspect && (
            <div className="verdict-breakdown__item">
              <span>STRONGEST NEGATIVE</span>

              <strong>
                {verdict.strongestNegativeAspect}
              </strong>
            </div>
          )}
        </div>
      )}

      <div className="verdict-panel__footer">
        <span>MODEL: CINEVIEW GRU</span>
        <span>IMDb SENTIMENT CLASSIFIER</span>
        <span>VERDICT GENERATED</span>
      </div>
    </section>
  );
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

export default VerdictPanel;