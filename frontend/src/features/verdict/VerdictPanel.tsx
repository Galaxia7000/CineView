import type { ReviewAnalysisResult } from "../../types/analysis";
import { calculateVerdict } from "./verdict";

interface VerdictPanelProps {
  analysis: ReviewAnalysisResult;
}

function VerdictPanel({
  analysis,
}: VerdictPanelProps) {
  const verdict = calculateVerdict(analysis);

  const polarityPosition =
    ((verdict.sentimentIndex + 100) / 200) * 100;

  return (
    <section className="verdict-panel">
      <div className="verdict-panel__header">
        <div>
          <span className="panel-overline">
            THE CINEVIEW VERDICT
          </span>

          <h2>{verdict.toneLabel.toUpperCase()}</h2>
        </div>

        <div className="verdict-panel__signal">
          <span
            className={`signal-indicator signal-indicator--${verdict.signalLevel}`}
          />

          <div>
            <strong>{verdict.signalLabel}</strong>
            <span>MODEL SIGNAL</span>
          </div>
        </div>
      </div>

      <div className="verdict-panel__divider" />

      <div className="verdict-panel__body">
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
            {verdict.sentimentIndex > 0 ? "+" : ""}
            {verdict.sentimentIndex}
          </strong>

          <span className="verdict-score__range">
            -100 TO +100
          </span>
        </div>

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
          <span>CONFIDENCE</span>

          <strong>
            {analysis.confidence.toFixed(2)}%
          </strong>
        </div>
      </div>

      <div className="verdict-panel__description">
        <span>INTERPRETATION</span>

        <p>{verdict.description}</p>
      </div>

      <div className="verdict-panel__footer">
        <span>MODEL: CINEVIEW GRU</span>
        <span>IMDb SENTIMENT CLASSIFIER</span>
        <span>VERDICT GENERATED</span>
      </div>
    </section>
  );
}

export default VerdictPanel;