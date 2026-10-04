import type { AspectInsight } from "../../types/analysis";

interface AspectPanelProps {
  aspects: AspectInsight[];
}

function AspectPanel({
  aspects,
}: AspectPanelProps) {
  if (aspects.length === 0) {
    return (
      <section className="aspect-panel aspect-panel--empty">
        <div className="aspect-panel__header">
          <div>
            <span className="panel-overline">
              WHAT STOOD OUT
            </span>

            <h2>NO CLEAR ASPECTS DETECTED</h2>
          </div>
        </div>

        <p>
          The review did not contain enough recognizable
          cinema-related aspect language for a detailed
          breakdown.
        </p>
      </section>
    );
  }

  return (
    <section className="aspect-panel">
      <div className="aspect-panel__header">
        <div>
          <span className="panel-overline">
            WHAT STOOD OUT
          </span>

          <h2>REVIEW ASPECTS</h2>
        </div>

        <span className="aspect-panel__count">
          {aspects.length.toString().padStart(2, "0")}
          {" "}
          DETECTED
        </span>
      </div>

      <div className="aspect-panel__grid">
        {aspects.map((aspect) => (
          <article
            className="aspect-card"
            key={aspect.name}
          >
            <div className="aspect-card__top">
              <span className="aspect-card__name">
                {aspect.name}
              </span>

              <span
                className={`aspect-card__sentiment aspect-card__sentiment--${aspect.sentiment.toLowerCase()}`}
              >
                {aspect.sentiment}
              </span>
            </div>

            <div className="aspect-card__score">
              <div className="aspect-card__score-track">
                <div
                  className={`aspect-card__score-fill aspect-card__score-fill--${aspect.sentiment.toLowerCase()}`}
                  style={{
                    width: `${Math.min(
                      Math.abs(aspect.score),
                      100
                    )}%`,
                  }}
                />
              </div>

              <span>
                {aspect.score > 0 ? "+" : ""}
                {aspect.score}
              </span>
            </div>

            {aspect.evidence.length > 0 && (
              <div className="aspect-card__evidence">
                <span>SIGNAL WORDS</span>

                <div>
                  {aspect.evidence.map((word) => (
                    <span key={`${aspect.name}-${word}`}>
                      {word}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </article>
        ))}
      </div>

      <div className="aspect-panel__footer">
        <span>
          ASPECT ANALYSIS / LEXICAL SIGNALS
        </span>

        <span>
          SEPARATE FROM GRU OVERALL SENTIMENT
        </span>
      </div>
    </section>
  );
}

export default AspectPanel;