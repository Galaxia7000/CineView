import { useState } from "react";
import FilmStrip from "../components/cinema/FilmStrip";
import { analyzeReview } from "../services/api";
import AspectPanel from "../features/aspects/AspectPanel";
import type { ReviewAnalysisResult } from "../types/analysis";
import VerdictPanel from "../features/verdict/VerdictPanel";
function HomePage() {
  const [review, setReview] = useState("");
  const [analysis, setAnalysis] =
    useState<ReviewAnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");

  async function handleAnalyze() {
    const trimmedReview = review.trim();

    if (!trimmedReview) {
      setError("Please enter a movie review first.");
      setAnalysis(null);
      return;
    }

    setIsAnalyzing(true);
    setError("");
    setAnalysis(null);

    try {
      const response = await analyzeReview(trimmedReview);

      setAnalysis(response.result);
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "Unable to analyze the review.";

      setError(message);
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <main className="cineview-page">
      <FilmStrip position="top" />

      <div className="theatre-curtain theatre-curtain--left" />
      <div className="theatre-curtain theatre-curtain--right" />

      {/* ======================================================
          HEADER
          ====================================================== */}

      <header className="cineview-header">
        <a className="cineview-brand" href="/">
          <span className="cineview-brand__title">CINEVIEW</span>

          <span className="cineview-brand__line">
            <i />
            <small>MOVIE REVIEW INTELLIGENCE</small>
            <i />
          </span>
        </a>

        <nav
          className="cineview-nav"
          aria-label="Primary navigation"
        >
          <a href="#analyze">Analyze</a>
          <a href="#features">Features</a>
          <a href="#about">About</a>
        </nav>

        <div className="header-status">
          <span className="status-dot" />
          <span>GRU ENGINE ONLINE</span>
        </div>
      </header>

      {/* ======================================================
          HERO
          ====================================================== */}

      <section className="hero-section">
        {/* ----------------------------------------------------
            Hero Copy
            ---------------------------------------------------- */}

        <div className="hero-copy">
          <p className="eyebrow">THE DIGITAL SCREENING ROOM</p>

          <h1>
            READ THE
            <br />
            <span>STORY BEHIND</span>
            <br />
            THE REVIEW
          </h1>

          <p className="hero-description">
            CineView transforms movie reviews into intelligent
            cinematic insights using a GRU-powered sentiment engine.
          </p>

          <div className="hero-meta">
            <span>SEQUENTIAL NLP</span>

            <span className="hero-meta__divider" />

            <span>GRU NEURAL NETWORK</span>

            <span className="hero-meta__divider" />

            <span>IMDb DATASET</span>
          </div>
        </div>

        {/* ----------------------------------------------------
            Analysis Column
            ---------------------------------------------------- */}

        <div className="analysis-column">
          {/* --------------------------------------------------
              Review Input Ticket
              -------------------------------------------------- */}

          <div className="screening-panel" id="analyze">
            <div className="screening-panel__header">
              <div>
                <span className="panel-overline">
                  NOW SCREENING
                </span>

                <h2>REVIEW ANALYSIS</h2>
              </div>

              <div className="reel-mark">
                <span />
                <span />
                <span />
                <span />
                <i />
              </div>
            </div>

            <div className="ticket-perforation" />

            <div className="review-paper">
              <div className="review-paper__label">
                YOUR REVIEW
              </div>

              <textarea
                value={review}
                onChange={(event) => {
                  setReview(event.target.value);

                  if (error) {
                    setError("");
                  }
                }}
                placeholder="Write or paste a movie review here..."
                aria-label="Movie review"
                maxLength={10000}
                disabled={isAnalyzing}
              />

              <div className="review-paper__footer">
                <span>
                  {review.length.toLocaleString()} / 10,000
                  CHARACTERS
                </span>

                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing
                    ? "ANALYZING..."
                    : "ANALYZE REVIEW"}

                  <span className="button-arrow">→</span>
                </button>
              </div>
            </div>

            <div className="screening-panel__footer">
              <span>SCREENING NO. 001</span>
              <span>CINEVIEW / AI ANALYSIS</span>
            </div>
          </div>

          {/* --------------------------------------------------
              Error State
              -------------------------------------------------- */}

          {error && (
            <div
              className="analysis-error"
              role="alert"
            >
              {error}
            </div>
          )}

          {/* --------------------------------------------------
              Analysis Result
              -------------------------------------------------- */}

          {analysis && (
              <>
           <VerdictPanel analysis={analysis} />

            <AspectPanel aspects={analysis.aspects}
                    />
              </>
            )}
        </div>
      </section>

      {/* ======================================================
          FEATURES
          ====================================================== */}

      <section
        className="feature-section"
        id="features"
      >
        <div className="section-heading">
          <p className="eyebrow">
            BEYOND A SIMPLE VERDICT
          </p>

          <h2>CINEMATIC INTELLIGENCE</h2>

          <p>
            A review is more than a positive or negative label.
            CineView is designed to uncover the layers behind what
            a viewer feels.
          </p>
        </div>

        <div className="feature-grid">
          {/* --------------------------------------------------
              Feature 01
              -------------------------------------------------- */}

          <article className="feature-card">
            <span className="feature-number">01</span>

            <div className="feature-card__film-frame">
              <span className="frame-hole" />
              <span className="frame-hole" />
              <span className="frame-hole" />
            </div>

            <h3>SENTIMENT</h3>

            <p>
              Understand whether a review leans positive or
              negative and see how confident the GRU model is in
              its prediction.
            </p>

            <span className="feature-label">
              GRU CLASSIFICATION
            </span>
          </article>

          {/* --------------------------------------------------
              Feature 02
              -------------------------------------------------- */}

          <article className="feature-card">
            <span className="feature-number">02</span>

            <div className="feature-card__film-frame feature-card__film-frame--gold">
              <span className="frame-hole" />
              <span className="frame-hole" />
              <span className="frame-hole" />
            </div>

            <h3>ASPECTS</h3>

            <p>
              Discover what the reviewer is actually discussing
              — story, acting, visuals, music, direction and
              more.
            </p>

            <span className="feature-label">
              REVIEW INTELLIGENCE
            </span>
          </article>

          {/* --------------------------------------------------
              Feature 03
              -------------------------------------------------- */}

          <article className="feature-card">
            <span className="feature-number">03</span>

            <div className="feature-card__film-frame feature-card__film-frame--cream">
              <span className="frame-hole" />
              <span className="frame-hole" />
              <span className="frame-hole" />
            </div>

            <h3>COMPARE</h3>

            <p>
              Place two reviews side by side and see how their
              sentiment and cinematic opinions differ.
            </p>

            <span className="feature-label">
              COMPARATIVE ANALYSIS
            </span>
          </article>
        </div>
      </section>

      {/* ======================================================
          ABOUT
          ====================================================== */}

      <section
        className="about-section"
        id="about"
      >
        <div className="about-ticket">
          <div className="about-ticket__content">
            <span className="panel-overline">
              THE CINEVIEW METHOD
            </span>

            <h2>
              FROM WORDS
              <br />
              TO VERDICT
            </h2>

            <p>
              A recurrent neural network reads the sequence of
              words within a review and transforms that language
              into a measurable sentiment prediction.
            </p>
          </div>

          <div className="about-ticket__stamp">
            <span>CV</span>
            <small>EST.</small>
            <strong>2026</strong>
          </div>
        </div>
      </section>

      {/* ======================================================
          FOOTER
          ====================================================== */}

      <footer className="cineview-footer">
        <div>
          <strong>CINEVIEW</strong>
          <span>
            Movie Review Intelligence System
          </span>
        </div>

        <span>BUILT FOR THE LOVE OF FILM</span>
      </footer>

      <FilmStrip position="bottom" />
    </main>
  );
}

export default HomePage;