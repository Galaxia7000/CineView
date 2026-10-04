interface FilmStripProps {
  position?: "top" | "bottom";
  className?: string;
}

function FilmStrip({
  position = "top",
  className = "",
}: FilmStripProps) {
  const frames = Array.from({ length: 12 });

  return (
    <div
      className={`film-strip film-strip--${position} ${className}`}
      aria-hidden="true"
    >
      <div className="film-strip__sprockets">
        {frames.map((_, index) => (
          <span key={`top-${index}`} />
        ))}
      </div>

      <div className="film-strip__frames">
        {frames.map((_, index) => (
          <span key={`frame-${index}`} />
        ))}
      </div>

      <div className="film-strip__sprockets">
        {frames.map((_, index) => (
          <span key={`bottom-${index}`} />
        ))}
      </div>
    </div>
  );
}

export default FilmStrip;