"use client";

interface LocateMeButtonProps {
  onClick: () => void;
  isLocating?: boolean;
  disabled?: boolean;
  className?: string;
}

/**
 * Dedicated "Locate Me" button for browser geolocation.
 *
 * Meets accessibility standards:
 * - Semantic `<button>` with clear `aria-label`
 * - `aria-busy="true"` and loading spinner while locating
 * - Keyboard accessible with visible focus rings
 * - Standard map control styling aligning with MapLibre controls
 */
export function LocateMeButton({
  onClick,
  isLocating = false,
  disabled = false,
  className = "",
}: LocateMeButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || isLocating}
      aria-label="Locate me"
      aria-busy={isLocating}
      title={isLocating ? "Locating…" : "Locate me"}
      className={`flex h-8 w-8 items-center justify-center rounded border border-slate-300 bg-white text-slate-700 shadow-sm transition-colors hover:bg-slate-50 hover:text-slate-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 disabled:opacity-60 sm:h-9 sm:w-9 ${className}`}
    >
      {isLocating ? (
        <svg
          className="h-4 w-4 animate-spin text-slate-600 sm:h-5 sm:w-5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          aria-hidden="true"
          focusable="false"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      ) : (
        <svg
          className="h-4 w-4 sm:h-5 sm:w-5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
          focusable="false"
        >
          {/* Crosshairs target icon */}
          <circle cx="12" cy="12" r="7" />
          <line x1="12" y1="2" x2="12" y2="5" />
          <line x1="12" y1="19" x2="12" y2="22" />
          <line x1="2" y1="12" x2="5" y2="12" />
          <line x1="19" y1="12" x2="22" y2="12" />
          <circle cx="12" cy="12" r="2" fill="currentColor" />
        </svg>
      )}
    </button>
  );
}
