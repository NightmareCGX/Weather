"use client";

interface ParticleToggleButtonProps {
  enabled: boolean;
  onToggle: () => void;
  disabled?: boolean;
  className?: string;
}

/**
 * Dedicated toggle button for 10-meter wind particle animation overlay.
 *
 * Meets accessibility standards:
 * - Semantic `<button>` with clear `aria-label` and `aria-pressed`
 * - Keyboard accessible with visible focus rings
 * - Standard map control styling aligning with MapLibre controls and LocateMeButton
 */
export function ParticleToggleButton({
  enabled,
  onToggle,
  disabled = false,
  className = "",
}: ParticleToggleButtonProps) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      aria-label="Toggle wind particle animation"
      aria-pressed={enabled}
      title={enabled ? "Pause wind particle animation" : "Enable wind particle animation"}
      data-testid="wind-particle-toggle"
      className={`flex h-8 w-8 items-center justify-center rounded-lg border shadow-xl backdrop-blur-md transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 disabled:opacity-60 sm:h-9 sm:w-9 ${
        enabled
          ? "border-cyan-500/60 bg-slate-900/95 text-cyan-400 hover:bg-slate-800 hover:text-cyan-300"
          : "border-slate-700/80 bg-slate-900/90 text-slate-500 hover:bg-slate-800 hover:text-slate-300"
      } ${className}`}
    >
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
        {/* Wind flow streams icon */}
        <path d="M12.8 19.6A2 2 0 1 0 14 16H2" />
        <path d="M17.5 8a2.5 2.5 0 1 1 2 4H2" />
        <path d="M9.8 4.4A2 2 0 1 1 11 8H2" />
        {!enabled && (
          <line
            x1="3"
            y1="21"
            x2="21"
            y2="3"
            stroke="currentColor"
            strokeWidth="2.2"
            className="text-rose-400/80"
          />
        )}
      </svg>
    </button>
  );
}
