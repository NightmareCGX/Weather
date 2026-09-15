interface ZeusWxBrandProps {
  className?: string;
  theme?: "light" | "dark" | "auto";
}

/**
 * Zeus Wx brand logotype (Concept C: Streamline Underpinned Wordmark).
 * Fuses the Zeus aerodynamic lightning bolt directly as the capital 'Z',
 * connects seamlessly into 'eus', and accents with professional meteorology 'Wx'
 * underpinned by an atmospheric jet-stream isobar line.
 */
export function ZeusWxBrand({ className = "" }: ZeusWxBrandProps) {
  return (
    <div
      className={`relative inline-flex items-baseline select-none ${className}`}
      aria-label="Zeus Wx"
    >
      {/* 1. Aerodynamic Lightning Bolt acting as initial 'Z' */}
      <svg
        width="32"
        height="32"
        viewBox="0 0 36 44"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="shrink-0 translate-y-1"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="brand-bolt-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="50%" stopColor="#818cf8" />
            <stop offset="100%" stopColor="#fbbf24" />
          </linearGradient>
          <filter id="brand-bolt-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#818cf8" floodOpacity="0.45" />
          </filter>
        </defs>
        <path
          d="M4 6H32L19.5 21.5H30.5L8 42L14.5 24H5.5L10 14H4V6Z"
          fill="url(#brand-bolt-grad)"
          stroke="rgba(255,255,255,0.35)"
          strokeWidth="0.8"
          strokeLinejoin="round"
          filter="url(#brand-bolt-glow)"
        />
      </svg>

      {/* 2. Seamless 'eus' Wordmark */}
      <span className="text-2xl font-extrabold tracking-tight text-slate-100 ml-0.5 font-sans">
        eus
      </span>

      {/* 3. High-Voltage Wx Meteorological Shorthand */}
      <span className="ml-2 inline-flex items-baseline font-black tracking-tight text-base font-sans">
        <span className="text-sky-400">W</span>
        <span className="text-amber-400">x</span>
      </span>

      {/* 4. Atmospheric Jet Streamline Underpinning */}
      <svg
        className="absolute -bottom-1 left-2.5 w-32 h-2.5 pointer-events-none"
        viewBox="0 0 120 10"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="brand-stream-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.9" />
            <stop offset="70%" stopColor="#c084fc" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#c084fc" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path
          d="M0 2C25 2 40 8 70 8C95 8 110 3 120 3"
          stroke="url(#brand-stream-grad)"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}

/**
 * Standalone Zeus Icon (for Favicons, app shortcuts, and compact avatar usages).
 */
export function ZeusLogo({ className = "h-8 w-8" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="zeus-icon-bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#0f172a" />
          <stop offset="100%" stopColor="#020617" />
        </linearGradient>
        <linearGradient id="zeus-icon-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#38bdf8" />
          <stop offset="50%" stopColor="#818cf8" />
          <stop offset="100%" stopColor="#fbbf24" />
        </linearGradient>
      </defs>
      <rect width="48" height="48" rx="12" fill="url(#zeus-icon-bg)" />
      <rect width="48" height="48" rx="12" stroke="rgba(255,255,255,0.12)" strokeWidth="1.2" />
      <path
        d="M14 14H34L25.5 25.5H33L17 38L21.5 27.5H14.5L18 19H14V14Z"
        fill="url(#zeus-icon-grad)"
        stroke="rgba(255,255,255,0.35)"
        strokeWidth="0.8"
        strokeLinejoin="round"
      />
    </svg>
  );
}
