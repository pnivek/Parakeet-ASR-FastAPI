/**
 * Header — 64px tall, hairline border-bottom, brand on the left, nav on
 * the right. Theme switching isn't here in Maison — the warm graphite
 * palette is the only look.
 */

const MicLogo = ({ size = 34 }: { size?: number }) => (
  <svg viewBox="0 0 32 32" width={size} height={size} fill="none" aria-hidden>
    <defs>
      <linearGradient id="pk-logo-grad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stopColor="oklch(0.78 0.13 55)" />
        <stop offset="100%" stopColor="oklch(0.8 0.085 80)" />
      </linearGradient>
    </defs>
    {/* Pill background with the brand gradient */}
    <rect x="0" y="0" width="32" height="32" rx="9" fill="url(#pk-logo-grad)" />
    {/* Subtle inner highlight along the top edge */}
    <rect x="0.5" y="0.5" width="31" height="31" rx="8.5" fill="none" stroke="rgba(255,255,255,0.25)" />
    {/* Microphone glyph, rendered in the page bg color so it pops on the gradient */}
    <g stroke="oklch(0.13 0.012 60)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none">
      <rect x="12.5" y="7" width="7" height="11" rx="3.5" fill="oklch(0.13 0.012 60)" stroke="none" />
      <path d="M9 14.5 a7 7 0 0 0 14 0" />
      <line x1="16" y1="21.5" x2="16" y2="25" />
      <line x1="12.5" y1="25" x2="19.5" y2="25" />
    </g>
  </svg>
)

export function Header() {
  return (
    <header className="header">
      <div className="header__brand">
        <MicLogo />
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
          <span className="header__name">Parakeet</span>
          <span className="header__sub">playground</span>
        </div>
      </div>
      <div className="header__right">
        <a
          className="ma-link header__link"
          href="https://github.com/pnivek/Parakeet-ASR-FastAPI#readme"
          target="_blank"
          rel="noreferrer"
        >
          Documentation
        </a>
        <button
          type="button"
          className="header__api pk-glow-btn"
          style={
            {
              ['--btn-accent' as never]: 'var(--accent)',
              ['--top-hl' as never]: 'rgba(255,255,255,0.15)',
              ['--stroke-pct' as never]: '38%',
              ['--bottom-pct' as never]: '18%',
              ['--glow-r' as never]: '14px',
            } as React.CSSProperties
          }
        >
          Get API key →
        </button>
      </div>
    </header>
  )
}
