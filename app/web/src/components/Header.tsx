/**
 * Header — 64px tall, hairline border-bottom, brand on the left, nav on
 * the right. Theme switching isn't here in Maison — the warm graphite
 * palette is the only look.
 */
const ParakeetLogo = () => (
  <svg viewBox="0 0 32 32" width={22} height={22} fill="none" aria-hidden>
    <defs>
      <linearGradient id="pk-logo-grad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stopColor="oklch(0.78 0.13 55)" />
        <stop offset="100%" stopColor="oklch(0.8 0.085 80)" />
      </linearGradient>
    </defs>
    <circle cx="16" cy="16" r="14" fill="url(#pk-logo-grad)" />
    <circle cx="16" cy="16" r="3.5" fill="white" />
    <circle cx="16" cy="16" r="1.4" fill="oklch(0.13 0.012 60)" />
  </svg>
)

export function Header() {
  return (
    <header className="header">
      <div className="header__brand">
        <ParakeetLogo />
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span className="header__name">Parakeet</span>
          <span className="header__sub">playground</span>
          <span className="header__chip">v5 · maison</span>
        </div>
      </div>
      <div className="header__right">
        <a className="ma-link header__link" href="#docs">Documentation</a>
        <a className="ma-link header__link" href="#pricing">Pricing</a>
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
