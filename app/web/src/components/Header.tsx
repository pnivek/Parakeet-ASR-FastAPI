/**
 * Header — 64 px tall, hairline border-bottom. Bigger detailed mic logo,
 * single Documentation link. No theme switcher, no API key CTA (self-hosted
 * app).
 */

const MicLogo = () => (
  <svg viewBox="0 0 32 32" width="34" height="34" fill="none" aria-label="mic">
    <defs>
      <linearGradient id="ma-mic-cap" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.95" />
        <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.55" />
      </linearGradient>
      <radialGradient id="ma-mic-glow" cx="50%" cy="35%" r="55%">
        <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.35" />
        <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
      </radialGradient>
    </defs>
    <circle cx="16" cy="14" r="13" fill="url(#ma-mic-glow)" />
    {/* mic capsule */}
    <rect x="11.5" y="4.5" width="9" height="14" rx="4.5" fill="url(#ma-mic-cap)" stroke="var(--accent)" strokeOpacity="0.7" strokeWidth="0.6" />
    {/* grille lines */}
    <line x1="13.4" y1="8.2" x2="18.6" y2="8.2" stroke="rgba(0,0,0,0.35)" strokeWidth="0.6" strokeLinecap="round" />
    <line x1="13.4" y1="10.8" x2="18.6" y2="10.8" stroke="rgba(0,0,0,0.35)" strokeWidth="0.6" strokeLinecap="round" />
    <line x1="13.4" y1="13.4" x2="18.6" y2="13.4" stroke="rgba(0,0,0,0.35)" strokeWidth="0.6" strokeLinecap="round" />
    {/* highlight */}
    <rect x="12.4" y="5.4" width="2" height="11.5" rx="1" fill="rgba(255,255,255,0.35)" />
    {/* yoke */}
    <path d="M7.5 14.5 a8.5 8.5 0 0 0 17 0" fill="none" stroke="var(--fg)" strokeOpacity="0.85" strokeWidth="1.4" strokeLinecap="round" />
    {/* stand */}
    <line x1="16" y1="23" x2="16" y2="27.5" stroke="var(--fg)" strokeOpacity="0.85" strokeWidth="1.4" strokeLinecap="round" />
    <line x1="12" y1="27.5" x2="20" y2="27.5" stroke="var(--fg)" strokeOpacity="0.85" strokeWidth="1.4" strokeLinecap="round" />
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
      </div>
    </header>
  )
}
