import { ThemeToggle } from './ThemeToggle'
import { StatusDot, type StreamState } from './StatusDot'

interface Props {
  state: StreamState
}

const ParakeetLogo = () => (
  <svg viewBox="0 0 32 32" width={28} height={28} fill="none" aria-hidden>
    <defs>
      <linearGradient id="pk-logo-grad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stopColor="oklch(0.78 0.13 55)" />
        <stop offset="100%" stopColor="oklch(0.65 0.17 215)" />
      </linearGradient>
    </defs>
    <circle cx="16" cy="16" r="14" fill="url(#pk-logo-grad)" />
    <circle cx="16" cy="16" r="3.5" fill="white" />
    <circle cx="16" cy="16" r="1.4" fill="oklch(0.18 0.025 250)" />
  </svg>
)

export function Header({ state }: Props) {
  return (
    <header className="header">
      <div className="header__brand">
        <ParakeetLogo />
        <div>
          <div className="header__title">
            Parakeet
            <span className="header__title--muted">Playground</span>
          </div>
          <div className="header__tagline">
            Whisper-compatible streaming ASR
            <span className="serif"> — listen, transcribe, ship.</span>
          </div>
        </div>
      </div>
      <div className="header__right">
        <ThemeToggle />
        <div className="header__divider" />
        <StatusDot state={state} />
      </div>
    </header>
  )
}
