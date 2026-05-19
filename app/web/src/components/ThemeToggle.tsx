import { useEffect } from 'react'
import { useSettings, type ThemeMode } from '../lib/settings'

const SystemIcon = () => (
  <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.6">
    <rect x="3" y="4" width="18" height="13" rx="2" />
    <path d="M8 21h8M12 17v4" />
  </svg>
)
const LightIcon = () => (
  <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
    <circle cx="12" cy="12" r="4" />
    <path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4l1.4-1.4M17 7l1.4-1.4" />
  </svg>
)
const DarkIcon = () => (
  <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 13A9 9 0 1 1 11 3a7 7 0 0 0 10 10z" />
  </svg>
)

const OPTIONS: { value: ThemeMode; icon: React.FC; label: string }[] = [
  { value: 'system', icon: SystemIcon, label: 'System' },
  { value: 'light', icon: LightIcon, label: 'Light' },
  { value: 'dark', icon: DarkIcon, label: 'Dark' },
]

export function ThemeToggle() {
  const theme = useSettings((s) => s.theme)
  const setTheme = useSettings((s) => s.set)

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
  }, [theme])

  return (
    <div className="theme" role="radiogroup" aria-label="Theme">
      {OPTIONS.map((opt) => {
        const Icon = opt.icon
        const active = theme === opt.value
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={opt.label}
            title={opt.label}
            className={active ? 'theme__btn theme__btn--active' : 'theme__btn'}
            onClick={() => setTheme('theme', opt.value)}
          >
            <Icon />
          </button>
        )
      })}
    </div>
  )
}
