import { useEffect } from 'react'
import { useSettings, type ThemeMode } from '../lib/settings'

const OPTIONS: { value: ThemeMode; label: string }[] = [
  { value: 'system', label: 'System' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
]

/**
 * Three-way theme switch (System / Light / Dark) persisted via the Zustand
 * store. Writes `data-theme="light"|"dark"` on <html> when an explicit
 * theme is chosen; clears it for system, letting `prefers-color-scheme`
 * win. The CSS reads `[data-theme="..."]` selectors in App.css to override
 * the media-query defaults.
 */
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
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          type="button"
          role="radio"
          aria-checked={theme === opt.value}
          className={theme === opt.value ? 'theme__btn theme__btn--active' : 'theme__btn'}
          onClick={() => setTheme('theme', opt.value)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}
