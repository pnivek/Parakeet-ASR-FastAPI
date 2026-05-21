/**
 * Maison-vocabulary dropdown — pill trigger + rounded menu panel.
 * Used for the Input, Format, and Strategy selectors in the sidebar.
 * Closes on outside-click and on Escape.
 */
import { useEffect, useRef, useState } from 'react'

const ChevIcon = () => (
  <svg
    className="ma-dd__chev"
    viewBox="0 0 24 24"
    width="11"
    height="11"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
    <path d="M6 9l6 6 6-6" />
  </svg>
)

const CheckIcon = () => (
  <svg
    className="ma-dd__check"
    viewBox="0 0 24 24"
    width="11"
    height="11"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
    <path d="M5 12l5 5L20 7" />
  </svg>
)

export interface DropdownOption<T extends string> {
  id: T
  label: string
  disabled?: boolean
  disabledReason?: string
}

export function Dropdown<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
}: {
  value: T
  options: DropdownOption<T>[]
  onChange: (next: T) => void
  ariaLabel?: string
}) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const current = options.find((o) => o.id === value)
  return (
    <div className="ma-dd" ref={wrapRef}>
      <button
        type="button"
        className="ma-dd__trigger"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={ariaLabel}
      >
        <span>{current ? current.label : value}</span>
        <ChevIcon />
      </button>
      {open && (
        <div className="ma-dd__menu" role="listbox">
          {options.map((o) => (
            <button
              key={o.id}
              type="button"
              role="option"
              aria-selected={o.id === value}
              disabled={o.disabled}
              title={o.disabled ? o.disabledReason : undefined}
              className={
                o.id === value ? 'ma-dd__item ma-dd__item--active' : 'ma-dd__item'
              }
              onClick={() => {
                if (o.disabled) return
                onChange(o.id)
                setOpen(false)
              }}
            >
              <span>{o.label}</span>
              <CheckIcon />
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
