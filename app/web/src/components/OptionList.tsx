/**
 * Vertical option list — the vertical version of .ma-tabs. Each
 * option is a click-to-select pill inside a rounded track. The active
 * pill fills with accent; matches the source/output/engine tabs at
 * the top of the sidebar so the whole selector vocabulary reads as
 * one consistent system.
 */

export interface OptionListItem<T extends string> {
  id: T
  label: string
  disabled?: boolean
  disabledReason?: string
}

export function OptionList<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T
  options: OptionListItem<T>[]
  onChange: (next: T) => void
}) {
  return (
    <div className="ma-option-list" role="radiogroup">
      {options.map((o) => {
        const active = o.id === value
        const cls = active ? 'ma-option ma-option--active' : 'ma-option'
        return (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={o.disabled}
            title={o.disabled ? o.disabledReason : undefined}
            className={cls}
            onClick={() => {
              if (o.disabled) return
              if (!active) onChange(o.id)
            }}
          >
            <span>{o.label}</span>
            <span className="ma-option__dot" aria-hidden />
          </button>
        )
      })}
    </div>
  )
}
