/**
 * Vertical option list — each option is a clickable row. Active row
 * gets a 2px accent left-bar + accent text + a filled dot indicator
 * on the right. Always-visible, replaces the dropdowns we had for
 * Input / Format / Strategy.
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
  mono = false,
}: {
  value: T
  options: OptionListItem<T>[]
  onChange: (next: T) => void
  /** Render labels in mono lowercase (used for Strategy / Format which
   * are technical identifiers; Input mode reads better in sans). */
  mono?: boolean
}) {
  return (
    <div className="ma-option-list" role="radiogroup">
      {options.map((o) => {
        const active = o.id === value
        const cls = [
          'ma-option',
          mono ? 'ma-option--mono' : '',
          active ? 'ma-option--active' : '',
        ]
          .filter(Boolean)
          .join(' ')
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
