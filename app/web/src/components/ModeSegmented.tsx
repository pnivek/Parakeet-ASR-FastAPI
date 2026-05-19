export type InputMode = 'file' | 'mic' | 'url'

interface Props {
  mode: InputMode
  onChange: (m: InputMode) => void
}

const UploadIcon = ({ size = 14 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M12 16V4M12 4l-4 4M12 4l4 4" />
    <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
  </svg>
)
const MicIcon = ({ size = 14 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="9" y="3" width="6" height="12" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
  </svg>
)
const LinkIcon = ({ size = 14 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1" />
    <path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" />
  </svg>
)

const OPTIONS: { id: InputMode; label: string; Icon: React.FC<{ size?: number }> }[] = [
  { id: 'file', label: 'File', Icon: UploadIcon },
  { id: 'mic', label: 'Live mic', Icon: MicIcon },
  { id: 'url', label: 'URL', Icon: LinkIcon },
]

export function ModeSegmented({ mode, onChange }: Props) {
  return (
    <div className="mode" role="tablist" aria-label="Input source">
      {OPTIONS.map((opt) => {
        const active = opt.id === mode
        return (
          <button
            key={opt.id}
            role="tab"
            aria-selected={active}
            className={active ? 'mode__btn mode__btn--active' : 'mode__btn'}
            onClick={() => onChange(opt.id)}
            type="button"
          >
            <opt.Icon /> {opt.label}
          </button>
        )
      })}
    </div>
  )
}
