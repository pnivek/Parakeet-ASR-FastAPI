import { useState } from 'react'
import { useSettings } from '../lib/settings'
import type { ResponseFormat, Strategy, TimestampGranularity } from '../lib/types'

const FORMATS: { id: ResponseFormat; label: string; sub: string }[] = [
  { id: 'verbose_json', label: 'verbose_json', sub: 'segments + words + meta' },
  { id: 'json', label: 'json', sub: 'compact { text }' },
  { id: 'text', label: 'text', sub: 'plain text only' },
  { id: 'srt', label: 'srt', sub: 'subtitle file' },
  { id: 'vtt', label: 'vtt', sub: 'webvtt file' },
]

const STRATEGIES: { id: Strategy; label: string; sub: string }[] = [
  { id: 'auto', label: 'auto', sub: 'pick per duration' },
  { id: 'full', label: 'full', sub: 'one-shot decode' },
  { id: 'chunked', label: 'chunked', sub: 'sentence-bounded' },
  { id: 'progressive', label: 'progressive', sub: 'live · WS only' },
]

const ChevIcon = ({ up }: { up: boolean }) => (
  <svg
    viewBox="0 0 24 24"
    width={12}
    height={12}
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ transform: up ? 'rotate(180deg)' : 'none', transition: 'transform .15s' }}
    aria-hidden
  >
    <path d="M6 9l6 6 6-6" />
  </svg>
)
const ResetIcon = () => (
  <svg viewBox="0 0 24 24" width={12} height={12} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
    <path d="M19 12a7 7 0 1 1-2-4.9" />
    <path d="M19 4v5h-5" />
  </svg>
)
const CheckIcon = () => (
  <svg viewBox="0 0 24 24" width={10} height={10} fill="none" stroke="oklch(0.15 0.02 250)" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M5 12l5 5L20 7" />
  </svg>
)

export function SettingsPanel() {
  const s = useSettings()
  const [advancedOpen, setAdvancedOpen] = useState(false)

  const toggleGranularity = (g: TimestampGranularity) => {
    const current = new Set(s.timestampGranularities)
    if (current.has(g)) current.delete(g)
    else current.add(g)
    if (current.size === 0) current.add('segment')
    s.set('timestampGranularities', Array.from(current))
  }

  const granDisabled = s.responseFormat !== 'verbose_json'
  const currentFmt = FORMATS.find((f) => f.id === s.responseFormat)

  return (
    <section className="glass input-card">
      {/* Response format */}
      <div className="settings__group">
        <div className="settings__label">Response format</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
          {FORMATS.map((f) => (
            <button
              key={f.id}
              type="button"
              title={f.sub}
              className={f.id === s.responseFormat ? 'chip chip--active' : 'chip'}
              onClick={() => s.set('responseFormat', f.id)}
            >
              {f.label}
            </button>
          ))}
        </div>
        {currentFmt && <div className="settings__hint">{currentFmt.sub}</div>}
      </div>

      <div className="settings__divider" />

      {/* Timestamps */}
      <div className="settings__group">
        <div className="settings__label">Timestamp granularities</div>
        <div style={{ display: 'flex', gap: 8 }}>
          {(['segment', 'word'] as TimestampGranularity[]).map((id) => {
            const on = s.timestampGranularities.includes(id)
            return (
              <button
                key={id}
                type="button"
                disabled={granDisabled}
                className={on ? 'gran gran--on' : 'gran'}
                onClick={() => toggleGranularity(id)}
              >
                <span style={{ textTransform: 'capitalize' }}>{id}</span>
                <span className="gran__box">{on && <CheckIcon />}</span>
              </button>
            )
          })}
        </div>
        {granDisabled && (
          <div className="settings__hint">
            Only available with <span className="mono">verbose_json</span>
          </div>
        )}
      </div>

      <div className="settings__divider" />

      {/* Strategy */}
      <div className="settings__group">
        <div className="settings__label">Strategy</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 5 }}>
          {STRATEGIES.map((opt) => (
            <button
              key={opt.id}
              type="button"
              className={s.strategy === opt.id ? 'strat strat--active' : 'strat'}
              onClick={() => s.set('strategy', opt.id)}
            >
              <div className="strat__name">{opt.label}</div>
              <div className="strat__sub">{opt.sub}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="settings__divider" />

      <button
        type="button"
        className="adv-toggle"
        onClick={() => setAdvancedOpen((o) => !o)}
        aria-expanded={advancedOpen}
      >
        <span className="settings__label">Advanced</span>
        <span className="adv-toggle__right">
          {advancedOpen ? 'hide' : 'show'}
          <ChevIcon up={advancedOpen} />
        </span>
      </button>

      {advancedOpen && (
        <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 14 }}>
          <DarkSlider
            label="long_audio_threshold"
            value={s.longAudioThreshold ?? 480}
            min={30}
            max={1800}
            step={30}
            onChange={(v) => s.set('longAudioThreshold', v)}
            fmt={(v) => `${v}s`}
          />
          <DarkSlider
            label="batch_size"
            value={s.batchSize ?? 4}
            min={1}
            max={16}
            step={1}
            onChange={(v) => s.set('batchSize', v)}
            fmt={(v) => v.toFixed(0)}
          />
          <DarkSlider
            label="chunk_length"
            value={s.chunkLength ?? 30}
            min={5}
            max={60}
            step={5}
            onChange={(v) => s.set('chunkLength', v)}
            fmt={(v) => `${v}s`}
          />
          <DarkToggle
            label="live_latency"
            sub="10-2-2 preset (~6 s lag) — WS only"
            value={s.liveLatency}
            onChange={(v) => s.set('liveLatency', v)}
          />
          <DarkToggle
            label="progressive_refinement"
            sub="EOF full pass replaces streamed segments"
            value={s.progressiveRefinement}
            onChange={(v) => s.set('progressiveRefinement', v)}
          />
          <button
            type="button"
            onClick={s.reset}
            style={{
              marginTop: 2,
              padding: '7px 12px',
              borderRadius: 8,
              cursor: 'pointer',
              border: '1px solid var(--border-soft)',
              background: 'rgba(255,255,255,0.025)',
              color: 'var(--fg-dim)',
              fontSize: 11.5,
              alignSelf: 'flex-start',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontFamily: 'inherit',
            }}
          >
            <ResetIcon />
            Reset defaults
          </button>
        </div>
      )}
    </section>
  )
}

function DarkSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  fmt,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  fmt?: (v: number) => string
}) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div>
      <div className="slider__head">
        <div className="slider__name">{label}</div>
        <div className="slider__val">{fmt ? fmt(value) : value.toFixed(2)}</div>
      </div>
      <div className="slider__track">
        <div className="slider__fill" style={{ width: `${pct}%` }} />
        <div className="slider__thumb" style={{ left: `${pct}%` }} />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="slider__input"
          aria-label={label}
        />
      </div>
    </div>
  )
}

function DarkToggle({
  label,
  sub,
  value,
  onChange,
}: {
  label: string
  sub?: string
  value: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="toggle-row">
      <div>
        <div className="toggle-row__label">{label}</div>
        {sub && <div className="toggle-row__sub">{sub}</div>}
      </div>
      <button
        type="button"
        className={value ? 'toggle toggle--on' : 'toggle'}
        onClick={() => onChange(!value)}
        aria-pressed={value}
        aria-label={label}
      >
        <span className="toggle__knob" />
      </button>
    </div>
  )
}
