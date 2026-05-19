import { useState } from 'react'
import { useSettings } from '../lib/settings'
import type { ResponseFormat, Strategy, TimestampGranularity } from '../lib/types'

const RESPONSE_FORMATS: { value: ResponseFormat; label: string; help: string }[] = [
  { value: 'json', label: 'json (compact)', help: 'OpenAI default — just { "text": ... }.' },
  { value: 'verbose_json', label: 'verbose_json', help: 'Full Whisper shape: segments, tokens, timestamps, words (if enabled).' },
  { value: 'text', label: 'text', help: 'Plain text body.' },
  { value: 'srt', label: 'srt', help: 'SubRip subtitle file.' },
  { value: 'vtt', label: 'vtt', help: 'WebVTT subtitle file.' },
]

const STRATEGIES: { value: Strategy; label: string; help: string }[] = [
  { value: 'auto', label: 'auto', help: 'Server picks — chunked for REST, progressive for WS.' },
  { value: 'full', label: 'full', help: 'Single-pass through encoder + decoder. Fastest on long offline files.' },
  { value: 'chunked', label: 'chunked', help: 'Streaming-style sentence-bounded segments. Default for REST.' },
  { value: 'progressive', label: 'progressive', help: 'For live WS only. Errors on REST.' },
]

/**
 * Compact form of all transcription settings. State lives in the persisted
 * Zustand store (`settings.ts`) so reloads keep the user's choices.
 */
export function SettingsPanel() {
  const s = useSettings()
  const [advancedOpen, setAdvancedOpen] = useState(false)

  const toggleGranularity = (g: TimestampGranularity) => {
    const current = new Set(s.timestampGranularities)
    if (current.has(g)) current.delete(g)
    else current.add(g)
    // Always keep at least "segment" — the server defaults to it anyway.
    if (current.size === 0) current.add('segment')
    s.set('timestampGranularities', Array.from(current))
  }

  return (
    <section className="settings">
      <h2>Settings</h2>

      <label className="settings__row">
        <span>Response format</span>
        <select
          value={s.responseFormat}
          onChange={(e) => s.set('responseFormat', e.target.value as ResponseFormat)}
        >
          {RESPONSE_FORMATS.map((f) => (
            <option key={f.value} value={f.value}>
              {f.label}
            </option>
          ))}
        </select>
        <small>{RESPONSE_FORMATS.find((f) => f.value === s.responseFormat)?.help}</small>
      </label>

      <fieldset className="settings__row settings__row--checks">
        <legend>Timestamps</legend>
        <label>
          <input
            type="checkbox"
            checked={s.timestampGranularities.includes('segment')}
            onChange={() => toggleGranularity('segment')}
          />
          segment (default)
        </label>
        <label>
          <input
            type="checkbox"
            checked={s.timestampGranularities.includes('word')}
            onChange={() => toggleGranularity('word')}
          />
          word
        </label>
        <small>
          Word-level timestamps only appear in <code>verbose_json</code>. Aggregated client-side from
          SentencePiece word boundaries.
        </small>
      </fieldset>

      <label className="settings__row">
        <span>Strategy</span>
        <select
          value={s.strategy}
          onChange={(e) => s.set('strategy', e.target.value as Strategy)}
        >
          {STRATEGIES.map((st) => (
            <option key={st.value} value={st.value}>
              {st.label}
            </option>
          ))}
        </select>
        <small>{STRATEGIES.find((st) => st.value === s.strategy)?.help}</small>
      </label>

      <button
        type="button"
        className="settings__toggle"
        onClick={() => setAdvancedOpen((o) => !o)}
        aria-expanded={advancedOpen}
      >
        {advancedOpen ? '▾' : '▸'} Advanced
      </button>

      {advancedOpen && (
        <div className="settings__advanced">
          <label className="settings__row">
            <span>Long-audio threshold (s)</span>
            <input
              type="number"
              min={0}
              max={3600}
              step={10}
              value={s.longAudioThreshold ?? ''}
              placeholder="server default"
              onChange={(e) =>
                s.set('longAudioThreshold', e.target.value === '' ? null : Number(e.target.value))
              }
            />
            <small>Audio above this duration switches the encoder to local-window attention.</small>
          </label>

          <label className="settings__row">
            <span>Batch size</span>
            <input
              type="number"
              min={1}
              max={32}
              value={s.batchSize ?? ''}
              placeholder="server default (4)"
              onChange={(e) =>
                s.set('batchSize', e.target.value === '' ? null : Number(e.target.value))
              }
            />
          </label>

          <label className="settings__row">
            <span>Chunk length (s)</span>
            <input
              type="number"
              min={1}
              max={300}
              value={s.chunkLength ?? ''}
              placeholder="server default"
              onChange={(e) =>
                s.set('chunkLength', e.target.value === '' ? null : Number(e.target.value))
              }
            />
          </label>

          <label className="settings__row settings__row--inline">
            <input
              type="checkbox"
              checked={s.liveLatency}
              onChange={(e) => s.set('liveLatency', e.target.checked)}
            />
            <span>live_latency (WS)</span>
            <small>
              Use the 10-2-2 streaming preset (~6 s emission lag) instead of 10-10-5 (~7.5 s,
              offline-like quality).
            </small>
          </label>

          <label className="settings__row settings__row--inline">
            <input
              type="checkbox"
              checked={s.progressiveRefinement}
              onChange={(e) => s.set('progressiveRefinement', e.target.checked)}
            />
            <span>progressive_refinement (WS)</span>
            <small>On EOF, run a single FULL pass over the accumulated PCM and replace streamed segments with offline-quality output.</small>
          </label>

          <button type="button" className="settings__reset" onClick={s.reset}>
            Reset to defaults
          </button>
        </div>
      )}
    </section>
  )
}
