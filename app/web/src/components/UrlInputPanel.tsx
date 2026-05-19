import { useCallback, useState } from 'react'
import { postTranscriptionUrl, type TranscriptionResponse } from '../lib/api'
import { useSettings } from '../lib/settings'

interface Props {
  onResult: (filename: string, result: TranscriptionResponse) => void
  onError: (message: string) => void
  onBusyChange?: (busy: boolean) => void
}

const LinkIcon = () => (
  <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1" />
    <path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" />
  </svg>
)
const SparkleIcon = () => (
  <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8" />
  </svg>
)

export function UrlInputPanel({ onResult, onError, onBusyChange }: Props) {
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const settings = useSettings()

  const transcribe = useCallback(async () => {
    if (!url.trim()) return
    setBusy(true)
    onBusyChange?.(true)
    try {
      const result = await postTranscriptionUrl(url.trim(), {
        response_format: settings.responseFormat,
        timestamp_granularities: settings.timestampGranularities,
        strategy: settings.strategy,
        chunk_length: settings.chunkLength ?? undefined,
        chunk_overlap: settings.chunkOverlap ?? undefined,
        batch_size: settings.batchSize ?? undefined,
        long_audio_threshold: settings.longAudioThreshold ?? undefined,
      })
      // Derive a filename for the player from the URL's path tail.
      let name = url
      try {
        const u = new URL(url)
        name = u.pathname.split('/').filter(Boolean).pop() || u.host
      } catch {
        /* keep url as-is */
      }
      onResult(name, result)
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      onBusyChange?.(false)
    }
  }, [url, settings, onResult, onError, onBusyChange])

  return (
    <section className="glass input-card">
      <div className="mono" style={{ fontSize: 10, color: 'var(--muted-deep)', letterSpacing: 0.6, marginBottom: 8 }}>
        AUDIO URL
      </div>
      <div className="url-input">
        <LinkIcon />
        <input
          type="url"
          value={url}
          placeholder="https://example.com/audio.wav"
          onChange={(e) => setUrl(e.target.value)}
          spellCheck={false}
          autoComplete="off"
        />
      </div>
      <div className="mono" style={{ fontSize: 10, color: 'var(--muted-deep)', marginTop: 8, letterSpacing: 0.3 }}>
        Server fetches the audio · public http(s) only · 512 MB cap
      </div>
      <button
        type="button"
        onClick={transcribe}
        disabled={!url.trim() || busy}
        className="btn-primary pk-glow-btn"
        style={
          {
            marginTop: 12,
            ['--btn-accent' as never]: 'var(--accent)',
            ['--top-hl' as never]: 'rgba(255,255,255,0.18)',
            ['--stroke-pct' as never]: '45%',
            ['--bottom-pct' as never]: '22%',
            ['--glow-r' as never]: '26px',
          } as React.CSSProperties
        }
      >
        <SparkleIcon /> {busy ? 'Transcribing…' : 'Transcribe from URL'}
      </button>
    </section>
  )
}
