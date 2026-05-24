import { useEffect, useRef, useState } from 'react'
import { Waveform } from './Waveform'
import { formatTime } from '../lib/format'
import {
  pause,
  play,
  useCurrentTime,
  useDuration,
  useIsPlaying,
} from '../lib/playback'
import {
  downloadAs,
  downloadAudio,
  type DownloadFormat,
  type LoadedAudio,
} from '../lib/download'
import type { TranscriptionResponse } from '../lib/api'

export type HeroState = 'idle' | 'streaming' | 'done' | 'error'

interface Props {
  loaded: LoadedAudio | null
  peaks: number[] | null
  state: HeroState
  language?: string
  /** Current transcription result — drives the download flyout. */
  result: TranscriptionResponse | null
  /** True if we have at least one committed segment in the result.
   * Disables prev/next buttons when false. */
  hasSegments: boolean
  /** Seek to the previous / next segment start. */
  onPrevSegment: () => void
  onNextSegment: () => void
  /** Pause + seek 0 (what the old single "Reset" button did). */
  onRewind: () => void
  /** Jump a live URL stream's <audio> element to its live edge. */
  onLiveEdge: () => void
  /** True when audio playhead is within ~2s of server's audio_received_s.
   * Drives the red-dot indicator on the Live button — lit when you ARE
   * live, hollow when you have catching up to do. */
  atLiveEdge: boolean
  /** True while partials are still arriving — used to show the Live
   * jump button (URL streams only). */
  live: boolean
}

const PlayIcon = ({ size = 20 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden>
    <path d="M8 5.5v13a.5.5 0 0 0 .77.42l10-6.5a.5.5 0 0 0 0-.84l-10-6.5A.5.5 0 0 0 8 5.5z" />
  </svg>
)
const PauseIcon = ({ size = 20 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden>
    <rect x="6" y="5" width="4" height="14" rx="1" />
    <rect x="14" y="5" width="4" height="14" rx="1" />
  </svg>
)
const PrevIcon = () => (
  <svg viewBox="0 0 24 24" width={13} height={13} fill="currentColor" aria-hidden>
    <path d="M6 5h2v14H6zM20 5L10 12l10 7z" />
  </svg>
)
const NextIcon = () => (
  <svg viewBox="0 0 24 24" width={13} height={13} fill="currentColor" aria-hidden>
    <path d="M16 5h2v14h-2zM4 5l10 7-10 7z" />
  </svg>
)
const RewindIcon = () => (
  <svg viewBox="0 0 24 24" width={13} height={13} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden>
    <path d="M19 12a7 7 0 1 1-2-4.9" />
    <path d="M19 4v5h-5" />
  </svg>
)
const LiveDotIcon = ({ filled }: { filled: boolean }) => (
  <svg viewBox="0 0 24 24" width={9} height={9} aria-hidden>
    <circle
      cx="12"
      cy="12"
      r="6"
      fill={filled ? 'oklch(0.62 0.22 25)' : 'transparent'}
      stroke={filled ? 'oklch(0.62 0.22 25)' : 'currentColor'}
      strokeWidth={filled ? 0 : 2}
    />
  </svg>
)
const DownloadIcon = () => (
  <svg viewBox="0 0 24 24" width={13} height={13} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M12 4v12M12 16l-4-4M12 16l4-4" />
    <path d="M4 20h16" />
  </svg>
)
const CaretIcon = () => (
  <svg viewBox="0 0 24 24" width={9} height={9} fill="currentColor" aria-hidden>
    <path d="M7 10l5 5 5-5z" />
  </svg>
)

const STATE_LABEL: Record<HeroState, string> = {
  idle: 'standby',
  streaming: 'streaming',
  done: 'complete',
  error: 'error',
}

export function HeroRow({
  loaded,
  peaks,
  state,
  language = 'en',
  result,
  hasSegments,
  onPrevSegment,
  onNextSegment,
  onRewind,
  onLiveEdge,
  atLiveEdge,
  live,
}: Props) {
  const t = useCurrentTime()
  const dur = useDuration()
  const playing = useIsPlaying()
  const progress = dur > 0 ? Math.round((t / dur) * 100) : 0
  // Live button is only meaningful for URL streams that are actively
  // being fed — a file or finished recording has no head to seek to.
  const showLive = loaded?.kind === 'url' && (live || playing)

  return (
    <section className="hero spin-in" key={loaded?.title ?? 'empty'}>
      <div>
        <div className="hero__id-row">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="label-eyebrow">NOW PLAYING</span>
            <span
              className={`status-pill ${
                state === 'streaming' ? 'status-pill--streaming' : state === 'idle' ? 'status-pill--idle' : ''
              }`}
            >
              <span className="status-pill__dot" />
              {STATE_LABEL[state]}
            </span>
          </div>
          <div className="hero__position">
            <span className="hero__position-l">POSITION</span>
            <span className="hero__position-v num">
              {formatTime(t)} <span className="hero__position-v--dim">/ {formatTime(dur)}</span>
            </span>
          </div>
        </div>
        <h1 className="hero__title">
          {loaded?.title ||
            (state === 'streaming'
              ? 'Recording…'
              : 'Drop audio, paste a URL, or record to begin.')}
        </h1>
        <div className="hero__meta">
          {loaded?.source && (
            <>
              <span className="hero__meta-mono">{loaded.source}</span>
              <span className="hero__meta-dot" />
            </>
          )}
          {dur > 0 && (
            <>
              <span className="mono num" style={{ letterSpacing: 0.4 }}>{dur.toFixed(2)}s</span>
              <span className="hero__meta-dot" />
            </>
          )}
          <span className="hero__meta-mono">{language}</span>
          <span className="hero__meta-dot" />
          <span className="mono" style={{ letterSpacing: 0.4 }}>parakeet-tdt-0.6b-v2</span>
        </div>
      </div>

      <div style={{ position: 'relative', padding: '4px 0 6px' }}>
        {/* Key flip on peaks ready triggers the CSS fade-in. Static
            keys keep the placeholder bars stable. */}
        <div key={peaks ? 'wf-ready' : 'wf-placeholder'} className={peaks ? 'wf-fade-in' : undefined}>
          <Waveform peaks={peaks} currentTime={t} duration={dur} height={90} />
        </div>
        <div className="hero__wf-times">
          <span>0:00</span>
          <span>{formatTime(dur * 0.25)}</span>
          <span>{formatTime(dur * 0.5)}</span>
          <span>{formatTime(dur * 0.75)}</span>
          <span>{formatTime(dur)}</span>
        </div>
      </div>

      <div className="transport">
        <button
          type="button"
          className="ma-pill ma-pill--lg"
          onClick={onPrevSegment}
          disabled={!loaded || !hasSegments}
          aria-label="Previous segment"
          title="Previous segment"
        >
          <PrevIcon />
        </button>
        <button
          type="button"
          className={
            playing
              ? 'ma-pill ma-pill--active transport__play'
              : 'ma-pill transport__play'
          }
          onClick={() => (playing ? pause() : play())}
          disabled={!loaded}
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? <PauseIcon /> : <PlayIcon />}
        </button>
        <button
          type="button"
          className="ma-pill ma-pill--lg"
          onClick={onNextSegment}
          disabled={!loaded || !hasSegments}
          aria-label="Next segment"
          title="Next segment"
        >
          <NextIcon />
        </button>
        <button
          type="button"
          className="ma-pill ma-pill--lg"
          onClick={onRewind}
          disabled={!loaded}
          aria-label="Rewind to start"
          title="Rewind"
        >
          <RewindIcon />
          Rewind
        </button>
        {showLive && (
          <button
            type="button"
            className={
              atLiveEdge
                ? 'ma-pill ma-pill--lg ma-pill--active'
                : 'ma-pill ma-pill--lg'
            }
            onClick={onLiveEdge}
            aria-label="Jump to live edge"
            title={atLiveEdge ? 'You are live' : 'Jump to live edge'}
          >
            <LiveDotIcon filled={atLiveEdge} />
            Live
          </button>
        )}
        {result && loaded && (
          <DownloadFlyout result={result} loaded={loaded} />
        )}

        <div className="transport__progress">
          <span className="transport__progress-l">PROGRESS</span>
          <span className="transport__progress-v num">{progress}%</span>
        </div>
      </div>
    </section>
  )
}

const FORMAT_LABELS: Array<{ id: DownloadFormat; label: string }> = [
  { id: 'json', label: 'JSON' },
  { id: 'text', label: 'Text' },
  { id: 'srt', label: 'SRT' },
  { id: 'vtt', label: 'VTT' },
  { id: 'csv', label: 'CSV' },
]

function DownloadFlyout({
  result,
  loaded,
}: {
  result: TranscriptionResponse
  loaded: LoadedAudio
}) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDocClick = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const baseName = loaded.title || 'transcript'
  const isVerbose = result.format === 'verbose_json'

  return (
    <div ref={wrapRef} className="hero__download" style={{ position: 'relative' }}>
      <button
        type="button"
        className="ma-pill ma-pill--lg"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <DownloadIcon />
        Download
        <CaretIcon />
      </button>
      {open && (
        <div className="hero__download-menu" role="menu">
          {FORMAT_LABELS.map((f) => (
            <button
              key={f.id}
              type="button"
              role="menuitem"
              className="hero__download-item"
              disabled={!isVerbose}
              onClick={() => {
                downloadAs(result, f.id, baseName)
                setOpen(false)
              }}
            >
              {f.label}
            </button>
          ))}
          <div className="hero__download-divider" />
          <button
            type="button"
            role="menuitem"
            className="hero__download-item"
            onClick={() => {
              downloadAudio(loaded)
              setOpen(false)
            }}
          >
            Audio
          </button>
        </div>
      )}
    </div>
  )
}
