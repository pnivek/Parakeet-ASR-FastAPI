import { Waveform } from './Waveform'
import { formatTime } from '../lib/format'
import { pause, play, resetPlayback, useCurrentTime, useDuration, useIsPlaying } from '../lib/playback'
import { downloadAudio, type LoadedAudio } from '../lib/download'

export type HeroState = 'idle' | 'streaming' | 'done' | 'error'

interface Props {
  /** Currently-loaded audio info — drives title, meta, downloads. */
  loaded: LoadedAudio | null
  /** Peaks for the waveform. */
  peaks: number[] | null
  /** Stream state pill in the eyebrow row. */
  state: HeroState
  /** Optional language label (defaults to "en"). */
  language?: string
}

const PlayIcon = ({ size = 17 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden>
    <path d="M8 5.5v13a.5.5 0 0 0 .77.42l10-6.5a.5.5 0 0 0 0-.84l-10-6.5A.5.5 0 0 0 8 5.5z" />
  </svg>
)
const PauseIcon = ({ size = 17 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden>
    <rect x="6" y="5" width="4" height="14" rx="1" />
    <rect x="14" y="5" width="4" height="14" rx="1" />
  </svg>
)
const ResetIcon = () => (
  <svg viewBox="0 0 24 24" width={11} height={11} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
    <path d="M19 12a7 7 0 1 1-2-4.9" />
    <path d="M19 4v5h-5" />
  </svg>
)
const DownloadIcon = () => (
  <svg viewBox="0 0 24 24" width={11} height={11} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M12 4v12M12 16l-4-4M12 16l4-4" />
    <path d="M4 20h16" />
  </svg>
)

const STATE_LABEL: Record<HeroState, string> = {
  idle: 'standby',
  streaming: 'streaming',
  done: 'complete',
  error: 'error',
}

export function HeroRow({ loaded, peaks, state, language = 'en' }: Props) {
  const t = useCurrentTime()
  const dur = useDuration()
  const playing = useIsPlaying()
  const progress = dur > 0 ? Math.round((t / dur) * 100) : 0
  const ext = loaded?.kind === 'file' && loaded.file ? loaded.file.name.split('.').pop() : 'wav'

  return (
    <section className="hero spin-in" key={loaded?.title ?? 'empty'}>
      <div className="hero__id">
        <div className="hero__eyebrow-row">
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
        <h1 className="hero__title">
          {loaded?.title || 'Drop audio, paste a URL, or record to begin.'}
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
        <Waveform peaks={peaks} currentTime={t} duration={dur} height={90} />
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
          className="transport__play pk-glow-btn"
          onClick={() => (playing ? pause() : play())}
          disabled={!loaded}
          aria-label={playing ? 'Pause' : 'Play'}
          style={
            {
              ['--btn-accent' as never]: 'var(--accent)',
              ['--top-hl' as never]: 'rgba(255,255,255,0.2)',
              ['--stroke-pct' as never]: '50%',
              ['--bottom-pct' as never]: '22%',
              ['--glow-r' as never]: '18px',
            } as React.CSSProperties
          }
        >
          {playing ? <PauseIcon /> : <PlayIcon />}
        </button>

        <div className="transport__pos">
          <span className="transport__pos-l">POSITION</span>
          <span className="transport__pos-v num">
            {formatTime(t)} <span className="transport__pos-v--dim">/ {formatTime(dur)}</span>
          </span>
        </div>

        <span className="transport__divider" />

        <button
          type="button"
          className="glass-pill pk-glow-btn"
          onClick={resetPlayback}
          disabled={!loaded}
          style={
            {
              ['--btn-accent' as never]: 'var(--accent)',
              ['--top-hl' as never]: 'rgba(255,255,255,0.14)',
              ['--stroke-pct' as never]: '18%',
              ['--bottom-pct' as never]: '0%',
              ['--glow-r' as never]: '14px',
            } as React.CSSProperties
          }
        >
          <ResetIcon />
          Reset
        </button>
        {loaded && (
          <button
            type="button"
            className="glass-pill pk-glow-btn"
            onClick={() => downloadAudio(loaded)}
            style={
              {
                ['--btn-accent' as never]: 'var(--accent)',
                ['--top-hl' as never]: 'rgba(255,255,255,0.14)',
                ['--stroke-pct' as never]: '18%',
                ['--bottom-pct' as never]: '0%',
                ['--glow-r' as never]: '14px',
              } as React.CSSProperties
            }
          >
            <DownloadIcon />
            .{ext}
          </button>
        )}

        <div className="transport__progress">
          <span className="transport__progress-l">PROGRESS</span>
          <span className="transport__progress-v num">{progress}%</span>
        </div>
      </div>
    </section>
  )
}
