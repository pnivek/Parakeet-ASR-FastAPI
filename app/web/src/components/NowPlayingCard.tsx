import { useEffect, useState } from 'react'
import { Waveform } from './Waveform'
import { StatusDot, type StreamState } from './StatusDot'
import { pause, play, useCurrentTime, useDuration, useIsPlaying } from '../lib/playback'
import { formatTime } from '../lib/format'
import type { Word } from '../lib/types'

interface Props {
  /** Display title for the audio (filename or URL). */
  title: string
  /** Subtitle/meta line (size · format). */
  meta?: string
  /** Words array from verbose_json result — drives the floating word pill. */
  words?: Word[]
  /** Pre-computed peaks (0..1). */
  peaks: number[] | null
  /** Current streaming state (used for the status pill). */
  state: StreamState
}

const PlayIcon = ({ size = 20 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor">
    <path d="M8 5.5v13a.5.5 0 0 0 .77.42l10-6.5a.5.5 0 0 0 0-.84l-10-6.5A.5.5 0 0 0 8 5.5z" />
  </svg>
)
const PauseIcon = ({ size = 20 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor">
    <rect x="6" y="5" width="4" height="14" rx="1" />
    <rect x="14" y="5" width="4" height="14" rx="1" />
  </svg>
)

export function NowPlayingCard({ title, meta, words, peaks, state }: Props) {
  const currentTime = useCurrentTime()
  const dur = useDuration()
  const isPlaying = useIsPlaying()
  const progress = dur > 0 ? Math.min(1, Math.max(0, currentTime / dur)) : 0

  const currentWord =
    words && words.length > 0
      ? words.find((w) => currentTime >= w.start && currentTime < w.end)?.word
      : undefined

  return (
    <section className="glass np">
      <div
        className="np__halo"
        style={{
          background: `radial-gradient(45% 70% at ${progress * 100}% 100%, color-mix(in oklch, var(--accent) 22%, transparent), transparent 70%)`,
        }}
      />
      <div className="np__hairline" />
      <div className="np__body">
        <div className="np__id">
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
              <span className="mono" style={{ fontSize: 10, color: 'var(--muted-deep)', letterSpacing: '0.8px' }}>
                NOW PLAYING
              </span>
              <StatusDot state={state} />
            </div>
            <div className="np__title">{title}</div>
            {meta && (
              <div className="np__sub">
                <span className="mono" style={{ letterSpacing: '0.3px' }}>{meta}</span>
              </div>
            )}
          </div>
          <SpectrumPulse active={isPlaying} />
        </div>

        <div style={{ position: 'relative', height: 22, marginBottom: -6 }}>
          {currentWord && (
            <div
              className="np__wordpill"
              style={{ left: `${Math.min(95, Math.max(5, progress * 100))}%`, opacity: isPlaying ? 1 : 0.6 }}
            >
              {currentWord}
            </div>
          )}
        </div>

        <div style={{ position: 'relative' }}>
          <Waveform peaks={peaks} currentTime={currentTime} duration={dur} height={110} />
          <div className="np__wf-times">
            <span>0:00</span>
            <span>{formatTime(dur * 0.25)}</span>
            <span>{formatTime(dur * 0.5)}</span>
            <span>{formatTime(dur * 0.75)}</span>
            <span>{formatTime(dur)}</span>
          </div>
        </div>

        <div className="np__transport">
          <button
            type="button"
            className="np__play pk-glow-btn"
            onClick={() => (isPlaying ? pause() : play())}
            style={{
              ['--btn-accent' as never]: 'var(--accent)',
              ['--top-hl' as never]: 'rgba(255,255,255,0.2)',
              ['--stroke-pct' as never]: '50%',
              ['--bottom-pct' as never]: '22%',
              ['--glow-r' as never]: '24px',
            } as React.CSSProperties}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <PauseIcon /> : <PlayIcon />}
          </button>
          <div className="np__pos">
            <span className="np__pos-l">POSITION</span>
            <span className="np__pos-v">
              {formatTime(currentTime)} <span className="np__pos-v--dim">/ {formatTime(dur)}</span>
            </span>
          </div>
        </div>
      </div>
    </section>
  )
}

function SpectrumPulse({ active }: { active: boolean }) {
  const [tick, setTick] = useState(0)
  useEffect(() => {
    if (!active) return
    const i = setInterval(() => setTick((t) => t + 1), 100)
    return () => clearInterval(i)
  }, [active])
  const N = 14
  return (
    <div className="spectrum">
      {Array.from({ length: N }).map((_, i) => {
        const phase = (tick + i * 1.4) * 0.5
        const env = active ? (Math.sin(phase) * 0.5 + 0.5) * 0.85 + 0.15 : 0.18
        const h = Math.max(3, env * 36)
        return (
          <div
            key={i}
            className={active ? 'spectrum__bar spectrum__bar--on' : 'spectrum__bar'}
            style={{
              height: h,
              boxShadow: active && h > 20 ? '0 0 6px -1px var(--accent)' : 'none',
              opacity: active ? 0.7 + env * 0.3 : 0.5,
            }}
          />
        )
      })}
    </div>
  )
}
