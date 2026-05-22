import { useCallback, useMemo } from 'react'
import { play, seek } from '../lib/playback'

interface Props {
  peaks: number[] | null
  currentTime: number
  duration: number
  height?: number
  barWidth?: number
  gap?: number
  /** Bar color when played (default: var(--accent)). */
  accent?: string
  /** Bar color when not yet played. */
  dim?: string
}

/**
 * Horizontal SVG bar waveform. Played portion uses `accent`; the rest is
 * `dim`. Click anywhere along the strip to seek the audio.
 */
export function Waveform({
  peaks,
  currentTime,
  duration,
  height = 90,
  barWidth = 2.6,
  gap = 1.4,
  accent = 'var(--accent)',
  dim = 'rgba(255,255,255,0.10)',
}: Props) {
  const data = useMemo(() => peaks ?? new Array(220).fill(0.18), [peaks])
  const N = data.length
  const totalW = N * (barWidth + gap)
  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0
  const cutoff = progress * N

  const onClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (duration <= 0) return
      const r = e.currentTarget.getBoundingClientRect()
      const ratio = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
      seek(ratio * duration)
      play()
    },
    [duration],
  )

  return (
    <svg
      viewBox={`0 0 ${totalW} ${height}`}
      width="100%"
      height={height}
      preserveAspectRatio="none"
      style={{ display: 'block', overflow: 'visible', cursor: duration > 0 ? 'pointer' : 'default' }}
      onClick={onClick}
    >
      {data.map((p, i) => {
        const h = Math.max(2, p * (height - 4))
        const y = (height - h) / 2
        const x = i * (barWidth + gap)
        const on = i < cutoff
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={barWidth}
            height={h}
            rx={barWidth / 2}
            fill={on ? accent : dim}
          />
        )
      })}
      {/* Playhead — only once playback has actually moved. At progress 0
          (live recording, or a loaded-but-unplayed file) a full-height
          line at x=0 reads like a planted peak bar at the very start, so
          we omit it; the played/unplayed bar coloring already conveys
          position. */}
      {progress > 0 && (
        <line
          x1={progress * totalW}
          x2={progress * totalW}
          y1={-4}
          y2={height + 4}
          stroke={accent}
          strokeWidth={1.5}
          strokeLinecap="round"
          opacity={0.9}
        />
      )}
    </svg>
  )
}
