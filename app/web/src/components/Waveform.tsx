import { useCallback, useMemo } from 'react'
import { seek } from '../lib/playback'

interface Props {
  peaks: number[] | null
  currentTime: number
  duration: number
  height?: number
  barWidth?: number
  gap?: number
  /** Color override; defaults to `var(--accent)`. */
  accent?: string
  /** Color for un-played bars; defaults to a dim surface tone. */
  dim?: string
  clickable?: boolean
}

/**
 * SVG bar waveform. Bars before the playhead use `accent`; bars after use
 * `dim`. Click anywhere along the strip to seek the audio.
 */
export function Waveform({
  peaks,
  currentTime,
  duration,
  height = 110,
  barWidth = 2.2,
  gap = 1.6,
  accent = 'var(--accent)',
  dim = 'oklch(0.34 0.025 250)',
  clickable = true,
}: Props) {
  const data = useMemo(() => peaks ?? new Array(200).fill(0.15), [peaks])
  const N = data.length
  const totalW = N * (barWidth + gap)
  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0
  const cutoff = progress * N

  const onClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!clickable || duration <= 0) return
      const r = e.currentTarget.getBoundingClientRect()
      const ratio = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
      seek(ratio * duration)
    },
    [clickable, duration],
  )

  return (
    <svg
      viewBox={`0 0 ${totalW} ${height}`}
      width="100%"
      height={height}
      preserveAspectRatio="none"
      style={{ display: 'block', overflow: 'visible', cursor: clickable ? 'pointer' : 'default' }}
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
    </svg>
  )
}
