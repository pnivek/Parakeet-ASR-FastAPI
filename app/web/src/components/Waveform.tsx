import { useCallback, useEffect, useRef } from 'react'
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
 * Horizontal bar waveform rendered to a single canvas. Played portion
 * uses `accent`; the rest is `dim`. Click anywhere along the strip to
 * seek the audio.
 *
 * Canvas-based on purpose: the SVG version reconciled 220 <rect> nodes
 * per `currentTime` tick (~60 Hz) because each bar's fill depended on
 * the cutoff position. Canvas turns that into a single imperative paint
 * per change — no React reconcile, no per-bar prop diff.
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
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  // CSS-pixel width of the canvas — measured from the parent so the
  // canvas fills its container and stays sharp on Retina (DPR scaling
  // is handled inside draw()).
  const widthRef = useRef<number>(0)

  // Resolve the CSS color tokens (`var(--accent)`) into actual color
  // strings exactly once per mount. Canvas 2D doesn't accept CSS
  // variables, so we read them off a probe element. The values are
  // cached because the accent color doesn't change at runtime.
  const colorsRef = useRef<{ accent: string; dim: string } | null>(null)
  const resolveColors = useCallback(() => {
    if (colorsRef.current) return colorsRef.current
    const probe = document.createElement('span')
    probe.style.color = accent
    document.body.appendChild(probe)
    const accentResolved = getComputedStyle(probe).color
    probe.style.color = dim
    const dimResolved = getComputedStyle(probe).color
    document.body.removeChild(probe)
    colorsRef.current = { accent: accentResolved, dim: dimResolved }
    return colorsRef.current
  }, [accent, dim])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const w = widthRef.current || canvas.clientWidth || 0
    if (w <= 0) return
    const dpr = window.devicePixelRatio || 1
    // Keep the backing store at device-pixel resolution.
    const targetW = Math.round(w * dpr)
    const targetH = Math.round(height * dpr)
    if (canvas.width !== targetW) canvas.width = targetW
    if (canvas.height !== targetH) canvas.height = targetH
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.save()
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w, height)

    const data = peaks ?? new Array(220).fill(0.18)
    const N = data.length
    // Fit the bars to the actual canvas width — keep barWidth/gap ratio
    // but scale so the strip ends flush with the right edge.
    const totalNatural = N * (barWidth + gap)
    const scale = totalNatural > 0 ? w / totalNatural : 1
    const sBw = barWidth * scale
    const sGap = gap * scale
    const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0
    const cutoff = progress * N
    const { accent: accentColor, dim: dimColor } = resolveColors()

    // Two passes: dim bars first, then accent. Avoids per-bar fillStyle
    // churn (more cache-friendly on the GPU compositor).
    const radius = sBw / 2
    for (let pass = 0; pass < 2; pass++) {
      ctx.fillStyle = pass === 0 ? dimColor : accentColor
      ctx.beginPath()
      for (let i = 0; i < N; i++) {
        const on = i < cutoff
        if (pass === 0 ? on : !on) continue
        const h = Math.max(2, data[i] * (height - 4))
        const y = (height - h) / 2
        const x = i * (sBw + sGap)
        // roundRect is widely supported in modern browsers.
        ctx.roundRect(x, y, sBw, h, radius)
      }
      ctx.fill()
    }

    // Playhead — a thin accent line at the progress position.
    const px = progress * w
    ctx.strokeStyle = accentColor
    ctx.lineWidth = 1.5
    ctx.lineCap = 'round'
    ctx.globalAlpha = 0.9
    ctx.beginPath()
    ctx.moveTo(px, -4)
    ctx.lineTo(px, height + 4)
    ctx.stroke()
    ctx.globalAlpha = 1

    ctx.restore()
  }, [peaks, currentTime, duration, height, barWidth, gap, resolveColors])

  // Repaint on any prop change.
  useEffect(() => {
    draw()
  }, [draw])

  // Observe the container so the canvas re-fits on window/layout
  // resizes (e.g. sidebar toggling, viewport changes).
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const next = Math.round(entry.contentRect.width)
        if (next > 0 && next !== widthRef.current) {
          widthRef.current = next
          draw()
        }
      }
    })
    ro.observe(canvas)
    // Seed with the current width.
    widthRef.current = canvas.clientWidth
    draw()
    return () => ro.disconnect()
  }, [draw])

  const onClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (duration <= 0) return
      const r = e.currentTarget.getBoundingClientRect()
      const ratio = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
      seek(ratio * duration)
      play()
    },
    [duration],
  )

  return (
    <canvas
      ref={canvasRef}
      onClick={onClick}
      style={{
        display: 'block',
        width: '100%',
        height,
        cursor: duration > 0 ? 'pointer' : 'default',
      }}
    />
  )
}
