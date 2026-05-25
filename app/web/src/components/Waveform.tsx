import { useCallback, useEffect, useRef } from 'react'
import { getAudioElement, play, seek } from '../lib/playback'

interface Props {
  peaks: number[] | null
  height?: number
  barWidth?: number
  gap?: number
  /** Bar color when played (default: var(--accent)). */
  accent?: string
  /** Bar color when not yet played. */
  dim?: string
}

/**
 * Horizontal bar waveform on a single canvas. Played portion uses
 * `accent`; the rest is `dim`. Click anywhere to seek the audio.
 *
 * **Self-driven via requestAnimationFrame** — the canvas draws every
 * frame from the latest peaks (via ref) + audio element's currentTime
 * + duration (read imperatively). This decouples the canvas update
 * from React's commit + browser layout cycle: even if the main thread
 * is briefly blocked by heavy DOM work elsewhere, the rAF loop still
 * fires per frame and keeps the waveform smooth.
 *
 * Skips redundant draws via hash comparison so 60Hz ticks with no
 * change are cheap (just a cache check, no canvas re-paint).
 */
export function Waveform({
  peaks,
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
  // Latest peaks ref — refreshed every render so the rAF loop reads
  // the current value without React re-subscribing.
  const peaksRef = useRef<number[] | null>(peaks)
  peaksRef.current = peaks

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

  // Imperative draw. Reads all dynamic inputs at call time so the rAF
  // caller doesn't need to thread them in.
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const w = widthRef.current || canvas.clientWidth || 0
    if (w <= 0) return
    const peaks = peaksRef.current
    const el = getAudioElement()
    const currentTime = el.currentTime
    const duration = isFinite(el.duration) && el.duration > 0 ? el.duration : 0
    const dpr = window.devicePixelRatio || 1
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
    const totalNatural = N * (barWidth + gap)
    const scale = totalNatural > 0 ? w / totalNatural : 1
    const sBw = barWidth * scale
    const sGap = gap * scale
    const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0
    const cutoff = progress * N
    const { accent: accentColor, dim: dimColor } = resolveColors()

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
        ctx.roundRect(x, y, sBw, h, radius)
      }
      ctx.fill()
    }

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
  }, [height, barWidth, gap, resolveColors])

  // Self-driven rAF loop. Runs continuously while the component is
  // mounted; skips actual canvas painting when nothing changed (hash
  // comparison). Cheap when idle (~one comparison per frame), runs in
  // the compositor-friendly rAF callback queue so it's never blocked
  // by React commit timing.
  useEffect(() => {
    let raf = 0
    let lastHash = ''
    const tick = () => {
      const canvas = canvasRef.current
      const el = getAudioElement()
      const peaks = peaksRef.current
      const w = canvas?.clientWidth ?? 0
      // Round to a coarse-enough hash that tiny float jitter doesn't
      // trigger a paint, but precise enough to look smooth.
      const hash = `${peaks?.length ?? 0}|${el.currentTime.toFixed(2)}|${el.duration.toFixed(0)}|${w}`
      if (hash !== lastHash) {
        draw()
        lastHash = hash
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [draw])

  // Observe the container for resize → trigger a fresh draw via the
  // width change. The rAF loop's hash includes width so it picks up
  // the change on the next tick.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const next = Math.round(entry.contentRect.width)
        if (next > 0 && next !== widthRef.current) {
          widthRef.current = next
        }
      }
    })
    ro.observe(canvas)
    widthRef.current = canvas.clientWidth
    return () => ro.disconnect()
  }, [])

  const onClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const el = getAudioElement()
    const duration = isFinite(el.duration) && el.duration > 0 ? el.duration : 0
    if (duration <= 0) return
    const r = e.currentTarget.getBoundingClientRect()
    const ratio = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
    seek(ratio * duration)
    play()
  }, [])

  return (
    <canvas
      ref={canvasRef}
      onClick={onClick}
      style={{
        display: 'block',
        width: '100%',
        height,
        cursor: 'pointer',
      }}
    />
  )
}
