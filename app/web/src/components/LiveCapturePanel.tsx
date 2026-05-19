import { useCallback, useEffect, useRef, useState } from 'react'
import { connectLiveWS, type LiveWSHandle, type TranscriptionResponse } from '../lib/api'
import { MIC_FORMAT_HINT, MIC_SAMPLE_RATE, MIC_MIME_TYPE, useMic } from '../lib/mic'
import { useSettings } from '../lib/settings'
import type { WhisperSegment, WSMessage } from '../lib/types'
import { formatTime } from '../lib/format'

interface Props {
  onResult: (audio: File, result: TranscriptionResponse) => void
  onPartial: (result: TranscriptionResponse) => void
  onError: (message: string) => void
  onBusyChange?: (busy: boolean) => void
}

const MicIcon = ({ size = 28 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="9" y="3" width="6" height="12" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
  </svg>
)
const BoltIcon = () => (
  <svg viewBox="0 0 24 24" width={12} height={12} fill="currentColor" aria-hidden>
    <path d="M13 2L3 14h7l-1 8 10-12h-7l1-8z" />
  </svg>
)

const LEVEL_BARS = 28

export function LiveCapturePanel({ onResult, onPartial, onError, onBusyChange }: Props) {
  const settings = useSettings()
  const wsRef = useRef<LiveWSHandle | null>(null)
  const segmentsRef = useRef<WhisperSegment[]>([])
  const chunksRef = useRef<Blob[]>([])
  const recordStartRef = useRef<number>(0)
  const [segmentCount, setSegmentCount] = useState(0)
  const [elapsed, setElapsed] = useState(0)

  const buildPartialResult = useCallback((): TranscriptionResponse => {
    const segs = segmentsRef.current
    const lastEnd = segs.length > 0 ? segs[segs.length - 1].end : 0
    const wall = (Date.now() - recordStartRef.current) / 1000
    return {
      format: 'verbose_json',
      body: {
        task: 'transcribe',
        language: 'en',
        duration: lastEnd,
        text: segs.map((s) => s.text).join(' ').trim(),
        segments: segs,
        strategy: 'progressive',
        transcription_time_seconds: wall,
      },
    }
  }, [])

  const handleMessage = useCallback(
    (msg: WSMessage) => {
      switch (msg.type) {
        case 'segments_batch':
          segmentsRef.current = [...segmentsRef.current, ...msg.segments]
          setSegmentCount(segmentsRef.current.length)
          onPartial(buildPartialResult())
          break
        case 'refined_transcription':
          segmentsRef.current = msg.segments
          setSegmentCount(msg.segments.length)
          onPartial({
            format: 'verbose_json',
            body: {
              task: 'transcribe',
              language: 'en',
              duration: msg.audio_duration_seconds,
              text: msg.text,
              segments: msg.segments,
              strategy: 'progressive',
              transcription_time_seconds: msg.transcription_time,
            },
          })
          break
        case 'final_transcription': {
          const blob = new Blob(chunksRef.current, { type: MIC_MIME_TYPE })
          const file = new File([blob], `mic-${Date.now()}.webm`, { type: MIC_MIME_TYPE })
          onResult(file, {
            format: 'verbose_json',
            body: {
              task: 'transcribe',
              language: msg.language,
              duration: msg.duration,
              text: msg.text,
              segments: msg.segments,
              strategy: msg.strategy,
              transcription_time_seconds: msg.transcription_time,
              csv_content: msg.csv_content,
              srt_content: msg.srt_content,
            },
          })
          break
        }
        case 'error':
          onError(msg.error)
          break
      }
    },
    [buildPartialResult, onPartial, onResult, onError],
  )

  const mic = useMic({
    onChunk: (blob) => {
      chunksRef.current.push(blob)
      wsRef.current?.sendBinary(blob)
    },
    onStop: () => {
      wsRef.current?.finish()
    },
    onError: (err) => {
      onError(err.message)
      wsRef.current?.abort()
      wsRef.current = null
    },
  })

  const recording = mic.state === 'recording'
  const busy = mic.state === 'starting' || mic.state === 'stopping'

  useEffect(() => {
    onBusyChange?.(recording || busy)
  }, [recording, busy, onBusyChange])

  useEffect(() => {
    if (!recording) return
    const t0 = recordStartRef.current
    const i = setInterval(() => setElapsed((Date.now() - t0) / 1000), 100)
    return () => clearInterval(i)
  }, [recording])

  const start = useCallback(async () => {
    segmentsRef.current = []
    chunksRef.current = []
    setSegmentCount(0)
    setElapsed(0)
    recordStartRef.current = Date.now()

    const ws = connectLiveWS(
      {
        sample_rate: MIC_SAMPLE_RATE,
        channels: 1,
        bytes_per_sample: 2,
        format: MIC_FORMAT_HINT,
        strategy: settings.strategy === 'auto' ? 'progressive' : settings.strategy,
        chunk_length: settings.chunkLength ?? undefined,
        chunk_overlap: settings.chunkOverlap ?? undefined,
        batch_size: settings.batchSize ?? undefined,
        long_audio_threshold: settings.longAudioThreshold ?? undefined,
        live_latency: settings.liveLatency,
        progressive_refinement: settings.progressiveRefinement,
      },
      {
        onMessage: handleMessage,
        onError: () => onError('WebSocket error — connection failed.'),
        onClose: (code, reason) => {
          if (code !== 1000 && code !== 1005) {
            onError(`WebSocket closed: code=${code} reason=${reason || 'no reason'}`)
          }
          wsRef.current = null
        },
      },
    )
    wsRef.current = ws
    await mic.start()
  }, [settings, handleMessage, mic, onError])

  const stop = useCallback(() => {
    mic.stop()
  }, [mic])

  // Stable per-mount jitter for the level bars so they don't look like a
  // uniform sine wave on quiet input.
  const jitterRef = useRef<number[]>([])
  if (jitterRef.current.length !== LEVEL_BARS) {
    jitterRef.current = Array.from({ length: LEVEL_BARS }, () => Math.random() * 0.5 + 0.5)
  }
  const lvl = mic.level

  return (
    <section className="glass input-card" style={{ padding: '20px 18px 18px' }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, padding: '8px 0 4px' }}>
        <button
          type="button"
          className={recording ? 'mic-button mic-button--recording pk-glow-btn' : 'mic-button pk-glow-btn'}
          onClick={() => (recording ? stop() : start())}
          disabled={busy}
          aria-label={recording ? 'Stop recording' : 'Start recording'}
          style={
            {
              ['--btn-accent' as never]: 'var(--accent)',
              ['--top-hl' as never]: recording ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.2)',
              ['--stroke-pct' as never]: recording ? '55%' : '50%',
              ['--bottom-pct' as never]: recording ? '0%' : '22%',
              ['--glow-r' as never]: '24px',
              ['--glow-pct' as never]: recording ? '55%' : '10%',
            } as React.CSSProperties
          }
        >
          {recording ? <span className="mic-button__square" /> : <MicIcon />}
          {recording && <span className="mic-button__halo" aria-hidden />}
        </button>
        <div style={{ textAlign: 'center' }}>
          <div className="mic-status">
            {mic.state === 'idle' && 'READY'}
            {mic.state === 'starting' && 'CONNECTING'}
            {mic.state === 'recording' && 'RECORDING'}
            {mic.state === 'stopping' && 'FINALIZING'}
            {mic.state === 'error' && 'ERROR'}
          </div>
          <div className="mic-timer" style={{ color: recording ? 'var(--accent)' : 'var(--fg-dim)' }}>
            {formatTime(elapsed)}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <span className="mono" style={{ fontSize: 9.5, color: 'var(--muted-deep)', letterSpacing: 0.6 }}>
            INPUT LEVEL
          </span>
          <span className="mono" style={{ fontSize: 9.5, color: 'var(--muted)', letterSpacing: 0.3 }}>
            {recording ? `${Math.round(lvl * 100)}%` : 'idle'}
          </span>
        </div>
        <div className="mic-meter">
          {jitterRef.current.map((jit, i) => {
            const k = i / (LEVEL_BARS - 1)
            // Bars from left to right grow with the level; jitter prevents a perfect ramp.
            const h = recording ? Math.max(8, Math.min(100, lvl * 100 * jit * (k * 0.6 + 0.7))) : 18
            const hot = recording && lvl * jit > 0.7
            const cls = !recording
              ? 'mic-meter__bar'
              : hot
                ? 'mic-meter__bar mic-meter__bar--hot'
                : 'mic-meter__bar mic-meter__bar--on'
            return <div key={i} className={cls} style={{ height: `${h}%` }} />
          })}
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginTop: 14,
          padding: '8px 10px',
          borderRadius: 8,
          background: 'rgba(255,255,255,0.022)',
          border: '1px solid var(--border-soft)',
        }}
      >
        <BoltIcon />
        <span style={{ fontSize: 11.5, color: 'var(--fg-dim)' }}>Segments captured</span>
        <span
          className="mono"
          style={{
            marginLeft: 'auto',
            fontSize: 12,
            fontWeight: 500,
            color: recording ? 'var(--accent)' : 'var(--muted)',
          }}
        >
          {segmentCount}
        </span>
      </div>
      {mic.state === 'error' && mic.error && (
        <div className="error" style={{ marginTop: 12 }}>
          {mic.error}
        </div>
      )}
    </section>
  )
}
