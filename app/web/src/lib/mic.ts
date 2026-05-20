/**
 * Microphone capture for the live transcription panel.
 *
 * Wraps `getUserMedia` + `MediaRecorder(audio/webm;codecs=opus)` in a small
 * React hook. The recorder emits a Blob every `timeslice` ms; the caller
 * forwards it to the WS as a binary frame. ffmpeg decodes webm/opus →
 * 16 kHz mono PCM in the backend.
 *
 * Server owns voice activity detection now (Silero VAD on the decoded
 * PCM); this module stays a dumb capture pipeline so the WebM container
 * stays intact end-to-end.
 *
 * Exposes an AnalyserNode-derived level (0..1) for the meter, sampled
 * via requestAnimationFrame.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

export type MicState = 'idle' | 'starting' | 'recording' | 'stopping' | 'error'

/** Browser default for MediaRecorder(audio/webm;codecs=opus) — Opus's native rate. */
export const MIC_SAMPLE_RATE = 48000
export const MIC_MIME_TYPE = 'audio/webm;codecs=opus'
/** Container/codec hint passed to ffmpeg over the WS config first frame. */
export const MIC_FORMAT_HINT = 'webm'
/**
 * How often MediaRecorder emits a Blob (ms). 250 ms gives the server data
 * to chew on much sooner — the first chunk leaves the browser within a
 * quarter-second of pressing Record instead of waiting a full second.
 */
export const MIC_TIMESLICE_MS = 250

export interface UseMicOptions {
  /** Called once per MediaRecorder timeslice with the recorded blob. */
  onChunk: (blob: Blob) => void
  /** Called after the final blob has been emitted following stop(). */
  onStop?: () => void
  /** Called on any unrecoverable error during start/record. */
  onError?: (err: Error) => void
  /** Forwarded to `getUserMedia({ audio: { noiseSuppression: ... } })`. */
  noiseSuppression?: boolean
  /** Called at ~10 Hz with the analyser's current peak (0..1) while
   * recording. Drives the hero waveform's live-growing bars without
   * needing a round-trip to the server. */
  onPeakSample?: (peak: number) => void
}

export interface UseMicResult {
  state: MicState
  /** Last error message, cleared on successful start. */
  error: string | null
  /** Live mic level, 0..1. Updated while recording, coalesced per rAF. */
  level: number
  start: () => Promise<void>
  stop: () => void
}

/**
 * Owns the MediaRecorder + AudioContext lifecycle. Idempotent across React
 * StrictMode double-invocations — the actual stream is keyed by an internal
 * ref, not by state.
 */
export function useMic({
  onChunk,
  onStop,
  onError,
  noiseSuppression = true,
  onPeakSample,
}: UseMicOptions): UseMicResult {
  const [state, setState] = useState<MicState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [level, setLevel] = useState(0)

  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const rafRef = useRef<number | null>(null)
  /** Last setLevel value — used to coalesce rAF updates that don't move the
   * peak meaningfully. Cuts the re-render rate of the sidebar by ~5×. */
  const lastLevelRef = useRef<number>(0)

  const cleanup = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect()
      analyserRef.current = null
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close().catch(() => {})
    }
    audioCtxRef.current = null
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) track.stop()
      streamRef.current = null
    }
    recorderRef.current = null
    lastLevelRef.current = 0
    setLevel(0)
  }, [])

  const start = useCallback(async () => {
    if (state === 'recording' || state === 'starting') return
    setError(null)
    setState('starting')
    try {
      // navigator.mediaDevices is only exposed in a secure context
      // (HTTPS or localhost). On plain HTTP over a LAN address the API is
      // undefined and there's no client-side workaround — explain it.
      if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
        const insecure = typeof window !== 'undefined' && !window.isSecureContext
        throw new Error(
          insecure
            ? `Microphone access requires HTTPS or localhost. This page is being served over plain HTTP (${window.location.host}). Open it via https:// or http://localhost:<port> to enable recording.`
            : 'Microphone API not available in this browser.',
        )
      }
      if (!window.MediaRecorder || !MediaRecorder.isTypeSupported(MIC_MIME_TYPE)) {
        throw new Error(
          `Browser does not support ${MIC_MIME_TYPE}. ` +
            `Try Chrome, Firefox, or another Chromium-based browser.`,
        )
      }
      let stream: MediaStream
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression,
          },
        })
      } catch (e) {
        const err = e as DOMException
        if (err?.name === 'NotAllowedError') {
          throw new Error('Microphone permission denied. Allow access in the browser site settings and retry.')
        }
        if (err?.name === 'NotFoundError') {
          throw new Error('No microphone detected. Plug one in (or check system input settings) and retry.')
        }
        throw new Error(`Microphone access failed: ${err?.message || String(e)}`)
      }
      streamRef.current = stream

      const recorder = new MediaRecorder(stream, { mimeType: MIC_MIME_TYPE })
      recorderRef.current = recorder

      recorder.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) onChunk(ev.data)
      }
      recorder.onstop = () => {
        onStop?.()
        cleanup()
        setState('idle')
      }
      recorder.onerror = (ev) => {
        const err = (ev as unknown as { error?: Error }).error ?? new Error('MediaRecorder error')
        setError(err.message)
        onError?.(err)
        cleanup()
        setState('error')
      }
      recorder.start(MIC_TIMESLICE_MS)

      // Audio analyser for the level meter. Separate from the recorder so the
      // recorder's encoder stays untouched.
      const ctx = new AudioContext()
      audioCtxRef.current = ctx
      const source = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 1024
      analyser.smoothingTimeConstant = 0.4
      source.connect(analyser)
      analyserRef.current = analyser

      const buf = new Uint8Array(analyser.fftSize)
      const LEVEL_THRESHOLD = 0.025 // coalesce sub-threshold rAF ticks
      // Waveform sampling cadence — one bar per ~100ms gives a
      // pleasantly smooth growing waveform (10 bars/s) without flooding
      // React with per-frame state updates. Tracked between rAF ticks
      // by holding the max peak we've seen since the last sample.
      const PEAK_SAMPLE_INTERVAL_MS = 100
      let lastPeakSampleMs = performance.now()
      let runningPeakSinceLastSample = 0
      const tick = () => {
        if (!analyserRef.current) return
        // Cast: lib.dom.d.ts in TS6 narrowed this to ArrayBuffer-only views.
        analyserRef.current.getByteTimeDomainData(buf as unknown as Uint8Array<ArrayBuffer>)
        let peak = 0
        for (let i = 0; i < buf.length; i++) {
          const v = Math.abs(buf[i] - 128) / 128
          if (v > peak) peak = v
        }
        if (peak > runningPeakSinceLastSample) runningPeakSinceLastSample = peak
        if (Math.abs(peak - lastLevelRef.current) >= LEVEL_THRESHOLD) {
          lastLevelRef.current = peak
          setLevel(peak)
        }
        const now = performance.now()
        if (now - lastPeakSampleMs >= PEAK_SAMPLE_INTERVAL_MS) {
          // Modest gain so typical speech (peak ≈ 0.2–0.4 in the
          // analyser's 0..1 range) lands as visible-but-not-clipped
          // bars. clamp(0..1) keeps the renderer's geometry honest.
          const sampled = Math.min(1, runningPeakSinceLastSample * 3)
          onPeakSample?.(sampled)
          runningPeakSinceLastSample = 0
          lastPeakSampleMs = now
        }
        rafRef.current = requestAnimationFrame(tick)
      }
      rafRef.current = requestAnimationFrame(tick)

      setState('recording')
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      onError?.(e instanceof Error ? e : new Error(msg))
      cleanup()
      setState('error')
    }
  }, [state, onChunk, onStop, onError, noiseSuppression, onPeakSample, cleanup])

  const stop = useCallback(() => {
    const rec = recorderRef.current
    if (!rec || rec.state === 'inactive') {
      cleanup()
      setState('idle')
      return
    }
    setState('stopping')
    try {
      rec.requestData()
      rec.stop()
    } catch {
      cleanup()
      setState('idle')
    }
  }, [cleanup])

  useEffect(
    () => () => {
      // Unmount safety net — stop the stream so the mic light goes off.
      cleanup()
    },
    [cleanup],
  )

  return { state, error, level, start, stop }
}
