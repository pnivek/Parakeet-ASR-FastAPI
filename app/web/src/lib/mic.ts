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

export type MicState =
  | 'idle'
  | 'listening' // preview: mic open + level meter live, but NOT recording
  | 'starting'
  | 'recording'
  | 'stopping'
  | 'error'

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
  /** Live mic level, 0..1. Updated while listening or recording, coalesced per rAF. */
  level: number
  start: () => Promise<void>
  stop: () => void
  /** Open the mic + level meter WITHOUT recording — used as a live
   * preview in mic+live mode so the meter reflects real speech before
   * the user commits to transcribing. No-op if already
   * listening/recording. */
  listen: () => Promise<void>
  /** Tear down a preview-only listen (does nothing while recording). */
  stopListening: () => void
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
  /** Whether the analyser tick should emit waveform peak samples. True
   * only while recording — a preview listen drives the level meter but
   * shouldn't populate the hero waveform. Mutable so the single tick
   * loop survives a listen→record upgrade without restarting. */
  const emitPeaksRef = useRef<boolean>(false)

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
    emitPeaksRef.current = false
    setLevel(0)
  }, [])

  /** getUserMedia helper with friendly error messages. */
  const acquireStream = useCallback(async (): Promise<MediaStream> => {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      const insecure = typeof window !== 'undefined' && !window.isSecureContext
      throw new Error(
        insecure
          ? `Microphone access requires HTTPS or localhost. This page is being served over plain HTTP (${window.location.host}). Open it via https:// or http://localhost:<port> to enable recording.`
          : 'Microphone API not available in this browser.',
      )
    }
    try {
      return await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression },
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
  }, [noiseSuppression])

  /** Attach the AnalyserNode + rAF level loop to a stream. Idempotent —
   * skips if an analyser is already running (e.g. a preview listen that
   * a recording start is upgrading). */
  const attachAnalyser = useCallback(
    (stream: MediaStream) => {
      if (analyserRef.current) return
      const ctx = new AudioContext()
      audioCtxRef.current = ctx
      const source = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 1024
      analyser.smoothingTimeConstant = 0.4
      source.connect(analyser)
      analyserRef.current = analyser

      const buf = new Uint8Array(analyser.fftSize)
      const LEVEL_THRESHOLD = 0.012
      const PEAK_SAMPLE_INTERVAL_MS = 100
      let lastPeakSampleMs = performance.now()
      let runningPeakSinceLastSample = 0
      const tick = () => {
        if (!analyserRef.current) return
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
          // Only feed the hero waveform while actually recording.
          if (emitPeaksRef.current) {
            onPeakSample?.(Math.min(1, runningPeakSinceLastSample * 3))
          }
          runningPeakSinceLastSample = 0
          lastPeakSampleMs = now
        }
        rafRef.current = requestAnimationFrame(tick)
      }
      rafRef.current = requestAnimationFrame(tick)
    },
    [onPeakSample],
  )

  const listen = useCallback(async () => {
    // Already capturing (preview or recording) — nothing to do.
    if (streamRef.current) return
    setError(null)
    try {
      const stream = await acquireStream()
      streamRef.current = stream
      emitPeaksRef.current = false
      attachAnalyser(stream)
      setState('listening')
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      onError?.(e instanceof Error ? e : new Error(msg))
      cleanup()
      setState('error')
    }
  }, [acquireStream, attachAnalyser, onError, cleanup])

  const stopListening = useCallback(() => {
    // Only tear down a preview listen; never interrupt a recording.
    if (recorderRef.current) return
    cleanup()
    setState('idle')
  }, [cleanup])

  const start = useCallback(async () => {
    if (state === 'recording' || state === 'starting') return
    setError(null)
    setState('starting')
    try {
      if (!window.MediaRecorder || !MediaRecorder.isTypeSupported(MIC_MIME_TYPE)) {
        throw new Error(
          `Browser does not support ${MIC_MIME_TYPE}. ` +
            `Try Chrome, Firefox, or another Chromium-based browser.`,
        )
      }
      // Reuse the preview stream if we're already listening; otherwise
      // acquire one now. The analyser may already be running (preview)
      // — attachAnalyser is idempotent, and flipping emitPeaksRef makes
      // the existing tick start feeding the hero waveform.
      const stream = streamRef.current ?? (await acquireStream())
      streamRef.current = stream
      emitPeaksRef.current = true
      attachAnalyser(stream)

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
      setState('recording')
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      onError?.(e instanceof Error ? e : new Error(msg))
      cleanup()
      setState('error')
    }
  }, [state, onChunk, onStop, onError, acquireStream, attachAnalyser, cleanup])

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

  return { state, error, level, start, stop, listen, stopListening }
}
