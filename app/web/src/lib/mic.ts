/**
 * Microphone capture for the live transcription panel.
 *
 * Wraps `getUserMedia` + `MediaRecorder(audio/webm;codecs=opus)` in a small
 * React hook. The recorder emits a Blob every `timeslice` ms (≈ 1s), which
 * the caller forwards to the WS as a binary frame. ffmpeg in the backend
 * image decodes webm/opus to 16 kHz mono PCM.
 *
 * Also exposes an AnalyserNode-derived level (0..1) for a VU-style meter,
 * sampled via requestAnimationFrame inside the hook.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

export type MicState = 'idle' | 'starting' | 'recording' | 'stopping' | 'error'

/** Browser default for MediaRecorder(audio/webm;codecs=opus) — Opus's native rate. */
export const MIC_SAMPLE_RATE = 48000
export const MIC_MIME_TYPE = 'audio/webm;codecs=opus'
/** Container/codec hint passed to ffmpeg over the WS config first frame. */
export const MIC_FORMAT_HINT = 'webm'
/** How often MediaRecorder emits a Blob (ms). 1 s keeps latency low without choking ffmpeg. */
export const MIC_TIMESLICE_MS = 1000

export interface UseMicOptions {
  /** Called once per MediaRecorder timeslice. Forward the blob to the WS. */
  onChunk: (blob: Blob) => void
  /** Called after the final blob has been emitted following stop(). */
  onStop?: () => void
  /** Called on any unrecoverable error during start/record. */
  onError?: (err: Error) => void
}

export interface UseMicResult {
  state: MicState
  /** Last error message, cleared on successful start. */
  error: string | null
  /** Live mic level, 0..1. Updated each animation frame while recording. */
  level: number
  start: () => Promise<void>
  stop: () => void
}

/**
 * Owns the MediaRecorder + AudioContext lifecycle. Idempotent across React
 * StrictMode double-invocations — the actual stream is keyed by an internal
 * ref, not by state.
 */
export function useMic({ onChunk, onStop, onError }: UseMicOptions): UseMicResult {
  const [state, setState] = useState<MicState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [level, setLevel] = useState(0)

  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const rafRef = useRef<number | null>(null)

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
    setLevel(0)
  }, [])

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
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      })
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
      const tick = () => {
        if (!analyserRef.current) return
        // Cast: lib.dom.d.ts in TS6 narrowed this to ArrayBuffer-only views.
        analyserRef.current.getByteTimeDomainData(buf as unknown as Uint8Array<ArrayBuffer>)
        let peak = 0
        for (let i = 0; i < buf.length; i++) {
          const v = Math.abs(buf[i] - 128) / 128
          if (v > peak) peak = v
        }
        setLevel(peak)
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
  }, [state, onChunk, onStop, onError, cleanup])

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
