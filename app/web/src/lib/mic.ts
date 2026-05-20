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
/**
 * How often MediaRecorder emits a Blob (ms). 250 ms gives the server data
 * to chew on much sooner — the first chunk leaves the browser within a
 * quarter-second of pressing Record instead of waiting a full second.
 * The server's chunk_queue handles the smaller-but-more-frequent frames
 * fine; ffmpeg sees the same bytestream either way.
 */
export const MIC_TIMESLICE_MS = 250

export interface UseMicOptions {
  /** Called once per MediaRecorder timeslice when audio is forwarded. */
  onChunk: (blob: Blob) => void
  /** Called after the final blob has been emitted following stop(). */
  onStop?: () => void
  /** Called on any unrecoverable error during start/record. */
  onError?: (err: Error) => void
  /**
   * Voice-activity detection. When true (default), silent chunks are
   * dropped before being forwarded to the WS — keeps the server's
   * inference queue drained so speech gets processed immediately
   * instead of behind silent windows.
   */
  vad?: boolean
  /** RMS threshold in [0..1] above which audio is considered speech. */
  vadThreshold?: number
  /** Once speech is detected, keep forwarding chunks for this many ms
   * after the last "loud" sample. Avoids clipping trailing words. */
  vadHangoverMs?: number
}

export interface UseMicResult {
  state: MicState
  /** Last error message, cleared on successful start. */
  error: string | null
  /** Live mic level, 0..1. Updated each animation frame while recording. */
  level: number
  /** True when VAD currently classifies the input as speech (or during
   * the hangover window). Drives the recording indicator. */
  voiceActive: boolean
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
  vad = true,
  vadThreshold = 0.04,
  vadHangoverMs = 800,
}: UseMicOptions): UseMicResult {
  const [state, setState] = useState<MicState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [level, setLevel] = useState(0)
  const [voiceActive, setVoiceActive] = useState(false)

  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const rafRef = useRef<number | null>(null)
  /** Last setLevel value — used to coalesce rAF updates that don't move the
   * peak meaningfully. Cuts the re-render rate of the sidebar by ~5×. */
  const lastLevelRef = useRef<number>(0)
  /** Last performance.now() at which we observed speech (RMS above
   * vadThreshold). Used for hangover-based VAD gating. */
  const lastVoiceTsRef = useRef<number>(0)
  /** Mirror of voiceActive in a ref so ondataavailable (a stale closure)
   * can read the latest value without re-binding the listener. */
  const voiceActiveRef = useRef<boolean>(false)
  /** Mirror of the latest options so the recorder listener picks up
   * runtime changes without being torn down. */
  const optsRef = useRef({ vad, vadThreshold, vadHangoverMs })
  optsRef.current = { vad, vadThreshold, vadHangoverMs }
  /** Chunks forwarded so far this session. The first N are always
   * forwarded regardless of VAD: MediaRecorder's WebM/Opus stream is
   * stateful — the very first blob carries the EBML init segment +
   * codec private data, without which ffmpeg can't decode ANY
   * subsequent blob. Dropping early "silent" chunks breaks the whole
   * container. */
  const forwardedCountRef = useRef<number>(0)
  const BOOTSTRAP_CHUNKS = 3

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
    lastVoiceTsRef.current = 0
    voiceActiveRef.current = false
    forwardedCountRef.current = 0
    setLevel(0)
    setVoiceActive(false)
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
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
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
        if (!ev.data || ev.data.size === 0) return
        const o = optsRef.current
        // ALWAYS forward the first BOOTSTRAP_CHUNKS regardless of VAD —
        // they carry the WebM EBML init / Opus codec private data that
        // ffmpeg needs to parse anything that follows. If we drop them,
        // every later chunk fails with "Invalid data" on the server.
        if (forwardedCountRef.current < BOOTSTRAP_CHUNKS) {
          forwardedCountRef.current++
          onChunk(ev.data)
          return
        }
        if (!o.vad) {
          onChunk(ev.data)
          return
        }
        const now = performance.now()
        const inHangover = now - lastVoiceTsRef.current <= o.vadHangoverMs
        if (inHangover) onChunk(ev.data)
        // else: silent chunk — drop it. The server engine doesn't know
        // about the gap; it'll see a contiguous (speech-only) stream.
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
      const tick = () => {
        if (!analyserRef.current) return
        // Cast: lib.dom.d.ts in TS6 narrowed this to ArrayBuffer-only views.
        analyserRef.current.getByteTimeDomainData(buf as unknown as Uint8Array<ArrayBuffer>)
        let peak = 0
        let sumSq = 0
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128
          const a = Math.abs(v)
          if (a > peak) peak = a
          sumSq += v * v
        }
        if (Math.abs(peak - lastLevelRef.current) >= LEVEL_THRESHOLD) {
          lastLevelRef.current = peak
          setLevel(peak)
        }
        // VAD: use RMS energy (less spiky than peak). If above threshold,
        // refresh the lastVoice timestamp. The ondataavailable handler
        // reads this + the hangover window to gate forwarding.
        const rms = Math.sqrt(sumSq / buf.length)
        const o = optsRef.current
        if (rms >= o.vadThreshold) {
          lastVoiceTsRef.current = performance.now()
          if (!voiceActiveRef.current) {
            voiceActiveRef.current = true
            setVoiceActive(true)
          }
        } else if (voiceActiveRef.current) {
          // Inside hangover → still "active"; outside → flip off.
          const inHang = performance.now() - lastVoiceTsRef.current <= o.vadHangoverMs
          if (!inHang) {
            voiceActiveRef.current = false
            setVoiceActive(false)
          }
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

  return { state, error, level, voiceActive, start, stop }
}
