/**
 * Audio playback singleton. One <audio> element per page, hidden via
 * `display: none` since the Maison hero row owns the visual transport.
 * Hooks back into React via `useSyncExternalStore` so only subscribed
 * components re-render on timeupdate / play / pause / duration.
 */
import { useEffect, useSyncExternalStore } from 'react'

let audioEl: HTMLAudioElement | null = null
let currentTime = 0
let duration = 0
/** Externally-supplied duration (e.g. parsed from a WAV header) used
 * as a fallback when the <audio> element hasn't reported `duration`
 * yet — keeps the hero's duration readout populated for very long
 * uploads where metadata loads slowly. */
let durationHint = 0
let isPlaying = false
const listeners = new Set<() => void>()
const notify = () => listeners.forEach((l) => l())
const effectiveDuration = () => (duration > 0 ? duration : durationHint)

/** Set a duration fallback. Cleared on every setAudioFile. */
export function setDurationHint(seconds: number) {
  durationHint = isFinite(seconds) && seconds > 0 ? seconds : 0
  notify()
}

function ensureEl(): HTMLAudioElement {
  if (audioEl) return audioEl
  const el = document.createElement('audio')
  el.preload = 'auto'
  el.style.display = 'none'
  el.addEventListener('timeupdate', () => {
    currentTime = el.currentTime
    notify()
  })
  el.addEventListener('seeked', () => {
    currentTime = el.currentTime
    notify()
  })
  el.addEventListener('loadedmetadata', () => {
    duration = isFinite(el.duration) ? el.duration : 0
    notify()
  })
  el.addEventListener('durationchange', () => {
    duration = isFinite(el.duration) ? el.duration : 0
    notify()
  })
  el.addEventListener('play', () => {
    isPlaying = true
    notify()
  })
  el.addEventListener('pause', () => {
    isPlaying = false
    notify()
  })
  el.addEventListener('ended', () => {
    isPlaying = false
    notify()
  })
  el.addEventListener('error', () => {
    // CORS rejected our crossOrigin='anonymous' request — retry without it.
    // Cost: playback works but the AnalyserNode samples come back as 128 (tainted),
    // so the waveform won't fill. Better than no playback.
    if (pendingCorsRetryUrl && el.crossOrigin) {
      const retry = pendingCorsRetryUrl
      pendingCorsRetryUrl = null
      el.removeAttribute('crossorigin')
      el.src = retry
      el.load()
    }
  })
  el.addEventListener('loadedmetadata', () => {
    // Successful load — clear the retry marker so a later error event
    // doesn't accidentally re-trigger.
    pendingCorsRetryUrl = null
  })
  audioEl = el
  return el
}

// ── Playback analyser (rolling waveform for URL sources) ──────────
// Lazy-built on first request. Tap point: MediaElementAudioSourceNode on
// the singleton <audio> → AnalyserNode → AudioContext.destination so the
// audio still plays. Only one MediaElementSource is permitted per element,
// so we keep this setup global + idempotent.
let audioCtx: AudioContext | null = null
let mediaSourceNode: MediaElementAudioSourceNode | null = null
let analyserNode: AnalyserNode | null = null

/** Returns the playback analyser, creating it on first call.
 *
 * Once this runs, the audio element's output is routed through the
 * AudioContext. If the context is suspended (no user gesture yet), audio
 * cuts out — so we resume it best-effort. Calls from inside a click /
 * play handler are safe.
 *
 * Returns null if WebAudio setup fails entirely (rare). Cross-origin
 * tainting doesn't fail — the analyser just returns 128 for every byte. */
export function getPlaybackAnalyser(): AnalyserNode | null {
  if (analyserNode) return analyserNode
  const el = ensureEl()
  try {
    if (!audioCtx) {
      type WindowWithWebkit = Window & { webkitAudioContext?: typeof AudioContext }
      const w = window as WindowWithWebkit
      const Ctor = window.AudioContext ?? w.webkitAudioContext
      if (!Ctor) return null
      audioCtx = new Ctor()
    }
    if (!mediaSourceNode) {
      mediaSourceNode = audioCtx.createMediaElementSource(el)
    }
    const a = audioCtx.createAnalyser()
    a.fftSize = 1024
    a.smoothingTimeConstant = 0.3
    mediaSourceNode.connect(a)
    a.connect(audioCtx.destination)
    analyserNode = a
    if (audioCtx.state === 'suspended') {
      void audioCtx.resume()
    }
    return a
  } catch (e) {
    console.warn('playback analyser setup failed', e)
    return null
  }
}

let currentBlobUrl: string | null = null
/** URL we should retry without crossOrigin if the CORS-anonymous load fails.
 * Cleared on successful load (loadedmetadata) or on the next `setAudioFile`. */
let pendingCorsRetryUrl: string | null = null

/** Replace the audio source. Null clears it. */
export function setAudioFile(file: File | Blob | null) {
  const el = ensureEl()
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl)
    currentBlobUrl = null
  }
  // Local blob URLs don't need crossOrigin; clearing keeps the load from
  // being treated as a CORS request.
  el.removeAttribute('crossorigin')
  pendingCorsRetryUrl = null
  if (!file) {
    el.removeAttribute('src')
    el.load()
    currentTime = 0
    duration = 0
    durationHint = 0
    isPlaying = false
    notify()
    return
  }
  currentBlobUrl = URL.createObjectURL(file)
  el.src = currentBlobUrl
  el.load()
  currentTime = 0
  duration = 0
  durationHint = 0
  isPlaying = false
  notify()
}

/** Point the audio element at a remote URL — used for URL-ingested
 * sources where we don't hold the bytes locally. Browser handles
 * playback natively (Range requests for finite files, live decode for
 * streamable formats).
 *
 * `crossOrigin='anonymous'` is set so the playback AnalyserNode can read
 * samples for the rolling-waveform visual. Sources without CORS headers
 * will fail to load — at that point we fall back to no-crossOrigin
 * (audio plays, no waveform) via the `error` handler below. */
export function setAudioUrl(url: string) {
  const el = ensureEl()
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl)
    currentBlobUrl = null
  }
  // Try with crossOrigin first; auto-retry without if the source rejects CORS.
  el.crossOrigin = 'anonymous'
  pendingCorsRetryUrl = url
  el.src = url
  el.load()
  currentTime = 0
  duration = 0
  durationHint = 0
  isPlaying = false
  notify()
}

export function seek(t: number) {
  const el = ensureEl()
  const target = Math.max(0, t)
  // For URL+WS live streams, the audio element's `seekable` range is
  // tiny (HLS playlist window or icecast buffer). Clicking on a word
  // from hours ago would be way outside that range — the browser
  // tries to fulfill the seek, stalls indefinitely waiting for
  // unreachable data, and the audio element enters a broken state
  // that can take minutes to recover from (if at all). Silently
  // refuse seeks to positions we know are unreachable. For files
  // and recorded audio, seekable spans the full duration so this
  // never trips.
  //
  // Defensive on empty seekable (just after src change, before any
  // data has loaded) — allow the seek and let the browser handle
  // it normally. Otherwise newly-loaded sources couldn't be seeked.
  const sk = el.seekable
  if (sk.length === 0) {
    el.currentTime = target
    return
  }
  for (let i = 0; i < sk.length; i++) {
    if (target >= sk.start(i) && target <= sk.end(i)) {
      el.currentTime = target
      return
    }
  }
  // Out of all seekable ranges — no-op. The audio continues from
  // wherever it currently is.
}

export function play() {
  // Resume any suspended context (no-op if not built or already running).
  // Required because once `getPlaybackAnalyser` runs, audio routing goes
  // through the AudioContext — a suspended context silences playback.
  if (audioCtx && audioCtx.state === 'suspended') {
    void audioCtx.resume()
  }
  ensureEl()
    .play()
    .catch(() => {})
}

export function pause() {
  ensureEl().pause()
}

export function resetPlayback() {
  const el = ensureEl()
  el.pause()
  el.currentTime = 0
}

/** Direct access to the singleton <audio> for callers that need to read
 * its seekable range or duration semantics that don't round-trip well
 * through the hook layer (e.g. the Live button needs `audio.seekable`
 * to land at the head of an HLS / icecast stream). */
export function getAudioElement(): HTMLAudioElement {
  return ensureEl()
}

function subscribe(cb: () => void) {
  listeners.add(cb)
  return () => {
    listeners.delete(cb)
  }
}

export function useCurrentTime(): number {
  return useSyncExternalStore(
    subscribe,
    () => currentTime,
    () => 0,
  )
}
export function useDuration(): number {
  return useSyncExternalStore(
    subscribe,
    effectiveDuration,
    () => 0,
  )
}
export function useIsPlaying(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => isPlaying,
    () => false,
  )
}

/** Mount the singleton into a hidden container. */
export function useAudioContainer(ref: React.RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    if (!ref.current) return
    const el = ensureEl()
    if (el.parentNode !== ref.current) ref.current.appendChild(el)
  }, [ref])
}
