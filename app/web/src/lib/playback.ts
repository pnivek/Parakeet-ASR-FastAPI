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
  audioEl = el
  return el
}

let currentBlobUrl: string | null = null

/** Replace the audio source. Null clears it. */
export function setAudioFile(file: File | Blob | null) {
  const el = ensureEl()
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl)
    currentBlobUrl = null
  }
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
 * streamable formats). */
export function setAudioUrl(url: string) {
  const el = ensureEl()
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl)
    currentBlobUrl = null
  }
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
  el.currentTime = Math.max(0, t)
}

export function play() {
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
