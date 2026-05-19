/**
 * Audio playback state for the transcription result.
 *
 * One <audio> element per page — kept as a module-level singleton so the
 * mounted element survives parent re-renders without re-loading the blob.
 * Exposes hooks for current time, duration, and play/pause state, all backed
 * by useSyncExternalStore so only components that subscribe re-render.
 */
import { useEffect, useSyncExternalStore } from 'react'

let audioEl: HTMLAudioElement | null = null
let currentTime = 0
let duration = 0
let isPlaying = false
const listeners = new Set<() => void>()
const notify = () => listeners.forEach((l) => l())

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

/**
 * Replace the audio source. Frees the previous blob URL, sets a new one
 * from the given File. No-op if `file` is null.
 */
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
    isPlaying = false
    notify()
    return
  }
  currentBlobUrl = URL.createObjectURL(file)
  el.src = currentBlobUrl
  el.load()
  currentTime = 0
  duration = 0
  isPlaying = false
  notify()
}

export function seek(t: number) {
  const el = ensureEl()
  el.currentTime = Math.max(0, t)
}

export function play() {
  const el = ensureEl()
  el.play().catch(() => {})
}

export function pause() {
  const el = ensureEl()
  el.pause()
}

export function togglePlay() {
  isPlaying ? pause() : play()
}

export function getAudioEl(): HTMLAudioElement {
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
    () => duration,
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

/**
 * Mount the singleton audio element into the given container. The element is
 * `display: none` — its visual stand-in is the waveform + transport in the
 * Now Playing card. We keep it mounted so playback survives parent
 * re-renders.
 */
export function useAudioContainer(ref: React.RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    if (!ref.current) return
    const el = ensureEl()
    if (el.parentNode !== ref.current) {
      ref.current.appendChild(el)
    }
  }, [ref])
}
