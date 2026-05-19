/**
 * Audio playback state for the transcription result.
 *
 * Exposes a single `useAudio()` hook returning the shared <audio> element,
 * the current playback time (via `useSyncExternalStore` so only components
 * that subscribe re-render on `timeupdate`), and a `seek(t)` helper.
 *
 * One audio instance per page — kept as a module-level singleton so the
 * mounted <audio> survives parent re-renders without re-loading the blob.
 */
import { useEffect, useSyncExternalStore } from 'react'

let audioEl: HTMLAudioElement | null = null
let currentTime = 0
const listeners = new Set<() => void>()

function ensureEl(): HTMLAudioElement {
  if (audioEl) return audioEl
  const el = document.createElement('audio')
  el.controls = true
  el.preload = 'auto'
  el.style.width = '100%'
  el.addEventListener('timeupdate', () => {
    currentTime = el.currentTime
    for (const l of listeners) l()
  })
  el.addEventListener('seeked', () => {
    currentTime = el.currentTime
    for (const l of listeners) l()
  })
  audioEl = el
  return el
}

let currentBlobUrl: string | null = null

/**
 * Replace the audio source. Frees the previous blob URL, sets a new one
 * from the given File. No-op if `file` is null.
 */
export function setAudioFile(file: File | null) {
  const el = ensureEl()
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl)
    currentBlobUrl = null
  }
  if (!file) {
    el.removeAttribute('src')
    el.load()
    currentTime = 0
    for (const l of listeners) l()
    return
  }
  currentBlobUrl = URL.createObjectURL(file)
  el.src = currentBlobUrl
  el.load()
  currentTime = 0
  for (const l of listeners) l()
}

export function seek(t: number) {
  const el = ensureEl()
  el.currentTime = Math.max(0, t)
  // Don't autoplay on seek — the user clicks segments to inspect, not to play.
}

export function getAudioEl(): HTMLAudioElement {
  return ensureEl()
}

/**
 * Subscribe to playback time. Backed by useSyncExternalStore so components
 * tracking the current time only re-render when it changes (≈ 4× per
 * second from the browser's `timeupdate` cadence).
 */
export function useCurrentTime(): number {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb)
      return () => {
        listeners.delete(cb)
      }
    },
    () => currentTime,
    () => 0,
  )
}

/**
 * Mount the singleton audio element into the given container. Use this in
 * the component that should own the player visually (we render it inside
 * the right pane above the transcription).
 */
export function useAudioContainer(ref: React.RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    if (!ref.current) return
    const el = ensureEl()
    if (el.parentNode !== ref.current) {
      ref.current.appendChild(el)
    }
    return () => {
      // Leave the element alive across remounts so playback state isn't lost.
    }
  }, [ref])
}
