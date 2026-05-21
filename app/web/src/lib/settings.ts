/**
 * Persisted client settings. Zustand store + the `persist` middleware writes
 * to localStorage so user choices survive reloads.
 *
 * Keep all knobs the server accepts in one place — keys map to the server's
 * WS / REST field names verbatim so `settings → PostParams / WSConfig` is a
 * trivial spread.
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ResponseFormat, Strategy, TimestampGranularity } from './types'

export type ThemeMode = 'system' | 'light' | 'dark'

export interface Settings {
  // OpenAI-shape knobs
  responseFormat: ResponseFormat
  timestampGranularities: TimestampGranularity[]

  // Server-extension knobs (REST query string / WS config field)
  strategy: Strategy
  chunkLength: number | null
  chunkOverlap: number | null
  batchSize: number | null
  longAudioThreshold: number | null

  // Progressive (WS) knobs
  liveLatency: boolean
  progressiveRefinement: boolean

  // Voice activity detection + noise (mic / WS path)
  vadEnabled: boolean
  /** Silero probability cutoff (0..1). Higher = pickier. */
  vadThreshold: number
  /** Frames of sustained speech required to flip silent → speech. */
  vadConsecutive: number
  /** Hangover (ms) — keep forwarding for this long after last loud frame. */
  vadHangoverMs: number
  /** Pre-utterance padding fires only when prior silence exceeded this. */
  vadPadMinGapMs: number
  /** Length of the low-noise padding injected at speech onset. */
  vadPadDurationMs: number
  /** ffmpeg `highpass=f=N`. Set 0 to disable. */
  hpfHz: number
  /** Forwarded to `getUserMedia({ audio: { noiseSuppression } })`. */
  noiseSuppression: boolean

  /** Mic capture mode: 'live' streams via WS during recording; 'record'
   * just captures locally then transcribes the finished blob as a file
   * (REST upload). Default 'live'. */
  micCaptureMode: 'live' | 'record'

  // UI
  theme: ThemeMode

  // Mutators
  set: <K extends keyof Settings>(key: K, value: Settings[K]) => void
  reset: () => void
}

const DEFAULTS: Omit<Settings, 'set' | 'reset'> = {
  responseFormat: 'verbose_json',
  timestampGranularities: ['segment', 'word'],
  strategy: 'auto',
  chunkLength: null,
  chunkOverlap: null,
  batchSize: null,
  longAudioThreshold: null,
  liveLatency: true,
  progressiveRefinement: false,

  vadEnabled: true,
  vadThreshold: 0.5,
  vadConsecutive: 3,
  vadHangoverMs: 500,
  vadPadMinGapMs: 400,
  vadPadDurationMs: 0,
  hpfHz: 100,
  noiseSuppression: true,

  micCaptureMode: 'live',

  theme: 'system',
}

export const useSettings = create<Settings>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      set: (key, value) => set({ [key]: value } as Partial<Settings>),
      reset: () => set(DEFAULTS),
    }),
    {
      name: 'parakeet-settings',
      // When the persisted shape lacks newly-added fields (e.g. user
      // hasn't reset since we added vadThreshold), merge DEFAULTS in so
      // those knobs get sensible values without forcing a reset.
      merge: (persisted, current) => ({ ...current, ...(persisted as Partial<Settings>) }),
    },
  ),
)
