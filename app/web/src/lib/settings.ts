/**
 * Persisted client settings. Zustand store + the `persist` middleware writes
 * to localStorage so user choices survive reloads.
 *
 * Keep all knobs the server accepts in one place — keys match the server
 * field names verbatim so `settings → PostParams` is a trivial spread.
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ResponseFormat, Strategy, TimestampGranularity } from './types'

export type ThemeMode = 'system' | 'light' | 'dark'

export interface Settings {
  // OpenAI-shape knobs
  responseFormat: ResponseFormat
  timestampGranularities: TimestampGranularity[]

  // Server-extension knobs (query string)
  strategy: Strategy
  chunkLength: number | null
  chunkOverlap: number | null
  batchSize: number | null
  longAudioThreshold: number | null

  // Progressive (WS) knobs
  liveLatency: boolean
  progressiveRefinement: boolean
  /** Client-side voice activity detection on the mic capture pipeline.
   * When true, silent chunks aren't forwarded to the server, keeping the
   * engine queue drained so real speech is processed immediately. */
  vad: boolean

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
  // Use the 10-2-2 streaming preset by default — ~6 s emission lag vs
  // ~7.5 s for 10-10-5. The quality regression at chunk boundaries is
  // small enough that the snappier default is the right tradeoff for an
  // interactive playground.
  liveLatency: true,
  vad: true,
  // Default off — refinement runs an extra FULL pass at EOF which adds
  // latency on long recordings. Users who want offline-quality final
  // output can opt in.
  progressiveRefinement: false,
  theme: 'system',
}

export const useSettings = create<Settings>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      set: (key, value) => set({ [key]: value } as Partial<Settings>),
      reset: () => set(DEFAULTS),
    }),
    { name: 'parakeet-settings' },
  ),
)
