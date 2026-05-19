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

  // UI
  theme: ThemeMode

  // Mutators
  set: <K extends keyof Settings>(key: K, value: Settings[K]) => void
  reset: () => void
}

const DEFAULTS: Omit<Settings, 'set' | 'reset'> = {
  responseFormat: 'verbose_json',
  timestampGranularities: ['segment'],
  strategy: 'auto',
  chunkLength: null,
  chunkOverlap: null,
  batchSize: null,
  longAudioThreshold: null,
  liveLatency: false,
  progressiveRefinement: true,
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
