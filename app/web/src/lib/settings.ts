/**
 * Persisted client settings. Zustand store + the `persist` middleware writes
 * to localStorage so user choices survive reloads.
 *
 * Settings are nested per-modality (`file` / `url` / `mic`) so each workflow
 * remembers its own engine, output format, and modality-specific knobs. The
 * active tab is persisted as `mode`. Mutate via `set(key, value)` for top-
 * level fields or `update('file', { engine: 'websocket' })` for a per-modality
 * patch.
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Engine, ResponseFormat, StrategyOverride, TimestampGranularity } from './types'

export type ThemeMode = 'system' | 'light' | 'dark'

export type Modality = 'file' | 'url' | 'mic'

/** Fields every modality carries. */
export interface CommonModality {
  engine: Engine
  /** REST-only override; ignored when engine === 'websocket'. */
  strategyOverride: StrategyOverride
  responseFormat: ResponseFormat
  timestampGranularities: TimestampGranularity[]
  /** REST-only knob; hidden when engine === 'websocket'. */
  longAudioThreshold: number | null
  /** WS-only knob — flips the streaming preset between 10-10-5 (offline-like,
   * higher throughput) and 10-2-2 (~6 s emission lag, partial_segment
   * messages). Defaults per-modality: false for `file` (throughput beats
   * partial responsiveness on a bounded upload), true for `url` and `mic`
   * (realtime sources benefit from faster commits + partials). */
  liveLatency: boolean
}

/** Mic modality also carries capture-quality knobs. */
export interface MicModality extends CommonModality {
  hpfHz: number
  noiseSuppression: boolean
  vadEnabled: boolean
  vadThreshold: number
  vadConsecutive: number
  vadHangoverMs: number
  vadPadMinGapMs: number
  vadPadDurationMs: number
}

export interface Settings {
  /** Active modality tab. */
  mode: Modality

  file: CommonModality
  url: CommonModality
  mic: MicModality

  theme: ThemeMode

  set: <K extends keyof Settings>(key: K, value: Settings[K]) => void
  /** Per-modality partial patcher — collapses prop-drilling of individual
   * setters in the sidebar. */
  update: <M extends Modality>(m: M, patch: Partial<Settings[M]>) => void
  reset: () => void
}

const COMMON_DEFAULTS: CommonModality = {
  engine: 'rest',
  strategyOverride: 'auto',
  responseFormat: 'verbose_json',
  timestampGranularities: ['segment', 'word'],
  longAudioThreshold: null,
  liveLatency: false,
}

const MIC_DEFAULTS: MicModality = {
  ...COMMON_DEFAULTS,
  engine: 'websocket', // live mic only delivers partials over WS
  liveLatency: true,
  hpfHz: 100,
  noiseSuppression: true,
  vadEnabled: true,
  vadThreshold: 0.5,
  vadConsecutive: 3,
  vadHangoverMs: 500,
  vadPadMinGapMs: 400,
  vadPadDurationMs: 0,
}

const DEFAULTS: Omit<Settings, 'set' | 'update' | 'reset'> = {
  mode: 'file',
  file: { ...COMMON_DEFAULTS },
  // URL streams default to live-latency so partials arrive promptly on
  // realtime sources; user can flip off via Advanced for cleaner commits.
  url: { ...COMMON_DEFAULTS, liveLatency: true },
  mic: { ...MIC_DEFAULTS },
  theme: 'system',
}

export const useSettings = create<Settings>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      set: (key, value) => set({ [key]: value } as Partial<Settings>),
      update: (m, patch) =>
        set((s) => ({ [m]: { ...s[m], ...patch } } as Partial<Settings>)),
      reset: () => set(DEFAULTS),
    }),
    {
      name: 'parakeet-settings',
    },
  ),
)
