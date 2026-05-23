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
import type { Engine, ResponseFormat, StrategyOverride, TimestampGranularity } from './types'

export type ThemeMode = 'system' | 'light' | 'dark'

export interface Settings {
  // OpenAI-shape knobs
  responseFormat: ResponseFormat
  timestampGranularities: TimestampGranularity[]

  // Two-axis transcription picker:
  //   engine          — transport: REST upload vs WebSocket live partials.
  //   strategyOverride — offline-engine implementation. `auto` lets the
  //                      server pick full ≤ MAX_FULL_WAVEFORM_S, else split_full.
  //                      Hidden in the UI when engine === 'streaming'.
  engine: Engine
  strategyOverride: StrategyOverride
  chunkLength: number | null
  chunkOverlap: number | null
  batchSize: number | null
  longAudioThreshold: number | null

  // Streaming (WS) knobs
  liveLatency: boolean

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
  engine: 'offline',
  strategyOverride: 'auto',
  chunkLength: null,
  chunkOverlap: null,
  batchSize: null,
  longAudioThreshold: null,
  liveLatency: true,

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

/** Coerce a persisted snapshot from the pre-split (single-`strategy`) era into
 * the new (engine, strategyOverride) pair. Idempotent: snapshots that already
 * carry `engine` pass through untouched. */
function migrateLegacyStrategy(snap: Record<string, unknown>): Record<string, unknown> {
  if (typeof snap.engine === 'string') return snap
  const legacy = typeof snap.strategy === 'string' ? snap.strategy.toLowerCase() : 'auto'
  const out: Record<string, unknown> = { ...snap }
  delete out.strategy
  delete out.progressiveRefinement
  if (legacy === 'progressive' || legacy === 'streaming') {
    out.engine = 'streaming'
    out.strategyOverride = 'auto'
  } else if (legacy === 'full' || legacy === 'chunked' || legacy === 'split_full') {
    out.engine = 'offline'
    out.strategyOverride = legacy as StrategyOverride
  } else {
    // 'auto' (or unknown) → safest default: offline + auto.
    out.engine = 'offline'
    out.strategyOverride = 'auto'
  }
  return out
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
      // Also coerce pre-split `strategy` snapshots into the new two-axis pair.
      merge: (persisted, current) => ({
        ...current,
        ...migrateLegacyStrategy((persisted ?? {}) as Record<string, unknown>),
      }),
    },
  ),
)
