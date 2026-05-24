/**
 * TypeScript shapes for the Parakeet ASR backend.
 *
 * Mirror the server-side definitions so client code reads them without casting:
 *   - segment shape  → app/streaming_v2.py:_whisper_segment
 *   - word shape     → app/streaming_v2.py:tokens_to_words
 *   - REST responses → app/main.py:_build_openai_response
 *   - WS messages    → app/main.py:_ws_accumulate_then_process,
 *                      app/main.py:handle_streaming_pcm
 *
 * Keep this file the source of truth for client typing.
 */

export type ResponseFormat = 'json' | 'text' | 'srt' | 'vtt' | 'verbose_json'

/** Engine — the transport axis the user picks per modality.
 * `rest`      → HTTP POST.
 * `websocket` → WebSocket; live partials while audio is in flight. */
export type Engine = 'rest' | 'websocket'

/** Strategy override for the REST engine. Default `auto` lets the server pick
 * `full` (≤ MAX_FULL_WAVEFORM_S) vs `split_full` (above). Explicit values force
 * a particular implementation. */
export type StrategyOverride = 'auto' | 'full' | 'split_full'

/** Wire-level strategy enum the server's `?strategy=` param accepts. The UI
 * derives it from (engine, strategyOverride) — see Sidebar `restStrategyParam`. */
export type Strategy = 'offline' | 'full' | 'split_full' | 'streaming'

export type TimestampGranularity = 'segment' | 'word'

/** Single segment in OpenAI's verbose_json shape (Whisper-compatible). */
export interface WhisperSegment {
  id: number
  seek: number
  start: number
  end: number
  text: string
  tokens: number[]
  temperature: number
  /** Null on FULL_GRAPH-mode CUDA-graph decode (NeMo 2.7.3 limitation). */
  avg_logprob: number | null
  compression_ratio: number
  /** Always null — Parakeet TDT has no `<|nospeech|>` equivalent. */
  no_speech_prob: number | null
  /** Per-token decoder times in seconds, parallel to `tokens`. Present
   * on streaming + FULL paths; absent on response shapes that didn't
   * carry it (e.g. older payloads). Wall-clock when VAD is on (server
   * translates from engine clock before emit). */
  token_times?: number[]
}

/** One word in the `words[]` array (when timestamp_granularities includes "word"). */
export interface Word {
  word: string
  start: number
  end: number
}

/** OpenAI `json` (compact). */
export interface CompactJsonResponse {
  text: string
}

/** OpenAI `verbose_json` plus our server-side extensions. */
export interface VerboseJsonResponse {
  // Whisper-compatible fields
  task: 'transcribe'
  language: string
  duration: number
  text: string
  segments: WhisperSegment[]
  /** Present only when `timestamp_granularities[]=word` was requested. */
  words?: Word[]

  // Server extensions (not in OpenAI's spec)
  strategy: Strategy
  transcription_time_seconds: number
  total_request_time_server_seconds?: number
  audio_duration_seconds?: number
  csv_content?: string
  srt_content?: string
  vtt_content?: string
  /** Live-streaming counters (present only on WS-streaming responses).
   * Drive the live RTFx in the footer:
   *   live_rtfx = speech_committed_s / speech_received_s
   * Both counters are server-side cumulative wall-clock measurements
   * derived from Silero VAD (received) and committed segment durations
   * (committed). Absent on REST / offline responses — the footer falls
   * back to the offline ratio there. */
  audio_received_s?: number
  speech_received_s?: number
  speech_committed_s?: number
}

/** Plain-text response body — `response_format=text|srt|vtt` returns a string. */
export type TextResponse = string

/* ----- WebSocket protocol ----- */

/** First frame sent over the WS — JSON text. */
export interface WSConfig {
  sample_rate: number
  channels: number
  bytes_per_sample: number
  /** Audio container/codec ffmpeg will be told to decode. */
  format: string
  /** When set, the server opens `ffmpeg -i <url>` and skips the binary-frame
   * reader. Suits HLS / icecast / RTSP / m3u8 live-stream URLs. */
  url?: string
  strategy?: Strategy
  chunk_length?: number
  chunk_overlap?: number
  batch_size?: number
  long_audio_threshold?: number
  live_latency?: boolean
  early_buffer_target_s?: number

  // VAD + noise knobs (server falls back to env defaults if absent).
  vad_enabled?: boolean
  vad_threshold?: number
  vad_consecutive?: number
  vad_hangover_ms?: number
  vad_pad_min_gap_ms?: number
  vad_pad_duration_ms?: number
  /** ffmpeg `highpass=f=N`. 0 = disabled. */
  hpf_hz?: number
}

/** Mid-stream batch of newly committed segments. Words for just those
 * segments are included (server-side interpolated from per-segment tokens). */
export interface WSSegmentsBatch {
  type: 'segments_batch'
  segments: WhisperSegment[]
  /** Word-level timestamps for the segments in this batch. */
  words?: Word[]
  /** Streaming counters — see VerboseJsonResponse for semantics. */
  audio_received_s?: number
  speech_received_s?: number
  speech_committed_s?: number
}

/** Final summary message at end of stream — mirrors verbose_json with extras. */
export interface WSFinalTranscription {
  type: 'final_transcription'
  task: 'transcribe'
  language: string
  duration: number
  text: string
  segments: WhisperSegment[]
  /** Word-level timestamps across the whole stream. */
  words?: Word[]
  strategy: Strategy
  transcription_time: number
  total_segments: number
  final_duration_processed_seconds: number
  csv_content: string
  srt_content: string
  vtt_content?: string
  streaming_mode: string
  /** Streaming counters — see VerboseJsonResponse for semantics. */
  audio_received_s?: number
  speech_received_s?: number
  speech_committed_s?: number
}

export interface WSError {
  type: 'error'
  error: string
}

/** Read-only peek of the engine's in-progress sentence buffer. Server
 * emits this after every chunk (live mic only) so the client can show
 * interim text while the model decodes — instead of waiting up to ~6s
 * for a `.!?` to commit. The same content will arrive as a real
 * segments_batch entry once the model terminates the sentence; the
 * client should clear the partial when that happens. Does NOT mutate
 * the model's outputs. */
export interface WSPartialSegment {
  type: 'partial_segment'
  segment: WhisperSegment
  words?: Word[]
  /** Streaming counters — see VerboseJsonResponse for semantics. */
  audio_received_s?: number
  speech_received_s?: number
  speech_committed_s?: number
}

export type WSMessage =
  | WSSegmentsBatch
  | WSFinalTranscription
  | WSError
  | WSPartialSegment

/* ----- Health endpoint ----- */

export interface HealthResponse {
  model_loaded: boolean
  model_name: string
  config: Record<string, unknown>
  decoding_strategy?: string
  device?: string
  dtype?: string
  status?: string
}
