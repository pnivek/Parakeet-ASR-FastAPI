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

export type Strategy = 'auto' | 'full' | 'chunked' | 'progressive'

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
  strategy?: Strategy
  chunk_length?: number
  chunk_overlap?: number
  batch_size?: number
  long_audio_threshold?: number
  live_latency?: boolean
  progressive_refinement?: boolean
  early_buffer_target_s?: number
}

/** Mid-stream batch of newly committed segments. Words for just those
 * segments are included (server-side interpolated from per-segment tokens). */
export interface WSSegmentsBatch {
  type: 'segments_batch'
  segments: WhisperSegment[]
  /** Word-level timestamps for the segments in this batch. */
  words?: Word[]
}

/** Optional post-EOF full-pass replacement segments. */
export interface WSRefinedTranscription {
  type: 'refined_transcription'
  segments: WhisperSegment[]
  /** Word-level timestamps for the refined segments. */
  words?: Word[]
  text: string
  transcription_time: number
  audio_duration_seconds: number
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
  streaming_mode: string
  refinement_applied: boolean
}

export interface WSError {
  type: 'error'
  error: string
}

export type WSMessage =
  | WSSegmentsBatch
  | WSRefinedTranscription
  | WSFinalTranscription
  | WSError

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
