/**
 * Thin client over the Parakeet ASR API. One source of truth for endpoint
 * URLs, request shaping, and abort plumbing.
 *
 * Two entry points:
 *   - `postTranscription(file, params)` — multipart POST to /v1/audio/transcriptions
 *   - `connectLiveWS(config, callbacks)` — WS to /v1/audio/transcriptions
 */

import type {
  CompactJsonResponse,
  HealthResponse,
  ResponseFormat,
  Strategy,
  TextResponse,
  TimestampGranularity,
  VerboseJsonResponse,
  WSConfig,
  WSMessage,
} from './types'

/** Form fields sent with the REST upload — server-side names verbatim. */
export interface PostParams {
  // OpenAI Whisper API form fields
  model?: string
  language?: string
  prompt?: string
  response_format?: ResponseFormat
  temperature?: number
  timestamp_granularities?: TimestampGranularity[]

  // Server extensions (sent as query params)
  strategy?: Strategy
  chunk_length?: number
  chunk_overlap?: number
  batch_size?: number
  long_audio_threshold?: number
}

export type TranscriptionResponse =
  | { format: 'json'; body: CompactJsonResponse }
  | { format: 'verbose_json'; body: VerboseJsonResponse }
  | { format: 'text' | 'srt' | 'vtt'; body: TextResponse }

export interface PostOptions {
  signal?: AbortSignal
  onProgress?: (loaded: number, total: number) => void
}

/**
 * POST /v1/audio/transcriptions.
 *
 * `onProgress` is wired via XHR (fetch doesn't expose upload progress in
 * any browser yet — Streams API for it is still draft). Aborts cancel the
 * XHR cleanly.
 */
export function postTranscription(
  file: File,
  params: PostParams = {},
  options: PostOptions = {},
): Promise<TranscriptionResponse> {
  const responseFormat: ResponseFormat = params.response_format ?? 'json'

  return new Promise((resolve, reject) => {
    const queryParts: string[] = []
    if (params.strategy) queryParts.push(`strategy=${encodeURIComponent(params.strategy)}`)
    if (params.chunk_length !== undefined) queryParts.push(`chunk_length=${params.chunk_length}`)
    if (params.chunk_overlap !== undefined) queryParts.push(`chunk_overlap=${params.chunk_overlap}`)
    if (params.batch_size !== undefined) queryParts.push(`batch_size=${params.batch_size}`)
    if (params.long_audio_threshold !== undefined)
      queryParts.push(`long_audio_threshold=${params.long_audio_threshold}`)
    const query = queryParts.length ? `?${queryParts.join('&')}` : ''

    const formData = new FormData()
    formData.append('file', file)
    if (params.model) formData.append('model', params.model)
    if (params.language) formData.append('language', params.language)
    if (params.prompt) formData.append('prompt', params.prompt)
    formData.append('response_format', responseFormat)
    if (params.temperature !== undefined) formData.append('temperature', String(params.temperature))
    if (params.timestamp_granularities) {
      for (const g of params.timestamp_granularities) {
        // OpenAI's spec uses bracketed keys; the backend reads both spellings.
        formData.append('timestamp_granularities[]', g)
      }
    }

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `/v1/audio/transcriptions${query}`)
    xhr.responseType = 'text'
    if (options.onProgress) {
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) options.onProgress?.(ev.loaded, ev.total)
      }
    }
    if (options.signal) {
      if (options.signal.aborted) {
        xhr.abort()
        reject(new DOMException('aborted', 'AbortError'))
        return
      }
      options.signal.addEventListener('abort', () => xhr.abort(), { once: true })
    }
    xhr.onerror = () => reject(new Error(`Network error contacting /v1/audio/transcriptions`))
    xhr.onabort = () => reject(new DOMException('aborted', 'AbortError'))
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(`POST returned ${xhr.status}: ${xhr.responseText.slice(0, 400)}`))
        return
      }
      const isJsonFormat = responseFormat === 'json' || responseFormat === 'verbose_json'
      if (isJsonFormat) {
        try {
          const body = JSON.parse(xhr.responseText)
          if (responseFormat === 'verbose_json') {
            resolve({ format: 'verbose_json', body: body as VerboseJsonResponse })
          } else {
            resolve({ format: 'json', body: body as CompactJsonResponse })
          }
        } catch (e) {
          reject(new Error(`Failed to parse JSON response: ${e}`))
        }
      } else {
        resolve({ format: responseFormat, body: xhr.responseText })
      }
    }
    xhr.send(formData)
  })
}

/** WS callbacks. `onMessage` runs on every JSON-typed frame; binary frames are ignored. */
export interface WSCallbacks {
  onOpen?: () => void
  onMessage: (msg: WSMessage) => void
  onClose?: (code: number, reason: string) => void
  onError?: (err: Event) => void
}

export interface LiveWSHandle {
  /** Send a binary chunk to the server. Blobs (MediaRecorder output) and bare ArrayBuffers are both fine. */
  sendBinary: (data: Blob | ArrayBuffer) => void
  /** Send the EOF sentinel (empty binary frame) and close after the final transcription arrives. */
  finish: () => void
  /** Immediately tear down. */
  abort: () => void
  /** Underlying socket — exposed for advanced use; prefer the helpers above. */
  socket: WebSocket
}

/**
 * Open WS /v1/audio/transcriptions, send the JSON config first frame, hand
 * back a handle the caller drives with binary chunks. `onMessage` fires for
 * every JSON-shaped server frame.
 *
 * The dev proxy entry in `vite.config.ts` upgrades this URL to ws://localhost:8777.
 * In prod, FastAPI serves the WS at the same origin.
 */
export function connectLiveWS(config: WSConfig, callbacks: WSCallbacks): LiveWSHandle {
  const wsUrl = (() => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${proto}//${window.location.host}/v1/audio/transcriptions`
  })()

  const socket = new WebSocket(wsUrl)

  let configSent = false
  socket.onopen = () => {
    socket.send(JSON.stringify(config))
    configSent = true
    callbacks.onOpen?.()
  }
  socket.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return // binary frames not used for control
    try {
      const msg = JSON.parse(ev.data) as WSMessage
      callbacks.onMessage(msg)
    } catch (e) {
      console.warn('Failed to parse WS message', e, ev.data)
    }
  }
  socket.onerror = (e) => callbacks.onError?.(e)
  socket.onclose = (ev) => callbacks.onClose?.(ev.code, ev.reason)

  return {
    sendBinary(data) {
      if (!configSent) {
        console.warn('connectLiveWS: dropped binary chunk — WS not open yet')
        return
      }
      socket.send(data)
    },
    finish() {
      // Empty binary frame is the documented EOF signal in the WS protocol.
      try {
        socket.send(new ArrayBuffer(0))
      } catch (e) {
        console.warn('connectLiveWS: finish() send failed', e)
      }
    },
    abort() {
      try {
        socket.close(1000, 'client abort')
      } catch (e) {
        console.warn('connectLiveWS: abort close failed', e)
      }
    },
    socket,
  }
}

/** Optional convenience for the health pane. */
export async function fetchHealth(): Promise<HealthResponse> {
  const r = await fetch('/health')
  if (!r.ok) throw new Error(`Health endpoint returned ${r.status}`)
  return (await r.json()) as HealthResponse
}
