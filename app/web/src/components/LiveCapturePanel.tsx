import { useCallback, useRef, useState } from 'react'
import { connectLiveWS, type LiveWSHandle, type TranscriptionResponse } from '../lib/api'
import { MIC_FORMAT_HINT, MIC_SAMPLE_RATE, MIC_MIME_TYPE, useMic } from '../lib/mic'
import { useSettings } from '../lib/settings'
import type { WhisperSegment, WSMessage } from '../lib/types'

interface Props {
  /** Called when a final transcription arrives (or stream is stopped). */
  onResult: (audio: File, result: TranscriptionResponse) => void
  /** Called on each segments_batch — drives the live-incremental view. */
  onPartial: (result: TranscriptionResponse) => void
  /** Called on any unrecoverable error. */
  onError: (message: string) => void
}

/**
 * Microphone capture + live WS transcription.
 *
 * Browser captures audio via MediaRecorder(webm/opus). Each timeslice blob
 * is forwarded to the WS as a binary frame. Server responds with
 * `segments_batch` mid-stream and `final_transcription` at EOF.
 *
 * We also accumulate the chunks locally so that after recording stops we
 * hand the parent a File for the audio player — letting the user replay
 * what they just spoke and click segments to seek, same UX as file upload.
 */
export function LiveCapturePanel({ onResult, onPartial, onError }: Props) {
  const settings = useSettings()
  const wsRef = useRef<LiveWSHandle | null>(null)
  const segmentsRef = useRef<WhisperSegment[]>([])
  const chunksRef = useRef<Blob[]>([])
  const recordStartRef = useRef<number>(0)
  const [segmentCount, setSegmentCount] = useState(0)

  const buildPartialResult = useCallback((): TranscriptionResponse => {
    const segs = segmentsRef.current
    const lastEnd = segs.length > 0 ? segs[segs.length - 1].end : 0
    const elapsed = (Date.now() - recordStartRef.current) / 1000
    return {
      format: 'verbose_json',
      body: {
        task: 'transcribe',
        language: 'en',
        duration: lastEnd,
        text: segs.map((s) => s.text).join(' ').trim(),
        segments: segs,
        strategy: 'progressive',
        transcription_time_seconds: elapsed,
      },
    }
  }, [])

  const handleMessage = useCallback(
    (msg: WSMessage) => {
      switch (msg.type) {
        case 'segments_batch':
          segmentsRef.current = [...segmentsRef.current, ...msg.segments]
          setSegmentCount(segmentsRef.current.length)
          onPartial(buildPartialResult())
          break
        case 'refined_transcription':
          // EOF full pass — replace segments with the higher-quality refinement.
          segmentsRef.current = msg.segments
          setSegmentCount(msg.segments.length)
          onPartial({
            format: 'verbose_json',
            body: {
              task: 'transcribe',
              language: 'en',
              duration: msg.audio_duration_seconds,
              text: msg.text,
              segments: msg.segments,
              strategy: 'progressive',
              transcription_time_seconds: msg.transcription_time,
            },
          })
          break
        case 'final_transcription': {
          const blob = new Blob(chunksRef.current, { type: MIC_MIME_TYPE })
          const file = new File([blob], `mic-${Date.now()}.webm`, { type: MIC_MIME_TYPE })
          const result: TranscriptionResponse = {
            format: 'verbose_json',
            body: {
              task: 'transcribe',
              language: msg.language,
              duration: msg.duration,
              text: msg.text,
              segments: msg.segments,
              strategy: msg.strategy,
              transcription_time_seconds: msg.transcription_time,
              csv_content: msg.csv_content,
              srt_content: msg.srt_content,
            },
          }
          onResult(file, result)
          break
        }
        case 'error':
          onError(msg.error)
          break
      }
    },
    [buildPartialResult, onPartial, onResult, onError],
  )

  const mic = useMic({
    onChunk: (blob) => {
      chunksRef.current.push(blob)
      wsRef.current?.sendBinary(blob)
    },
    onStop: () => {
      wsRef.current?.finish()
    },
    onError: (err) => {
      onError(err.message)
      wsRef.current?.abort()
      wsRef.current = null
    },
  })

  const start = useCallback(async () => {
    segmentsRef.current = []
    chunksRef.current = []
    setSegmentCount(0)
    recordStartRef.current = Date.now()

    const ws = connectLiveWS(
      {
        sample_rate: MIC_SAMPLE_RATE,
        channels: 1,
        bytes_per_sample: 2,
        format: MIC_FORMAT_HINT,
        strategy: settings.strategy === 'auto' ? 'progressive' : settings.strategy,
        chunk_length: settings.chunkLength ?? undefined,
        chunk_overlap: settings.chunkOverlap ?? undefined,
        batch_size: settings.batchSize ?? undefined,
        long_audio_threshold: settings.longAudioThreshold ?? undefined,
        live_latency: settings.liveLatency,
        progressive_refinement: settings.progressiveRefinement,
      },
      {
        onMessage: handleMessage,
        onError: () => onError('WebSocket error — connection failed.'),
        onClose: (code, reason) => {
          if (code !== 1000 && code !== 1005) {
            onError(`WebSocket closed: code=${code} reason=${reason || 'no reason'}`)
          }
          wsRef.current = null
        },
      },
    )
    wsRef.current = ws
    await mic.start()
  }, [settings, handleMessage, mic, onError])

  const stop = useCallback(() => {
    mic.stop()
  }, [mic])

  const recording = mic.state === 'recording'
  const busy = mic.state === 'starting' || mic.state === 'stopping'

  return (
    <section className="drop">
      <div className="drop__file">
        <div className="drop__filename">Live microphone</div>
        <div className="drop__filemeta">
          {mic.state === 'idle' && 'Click record to start capturing.'}
          {mic.state === 'starting' && 'Requesting microphone access…'}
          {mic.state === 'recording' && `Recording · ${segmentCount} segment${segmentCount === 1 ? '' : 's'} so far`}
          {mic.state === 'stopping' && 'Finalizing transcription…'}
          {mic.state === 'error' && mic.error}
        </div>
        <div className="drop__actions">
          {!recording && (
            <button type="button" onClick={start} disabled={busy} className="primary">
              {busy ? 'Working…' : 'Record'}
            </button>
          )}
          {recording && (
            <button type="button" onClick={stop} disabled={busy}>
              Stop
            </button>
          )}
        </div>
        {(recording || busy) && (
          <div className="mic-meter" aria-label="microphone level">
            <div
              className="mic-meter__bar"
              style={{ width: `${Math.min(100, Math.round(mic.level * 140))}%` }}
            />
          </div>
        )}
      </div>
    </section>
  )
}
