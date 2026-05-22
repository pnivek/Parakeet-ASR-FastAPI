import { useCallback, useEffect, useRef, useState } from 'react'
import {
  postTranscription,
  postTranscriptionUrl,
  streamFileViaWS,
  type TranscriptionResponse,
  connectLiveWS,
  type LiveWSHandle,
} from '../lib/api'
import { useSettings } from '../lib/settings'
import { MIC_FORMAT_HINT, MIC_SAMPLE_RATE, MIC_MIME_TYPE, useMic } from '../lib/mic'
import { formatBytes } from '../lib/format'
import type { ResponseFormat, Strategy, TimestampGranularity, WhisperSegment, Word, WSMessage } from '../lib/types'
import type { LoadedAudio } from '../lib/download'
import { OptionList } from './OptionList'

export type InputMode = 'file' | 'mic' | 'url'

type SidebarTab = 'source' | 'output' | 'engine'

interface Props {
  mode: InputMode
  onModeChange: (m: InputMode) => void
  onResult: (loaded: LoadedAudio, result: TranscriptionResponse) => void
  onPartial: (result: TranscriptionResponse) => void
  onError: (message: string) => void
  onBusyChange: (busy: boolean) => void
  /** Called right before a new mic recording starts. Use to clear prior
   * result/loaded/peaks so the UI doesn't show the previous session. */
  onSessionStart?: () => void
  /** Called early in a streaming session to wire the audio source without
   * setting the final result (so the user can scrub/play during streaming). */
  onAudioReady?: (loaded: LoadedAudio) => void
  /** Progressive PCM peaks from the server — append or replace the hero
   * waveform's peaks array. Lets mic mode draw bars as the user speaks
   * instead of waiting for final_transcription + blob decode. */
  onPeaks?: (peaks: number[], cumulative: boolean) => void
  /** Read-only peek of the engine's in-flight sentence buffer. Null
   * clears it (e.g., the sentence just committed). */
  onPartialSegment?: (segment: WhisperSegment | null, words: Word[]) => void
}

const FORMATS: { id: ResponseFormat; label: string }[] = [
  { id: 'verbose_json', label: 'verbose_json' },
  { id: 'json', label: 'json' },
  { id: 'text', label: 'text' },
  { id: 'srt', label: 'srt' },
  { id: 'vtt', label: 'vtt' },
]
const STRATEGIES: Strategy[] = ['auto', 'full', 'chunked', 'progressive']

export function Sidebar({ mode, onModeChange, onResult, onPartial, onError, onBusyChange, onSessionStart, onAudioReady, onPeaks, onPartialSegment }: Props) {
  const s = useSettings()
  const [tab, setTab] = useState<SidebarTab>('source')
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)

  const setBusyAll = useCallback(
    (b: boolean) => {
      setBusy(b)
      onBusyChange(b)
    },
    [onBusyChange],
  )

  // File state
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [pickedFile, setPickedFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const onFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return
    const file = files[0]
    setPickedFile(file)
    // Wire the audio source + kick off peaks decode immediately. We
    // already have the entire audio locally — no reason to wait for the
    // user to click Transcribe. The waveform fades in as soon as
    // decodeAudioData resolves.
    onAudioReady?.({
      kind: 'file',
      title: file.name.replace(/\.[^.]+$/, ''),
      source: `${formatBytes(file.size)} · ${file.type || 'audio'}`,
      file,
    })
  }

  // URL state
  const [urlInput, setUrlInput] = useState('https://')

  // Mic state — live capture
  const wsRef = useRef<LiveWSHandle | null>(null)
  // AbortController for the in-flight REST upload (file/url, non-progressive).
  // Held in a ref because we need to reach in from the Stop button click.
  const restAbortRef = useRef<AbortController | null>(null)
  const segmentsRef = useRef<WhisperSegment[]>([])
  const wordsRef = useRef<Word[]>([])
  const chunksRef = useRef<Blob[]>([])
  const recordStartRef = useRef<number>(0)
  const [recordElapsed, setRecordElapsed] = useState(0)

  // ── Auto-drop progressive when mode is URL ───────────────────
  // progressive is available for mic (WS) and file (WS-stream). URL mode
  // uses server-side fetch over REST, so progressive isn't supported.
  useEffect(() => {
    if (mode === 'url' && s.strategy === 'progressive') {
      s.set('strategy', 'auto')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode])
  // Strategy actually sent over REST (file-non-progressive / URL). If the
  // user has progressive selected but we're falling back to REST (URL
  // mode), map to chunked so the backend doesn't 400.
  const restStrategy = (): Strategy =>
    s.strategy === 'progressive' ? 'chunked' : s.strategy

  /** Shared VAD + HPF config block for any WS config first frame.
   *
   * VAD is only honored in mic mode: it exists to keep the engine queue
   * drained when the producer is bandwidth-bound. File and URL paths
   * stream data faster than realtime, so VAD adds per-frame CPU cost
   * without any latency benefit there. Auto-disable it for non-mic
   * modes regardless of the persisted toggle. */
  const vadConfig = () => ({
    vad_enabled: mode === 'mic' ? s.vadEnabled : false,
    vad_threshold: s.vadThreshold,
    vad_consecutive: s.vadConsecutive,
    vad_hangover_ms: s.vadHangoverMs,
    vad_pad_min_gap_ms: s.vadPadMinGapMs,
    vad_pad_duration_ms: s.vadPadDurationMs,
    hpf_hz: s.hpfHz,
  })

  // ── Throttle for onPartial during streaming ──────────────────────
  // Server emits segments_batch up to ~10/s during fast file processing.
  // Dispatching every one causes O(N) React re-renders of the transcript
  // view per partial (text.join over a growing array, full DOM diff).
  // Coalesce to ~4 Hz with leading + trailing edges; build the payload
  // off the refs at dispatch time so the trailing call always carries
  // the latest accumulated state.
  const PARTIAL_THROTTLE_MS = 250
  const lastPartialAtRef = useRef(0)
  const partialTimerRef = useRef<number | null>(null)
  const buildPartialPayload = useCallback((): TranscriptionResponse => {
    const wall = (Date.now() - recordStartRef.current) / 1000
    const segs = segmentsRef.current
    const words = wordsRef.current
    return {
      format: 'verbose_json',
      body: {
        task: 'transcribe',
        language: 'en',
        duration: segs[segs.length - 1]?.end ?? 0,
        text: segs.map((x) => x.text).join(' ').trim(),
        segments: segs,
        words: words.length > 0 ? words : undefined,
        strategy: 'progressive',
        transcription_time_seconds: wall,
      },
    }
  }, [])
  const schedulePartial = useCallback(() => {
    const now = Date.now()
    const since = now - lastPartialAtRef.current
    if (since >= PARTIAL_THROTTLE_MS) {
      lastPartialAtRef.current = now
      onPartial(buildPartialPayload())
      return
    }
    if (partialTimerRef.current === null) {
      partialTimerRef.current = window.setTimeout(() => {
        partialTimerRef.current = null
        lastPartialAtRef.current = Date.now()
        onPartial(buildPartialPayload())
      }, PARTIAL_THROTTLE_MS - since)
    }
  }, [onPartial, buildPartialPayload])
  const cancelPendingPartial = useCallback(() => {
    if (partialTimerRef.current !== null) {
      clearTimeout(partialTimerRef.current)
      partialTimerRef.current = null
    }
    lastPartialAtRef.current = 0
  }, [])

  // ── WS message → result/partial dispatch ─────────────────────────
  const handleMessage = useCallback(
    (msg: WSMessage) => {
      switch (msg.type) {
        case 'segments_batch': {
          segmentsRef.current = [...segmentsRef.current, ...msg.segments]
          if (msg.words && msg.words.length > 0) {
            wordsRef.current = [...wordsRef.current, ...msg.words]
          }
          // The committed segments include whatever was previously in
          // the partial buffer — clear the in-flight partial so the
          // text doesn't appear twice (once as partial, once as real).
          onPartialSegment?.(null, [])
          schedulePartial()
          break
        }
        case 'partial_segment':
          onPartialSegment?.(msg.segment, msg.words ?? [])
          break
        case 'refined_transcription':
          cancelPendingPartial()
          onPartialSegment?.(null, [])
          segmentsRef.current = msg.segments
          if (msg.words) wordsRef.current = msg.words
          onPartial({
            format: 'verbose_json',
            body: {
              task: 'transcribe',
              language: 'en',
              duration: msg.audio_duration_seconds,
              text: msg.text,
              segments: msg.segments,
              words: msg.words,
              strategy: 'progressive',
              transcription_time_seconds: msg.transcription_time,
            },
          })
          break
        case 'final_transcription': {
          cancelPendingPartial()
          onPartialSegment?.(null, [])
          const blob = new Blob(chunksRef.current, { type: MIC_MIME_TYPE })
          const file = new File([blob], `mic-${Date.now()}.webm`, { type: MIC_MIME_TYPE })
          onResult(
            {
              kind: 'file',
              title: `live recording — ${new Date().toLocaleTimeString()}`,
              source: 'live mic',
              file,
            },
            {
              format: 'verbose_json',
              body: {
                task: 'transcribe',
                language: msg.language,
                duration: msg.duration,
                text: msg.text,
                segments: msg.segments,
                words: msg.words,
                strategy: msg.strategy,
                transcription_time_seconds: msg.transcription_time,
                csv_content: msg.csv_content,
                srt_content: msg.srt_content,
              },
            },
          )
          // Final result is in — tear the socket down promptly rather
          // than waiting for the server's close frame. abort() closes
          // with code 1000 so the own-checked onClose won't error-toast.
          wsRef.current?.abort()
          wsRef.current = null
          break
        }
        // 'peaks' messages from the server are ignored here. Mic-mode
        // peaks are driven from the local AnalyserNode via
        // useMic({ onPeakSample }) — smoother + no round-trip. File mode
        // gets peaks from a client-side decode of the Blob in App.tsx.
        case 'error':
          onError(msg.error)
          break
      }
    },
    [onPartial, onResult, onError, schedulePartial, cancelPendingPartial, onPartialSegment],
  )

  const mic = useMic({
    onChunk: (blob) => {
      chunksRef.current.push(blob)
      // Live mode forwards each chunk to the server while recording.
      // Record mode just accumulates locally — wsRef is null, so this
      // is a no-op.
      wsRef.current?.sendBinary(blob)
    },
    onStop: () => {
      if (wsRef.current) {
        // Live mode: tell the server we're done. Final segments/words
        // arrive via the WS message handler.
        wsRef.current.finish()
        return
      }
      // Record mode: build a File from the accumulated chunks and
      // stage it as pickedFile. The user clicks Transcribe to send it
      // through the REST upload path.
      if (chunksRef.current.length === 0) return
      const blob = new Blob(chunksRef.current, { type: MIC_MIME_TYPE })
      const file = new File([blob], `recording-${Date.now()}.webm`, { type: MIC_MIME_TYPE })
      setPickedFile(file)
    },
    onError: (err) => {
      onError(err.message)
      wsRef.current?.abort()
      wsRef.current = null
    },
    noiseSuppression: s.noiseSuppression,
    onPeakSample: (peak) => onPeaks?.([peak], false),
  })

  const recording = mic.state === 'recording'
  const micBusy = mic.state === 'starting' || mic.state === 'stopping'

  useEffect(() => {
    onBusyChange(busy || recording || micBusy)
  }, [busy, recording, micBusy, onBusyChange])

  useEffect(() => {
    if (!recording) return
    const i = setInterval(() => setRecordElapsed((Date.now() - recordStartRef.current) / 1000), 100)
    return () => clearInterval(i)
  }, [recording])

  // Live-preview listening: in mic + Live mode the meter should reflect
  // real speech BEFORE the user hits Record (the design's "Listening"
  // state). Open a preview-only mic stream while idle in that mode;
  // tear it down whenever we leave it. start() reuses the preview
  // stream when the user commits to recording.
  //
  // Only attempt where getUserMedia is actually available (secure
  // context). On a plain-HTTP LAN deployment the API is undefined, so
  // we skip the preview entirely rather than surface a mic error on
  // mode switch — the meter just stays flat there.
  useEffect(() => {
    const micAvailable =
      typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia
    if (!micAvailable) return
    const wantListen = mode === 'mic' && s.micCaptureMode === 'live'
    if (wantListen && mic.state === 'idle') {
      void mic.listen()
    } else if (!wantListen && mic.state === 'listening') {
      mic.stopListening()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, s.micCaptureMode, mic.state])

  // Stop the preview when the component unmounts.
  useEffect(() => {
    return () => mic.stopListening()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const startMic = useCallback(
    async (captureMode: 'live' | 'record' = s.micCaptureMode) => {
      onSessionStart?.()
      cancelPendingPartial()
      segmentsRef.current = []
      wordsRef.current = []
      chunksRef.current = []
      recordStartRef.current = Date.now()
      setRecordElapsed(0)
      // Record mode skips the WS entirely — useMic still records to
      // chunksRef, and the onStop handler stages the resulting blob
      // as pickedFile + routes through the REST file path on
      // Transcribe click.
      if (captureMode === 'record') {
        // Make sure any prior recording / WS is cleared so the new
        // blob can be staged cleanly.
        wsRef.current = null
        setPickedFile(null)
        await mic.start()
        return
      }
      // Live mode: open the WS, then start the mic.
      const wsStrategy: Strategy =
        s.strategy === 'auto' || s.strategy === 'progressive' ? 'progressive' : s.strategy
      let liveWs: LiveWSHandle
      liveWs = connectLiveWS(
        {
          sample_rate: MIC_SAMPLE_RATE,
          channels: 1,
          bytes_per_sample: 2,
          format: MIC_FORMAT_HINT,
          strategy: wsStrategy,
          live_latency: s.liveLatency,
          progressive_refinement: s.progressiveRefinement,
          chunk_length: s.chunkLength ?? undefined,
          chunk_overlap: s.chunkOverlap ?? undefined,
          batch_size: s.batchSize ?? undefined,
          long_audio_threshold: s.longAudioThreshold ?? undefined,
          ...vadConfig(),
        },
        {
          onMessage: handleMessage,
          onError: () => onError('WebSocket error — connection failed.'),
          onClose: (code, reason) => {
            if (code !== 1000 && code !== 1005) {
              onError(`WebSocket closed: code=${code} reason=${reason || 'no reason'}`)
            }
            // Only clear the ref if it still points at THIS socket — a
            // newer recording may have already replaced it. Without this
            // guard a late close from a previous session nulls the live
            // socket and chunks stop being forwarded.
            if (wsRef.current === liveWs) wsRef.current = null
          },
        },
      )
      wsRef.current = liveWs
      await mic.start()
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [s, handleMessage, mic, onError, onSessionStart],
  )

  // Drive the bottom Transcribe button. In mic mode it doubles as
  // record/stop. In file mode + progressive, stream the bytes over the WS
  // for real-time partials. Otherwise REST.
  const transcribe = useCallback(async () => {
    if (mode === 'file') {
      if (!pickedFile) return
      onSessionStart?.()

      // File + progressive → stream over WebSocket for live partials.
      // The audio player gets the file immediately so the user can scrub
      // / play while transcription streams in.
      if (s.strategy === 'progressive') {
        cancelPendingPartial()
        segmentsRef.current = []
        wordsRef.current = []
        chunksRef.current = []
        recordStartRef.current = Date.now()
        const loaded: LoadedAudio = {
          kind: 'file',
          title: pickedFile.name.replace(/\.[^.]+$/, ''),
          source: `${formatBytes(pickedFile.size)} · ${pickedFile.type || 'audio'}`,
          file: pickedFile,
        }
        // Wire the audio source NOW so the user can scrub / play while
        // transcription streams. The transcript will fill in via partials,
        // and onResult fires when final_transcription arrives.
        onAudioReady?.(loaded)

        setBusyAll(true)
        try {
          const fileExt = pickedFile.name.split('.').pop()?.toLowerCase() || 'wav'
          let fileWs: LiveWSHandle
          fileWs = streamFileViaWS(
            pickedFile,
            {
              sample_rate: 16000,
              channels: 1,
              bytes_per_sample: 2,
              format: fileExt,
              strategy: 'progressive',
              // Force the offline 10s-chunk preset for file mode. The
              // user's `liveLatency` toggle exists for mic responsiveness
              // (2s chunks → 4s emission lag) but actively slows file
              // throughput ~4x because the engine runs more steps per
              // second of audio. Bench: 75x RTFx offline vs 18x live on
              // the same 25-min file.
              live_latency: false,
              progressive_refinement: s.progressiveRefinement,
              chunk_length: s.chunkLength ?? undefined,
              chunk_overlap: s.chunkOverlap ?? undefined,
              batch_size: s.batchSize ?? undefined,
              long_audio_threshold: s.longAudioThreshold ?? undefined,
              ...vadConfig(),
            },
            {
              onMessage: (msg) => {
                // Same as mic — accumulate segments + words, surface partials.
                if (msg.type === 'segments_batch') {
                  segmentsRef.current = [...segmentsRef.current, ...msg.segments]
                  if (msg.words && msg.words.length > 0)
                    wordsRef.current = [...wordsRef.current, ...msg.words]
                  schedulePartial()
                } else if (msg.type === 'refined_transcription') {
                  cancelPendingPartial()
                  segmentsRef.current = msg.segments
                  if (msg.words) wordsRef.current = msg.words
                  onPartial({
                    format: 'verbose_json',
                    body: {
                      task: 'transcribe',
                      language: 'en',
                      duration: msg.audio_duration_seconds,
                      text: msg.text,
                      segments: msg.segments,
                      words: msg.words,
                      strategy: 'progressive',
                      transcription_time_seconds: msg.transcription_time,
                    },
                  })
                } else if (msg.type === 'final_transcription') {
                  cancelPendingPartial()
                  onResult(loaded, {
                    format: 'verbose_json',
                    body: {
                      task: 'transcribe',
                      language: msg.language,
                      duration: msg.duration,
                      text: msg.text,
                      segments: msg.segments,
                      words: msg.words,
                      strategy: msg.strategy,
                      transcription_time_seconds: msg.transcription_time,
                      csv_content: msg.csv_content,
                      srt_content: msg.srt_content,
                    },
                  })
                } else if (msg.type === 'error') {
                  onError(msg.error)
                }
                // 'peaks' messages from the server are ignored — for
                // file+progressive the Blob is already local and
                // computePeaksFor decodes it client-side in App.tsx.
              },
              onError: () => onError('WebSocket error during file streaming.'),
              onClose: (code, reason) => {
                if (code !== 1000 && code !== 1005) {
                  onError(`WebSocket closed: code=${code} reason=${reason || 'no reason'}`)
                }
                if (wsRef.current === fileWs) wsRef.current = null
                setBusyAll(false)
              },
            },
          )
          wsRef.current = fileWs
        } catch (e) {
          onError(e instanceof Error ? e.message : String(e))
          setBusyAll(false)
        }
        return
      }

      // File + chunked/full/auto → standard REST upload.
      setBusyAll(true)
      setProgress(0)
      const ac = new AbortController()
      restAbortRef.current = ac
      try {
        const r = await postTranscription(
          pickedFile,
          {
            response_format: s.responseFormat,
            timestamp_granularities: s.timestampGranularities,
            strategy: restStrategy(),
            chunk_length: s.chunkLength ?? undefined,
            chunk_overlap: s.chunkOverlap ?? undefined,
            batch_size: s.batchSize ?? undefined,
            long_audio_threshold: s.longAudioThreshold ?? undefined,
          },
          { onProgress: (loaded, total) => setProgress(loaded / total), signal: ac.signal },
        )
        onResult(
          {
            kind: 'file',
            title: pickedFile.name.replace(/\.[^.]+$/, ''),
            source: `${formatBytes(pickedFile.size)} · ${pickedFile.type || 'audio'}`,
            file: pickedFile,
          },
          r,
        )
      } catch (e) {
        // User-initiated cancel — silent return, partials (if any) stay.
        if (e instanceof DOMException && e.name === 'AbortError') return
        onError(e instanceof Error ? e.message : String(e))
      } finally {
        if (restAbortRef.current === ac) restAbortRef.current = null
        setBusyAll(false)
        setProgress(null)
      }
    } else if (mode === 'url') {
      if (!urlInput.trim() || urlInput === 'https://') return
      onSessionStart?.()
      setBusyAll(true)
      const ac = new AbortController()
      restAbortRef.current = ac
      try {
        const r = await postTranscriptionUrl(
          urlInput.trim(),
          {
            response_format: s.responseFormat,
            timestamp_granularities: s.timestampGranularities,
            strategy: restStrategy(),
            chunk_length: s.chunkLength ?? undefined,
            chunk_overlap: s.chunkOverlap ?? undefined,
            batch_size: s.batchSize ?? undefined,
            long_audio_threshold: s.longAudioThreshold ?? undefined,
          },
          { signal: ac.signal },
        )
        let name = urlInput
        try {
          const u = new URL(urlInput)
          name = u.pathname.split('/').filter(Boolean).pop() || u.host
        } catch {
          /* keep input */
        }
        onResult({ kind: 'url', title: name, source: 'via URL', url: urlInput.trim() }, r)
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') return
        onError(e instanceof Error ? e.message : String(e))
      } finally {
        if (restAbortRef.current === ac) restAbortRef.current = null
        setBusyAll(false)
      }
    } else if (mode === 'mic') {
      if (recording) {
        mic.stop()
        return
      }
      // Record sub-mode with a staged blob: upload via REST so the
      // user gets the full pipeline (auto strategy, refinement, etc).
      if (s.micCaptureMode === 'record' && pickedFile) {
        onSessionStart?.()
        setBusyAll(true)
        setProgress(0)
        const ac = new AbortController()
        restAbortRef.current = ac
        try {
          const r = await postTranscription(
            pickedFile,
            {
              response_format: s.responseFormat,
              timestamp_granularities: s.timestampGranularities,
              strategy: restStrategy(),
              chunk_length: s.chunkLength ?? undefined,
              chunk_overlap: s.chunkOverlap ?? undefined,
              batch_size: s.batchSize ?? undefined,
              long_audio_threshold: s.longAudioThreshold ?? undefined,
            },
            { onProgress: (loaded, total) => setProgress(loaded / total), signal: ac.signal },
          )
          onResult(
            {
              kind: 'file',
              title: `recording — ${new Date().toLocaleTimeString()}`,
              source: 'mic · recorded',
              file: pickedFile,
            },
            r,
          )
        } catch (e) {
          if (e instanceof DOMException && e.name === 'AbortError') return
          onError(e instanceof Error ? e.message : String(e))
        } finally {
          if (restAbortRef.current === ac) restAbortRef.current = null
          setBusyAll(false)
          setProgress(null)
        }
        return
      }
      startMic(s.micCaptureMode).catch((e) => onError(e instanceof Error ? e.message : String(e)))
    }
  }, [mode, pickedFile, urlInput, s, recording, mic, startMic, onResult, onPartial, onError, setBusyAll, onSessionStart, onAudioReady])

  // Tear down whatever's in flight for file/url. Mic uses its own
  // record/stop path via mic.stop() — handled inside transcribe().
  const cancelInFlight = useCallback(() => {
    if (wsRef.current && mode !== 'mic') {
      // file + progressive. Close the WS with code 1000; even if onClose
      // doesn't fire in time (server taking a beat to ack), we force the
      // UI back to idle below.
      try {
        wsRef.current.abort()
      } catch (e) {
        console.warn('cancelInFlight: WS abort failed', e)
      }
      wsRef.current = null
    }
    if (restAbortRef.current) {
      // file/url REST — XHR.abort() rejects with AbortError; the
      // promise's `finally` would normally clear busy, but explicitly
      // doing it here keeps the button responsive.
      try {
        restAbortRef.current.abort()
      } catch (e) {
        console.warn('cancelInFlight: REST abort failed', e)
      }
      restAbortRef.current = null
    }
    // Force the UI back to idle. The onClose handler also runs setBusyAll(false)
    // but we don't want to wait for the server to ack the close frame —
    // the user clicked Stop, the UI should respond now.
    cancelPendingPartial()
    setBusyAll(false)
    setProgress(null)
  }, [mode, setBusyAll, cancelPendingPartial])

  // Plain-text labels — the pill's accent fill conveys active state,
  // we don't need unicode glyphs to decorate them. The "Transcribe"
  // label gets a small document icon at render time.
  const transcribeLabel = (() => {
    if (busy) return mode === 'mic' ? 'Working' : 'Stop'
    if (mode === 'mic') {
      if (recording) return 'Stop'
      if (s.micCaptureMode === 'record' && pickedFile) return 'Transcribe'
      return 'Record'
    }
    return 'Transcribe'
  })()
  const transcribeIsAccent = busy || recording
  const transcribeShowsIcon = transcribeLabel === 'Transcribe'
  const onCommitClick = () => {
    // Anything in flight (REST upload for file / url / mic-record, or a
    // file+progressive WS stream) → the button acts as a hard cancel.
    if (busy) {
      cancelInFlight()
      return
    }
    // A live mic recording → Stop means "finish & transcribe the rest".
    if (mode === 'mic' && recording) {
      mic.stop()
      return
    }
    void transcribe()
  }
  const transcribeDisabled = (() => {
    // While busy on file/url, keep the button enabled so it can act as Stop.
    if (busy) return mode === 'mic'
    if (mode === 'file') return !pickedFile
    if (mode === 'url') return !urlInput.trim() || urlInput === 'https://'
    if (mode === 'mic') return micBusy
    return false
  })()

  const granDisabled = s.responseFormat !== 'verbose_json'
  const toggleGran = (g: TimestampGranularity) => {
    if (granDisabled) return
    const set = new Set(s.timestampGranularities)
    if (set.has(g)) set.delete(g)
    else set.add(g)
    if (set.size === 0) set.add('segment')
    s.set('timestampGranularities', Array.from(set))
  }

  return (
    <div className="sb">
      {/* Section tabs */}
      <div className="sb__tabs-row">
        <div className="ma-tabs">
          {(['source', 'output', 'engine'] as SidebarTab[]).map((id) => (
            <button
              key={id}
              type="button"
              className={tab === id ? 'ma-tab ma-tab--active' : 'ma-tab'}
              onClick={() => setTab(id)}
            >
              {id}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      <div className="sb__content">
        {tab === 'source' && (
          <SourcePane
            mode={mode}
            onModeChange={onModeChange}
            pickedFile={pickedFile}
            setPickedFile={setPickedFile}
            dragOver={dragOver}
            setDragOver={setDragOver}
            fileInputRef={fileInputRef}
            onFiles={onFiles}
            urlInput={urlInput}
            setUrlInput={setUrlInput}
            mic={mic}
            recording={recording}
            recordElapsed={recordElapsed}
            micCaptureMode={s.micCaptureMode}
            setMicCaptureMode={(v) => s.set('micCaptureMode', v)}
          />
        )}
        {tab === 'output' && (
          <OutputPane
            format={s.responseFormat}
            setFormat={(v) => s.set('responseFormat', v)}
            granularities={s.timestampGranularities}
            granDisabled={granDisabled}
            toggleGran={toggleGran}
          />
        )}
        {tab === 'engine' && (
          <EnginePane
            strategy={s.strategy}
            setStrategy={(v) => s.set('strategy', v)}
            mode={mode}
            micCaptureMode={s.micCaptureMode}
            longAudioThreshold={s.longAudioThreshold}
            liveLatency={s.liveLatency}
            progressiveRefinement={s.progressiveRefinement}
            vadEnabled={s.vadEnabled}
            vadThreshold={s.vadThreshold}
            vadConsecutive={s.vadConsecutive}
            vadHangoverMs={s.vadHangoverMs}
            vadPadMinGapMs={s.vadPadMinGapMs}
            vadPadDurationMs={s.vadPadDurationMs}
            hpfHz={s.hpfHz}
            noiseSuppression={s.noiseSuppression}
            setLong={(v) => s.set('longAudioThreshold', v)}
            setLiveLatency={(v) => s.set('liveLatency', v)}
            setProgRefine={(v) => s.set('progressiveRefinement', v)}
            setVadEnabled={(v) => s.set('vadEnabled', v)}
            setVadThreshold={(v) => s.set('vadThreshold', v ?? 0.5)}
            setVadConsecutive={(v) => s.set('vadConsecutive', v ?? 3)}
            setVadHangoverMs={(v) => s.set('vadHangoverMs', v ?? 500)}
            setVadPadMinGap={(v) => s.set('vadPadMinGapMs', v ?? 400)}
            setVadPadDuration={(v) => s.set('vadPadDurationMs', v ?? 0)}
            setHpfHz={(v) => s.set('hpfHz', v ?? 100)}
            setNoiseSuppression={(v) => s.set('noiseSuppression', v)}
          />
        )}
      </div>

      {progress !== null && (
        <>
          <div className="sb__progress">
            <div className="sb__progress-bar" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        </>
      )}

      {/* Commit button — pinned to bottom of the sidebar */}
      <div className="sb__commit">
        <button
          type="button"
          className={transcribeIsAccent ? 'ma-pill ma-pill--active' : 'ma-pill'}
          onClick={onCommitClick}
          disabled={transcribeDisabled}
        >
          {transcribeShowsIcon && (
            <svg
              viewBox="0 0 24 24"
              width="13"
              height="13"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
              <path d="M14 3v5h5" />
              <line x1="8.5" y1="12.5" x2="15" y2="12.5" />
              <line x1="8.5" y1="16" x2="14" y2="16" />
            </svg>
          )}
          {transcribeLabel}
        </button>
      </div>
      {mic.state === 'error' && mic.error && <div className="error">{mic.error}</div>}
    </div>
  )
}

// ── Section label ─────────────────────────────────────────────────
function SBLabel({ children, top }: { children: React.ReactNode; top?: number }) {
  return (
    <div className="label-eyebrow-row" style={{ marginTop: top ?? 0 }}>
      <span className="label-eyebrow">{children}</span>
      <span className="rule-extend" />
    </div>
  )
}

// ── Source pane ───────────────────────────────────────────────────
type MicReturn = ReturnType<typeof useMic>
function SourcePane({
  mode,
  onModeChange,
  pickedFile,
  setPickedFile,
  dragOver,
  setDragOver,
  fileInputRef,
  onFiles,
  urlInput,
  setUrlInput,
  mic,
  recording,
  recordElapsed,
  micCaptureMode,
  setMicCaptureMode,
}: {
  mode: InputMode
  onModeChange: (m: InputMode) => void
  pickedFile: File | null
  setPickedFile: (f: File | null) => void
  dragOver: boolean
  setDragOver: (b: boolean) => void
  fileInputRef: React.RefObject<HTMLInputElement | null>
  onFiles: (files: FileList | null) => void
  urlInput: string
  setUrlInput: (v: string) => void
  mic: MicReturn
  recording: boolean
  recordElapsed: number
  micCaptureMode: 'live' | 'record'
  setMicCaptureMode: (v: 'live' | 'record') => void
}) {
  const urlValid = /^https?:\/\/\S+/i.test(urlInput.trim()) && urlInput.trim() !== 'https://'
  return (
    <div>
      <SBLabel>Input</SBLabel>
      <OptionList<InputMode>
        value={mode}
        options={[
          { id: 'file', label: 'File upload' },
          { id: 'mic', label: 'Microphone' },
          { id: 'url', label: 'URL' },
        ]}
        onChange={onModeChange}
      />

      {mode === 'file' && (
        <div style={{ marginTop: 16 }}>
          <div
            className={dragOver ? 'sb__file-card sb__file-card--drag' : 'sb__file-card'}
            onClick={() => fileInputRef.current?.click()}
            onDragEnter={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={(e) => {
              e.preventDefault()
              setDragOver(false)
            }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              setDragOver(false)
              onFiles(e.dataTransfer.files)
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click()
            }}
          >
            <div className="sb__file-card__icon">
              <svg
                viewBox="0 0 24 24"
                width="16"
                height="16"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden
              >
                <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
                <path d="M14 3v5h5" />
              </svg>
            </div>
            <div className="sb__file-card__body">
              {pickedFile ? (
                <>
                  <div className="sb__file-card__name">{pickedFile.name}</div>
                  <div className="sb__file-card__meta">
                    {formatBytes(pickedFile.size)} · {pickedFile.type || 'audio'}
                  </div>
                </>
              ) : (
                <>
                  <div className="sb__file-card__name sb__file-card__name--empty">
                    Drop a file or browse
                  </div>
                  <div className="sb__file-card__meta">No file selected</div>
                </>
              )}
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,video/*"
            hidden
            onChange={(e) => onFiles(e.target.files)}
          />
          <div className="sb__file-actions">
            <button
              type="button"
              className="ma-pill ma-pill--grow"
              onClick={() => fileInputRef.current?.click()}
            >
              Browse…
            </button>
            {pickedFile && (
              <button
                type="button"
                className="ma-pill"
                aria-label="Remove file"
                onClick={(e) => {
                  e.stopPropagation()
                  setPickedFile(null)
                  if (fileInputRef.current) fileInputRef.current.value = ''
                }}
              >
                <svg
                  viewBox="0 0 24 24"
                  width="11"
                  height="11"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.7"
                  strokeLinecap="round"
                  aria-hidden
                >
                  <path d="M19 6L6 19M6 6l13 13" />
                </svg>
              </button>
            )}
          </div>
          <div className="sb__file-formats">
            .wav · .mp3 · .flac · .m4a · .ogg · .webm
          </div>
        </div>
      )}

      {mode === 'mic' && (
        <div style={{ marginTop: 16 }}>
          <SBLabel>Capture</SBLabel>
          <OptionList<'live' | 'record'>
            value={micCaptureMode}
            options={[
              { id: 'live', label: 'Live transcription' },
              { id: 'record', label: 'Recording' },
            ]}
            onChange={setMicCaptureMode}
          />

          <div className="mic-card" style={{ marginTop: 16 }}>
            <MicCardBars
              level={mic.level}
              recording={recording}
              captureMode={micCaptureMode}
            />
            <div className="mic-card__row">
              <span className="mic-card__status">
                {recording
                  ? 'Recording'
                  : micCaptureMode === 'live'
                    ? 'Listening'
                    : 'Standby'}
              </span>
              <span
                className={
                  recording || (micCaptureMode === 'live' && !recording)
                    ? 'mic-card__timer mic-card__timer--rec'
                    : 'mic-card__timer'
                }
              >
                {micCaptureMode === 'live' && !recording
                  ? '——'
                  : formatMicTimer(recordElapsed)}
              </span>
            </div>
          </div>

          {!recording && pickedFile && micCaptureMode === 'record' && (
            <div className="sb__file-card" style={{ marginTop: 10 }}>
              <div className="sb__file-card__icon">
                <svg
                  viewBox="0 0 24 24"
                  width="16"
                  height="16"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden
                >
                  <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
                  <path d="M14 3v5h5" />
                </svg>
              </div>
              <div className="sb__file-card__body">
                <div className="sb__file-card__name">{pickedFile.name}</div>
                <div className="sb__file-card__meta">
                  {formatBytes(pickedFile.size)} · WebM Opus
                </div>
              </div>
              <button
                type="button"
                className="ma-pill ma-pill--sm"
                aria-label="Discard recording"
                onClick={(e) => {
                  e.stopPropagation()
                  setPickedFile(null)
                }}
              >
                ✕
              </button>
            </div>
          )}

          <div className="mic-card__hint">16 kHz · Mono · WebM Opus</div>
        </div>
      )}

      {mode === 'url' && (
        <div style={{ marginTop: 16 }}>
          <div className="sb__url-field">
            <svg
              viewBox="0 0 24 24"
              width="14"
              height="14"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <path d="M10 13a5 5 0 0 0 7.07 0l3.18-3.18a5 5 0 0 0-7.07-7.07L11.34 5" />
              <path d="M14 11a5 5 0 0 0-7.07 0L3.75 14.18a5 5 0 0 0 7.07 7.07L12.66 19" />
            </svg>
            <input
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              className="sb__url-input"
              spellCheck={false}
              autoComplete="off"
              placeholder="https://…"
            />
          </div>
          <div className="sb__url-status">
            <span className={urlValid ? 'sb__url-dot sb__url-dot--on' : 'sb__url-dot'} />
            <span className="sb__url-status-label">
              {urlValid ? 'Ready to fetch' : 'Enter a media URL'}
            </span>
          </div>
          <div className="sb__file-formats">Direct link to an audio or video file</div>
        </div>
      )}
    </div>
  )
}

/** Bar meter inside the mic card. Real audio only — driven by
 * mic.level (the AnalyserNode peak), not a synthetic animation. The
 * meter is "active" both while recording AND while previewing (live
 * mode opens the mic before the user hits Record), so it reflects
 * actual speech in both cases:
 *
 *   - recording, OR live-mode preview → bars scale with level.
 *   - record-mode idle               → bars sit flat (mic not open).
 *
 * Each bar has a static center-hump envelope + per-bar texture so the
 * meter reads as a wave rather than a flat block; the overall
 * amplitude tracks loudness. Quiet input = short bars, speech = tall. */
function MicCardBars({
  level,
  recording,
  captureMode,
}: {
  level: number
  recording: boolean
  captureMode: 'live' | 'record'
}) {
  const N = 28
  const active = recording || captureMode === 'live'
  const drive = active ? Math.min(1, level * 2.4) : 0
  return (
    <div className="mic-card__bars">
      {Array.from({ length: N }).map((_, i) => {
        const envelope = 0.45 + 0.55 * Math.sin((i / (N - 1)) * Math.PI)
        const variety = 0.65 + 0.35 * Math.abs(Math.sin(i * 1.7 + 0.6))
        const h = active ? Math.max(4, drive * 86 * envelope * variety) : 4
        return (
          <span
            key={i}
            className={active ? 'mic-card__bar mic-card__bar--on' : 'mic-card__bar'}
            style={{ height: `${h}%`, opacity: active ? 0.55 + 0.45 * variety : 1 }}
          />
        )
      })}
    </div>
  )
}

/** Tabular MM:SS.t format for the mic card timer. */
function formatMicTimer(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  const t = Math.floor((seconds * 10) % 10)
  return `${m}:${String(s).padStart(2, '0')}.${t}`
}

// ── Output pane ───────────────────────────────────────────────────
function OutputPane({
  format,
  setFormat,
  granularities,
  granDisabled,
  toggleGran,
}: {
  format: ResponseFormat
  setFormat: (v: ResponseFormat) => void
  granularities: TimestampGranularity[]
  granDisabled: boolean
  toggleGran: (g: TimestampGranularity) => void
}) {
  return (
    <div>
      <SBLabel>Format</SBLabel>
      <OptionList<ResponseFormat>
        value={format}
        options={FORMATS.map((f) => ({ id: f.id, label: f.label }))}
        onChange={setFormat}
      />

      <SBLabel top={22}>
        Timestamps
        {granDisabled && (
          <span
            style={{
              marginLeft: 10,
              fontFamily: 'Newsreader, serif',
              fontStyle: 'italic',
              fontSize: 11,
              color: 'var(--muted-deep)',
              textTransform: 'none',
              letterSpacing: 0,
            }}
          >
            verbose_json only
          </span>
        )}
      </SBLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {(['segment', 'word'] as TimestampGranularity[]).map((id) => {
          const on = !granDisabled && granularities.includes(id)
          return (
            <button
              key={id}
              type="button"
              disabled={granDisabled}
              className={on ? 'ma-check ma-check--on' : 'ma-check'}
              onClick={() => toggleGran(id)}
            >
              <span className="ma-check__box">
                {on && (
                  <svg
                    viewBox="0 0 24 24"
                    width="9"
                    height="9"
                    fill="none"
                    stroke="oklch(0.135 0.012 60)"
                    strokeWidth="4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden
                  >
                    <path d="M5 12l5 5L20 7" />
                  </svg>
                )}
              </span>
              <span style={{ textTransform: 'capitalize' }}>{id}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── Engine pane ───────────────────────────────────────────────────
function EnginePane({
  strategy,
  setStrategy,
  mode,
  micCaptureMode,
  longAudioThreshold,
  liveLatency,
  progressiveRefinement,
  vadEnabled,
  vadThreshold,
  vadConsecutive,
  vadHangoverMs,
  vadPadMinGapMs,
  vadPadDurationMs,
  hpfHz,
  noiseSuppression,
  setLong,
  setLiveLatency,
  setProgRefine,
  setVadEnabled,
  setVadThreshold,
  setVadConsecutive,
  setVadHangoverMs,
  setVadPadMinGap,
  setVadPadDuration,
  setHpfHz,
  setNoiseSuppression,
}: {
  strategy: Strategy
  setStrategy: (v: Strategy) => void
  mode: InputMode
  micCaptureMode: 'live' | 'record'
  longAudioThreshold: number | null
  liveLatency: boolean
  progressiveRefinement: boolean
  vadEnabled: boolean
  vadThreshold: number | null
  vadConsecutive: number | null
  vadHangoverMs: number | null
  vadPadMinGapMs: number | null
  vadPadDurationMs: number | null
  hpfHz: number | null
  noiseSuppression: boolean
  setLong: (v: number | null) => void
  setLiveLatency: (v: boolean) => void
  setProgRefine: (v: boolean) => void
  setVadEnabled: (v: boolean) => void
  setVadThreshold: (v: number | null) => void
  setVadConsecutive: (v: number | null) => void
  setVadHangoverMs: (v: number | null) => void
  setVadPadMinGap: (v: number | null) => void
  setVadPadDuration: (v: number | null) => void
  setHpfHz: (v: number | null) => void
  setNoiseSuppression: (v: boolean) => void
}) {
  // Each setting only shows on the path that actually uses it, so we
  // never present a knob that silently does nothing.
  //
  // Two server paths:
  //  - STREAMING (WebSocket): mic+live, and file+progressive. Processes
  //    audio chunk-by-chunk as it arrives. Uses progressive_refinement
  //    (+ live_latency / VAD for mic). Does NOT use batch_size /
  //    chunk_length / long_audio_threshold — those are batch knobs.
  //  - REST UPLOAD: file (chunked/full/auto), url, mic+record. The
  //    whole file is decoded server-side, so batch_size / chunk_length
  //    / long_audio_threshold apply here.
  const isMicLive = mode === 'mic' && micCaptureMode === 'live'
  const isMicAny = mode === 'mic'
  const isStreaming = isMicLive || (mode === 'file' && strategy === 'progressive')
  const isRestUpload = !isStreaming
  return (
    <div>
      <SBLabel>Strategy</SBLabel>
      <OptionList<Strategy>
        value={strategy}
        options={STRATEGIES.map((id) => ({
          id,
          label: id,
          disabled: id === 'progressive' && mode === 'url',
          disabledReason: 'progressive isn’t available for URL ingest',
        }))}
        onChange={setStrategy}
      />
      {strategy === 'progressive' && mode === 'url' && (
        <div className="sb__hint">progressive isn’t available for URL ingest — falling back to chunked.</div>
      )}
      {strategy === 'progressive' && mode === 'file' && (
        <div className="sb__hint">file + progressive streams the audio over a WebSocket so partials arrive live.</div>
      )}

      <SBLabel top={22}>Advanced</SBLabel>
      <div className="sb__adv-list">
        {isRestUpload && (
          <>
            {/* long_audio_threshold is the only offline knob the engine
                actually honors (picks long-audio model settings). The
                chunked engine processes sequentially at a fixed internal
                chunk size, so client batch_size / chunk_length had no
                effect — removed rather than ship dead controls. */}
            <NumKv k="long_audio_threshold" v={longAudioThreshold} placeholder={480} suffix="s" onSet={setLong} />
          </>
        )}
        {isMicLive && <ToggleKv k="live_latency" v={liveLatency} onSet={setLiveLatency} />}
        {isStreaming && (
          <ToggleKv k="progressive_refinement" v={progressiveRefinement} onSet={setProgRefine} />
        )}

        {isMicAny && (
          <>
            <div className="sb__adv-divider" />

            {/* Mic input quality (browser-side getUserMedia constraints +
                ffmpeg highpass). Applies to both Live and Record. */}
            <NumKv k="hpf_hz" v={hpfHz} placeholder={100} suffix="Hz" onSet={setHpfHz} />
            <ToggleKv k="noise_suppression (browser)" v={noiseSuppression} onSet={setNoiseSuppression} />
          </>
        )}

        {isMicLive && (
          <>
            <div className="sb__adv-divider" />

            {/* VAD settings only apply to the live WS streaming path —
                Record mode uploads the raw blob via REST and ffmpeg
                doesn't touch VAD on the server. */}
            <ToggleKv k="vad_enabled" v={vadEnabled} onSet={setVadEnabled} />
            <NumKv k="vad_threshold" v={vadThreshold} placeholder={0.5} step={0.05} onSet={setVadThreshold} />
            <NumKv k="vad_consecutive" v={vadConsecutive} placeholder={3} onSet={setVadConsecutive} />
            <NumKv k="vad_hangover_ms" v={vadHangoverMs} placeholder={500} suffix="ms" onSet={setVadHangoverMs} />
            <NumKv k="vad_pad_min_gap_ms" v={vadPadMinGapMs} placeholder={400} suffix="ms" onSet={setVadPadMinGap} />
            <NumKv k="vad_pad_duration_ms" v={vadPadDurationMs} placeholder={0} suffix="ms" onSet={setVadPadDuration} />
          </>
        )}
      </div>
    </div>
  )
}

function NumKv({
  k,
  v,
  placeholder,
  suffix,
  step,
  onSet,
}: {
  k: string
  v: number | null
  placeholder: number
  suffix?: string
  step?: number
  onSet: (v: number | null) => void
}) {
  return (
    <div className="ma-kv">
      <span className="ma-kv__k">{k}</span>
      <span className="ma-kv__v">
        <input
          type="number"
          className="ma-kv__input"
          value={v ?? ''}
          placeholder={`${placeholder}`}
          step={step ?? 1}
          spellCheck={false}
          onChange={(e) => onSet(e.target.value === '' ? null : Number(e.target.value))}
        />
        <span className="ma-kv__unit">{suffix ?? ''}</span>
      </span>
    </div>
  )
}

function ToggleKv({ k, v, onSet }: { k: string; v: boolean; onSet: (v: boolean) => void }) {
  return (
    <div className="ma-kv">
      <span className="ma-kv__k">{k}</span>
      <span className="ma-kv__v" style={{ justifyContent: 'flex-start' }}>
        <button
          type="button"
          className={v ? 'ma-switch ma-switch--on' : 'ma-switch'}
          onClick={() => onSet(!v)}
          aria-pressed={v}
        >
          <span className="ma-switch__label ma-switch__label--off">Off</span>
          <span className="ma-switch__label ma-switch__label--on">On</span>
          <span className="ma-switch__knob" />
        </button>
      </span>
    </div>
  )
}
