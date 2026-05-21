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
import { formatBytes, formatTime } from '../lib/format'
import type { ResponseFormat, Strategy, TimestampGranularity, WhisperSegment, Word, WSMessage } from '../lib/types'
import type { LoadedAudio } from '../lib/download'

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
}

const FORMATS: { id: ResponseFormat; label: string }[] = [
  { id: 'verbose_json', label: 'verbose_json' },
  { id: 'json', label: 'json' },
  { id: 'text', label: 'text' },
  { id: 'srt', label: 'srt' },
  { id: 'vtt', label: 'vtt' },
]
const STRATEGIES: Strategy[] = ['auto', 'full', 'chunked', 'progressive']

const ChevIcon = ({ up }: { up: boolean }) => (
  <svg
    viewBox="0 0 24 24"
    width={11}
    height={11}
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ transform: up ? 'rotate(180deg)' : 'none', transition: 'transform .15s' }}
    aria-hidden
  >
    <path d="M6 9l6 6 6-6" />
  </svg>
)
const MicIcon = ({ size = 22 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="9" y="3" width="6" height="12" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
  </svg>
)

export function Sidebar({ mode, onModeChange, onResult, onPartial, onError, onBusyChange, onSessionStart, onAudioReady, onPeaks }: Props) {
  const s = useSettings()
  const [tab, setTab] = useState<SidebarTab>('source')
  const [advOpen, setAdvOpen] = useState(false)
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
          schedulePartial()
          break
        }
        case 'refined_transcription':
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
          break
        case 'final_transcription': {
          cancelPendingPartial()
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
    [onPartial, onResult, onError, schedulePartial, cancelPendingPartial],
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

  const startMic = useCallback(async () => {
    onSessionStart?.()
    cancelPendingPartial()
    segmentsRef.current = []
    wordsRef.current = []
    chunksRef.current = []
    recordStartRef.current = Date.now()
    setRecordElapsed(0)
    // Honor the user's strategy choice. Live mic + chunked/full just means
    // "accumulate then process" — no partials, slow. progressive is the
    // real-time path. auto routes to progressive over WS.
    const wsStrategy: Strategy =
      s.strategy === 'auto' || s.strategy === 'progressive' ? 'progressive' : s.strategy
    const ws = connectLiveWS(
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
          wsRef.current = null
        },
      },
    )
    wsRef.current = ws
    await mic.start()
  }, [s, handleMessage, mic, onError, onSessionStart])

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
          const ws = streamFileViaWS(
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
                wsRef.current = null
                setBusyAll(false)
              },
            },
          )
          wsRef.current = ws
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
      if (recording) mic.stop()
      else startMic().catch((e) => onError(e instanceof Error ? e.message : String(e)))
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

  const transcribeLabel = (() => {
    if (busy) return mode === 'mic' ? 'Working…' : 'Stop ▣'
    if (mode === 'mic') return recording ? 'Stop ▣' : 'Record ●'
    return 'Transcribe →'
  })()
  const onCommitClick = () => {
    if (busy && mode !== 'mic') {
      cancelInFlight()
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
            micBusy={micBusy}
            recordElapsed={recordElapsed}
            startMic={startMic}
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
            advOpen={advOpen}
            setAdvOpen={setAdvOpen}
            longAudioThreshold={s.longAudioThreshold}
            batchSize={s.batchSize}
            chunkLength={s.chunkLength}
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
            setBatch={(v) => s.set('batchSize', v)}
            setChunkLen={(v) => s.set('chunkLength', v)}
            setLiveLatency={(v) => s.set('liveLatency', v)}
            setProgRefine={(v) => s.set('progressiveRefinement', v)}
            setVadEnabled={(v) => s.set('vadEnabled', v)}
            setVadThreshold={(v) => s.set('vadThreshold', v ?? 0.5)}
            setVadConsecutive={(v) => s.set('vadConsecutive', v ?? 3)}
            setVadHangoverMs={(v) => s.set('vadHangoverMs', v ?? 500)}
            setVadPadMinGap={(v) => s.set('vadPadMinGapMs', v ?? 400)}
            setVadPadDuration={(v) => s.set('vadPadDurationMs', v ?? 250)}
            setHpfHz={(v) => s.set('hpfHz', v ?? 100)}
            setNoiseSuppression={(v) => s.set('noiseSuppression', v)}
            ChevIcon={ChevIcon}
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
          className="ma-pill ma-pill--primary"
          onClick={onCommitClick}
          disabled={transcribeDisabled}
        >
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
  setPickedFile: _setPickedFile,
  dragOver,
  setDragOver,
  fileInputRef,
  onFiles,
  urlInput,
  setUrlInput,
  mic,
  recording,
  micBusy,
  recordElapsed,
  startMic,
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
  micBusy: boolean
  recordElapsed: number
  startMic: () => Promise<void>
}) {
  return (
    <div>
      <SBLabel>Input</SBLabel>
      <div className="ma-segmented ma-segmented--full">
        {(['file', 'mic', 'url'] as InputMode[]).map((id) => (
          <button
            key={id}
            type="button"
            className={mode === id ? 'ma-pill ma-pill--active' : 'ma-pill'}
            onClick={() => onModeChange(id)}
          >
            {id === 'file' ? 'file' : id === 'mic' ? 'live mic' : 'url'}
          </button>
        ))}
      </div>

      {mode === 'file' && (
        <>
          <div
            className={dragOver ? 'sb__drop sb__drop--active' : 'sb__drop'}
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
            <div className="sb__drop-title">
              Drop file · or <span className="sb__drop-browse">browse</span>
            </div>
            <div className="sb__drop-hint">wav · mp3 · flac · m4a · ogg · webm</div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,video/*"
            hidden
            onChange={(e) => onFiles(e.target.files)}
          />
          {pickedFile && (
            <div className="sb__file-info">
              <span className="sb__file-info-name">{pickedFile.name}</span>
              <span className="sb__file-info-meta">
                {formatBytes(pickedFile.size)} · {pickedFile.type || 'audio'}
              </span>
            </div>
          )}
        </>
      )}

      {mode === 'mic' && (
        <div className="sb__mic">
          <button
            type="button"
            className={recording ? 'mic-btn mic-btn--recording' : 'mic-btn'}
            onClick={() => (recording ? mic.stop() : startMic())}
            disabled={micBusy}
            aria-label={recording ? 'Stop recording' : 'Start recording'}
          >
            {recording ? <span className="mic-btn__square" /> : <MicIcon />}
            {recording && <span className="mic-btn__halo" />}
          </button>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <span className="mic__status">
              {mic.state === 'idle' && 'READY'}
              {mic.state === 'starting' && 'CONNECTING'}
              {mic.state === 'recording' && 'RECORDING'}
              {mic.state === 'stopping' && 'FINALIZING'}
              {mic.state === 'error' && 'ERROR'}
            </span>
            <span className={recording ? 'mic__timer mic__timer--on num' : 'mic__timer num'}>
              {formatTime(recordElapsed)}
            </span>
          </div>
          <SidebarMicMeter level={mic.level} recording={recording} />
          <div className="sb__mic-hint">Tap above to start. Tap Transcribe at the bottom to stop.</div>
        </div>
      )}

      {mode === 'url' && (
        <input
          type="url"
          value={urlInput}
          onChange={(e) => setUrlInput(e.target.value)}
          className="sb__url-input"
          spellCheck={false}
          autoComplete="off"
          placeholder="https://…"
        />
      )}
    </div>
  )
}

function SidebarMicMeter({ level, recording }: { level: number; recording: boolean }) {
  return (
    <div className="mic-meter">
      {Array.from({ length: 22 }).map((_, i) => {
        const k = i / 21
        const jit = 0.55 + ((i * 11.7) % 50) / 100
        const h = recording ? Math.max(8, Math.min(100, level * 110 * jit * (k * 0.6 + 0.7))) : 18
        const hot = recording && level * jit > 0.7
        const cls = !recording
          ? 'mic-meter__bar'
          : hot
            ? 'mic-meter__bar mic-meter__bar--hot'
            : 'mic-meter__bar mic-meter__bar--on'
        return <div key={i} className={cls} style={{ height: `${h}%` }} />
      })}
    </div>
  )
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
      <div className="ma-cluster">
        {FORMATS.map((f) => (
          <button
            key={f.id}
            type="button"
            className={format === f.id ? 'ma-pill ma-pill--active' : 'ma-pill'}
            onClick={() => setFormat(f.id)}
          >
            {f.label}
          </button>
        ))}
      </div>

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
      <div className="ma-segmented ma-segmented--full">
        {(['segment', 'word'] as TimestampGranularity[]).map((id) => {
          const on = !granDisabled && granularities.includes(id)
          return (
            <button
              key={id}
              type="button"
              disabled={granDisabled}
              className={on ? 'ma-pill ma-pill--active' : 'ma-pill'}
              onClick={() => toggleGran(id)}
            >
              {id}
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
  advOpen,
  setAdvOpen,
  longAudioThreshold,
  batchSize,
  chunkLength,
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
  setBatch,
  setChunkLen,
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
  ChevIcon,
}: {
  strategy: Strategy
  setStrategy: (v: Strategy) => void
  mode: InputMode
  advOpen: boolean
  setAdvOpen: (b: boolean) => void
  longAudioThreshold: number | null
  batchSize: number | null
  chunkLength: number | null
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
  setBatch: (v: number | null) => void
  setChunkLen: (v: number | null) => void
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
  ChevIcon: React.FC<{ up: boolean }>
}) {
  return (
    <div>
      <SBLabel>Strategy</SBLabel>
      <div className="ma-cluster">
        {STRATEGIES.map((id) => {
          // progressive: allowed for file (we stream over WS) and mic.
          // Not available for url (server fetches the audio — no WS path).
          const disabled = id === 'progressive' && mode === 'url'
          return (
            <button
              key={id}
              type="button"
              disabled={disabled}
              title={disabled ? 'progressive isn’t available for URL ingest' : undefined}
              className={strategy === id ? 'ma-pill ma-pill--active' : 'ma-pill'}
              onClick={() => !disabled && setStrategy(id)}
            >
              {id}
            </button>
          )
        })}
      </div>
      {strategy === 'progressive' && mode === 'url' && (
        <div className="sb__hint">progressive isn’t available for URL ingest — falling back to chunked.</div>
      )}
      {strategy === 'progressive' && mode === 'file' && (
        <div className="sb__hint">file + progressive streams the audio over a WebSocket so partials arrive live.</div>
      )}

      <div style={{ marginTop: 22 }}>
        <button type="button" className="sb__adv-toggle" onClick={() => setAdvOpen(!advOpen)}>
          <ChevIcon up={advOpen} />
          <span>Advanced · {advOpen ? 'hide' : 'show'}</span>
        </button>
        {advOpen && (
          <div className="sb__adv-body">
            <NumKv k="long_audio_threshold" v={longAudioThreshold} placeholder={480} suffix="s" onSet={setLong} />
            <NumKv k="batch_size" v={batchSize} placeholder={4} onSet={setBatch} />
            <NumKv k="chunk_length" v={chunkLength} placeholder={30} suffix="s" onSet={setChunkLen} />
            <ToggleKv k="live_latency" v={liveLatency} onSet={setLiveLatency} />
            <ToggleKv k="progressive_refinement" v={progressiveRefinement} onSet={setProgRefine} />

            <div className="sb__adv-divider" />

            <ToggleKv k="vad_enabled" v={vadEnabled} onSet={setVadEnabled} />
            {mode !== 'mic' && vadEnabled && (
              <div className="sb__hint">Auto-disabled for File and URL modes — the engine queue isn’t bandwidth-limited there, so VAD only adds per-frame CPU cost.</div>
            )}
            <NumKv k="vad_threshold" v={vadThreshold} placeholder={0.5} step={0.05} onSet={setVadThreshold} />
            <NumKv k="vad_consecutive" v={vadConsecutive} placeholder={3} onSet={setVadConsecutive} />
            <NumKv k="vad_hangover_ms" v={vadHangoverMs} placeholder={500} suffix="ms" onSet={setVadHangoverMs} />
            <NumKv k="vad_pad_min_gap_ms" v={vadPadMinGapMs} placeholder={400} suffix="ms" onSet={setVadPadMinGap} />
            <NumKv k="vad_pad_duration_ms" v={vadPadDurationMs} placeholder={250} suffix="ms" onSet={setVadPadDuration} />
            <NumKv k="hpf_hz" v={hpfHz} placeholder={100} suffix="Hz" onSet={setHpfHz} />
            <ToggleKv k="noise_suppression (browser)" v={noiseSuppression} onSet={setNoiseSuppression} />
          </div>
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
    <div className="sb__kv">
      <span className="sb__kv-k">{k}</span>
      <input
        type="number"
        value={v ?? ''}
        placeholder={`${placeholder}${suffix ?? ''}`}
        step={step ?? 1}
        onChange={(e) => onSet(e.target.value === '' ? null : Number(e.target.value))}
        className="sb__kv-v"
        style={{ background: 'transparent' }}
      />
    </div>
  )
}

function ToggleKv({ k, v, onSet }: { k: string; v: boolean; onSet: (v: boolean) => void }) {
  return (
    <div className="sb__kv">
      <span className="sb__kv-k">{k}</span>
      <button
        type="button"
        className={v ? 'sb__kv-v sb__kv-v--toggle is-on' : 'sb__kv-v sb__kv-v--toggle'}
        onClick={() => onSet(!v)}
      >
        {v ? 'on' : 'off'}
      </button>
    </div>
  )
}
