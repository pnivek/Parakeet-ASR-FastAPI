import { useCallback, useEffect, useRef, useState } from 'react'
import {
  postTranscription,
  postTranscriptionUrl,
  type TranscriptionResponse,
  connectLiveWS,
  type LiveWSHandle,
} from '../lib/api'
import { useSettings } from '../lib/settings'
import { MIC_FORMAT_HINT, MIC_SAMPLE_RATE, MIC_MIME_TYPE, useMic } from '../lib/mic'
import { formatBytes, formatTime } from '../lib/format'
import type { ResponseFormat, Strategy, TimestampGranularity, WhisperSegment, WSMessage } from '../lib/types'
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

export function Sidebar({ mode, onModeChange, onResult, onPartial, onError, onBusyChange }: Props) {
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
    setPickedFile(files[0])
  }

  // URL state
  const [urlInput, setUrlInput] = useState('https://')

  // Mic state — live capture
  const wsRef = useRef<LiveWSHandle | null>(null)
  const segmentsRef = useRef<WhisperSegment[]>([])
  const chunksRef = useRef<Blob[]>([])
  const recordStartRef = useRef<number>(0)
  const [recordElapsed, setRecordElapsed] = useState(0)

  // ── Auto-drop progressive when mode leaves mic ───────────────────
  useEffect(() => {
    if (mode !== 'mic' && s.strategy === 'progressive') {
      s.set('strategy', 'auto')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode])
  const restStrategy = (): Strategy =>
    mode === 'mic' ? s.strategy : s.strategy === 'progressive' ? 'chunked' : s.strategy

  // ── WS message → result/partial dispatch ─────────────────────────
  const handleMessage = useCallback(
    (msg: WSMessage) => {
      switch (msg.type) {
        case 'segments_batch': {
          segmentsRef.current = [...segmentsRef.current, ...msg.segments]
          const wall = (Date.now() - recordStartRef.current) / 1000
          onPartial({
            format: 'verbose_json',
            body: {
              task: 'transcribe',
              language: 'en',
              duration: segmentsRef.current[segmentsRef.current.length - 1]?.end ?? 0,
              text: segmentsRef.current.map((x) => x.text).join(' ').trim(),
              segments: segmentsRef.current,
              strategy: 'progressive',
              transcription_time_seconds: wall,
            },
          })
          break
        }
        case 'refined_transcription':
          segmentsRef.current = msg.segments
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
                strategy: msg.strategy,
                transcription_time_seconds: msg.transcription_time,
                csv_content: msg.csv_content,
                srt_content: msg.srt_content,
              },
            },
          )
          break
        }
        case 'error':
          onError(msg.error)
          break
      }
    },
    [onPartial, onResult, onError],
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
    segmentsRef.current = []
    chunksRef.current = []
    recordStartRef.current = Date.now()
    setRecordElapsed(0)
    const ws = connectLiveWS(
      {
        sample_rate: MIC_SAMPLE_RATE,
        channels: 1,
        bytes_per_sample: 2,
        format: MIC_FORMAT_HINT,
        strategy: 'progressive',
        live_latency: s.liveLatency,
        progressive_refinement: s.progressiveRefinement,
        chunk_length: s.chunkLength ?? undefined,
        chunk_overlap: s.chunkOverlap ?? undefined,
        batch_size: s.batchSize ?? undefined,
        long_audio_threshold: s.longAudioThreshold ?? undefined,
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
  }, [s, handleMessage, mic, onError])

  // Drive the bottom Transcribe button. In mic mode it doubles as
  // record/stop. In file/url mode it kicks off the REST upload.
  const transcribe = useCallback(async () => {
    if (mode === 'file') {
      if (!pickedFile) return
      setBusyAll(true)
      setProgress(0)
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
          { onProgress: (loaded, total) => setProgress(loaded / total) },
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
        onError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusyAll(false)
        setProgress(null)
      }
    } else if (mode === 'url') {
      if (!urlInput.trim() || urlInput === 'https://') return
      setBusyAll(true)
      try {
        const r = await postTranscriptionUrl(urlInput.trim(), {
          response_format: s.responseFormat,
          timestamp_granularities: s.timestampGranularities,
          strategy: restStrategy(),
          chunk_length: s.chunkLength ?? undefined,
          chunk_overlap: s.chunkOverlap ?? undefined,
          batch_size: s.batchSize ?? undefined,
          long_audio_threshold: s.longAudioThreshold ?? undefined,
        })
        let name = urlInput
        try {
          const u = new URL(urlInput)
          name = u.pathname.split('/').filter(Boolean).pop() || u.host
        } catch {
          /* keep input */
        }
        onResult({ kind: 'url', title: name, source: 'via URL', url: urlInput.trim() }, r)
      } catch (e) {
        onError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusyAll(false)
      }
    } else if (mode === 'mic') {
      if (recording) mic.stop()
      else startMic().catch((e) => onError(e instanceof Error ? e.message : String(e)))
    }
  }, [mode, pickedFile, urlInput, s, recording, mic, startMic, onResult, onError, setBusyAll])

  const transcribeLabel = (() => {
    if (busy) return 'Working…'
    if (mode === 'mic') return recording ? 'Stop ▣' : 'Record ●'
    return 'Transcribe →'
  })()
  const transcribeDisabled = (() => {
    if (busy) return true
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
            setLong={(v) => s.set('longAudioThreshold', v)}
            setBatch={(v) => s.set('batchSize', v)}
            setChunkLen={(v) => s.set('chunkLength', v)}
            setLiveLatency={(v) => s.set('liveLatency', v)}
            setProgRefine={(v) => s.set('progressiveRefinement', v)}
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
          onClick={transcribe}
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
  setLong,
  setBatch,
  setChunkLen,
  setLiveLatency,
  setProgRefine,
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
  setLong: (v: number | null) => void
  setBatch: (v: number | null) => void
  setChunkLen: (v: number | null) => void
  setLiveLatency: (v: boolean) => void
  setProgRefine: (v: boolean) => void
  ChevIcon: React.FC<{ up: boolean }>
}) {
  return (
    <div>
      <SBLabel>Strategy</SBLabel>
      <div className="ma-cluster">
        {STRATEGIES.map((id) => {
          const disabled = id === 'progressive' && mode !== 'mic'
          return (
            <button
              key={id}
              type="button"
              disabled={disabled}
              title={disabled ? 'progressive requires Live mic (WebSocket)' : undefined}
              className={strategy === id ? 'ma-pill ma-pill--active' : 'ma-pill'}
              onClick={() => !disabled && setStrategy(id)}
            >
              {id}
            </button>
          )
        })}
      </div>
      {strategy === 'progressive' && mode !== 'mic' && (
        <div className="sb__hint">progressive requires Live mic — REST uploads use chunked.</div>
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
  onSet,
}: {
  k: string
  v: number | null
  placeholder: number
  suffix?: string
  onSet: (v: number | null) => void
}) {
  return (
    <div className="sb__kv">
      <span className="sb__kv-k">{k}</span>
      <input
        type="number"
        value={v ?? ''}
        placeholder={`${placeholder}${suffix ?? ''}`}
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
