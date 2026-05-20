import { useCallback, useEffect, useRef, useState } from 'react'
import { postTranscription, postTranscriptionUrl, type TranscriptionResponse, connectLiveWS, type LiveWSHandle } from '../lib/api'
import { useSettings } from '../lib/settings'
import { MIC_FORMAT_HINT, MIC_SAMPLE_RATE, MIC_MIME_TYPE, useMic } from '../lib/mic'
import { formatBytes, formatTime } from '../lib/format'
import type { ResponseFormat, Strategy, TimestampGranularity, WhisperSegment, WSMessage } from '../lib/types'
import type { LoadedAudio } from '../lib/download'

export type InputMode = 'file' | 'mic' | 'url'

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
const CheckIcon = () => (
  <svg viewBox="0 0 24 24" width={8} height={8} fill="none" stroke="oklch(0.13 0.012 60)" strokeWidth={4} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M5 12l5 5L20 7" />
  </svg>
)
const MicIcon = ({ size = 22 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="9" y="3" width="6" height="12" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
  </svg>
)

const LEVEL_BARS = 22

export function Sidebar({ mode, onModeChange, onResult, onPartial, onError, onBusyChange }: Props) {
  const s = useSettings()
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
            { kind: 'file', title: `live recording — ${new Date().toLocaleTimeString()}`, source: 'live mic', file },
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

  // Drive Transcribe based on mode
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
            strategy: s.strategy,
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
          strategy: s.strategy,
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
      if (recording) {
        mic.stop()
      } else {
        startMic().catch((e) => onError(e instanceof Error ? e.message : String(e)))
      }
    }
  }, [mode, pickedFile, urlInput, s, recording, mic, startMic, onResult, onError, setBusyAll])

  const transcribeLabel = (() => {
    if (busy) return 'Working…'
    if (mode === 'mic') return recording ? 'Stop' : 'Record'
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
    const set = new Set(s.timestampGranularities)
    if (set.has(g)) set.delete(g)
    else set.add(g)
    if (set.size === 0) set.add('segment')
    s.set('timestampGranularities', Array.from(set))
  }

  return (
    <div className="sb">
      {/* Input section */}
      <Section label="Input" first>
        <div className="sb__list">
          {(['file', 'mic', 'url'] as InputMode[]).map((id) => (
            <Radio
              key={id}
              active={mode === id}
              onClick={() => onModeChange(id)}
              label={id === 'file' ? 'File' : id === 'mic' ? 'Live mic' : 'URL'}
            />
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
                <div>{pickedFile.name}</div>
                <div className="sb__file-info-meta">
                  {formatBytes(pickedFile.size)} · {pickedFile.type || 'audio'}
                </div>
              </div>
            )}
          </>
        )}

        {mode === 'mic' && (
          <div className="sb__mic">
            <button
              type="button"
              className={recording ? 'mic-btn mic-btn--recording pk-glow-btn' : 'mic-btn pk-glow-btn'}
              onClick={() => (recording ? mic.stop() : startMic())}
              disabled={micBusy}
              aria-label={recording ? 'Stop recording' : 'Start recording'}
              style={
                {
                  ['--btn-accent' as never]: 'var(--accent)',
                  ['--top-hl' as never]: recording ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.2)',
                  ['--stroke-pct' as never]: recording ? '55%' : '50%',
                  ['--bottom-pct' as never]: recording ? '0%' : '22%',
                  ['--glow-r' as never]: '20px',
                  ['--glow-pct' as never]: recording ? '50%' : '10%',
                } as React.CSSProperties
              }
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
            <div className="mic-meter">
              {Array.from({ length: LEVEL_BARS }).map((_, i) => {
                const k = i / (LEVEL_BARS - 1)
                const lvl = mic.level
                const jit = 0.55 + (i * 0.117) % 0.5
                const h = recording ? Math.max(8, Math.min(100, lvl * 110 * jit * (k * 0.6 + 0.7))) : 18
                const hot = recording && lvl * jit > 0.7
                const cls = !recording
                  ? 'mic-meter__bar'
                  : hot
                    ? 'mic-meter__bar mic-meter__bar--hot'
                    : 'mic-meter__bar mic-meter__bar--on'
                return <div key={i} className={cls} style={{ height: `${h}%` }} />
              })}
            </div>
            {mic.state === 'error' && mic.error && <div className="error">{mic.error}</div>}
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

        <button
          type="button"
          onClick={transcribe}
          disabled={transcribeDisabled}
          className="sb__transcribe pk-glow-btn"
          style={
            {
              ['--btn-accent' as never]: 'var(--accent)',
              ['--top-hl' as never]: 'rgba(255,255,255,0.18)',
              ['--stroke-pct' as never]: '45%',
              ['--bottom-pct' as never]: '22%',
              ['--glow-r' as never]: '18px',
            } as React.CSSProperties
          }
        >
          {transcribeLabel}
        </button>

        {progress !== null && (
          <>
            <div className="sb__progress">
              <div className="sb__progress-bar" style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
            <div className="sb__progress-label">
              <span>UPLOAD</span>
              <span>{Math.round(progress * 100)}%</span>
            </div>
          </>
        )}
      </Section>

      {/* Format */}
      <Section label="Format">
        <div className="sb__list">
          {FORMATS.map((f) => (
            <Radio
              key={f.id}
              mono
              active={s.responseFormat === f.id}
              onClick={() => s.set('responseFormat', f.id)}
              label={f.label}
            />
          ))}
        </div>
      </Section>

      {/* Strategy — gate `progressive` on mic mode */}
      <Section label="Strategy">
        <div className="sb__list">
          {STRATEGIES.map((id) => {
            const disabled = id === 'progressive' && mode !== 'mic'
            return (
              <Radio
                key={id}
                mono
                active={s.strategy === id}
                onClick={() => !disabled && s.set('strategy', id)}
                disabled={disabled}
                title={disabled ? 'progressive requires Live mic (WebSocket)' : undefined}
                label={id}
              />
            )
          })}
        </div>
        {s.strategy === 'progressive' && mode !== 'mic' && (
          <div className="sb__hint">progressive requires Live mic — REST modes will use chunked.</div>
        )}
      </Section>

      {/* Timestamps */}
      <Section label="Timestamps">
        <div className="sb__list">
          {(['segment', 'word'] as TimestampGranularity[]).map((id) => (
            <Check
              key={id}
              active={s.timestampGranularities.includes(id)}
              disabled={granDisabled}
              onClick={() => toggleGran(id)}
              label={id === 'segment' ? 'Segment' : 'Word'}
            />
          ))}
        </div>
        {granDisabled && <div className="sb__hint">verbose_json only</div>}
      </Section>

      {/* Advanced */}
      <Section label="Advanced">
        <button type="button" className="sb__adv-toggle" onClick={() => setAdvOpen((o) => !o)}>
          <ChevIcon up={advOpen} />
          <span>{advOpen ? 'collapse' : 'expand'}</span>
        </button>
        {advOpen && (
          <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
            <NumKv
              k="long_audio_threshold"
              v={s.longAudioThreshold}
              defaultV={480}
              onSet={(v) => s.set('longAudioThreshold', v)}
              suffix="s"
            />
            <NumKv
              k="batch_size"
              v={s.batchSize}
              defaultV={4}
              onSet={(v) => s.set('batchSize', v)}
            />
            <NumKv
              k="chunk_length"
              v={s.chunkLength}
              defaultV={30}
              onSet={(v) => s.set('chunkLength', v)}
              suffix="s"
            />
            <ToggleKv
              k="live_latency"
              v={s.liveLatency}
              onSet={(v) => s.set('liveLatency', v)}
            />
            <ToggleKv
              k="progressive_refinement"
              v={s.progressiveRefinement}
              onSet={(v) => s.set('progressiveRefinement', v)}
            />
            <button
              type="button"
              onClick={s.reset}
              style={{
                marginTop: 4,
                padding: '6px 10px',
                background: 'transparent',
                border: '1px solid var(--rule)',
                color: 'var(--fg-dim)',
                fontSize: 10.5,
                letterSpacing: 0.7,
                textTransform: 'uppercase',
                fontFamily: 'var(--font-mono)',
                cursor: 'pointer',
                alignSelf: 'flex-start',
                borderRadius: 4,
              }}
            >
              Reset defaults
            </button>
          </div>
        )}
      </Section>
    </div>
  )
}

function Section({ label, children, first }: { label: string; children: React.ReactNode; first?: boolean }) {
  return (
    <div className={first ? 'sb__section sb__section--first' : 'sb__section'}>
      <div className="sb__label">{label}</div>
      {children}
    </div>
  )
}

function Radio({
  active,
  onClick,
  disabled,
  label,
  mono,
  title,
}: {
  active: boolean
  onClick: () => void
  disabled?: boolean
  label: string
  mono?: boolean
  title?: string
}) {
  return (
    <button
      type="button"
      className={`${active ? 'sb-radio sb-radio--active' : 'sb-radio'}${mono ? ' sb-radio--mono' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={title}
    >
      <span>{label}</span>
      <span className="sb-radio__dot" />
    </button>
  )
}

function Check({
  active,
  onClick,
  disabled,
  label,
}: {
  active: boolean
  onClick: () => void
  disabled?: boolean
  label: string
}) {
  return (
    <button
      type="button"
      className={active ? 'sb-check sb-check--active' : 'sb-check'}
      onClick={onClick}
      disabled={disabled}
    >
      <span className="sb-check__box">{active && <CheckIcon />}</span>
      <span>{label}</span>
    </button>
  )
}

function NumKv({
  k,
  v,
  defaultV,
  onSet,
  suffix = '',
}: {
  k: string
  v: number | null
  defaultV: number
  onSet: (v: number | null) => void
  suffix?: string
}) {
  return (
    <div className="sb__kv">
      <span className="sb__kv-k">{k}</span>
      <input
        type="number"
        value={v ?? ''}
        placeholder={`${defaultV}`}
        onChange={(e) => onSet(e.target.value === '' ? null : Number(e.target.value))}
        style={{
          width: 80,
          background: 'transparent',
          border: 'none',
          borderBottom: '1px solid var(--rule)',
          color: 'var(--fg-dim)',
          fontFamily: 'var(--font-mono)',
          fontVariantNumeric: 'tabular-nums',
          fontSize: 11,
          letterSpacing: 0.2,
          padding: '2px 0',
          outline: 'none',
          textAlign: 'right',
        }}
      />
      {suffix && <span className="sb__kv-v" style={{ marginLeft: -4, opacity: 0.6 }}>{suffix}</span>}
    </div>
  )
}

function ToggleKv({ k, v, onSet }: { k: string; v: boolean; onSet: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onSet(!v)}
      className="sb__kv"
      style={{ background: 'transparent', border: 'none', padding: 0, cursor: 'pointer', textAlign: 'left' }}
    >
      <span className="sb__kv-k">{k}</span>
      <span className="sb__kv-v" style={{ color: v ? 'var(--accent)' : 'var(--muted)' }}>
        {v ? 'on' : 'off'}
      </span>
    </button>
  )
}
