import { useCallback, useRef, useState } from 'react'
import { postTranscription, type TranscriptionResponse } from '../lib/api'
import { useSettings } from '../lib/settings'
import { formatBytes } from '../lib/format'

interface Props {
  /** Called when a successful transcription completes. */
  onResult: (file: File, result: TranscriptionResponse) => void
  /** Called when a request errors. */
  onError: (message: string) => void
  /** Called when busy state changes (so the parent can set the streaming state). */
  onBusyChange?: (busy: boolean) => void
}

const UploadIcon = () => (
  <svg viewBox="0 0 24 24" width={18} height={18} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M12 16V4M12 4l-4 4M12 4l4 4" />
    <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
  </svg>
)
const SparkleIcon = ({ size = 14 }: { size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8" />
  </svg>
)
const ClearIcon = () => (
  <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
)
const FileIcon = () => (
  <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5" />
  </svg>
)

export function FileUploadPanel({ onResult, onError, onBusyChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [progress, setProgress] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const settings = useSettings()

  const pickFile = useCallback(() => inputRef.current?.click(), [])

  const onFiles = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return
    setFile(files[0])
    setProgress(null)
  }, [])

  const transcribe = useCallback(async () => {
    if (!file) return
    setBusy(true)
    onBusyChange?.(true)
    setProgress(0)
    try {
      const result = await postTranscription(
        file,
        {
          response_format: settings.responseFormat,
          timestamp_granularities: settings.timestampGranularities,
          strategy: settings.strategy,
          chunk_length: settings.chunkLength ?? undefined,
          chunk_overlap: settings.chunkOverlap ?? undefined,
          batch_size: settings.batchSize ?? undefined,
          long_audio_threshold: settings.longAudioThreshold ?? undefined,
        },
        {
          onProgress: (loaded, total) => setProgress(loaded / total),
        },
      )
      onResult(file, result)
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      onBusyChange?.(false)
      setProgress(null)
    }
  }, [file, settings, onResult, onError, onBusyChange])

  const clear = useCallback(() => {
    setFile(null)
    setProgress(null)
  }, [])

  const pct = Math.round((progress ?? 0) * 100)

  return (
    <section className="glass input-card">
      <div
        className={dragOver ? 'drop drop--active' : 'drop'}
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
        onClick={pickFile}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') pickFile()
        }}
      >
        <div className="drop__icon">
          <UploadIcon />
        </div>
        <div className="drop__title">
          Drop audio · or <span className="drop__browse">browse</span>
        </div>
        <div className="drop__hint mono">wav · mp3 · flac · m4a · ogg · webm</div>
        <input
          ref={inputRef}
          type="file"
          accept="audio/*,video/*"
          onChange={(e) => onFiles(e.target.files)}
          hidden
        />
      </div>

      {file && (
        <div className="file-row">
          <div className="file-row__icon">
            <FileIcon />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="file-row__name">{file.name}</div>
            <div className="file-row__meta">
              {formatBytes(file.size)} · {file.type || 'audio'}
            </div>
          </div>
          <button type="button" className="file-row__clear" onClick={clear} aria-label="Clear file">
            <ClearIcon />
          </button>
        </div>
      )}

      <button
        type="button"
        onClick={transcribe}
        disabled={!file || busy}
        className="btn-primary pk-glow-btn"
        style={
          {
            marginTop: 12,
            ['--btn-accent' as never]: 'var(--accent)',
            ['--top-hl' as never]: 'rgba(255,255,255,0.18)',
            ['--stroke-pct' as never]: '45%',
            ['--bottom-pct' as never]: '22%',
            ['--glow-r' as never]: '26px',
          } as React.CSSProperties
        }
      >
        <SparkleIcon /> {busy ? 'Transcribing…' : 'Transcribe'}
      </button>

      {progress !== null && (
        <>
          <div className="progress">
            <div className="progress__bar" style={{ width: `${pct}%` }} />
          </div>
          <div className="progress__label">
            <span>UPLOAD</span>
            <span>
              {pct}% · {pct >= 100 ? 'complete' : 'sending'}
            </span>
          </div>
        </>
      )}
    </section>
  )
}
