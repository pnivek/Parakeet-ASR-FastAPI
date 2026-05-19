import { useCallback, useRef, useState } from 'react'
import { postTranscription, type TranscriptionResponse } from '../lib/api'
import { useSettings } from '../lib/settings'

interface Props {
  /** Called when a successful transcription completes. */
  onResult: (file: File, result: TranscriptionResponse) => void
  /** Called when a request errors. */
  onError: (message: string) => void
}

/**
 * Drag-and-drop or click-to-pick file upload, with a single Transcribe button
 * that POSTs to /v1/audio/transcriptions using the current settings store.
 * Upload progress is shown as a thin bar (we use XHR under the hood).
 */
export function FileUploadPanel({ onResult, onError }: Props) {
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
      setProgress(null)
    }
  }, [file, settings, onResult, onError])

  return (
    <section
      onDragEnter={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={(e) => {
        e.preventDefault()
        setDragOver(false)
      }}
      onDragOver={(e) => {
        e.preventDefault()
      }}
      onDrop={(e) => {
        e.preventDefault()
        setDragOver(false)
        onFiles(e.dataTransfer.files)
      }}
      className={`drop ${dragOver ? 'drop--active' : ''}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,video/*"
        onChange={(e) => onFiles(e.target.files)}
        hidden
      />
      {file ? (
        <div className="drop__file">
          <div className="drop__filename">{file.name}</div>
          <div className="drop__filemeta">
            {(file.size / 1024 / 1024).toFixed(2)} MB · {file.type || 'unknown type'}
          </div>
          <div className="drop__actions">
            <button type="button" onClick={transcribe} disabled={busy} className="primary">
              {busy ? 'Transcribing…' : 'Transcribe'}
            </button>
            <button type="button" onClick={pickFile} disabled={busy}>
              Pick different file
            </button>
          </div>
          {progress !== null && (
            <div className="progress">
              <div className="progress__bar" style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
          )}
        </div>
      ) : (
        <button type="button" onClick={pickFile} className="drop__placeholder">
          <strong>Drop an audio file here</strong>
          <span>or click to pick one — WAV, MP3, M4A, OGG, FLAC, webm</span>
        </button>
      )}
    </section>
  )
}
