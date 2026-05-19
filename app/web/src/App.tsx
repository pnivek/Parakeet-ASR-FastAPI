import { useRef, useState } from 'react'
import type { TranscriptionResponse } from './lib/api'
import { FileUploadPanel } from './components/FileUploadPanel'
import { LiveCapturePanel } from './components/LiveCapturePanel'
import { SettingsPanel } from './components/SettingsPanel'
import { OutputView } from './components/OutputView'
import { setAudioFile, seek, useAudioContainer, useCurrentTime } from './lib/playback'
import './App.css'

type Mode = 'file' | 'live'

export default function App() {
  const [mode, setMode] = useState<Mode>('file')
  const [result, setResult] = useState<TranscriptionResponse | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const playerRef = useRef<HTMLDivElement>(null)
  useAudioContainer(playerRef)
  const currentTime = useCurrentTime()

  return (
    <div className="app">
      <header className="app__header">
        <h1>Parakeet ASR</h1>
        <p className="app__tagline">
          OpenAI Whisper-compatible API · <code>nvidia/parakeet-tdt-0.6b-v2</code>
        </p>
      </header>

      <main className="app__main">
        <div className="app__left">
          <div className="modeswitch" role="tablist" aria-label="Input source">
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'file'}
              className={mode === 'file' ? 'modeswitch__btn modeswitch__btn--active' : 'modeswitch__btn'}
              onClick={() => setMode('file')}
            >
              File
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'live'}
              className={mode === 'live' ? 'modeswitch__btn modeswitch__btn--active' : 'modeswitch__btn'}
              onClick={() => setMode('live')}
            >
              Live mic
            </button>
          </div>
          {mode === 'file' ? (
            <FileUploadPanel
              onResult={(file, r) => {
                setFilename(file.name)
                setResult(r)
                setError(null)
                setAudioFile(file)
              }}
              onError={(msg) => {
                setError(msg)
              }}
            />
          ) : (
            <LiveCapturePanel
              onPartial={(r) => {
                setResult(r)
                setError(null)
              }}
              onResult={(file, r) => {
                setFilename(file.name)
                setResult(r)
                setError(null)
                setAudioFile(file)
              }}
              onError={(msg) => {
                setError(msg)
              }}
            />
          )}
          <SettingsPanel />
        </div>

        <div className="app__right">
          {error && (
            <div className="error">
              <strong>Error:</strong> {error}
            </div>
          )}
          {!result && !error && (
            <div className="empty">
              <p>Drop a file on the left, choose a response format, hit Transcribe.</p>
            </div>
          )}
          {result && (
            <>
              {filename && (
                <div className="result-filename">
                  <span>{filename}</span>
                </div>
              )}
              <div ref={playerRef} className="player" />
              <OutputView result={result} currentTime={currentTime} onSeek={seek} />
            </>
          )}
        </div>
      </main>
    </div>
  )
}
