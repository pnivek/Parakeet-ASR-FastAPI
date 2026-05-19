import { useState } from 'react'
import type { TranscriptionResponse } from './lib/api'
import { FileUploadPanel } from './components/FileUploadPanel'
import { SettingsPanel } from './components/SettingsPanel'
import { OutputView } from './components/OutputView'
import './App.css'

export default function App() {
  const [result, setResult] = useState<TranscriptionResponse | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

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
          <FileUploadPanel
            onResult={(file, r) => {
              setFilename(file.name)
              setResult(r)
              setError(null)
            }}
            onError={(msg) => setError(msg)}
          />
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
            <div>
              {filename && <div className="result-filename">{filename}</div>}
              <OutputView result={result} />
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
