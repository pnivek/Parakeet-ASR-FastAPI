import { useEffect, useRef, useState } from 'react'
import type { TranscriptionResponse } from './lib/api'
import { Atmosphere } from './components/Atmosphere'
import { Header } from './components/Header'
import { ModeSegmented, type InputMode } from './components/ModeSegmented'
import { FileUploadPanel } from './components/FileUploadPanel'
import { LiveCapturePanel } from './components/LiveCapturePanel'
import { UrlInputPanel } from './components/UrlInputPanel'
import { SettingsPanel } from './components/SettingsPanel'
import { NowPlayingCard } from './components/NowPlayingCard'
import { OutputView } from './components/OutputView'
import { setAudioFile, useAudioContainer, useCurrentTime } from './lib/playback'
import { computePeaks } from './lib/peaks'
import type { StreamState } from './components/StatusDot'
import { formatBytes } from './lib/format'
import './App.css'

export default function App() {
  const [mode, setMode] = useState<InputMode>('file')
  const [result, setResult] = useState<TranscriptionResponse | null>(null)
  const [filename, setFilename] = useState<string>('')
  const [fileMeta, setFileMeta] = useState<string>('')
  const [peaks, setPeaks] = useState<number[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // Singleton <audio> mount point — `display: none`, the visual transport
  // lives in NowPlayingCard.
  const audioMountRef = useRef<HTMLDivElement>(null)
  useAudioContainer(audioMountRef)
  const currentTime = useCurrentTime()

  // After a new audio source is set, compute waveform peaks in the background.
  const computeFor = async (file: Blob) => {
    setPeaks(null)
    try {
      const p = await computePeaks(file, 220)
      setPeaks(p)
    } catch (e) {
      // Audio formats that the browser can't decode (e.g. some webm/opus
      // edge cases) still play fine via <audio>, just no waveform.
      console.warn('Peak compute failed:', e)
      setPeaks(null)
    }
  }

  // Derive state for status pill from busy + result.
  const state: StreamState = error
    ? 'error'
    : busy
      ? 'streaming'
      : result
        ? 'done'
        : 'idle'

  const handleFileResult = (file: File, r: TranscriptionResponse) => {
    setFilename(file.name)
    setFileMeta(`${formatBytes(file.size)} · ${file.type || 'audio'}`)
    setResult(r)
    setError(null)
    setAudioFile(file)
    computeFor(file)
  }

  const handleUrlResult = (name: string, r: TranscriptionResponse) => {
    // URL mode: the audio plays from the original URL (no client file). We
    // still skip peaks since we don't have the bytes locally without re-fetch.
    setFilename(name)
    setFileMeta('via URL')
    setResult(r)
    setError(null)
    setAudioFile(null)
    setPeaks(null)
  }

  // Result words drive the floating word pill above the waveform.
  const words = result && result.format === 'verbose_json' ? result.body.words : undefined

  // Title for the Now Playing card: the filename, with a sensible default
  // before the first transcription completes.
  const npTitle = filename || 'Drop audio or hit record'

  // Clear the busy flag when error appears.
  useEffect(() => {
    if (error) setBusy(false)
  }, [error])

  return (
    <div className="app">
      <Atmosphere />
      <Header state={state} />

      <main className="shell">
        <div className="shell__main">
          <NowPlayingCard
            title={npTitle}
            meta={fileMeta}
            words={words}
            peaks={peaks}
            state={state}
          />

          {error && <div className="error">{error}</div>}

          {result && !error && (
            <OutputView result={result} filename={filename || 'transcript'} currentTime={currentTime} />
          )}

          {!result && !error && (
            <section className="glass output">
              <div className="empty">
                <p>Pick a file, paste a URL, or record live — the transcription appears here.</p>
              </div>
            </section>
          )}
        </div>

        <aside className="shell__inspector">
          <ModeSegmented mode={mode} onChange={setMode} />

          {mode === 'file' && (
            <FileUploadPanel
              onResult={handleFileResult}
              onError={(msg) => setError(msg)}
              onBusyChange={setBusy}
            />
          )}
          {mode === 'mic' && (
            <LiveCapturePanel
              onPartial={(r) => {
                setResult(r)
                setError(null)
              }}
              onResult={(file, r) => handleFileResult(file, r)}
              onError={(msg) => setError(msg)}
              onBusyChange={setBusy}
            />
          )}
          {mode === 'url' && (
            <UrlInputPanel
              onResult={handleUrlResult}
              onError={(msg) => setError(msg)}
              onBusyChange={setBusy}
            />
          )}

          <SettingsPanel />
        </aside>
      </main>

      {/* Hidden audio host — singleton survives parent re-renders. */}
      <div ref={audioMountRef} style={{ position: 'absolute', width: 0, height: 0, overflow: 'hidden' }} />
    </div>
  )
}
