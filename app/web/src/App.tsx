import { useEffect, useRef, useState } from 'react'
import type { TranscriptionResponse } from './lib/api'
import { setAudioFile, useAudioContainer, useCurrentTime } from './lib/playback'
import { computePeaks } from './lib/peaks'
import type { LoadedAudio } from './lib/download'
import { Header } from './components/Header'
import { HeroRow, type HeroState } from './components/HeroRow'
import { TranscriptSection } from './components/TranscriptSection'
import { Sidebar, type InputMode } from './components/Sidebar'
import { FooterRail } from './components/FooterRail'
import './App.css'

export default function App() {
  const [mode, setMode] = useState<InputMode>('file')
  const [loaded, setLoaded] = useState<LoadedAudio | null>(null)
  const [result, setResult] = useState<TranscriptionResponse | null>(null)
  const [peaks, setPeaks] = useState<number[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  /** True while we're receiving partials (mid-stream). False on final result
   * or no result. Drives the word-by-word reveal animation in the
   * transcript view — static results should not animate. */
  const [live, setLive] = useState(false)
  /**
   * Per-segment arrival timestamp (`performance.now()` at first observation).
   * Live segments fade in based on (now - arrivalMs). Reset on
   * handleSessionStart and on handleResult.
   */
  const arrivalRef = useRef<Map<number, number>>(new Map())

  // Hidden host for the singleton <audio>.
  const audioMountRef = useRef<HTMLDivElement>(null)
  useAudioContainer(audioMountRef)
  const currentTime = useCurrentTime()

  // Recompute peaks when a new file is loaded.
  const computePeaksFor = async (blob: Blob) => {
    setPeaks(null)
    try {
      const p = await computePeaks(blob, 220)
      setPeaks(p)
    } catch (e) {
      console.warn('Peaks decode failed:', e)
      setPeaks(null)
    }
  }

  const handleResult = (l: LoadedAudio, r: TranscriptionResponse) => {
    setLoaded(l)
    setResult(r)
    setError(null)
    setLive(false) // finalized — drop the streaming reveal
    arrivalRef.current = new Map() // settled results don't need arrival fades
    if (l.kind === 'file') {
      setAudioFile(l.file)
      computePeaksFor(l.file)
    } else {
      // URL ingest: we don't have the bytes locally, so no playback / peaks.
      setAudioFile(null)
      setPeaks(null)
    }
  }

  const handlePartial = (r: TranscriptionResponse) => {
    // Stamp any newly-seen segments with their arrival time. Existing ids
    // keep their original arrival timestamp so they don't re-animate.
    if (r.format === 'verbose_json') {
      const now = performance.now()
      for (const s of r.body.segments) {
        if (!arrivalRef.current.has(s.id)) {
          arrivalRef.current.set(s.id, now)
        }
      }
    }
    setResult(r)
    setError(null)
    setLive(true)
  }

  const handleError = (msg: string) => {
    setError(msg)
    setLive(false)
  }

  /** Clear prior session state right before a new transcribe / record fires.
   * Keeps the UI from flashing the previous transcript while the new one is
   * in flight. */
  const handleSessionStart = () => {
    setResult(null)
    setError(null)
    setLive(false)
    arrivalRef.current = new Map()
  }

  useEffect(() => {
    if (error) setBusy(false)
  }, [error])

  const state: HeroState = error
    ? 'error'
    : busy
      ? 'streaming'
      : result
        ? 'done'
        : 'idle'

  // language from verbose_json if available
  const language =
    result?.format === 'verbose_json' ? result.body.language || 'en' : 'en'

  return (
    <div className="app">
      <div className="glows" aria-hidden />
      <Header />

      <main className="shell">
        <div className="shell__main">
          <HeroRow loaded={loaded} peaks={peaks} state={state} language={language} />
          {error && <div className="error">{error}</div>}
          <TranscriptSection
            result={result}
            filename={loaded?.title ?? 'transcript'}
            currentTime={currentTime}
            live={live}
            segmentArrivals={arrivalRef.current}
          />
        </div>
        <aside className="shell__side">
          <Sidebar
            mode={mode}
            onModeChange={setMode}
            onResult={handleResult}
            onPartial={handlePartial}
            onError={handleError}
            onBusyChange={setBusy}
            onSessionStart={handleSessionStart}
          />
        </aside>
      </main>

      <FooterRail loaded={loaded} result={result} />

      <div
        ref={audioMountRef}
        style={{ position: 'absolute', width: 0, height: 0, overflow: 'hidden' }}
      />
    </div>
  )
}
