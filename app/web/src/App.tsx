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
  /** Per-word-index arrival timestamps, same idea — drives the streaming
   * word reveal in the text view. */
  const wordArrivalRef = useRef<Map<number, number>>(new Map())
  /** Highest segment.id we've already stamped — lets us O(new) the arrival
   * walk instead of O(total). Segments are emitted with monotonically
   * increasing ids; words are append-only by index. */
  const maxSeenSegIdRef = useRef<number>(-1)
  const wordArrivalCountRef = useRef<number>(0)
  /** The blob whose peaks are currently in state. Prevents the duplicate
   * decode where handleAudioReady fires one (decode 1) and handleResult
   * later fires another (decode 2) on the SAME file — which would clear
   * peaks=null on decode 2 start and make the waveform vanish + reappear
   * at the end of transcription. */
  const peaksForBlobRef = useRef<Blob | null>(null)
  /** Promise of the in-flight decode for the current blob, so handleResult
   * can join it instead of starting a fresh decode. */
  const peaksInFlightRef = useRef<Promise<void> | null>(null)

  // Hidden host for the singleton <audio>.
  const audioMountRef = useRef<HTMLDivElement>(null)
  useAudioContainer(audioMountRef)
  const currentTime = useCurrentTime()

  // decodeAudioData allocates ~10x the compressed file size in PCM — a 3hr
  // mp3 (≈180MB on disk) would need ~1.5GB of float32 PCM and may OOM the
  // tab. Skip the decode for huge files and leave peaks null (the Waveform
  // component renders placeholder bars).
  const PEAKS_MAX_BYTES = 200 * 1024 * 1024 // 200 MB compressed input ceiling

  /** Decode peaks for a blob. Idempotent: if we already started a decode
   * for this exact blob, return the same promise — no double-work, no
   * peaks-flicker. Caller can ignore the returned promise. */
  const computePeaksFor = (blob: Blob): Promise<void> => {
    if (peaksForBlobRef.current === blob && peaksInFlightRef.current) {
      return peaksInFlightRef.current
    }
    if (peaksForBlobRef.current === blob) {
      // Already decoded; peaks state still reflects this blob.
      return Promise.resolve()
    }
    peaksForBlobRef.current = blob
    setPeaks(null)
    if (blob.size > PEAKS_MAX_BYTES) {
      console.warn(
        `Peaks decode skipped: ${(blob.size / 1024 / 1024).toFixed(0)} MB > ${PEAKS_MAX_BYTES / 1024 / 1024} MB cap.`,
      )
      peaksInFlightRef.current = null
      return Promise.resolve()
    }
    const run = async (): Promise<void> => {
      try {
        const peaks = await computePeaks(blob, 220)
        if (peaksForBlobRef.current === blob) setPeaks(peaks)
      } catch (e) {
        console.warn('Peaks decode failed:', e)
        if (peaksForBlobRef.current === blob) setPeaks(null)
      }
    }
    const promise = run().finally(() => {
      if (peaksInFlightRef.current === promise) peaksInFlightRef.current = null
    })
    peaksInFlightRef.current = promise
    return promise
  }

  const handleResult = (l: LoadedAudio, r: TranscriptionResponse) => {
    setLoaded(l)
    setResult(r)
    setError(null)
    setLive(false) // finalized — drop the streaming reveal
    arrivalRef.current = new Map() // settled results don't need arrival fades
    wordArrivalRef.current = new Map()
    maxSeenSegIdRef.current = -1
    wordArrivalCountRef.current = 0
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
    // Stamp any newly-seen segments + words with their arrival time.
    // Segments are emitted with monotonically increasing ids and words
    // are append-only, so we only need to walk the NEW tail — not the
    // whole accumulated list. The latter would be O(N²) over a long
    // streaming session.
    if (r.format === 'verbose_json') {
      const now = performance.now()
      for (const s of r.body.segments) {
        if (s.id > maxSeenSegIdRef.current) {
          arrivalRef.current.set(s.id, now)
          maxSeenSegIdRef.current = s.id
        }
      }
      if (r.body.words) {
        const total = r.body.words.length
        for (let i = wordArrivalCountRef.current; i < total; i++) {
          wordArrivalRef.current.set(i, now)
        }
        if (total > wordArrivalCountRef.current) {
          wordArrivalCountRef.current = total
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
    setPeaks(null)
    peaksForBlobRef.current = null
    peaksInFlightRef.current = null
    arrivalRef.current = new Map()
    wordArrivalRef.current = new Map()
    maxSeenSegIdRef.current = -1
    wordArrivalCountRef.current = 0
  }

  /** Live peaks for the hero waveform. Currently driven by the mic's
   * AnalyserNode (via Sidebar's onPeakSample → onPeaks) at ~10 Hz. We
   * cap the array to a sliding window so a long mic recording doesn't
   * grow the peaks array (and the rendered SVG rects) without bound —
   * when handleResult fires we'll replace with a properly-binned
   * computePeaks on the final blob. */
  const PEAKS_LIVE_WINDOW = 300 // ~30s at 10 Hz, plenty of feedback
  const handlePeaks = (newPeaks: number[], cumulative: boolean) => {
    setPeaks((prev) => {
      if (cumulative || prev === null) return newPeaks.slice(-PEAKS_LIVE_WINDOW)
      const merged = prev.concat(newPeaks)
      return merged.length > PEAKS_LIVE_WINDOW
        ? merged.slice(merged.length - PEAKS_LIVE_WINDOW)
        : merged
    })
  }

  /**
   * Wire the picked file into the audio player + peak waveform early in a
   * streaming session (e.g. file + progressive over WS), so the user can
   * scrub and play while transcription is still arriving. Does NOT set
   * `result` — partials drive the transcript view; final result comes
   * through `handleResult`.
   */
  const handleAudioReady = (l: LoadedAudio) => {
    setLoaded(l)
    if (l.kind === 'file') {
      setAudioFile(l.file)
      computePeaksFor(l.file)
    }
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
            wordArrivals={wordArrivalRef.current}
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
            onAudioReady={handleAudioReady}
            onPeaks={handlePeaks}
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
