import { useEffect, useRef, useState } from 'react'
import type { TranscriptionResponse } from './lib/api'
import { setAudioFile, setDurationHint, useAudioContainer, useCurrentTime } from './lib/playback'
import { computePeaks, readWavDuration } from './lib/peaks'
import type { LoadedAudio } from './lib/download'
import type { WhisperSegment, Word } from './lib/types'
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
  /** Engine's in-flight sentence (uncommitted tokens). Streams in
   * during live mic so the UI shows text while the model is still
   * deciding where the sentence ends. Cleared when the matching
   * sentence-bounded segments_batch lands. */
  const [partialSegment, setPartialSegment] = useState<WhisperSegment | null>(null)
  const [partialWords, setPartialWords] = useState<Word[]>([])
  /** Time-to-first-segment (seconds) — wall clock from the transcribe /
   * record click to the first segment landing. `ttfsRef` guards the
   * one-shot capture against state-update races. */
  const [ttfs, setTtfs] = useState<number | null>(null)
  const ttfsRef = useRef<number | null>(null)
  const sessionStartRef = useRef<number>(0)
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

  // Hidden host for the singleton <audio>.
  const audioMountRef = useRef<HTMLDivElement>(null)
  useAudioContainer(audioMountRef)
  const currentTime = useCurrentTime()

  // Decode peaks for whichever file is currently loaded. One source of
  // truth: any time `loaded.file` becomes a new blob (file picked, mic
  // recording finalized, etc.), kick off a decode. We DON'T clear peaks
  // at the start of the decode — any prior peaks (from a live mic
  // session, or a previous file) stay visible until the new decode
  // resolves, avoiding a flash of empty bars. Cancellation flag prevents
  // a stale decode from clobbering a newer file's peaks.
  //
  // peaks.ts handles format dispatch + size caps. WAV files go through a
  // header-parsing fast path that reads only ~3MB per file regardless of
  // duration; other formats fall through to decodeAudioData (capped at
  // 200MB compressed input).
  const loadedFile = loaded?.kind === 'file' ? loaded.file : null
  useEffect(() => {
    if (!loadedFile) {
      setPeaks(null)
      setDurationHint(0)
      return
    }
    let cancelled = false
    computePeaks(loadedFile, 220).then(
      (p) => {
        if (!cancelled) setPeaks(p)
      },
      (e) => {
        console.warn('Peaks decode failed:', e)
        if (!cancelled) setPeaks(null)
      },
    )
    // Duration fallback for WAVs — the <audio> element can be slow (or
    // fail) to report `duration` on multi-hour uploads. Parse it from
    // the header so the hero meta shows it immediately.
    readWavDuration(loadedFile).then((s) => {
      if (!cancelled && s) setDurationHint(s)
    })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadedFile])

  const handleResult = (l: LoadedAudio, r: TranscriptionResponse) => {
    // Non-streaming paths (REST) deliver everything at once — capture
    // TTFS here too if no partial beat us to it.
    if (r.format === 'verbose_json') markFirstSegment(r.body.segments.length)
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
      // peaks decode runs in the useEffect above when loaded.file changes.
    } else {
      // URL ingest: we don't have the bytes locally, so no playback / peaks.
      setAudioFile(null)
    }
    // Duration fallback from the server's reported duration. Mic
    // recordings are WebM/Opus blobs whose <audio> element duration is
    // unreliable (often Infinity/0), which left the hero timeline stuck
    // at 0:00. setAudioFile above resets the hint, so set it after.
    if (r.format === 'verbose_json' && r.body.duration > 0) {
      setDurationHint(r.body.duration)
    }
  }

  /** One-shot capture of time-to-first-segment, the first time a
   * response carries at least one segment after a session start. */
  const markFirstSegment = (segCount: number) => {
    if (ttfsRef.current === null && segCount > 0) {
      const t = (performance.now() - sessionStartRef.current) / 1000
      ttfsRef.current = t
      setTtfs(t)
    }
  }

  const handlePartial = (r: TranscriptionResponse) => {
    if (r.format === 'verbose_json') markFirstSegment(r.body.segments.length)
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
  const handleSessionStart = (opts?: { resetHero?: boolean }) => {
    setResult(null)
    setError(null)
    setLive(false)
    setPartialSegment(null)
    setPartialWords([])
    sessionStartRef.current = performance.now()
    ttfsRef.current = null
    setTtfs(null)
    arrivalRef.current = new Map()
    wordArrivalRef.current = new Map()
    maxSeenSegIdRef.current = -1
    wordArrivalCountRef.current = 0
    // Only wipe the hero waveform/identity when explicitly asked (mic =
    // fresh recording, url = new source). File mode keeps the already-
    // decoded waveform: clearing it here would null `loaded`, re-trigger
    // the decode effect (file→null→file), and flash — and a cancelled
    // re-decode is exactly why the waveform sometimes never came back.
    if (opts?.resetHero) {
      setLoaded(null)
      setPeaks(null)
    }
  }

  const handlePartialSegment = (segment: WhisperSegment | null, words: Word[]) => {
    setPartialSegment(segment)
    setPartialWords(words)
  }

  /** Live peaks for the hero waveform, driven by the mic's AnalyserNode
   * (via Sidebar's onPeakSample → onPeaks) at ~10 Hz.
   *
   * We keep a FIXED-LENGTH buffer (not a growing array): the Waveform
   * stretches its viewBox to 100% width based on element count, so a
   * growing array makes every bar's position drift as it fills — which
   * read as "a spike stuck in place while the rest slides". A constant
   * length means bar positions never move; new audio scrolls in from the
   * right and old audio scrolls off the left, like an oscilloscope. The
   * buffer starts as a flat silent line and fills smoothly as you speak.
   * On final_transcription handleResult replaces it with a binned
   * computePeaks of the blob (same length → no jump). */
  const PEAKS_LIVE_BARS = 220
  const handlePeaks = (newPeaks: number[], cumulative: boolean) => {
    setPeaks((prev) => {
      // Server-pushed cumulative peaks (non-mic) — replace wholesale.
      if (cumulative) return newPeaks.slice(-PEAKS_LIVE_BARS)
      const base =
        prev && prev.length === PEAKS_LIVE_BARS ? prev : new Array(PEAKS_LIVE_BARS).fill(0)
      const next = base.concat(newPeaks)
      return next.slice(next.length - PEAKS_LIVE_BARS)
    })
  }

  /**
   * Wire the picked file into the audio player early in a streaming
   * session (e.g. file + progressive over WS), so the user can scrub and
   * play while transcription is still arriving. Setting `loaded` triggers
   * the peaks-decode effect above. Does NOT set `result` — partials
   * drive the transcript view; final result comes through `handleResult`.
   */
  const handleAudioReady = (l: LoadedAudio) => {
    setLoaded(l)
    if (l.kind === 'file') {
      setAudioFile(l.file)
    }
  }

  /** Wipe the hero (waveform + metadata + transcript) when the user
   * removes the staged source — e.g. clicking ✕ on the file card. */
  const handleClearSource = () => {
    setLoaded(null)
    setResult(null)
    setError(null)
    setLive(false)
    setPartialSegment(null)
    setPartialWords([])
    setPeaks(null)
    setTtfs(null)
    ttfsRef.current = null
    setAudioFile(null)
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
            partialSegment={partialSegment}
            partialWords={partialWords}
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
            onClearSource={handleClearSource}
            onPeaks={handlePeaks}
            onPartialSegment={handlePartialSegment}
          />
        </aside>
      </main>

      <FooterRail loaded={loaded} result={result} ttfs={ttfs} />

      <div
        ref={audioMountRef}
        style={{ position: 'absolute', width: 0, height: 0, overflow: 'hidden' }}
      />
    </div>
  )
}
