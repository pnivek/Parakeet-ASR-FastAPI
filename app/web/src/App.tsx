import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { TranscriptionResponse } from './lib/api'
import {
  getAudioElement,
  getPlaybackAnalyser,
  play,
  seek,
  setAudioFile,
  setAudioUrl,
  setDurationHint,
  useAudioContainer,
  useCurrentTime,
  useIsPlaying,
} from './lib/playback'
import { computePeaks, readWavDuration } from './lib/peaks'
import type { LoadedAudio } from './lib/download'
import type { WhisperSegment, Word } from './lib/types'
import { Header } from './components/Header'
import { HeroRow, type HeroState } from './components/HeroRow'
import { TranscriptSection } from './components/TranscriptSection'
import { Sidebar, type InputMode } from './components/Sidebar'
import { FooterRail } from './components/FooterRail'
import { useSettings } from './lib/settings'
import './App.css'

export default function App() {
  const mode = useSettings((s) => s.mode)
  const setSettingsMode = useSettings((s) => s.set)
  const setMode = (m: InputMode) => setSettingsMode('mode', m)
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
  /** performance.now() at the start of the active session. Used by
   * markFirstSegment to compute TTFS. The live RTFx no longer needs
   * wall-elapsed client-side — it reads server-tracked counters from
   * the verbose body — so we don't mirror this into state. */
  const sessionStartRef = useRef(0)
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
  const audioPlaying = useIsPlaying()

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
      // URL ingest: point the audio element at the URL directly so the
      // user can play / scrub. Peaks decode is gated to local blobs (see
      // useEffect on loadedFile) — no waveform for URL sources.
      setAudioUrl(l.url)
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
  const handleSessionStart = useCallback((opts?: { resetHero?: boolean }) => {
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
    setLiveAnchored(false)
    // Only wipe the hero waveform/identity when explicitly asked (mic =
    // fresh recording, url = new source). File mode keeps the already-
    // decoded waveform: clearing it here would null `loaded`, re-trigger
    // the decode effect (file→null→file), and flash — and a cancelled
    // re-decode is exactly why the waveform sometimes never came back.
    if (opts?.resetHero) {
      setLoaded(null)
      setPeaks(null)
    }
  }, [])

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

  // Rolling waveform for URL sources, driven by an AnalyserNode tapped
  // off the playing <audio> element. File modes already get full peaks
  // from a local decodeAudioData; mic modes drive peaks from a getUserMedia
  // analyser. URL has neither (bytes are remote / live-streamed), so we
  // sample the playback element while it's actually playing — bars scroll
  // in from the right like an oscilloscope, same handlePeaks pattern.
  //
  // Sample rate: ~15 Hz (~one frame per 4 rAFs at 60 Hz). Faster looks
  // jittery, slower looks coarse. handlePeaks rolls into the 220-bar buffer.
  useEffect(() => {
    if (loaded?.kind !== 'url' || !audioPlaying) return
    const analyser = getPlaybackAnalyser()
    if (!analyser) return
    const buf = new Uint8Array(analyser.fftSize)
    let raf = 0
    let frameSkip = 0
    const tick = () => {
      frameSkip = (frameSkip + 1) % 4
      if (frameSkip === 0) {
        analyser.getByteTimeDomainData(buf as unknown as Uint8Array<ArrayBuffer>)
        let max = 0
        for (let i = 0; i < buf.length; i++) {
          const v = Math.abs(buf[i] - 128) / 128
          if (v > max) max = v
        }
        handlePeaks([max], false)
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded?.kind, audioPlaying])

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
    } else {
      setAudioUrl(l.url)
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
    setLiveAnchored(false)
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

  // ── Transport helpers driven by hero buttons ────────────────────
  const segments = useMemo(
    () =>
      result?.format === 'verbose_json' ? result.body.segments : [],
    [result],
  )

  /** Seek to the start of the segment ENDING after (or AT) currentTime —
   * stepping forward by one transcript line. */
  const seekNextSegment = useCallback(() => {
    if (segments.length === 0) return
    // Find the first segment whose start is strictly after the playhead.
    // Floor at +0.05 so a click after a long pause still advances even
    // when currentTime is touching the previous segment's end.
    const target = segments.find((s) => s.start > currentTime + 0.05)
    if (target) seek(target.start)
    else seek(segments[segments.length - 1].start)
    play()
  }, [segments, currentTime])

  /** Seek to the start of the previous segment (or the start of the
   * current one if we're > ~1.2 s into it — the classic "double tap to
   * skip" affordance). */
  const seekPrevSegment = useCallback(() => {
    if (segments.length === 0) return
    const cur = segments.findIndex(
      (s) => currentTime >= s.start && currentTime <= s.end,
    )
    if (cur > 0) {
      // If well into the current segment, restart it first; otherwise jump.
      const into = currentTime - segments[cur].start
      const target = into > 1.2 ? segments[cur] : segments[cur - 1]
      seek(target.start)
    } else if (cur === 0) {
      seek(segments[0].start)
    } else {
      // Outside any segment — find the last that started before us.
      const prev = [...segments].reverse().find((s) => s.start < currentTime)
      if (prev) seek(prev.start)
      else seek(0)
    }
    play()
  }, [segments, currentTime])

  // "At live edge" is a STATE — true after Live click, false the moment
  // the user pauses or seeks (any direction). Survives micro-stalls in
  // the audio buffer that briefly inflate the audio_received - currentTime
  // diff. A 10s diff backstop kicks in only if the gap genuinely runs
  // away (rare; means the user's been silently stalling for ages).
  const [liveAnchored, setLiveAnchored] = useState(false)
  const suppressNextSeekRef = useRef(false)
  // Subscribe to the audio element's pause + seek events. Pause un-
  // anchors immediately; seek un-anchors unless we just triggered it
  // from Live click (suppression flag).
  useEffect(() => {
    const el = getAudioElement()
    const onPause = () => setLiveAnchored(false)
    const onSeeked = () => {
      if (suppressNextSeekRef.current) {
        suppressNextSeekRef.current = false
        return
      }
      setLiveAnchored(false)
    }
    el.addEventListener('pause', onPause)
    el.addEventListener('seeked', onSeeked)
    return () => {
      el.removeEventListener('pause', onPause)
      el.removeEventListener('seeked', onSeeked)
    }
  }, [])

  // Live indicator is purely the anchor state (set by Live click, cleared
  // on pause/seek). The 10s diff backstop only applies when we HAVE a
  // server-side audio_received signal — otherwise we trust liveAnchored
  // alone. This way the dot lights up the instant Live is clicked,
  // regardless of whether transcription has started feeding counters yet.
  const audioReceived =
    result?.format === 'verbose_json' ? result.body.audio_received_s ?? 0 : 0
  const atLiveEdge =
    loaded?.kind === 'url' &&
    liveAnchored &&
    (audioReceived <= 0 || audioReceived - currentTime < 10.0)

  /** Jump the playback element to the "live edge" — the latest audio
   * we have. Source of truth is the server's `audio_received_s` counter
   * (where ffmpeg has actually pulled to), shipped in every streaming
   * WS message. That's what aligns the client's audio clock with the
   * transcription pipeline's "now," and it survives the case where the
   * user paused for a while and the browser stopped buffering ahead.
   *
   * Falls back to `audio.seekable.end(last)` (browser buffer head) when
   * counters aren't available — e.g., URL+REST sessions or older
   * servers. `audio.duration` is the last resort because some HLS
   * sources report it as the playlist window rather than the broadcast
   * head, which would seek backwards. */
  const seekLiveEdge = useCallback(() => {
    if (loaded?.kind !== 'url') return
    const el = getAudioElement()
    const serverEdge =
      result?.format === 'verbose_json' ? result.body.audio_received_s ?? 0 : 0
    const seekableEnd =
      el.seekable && el.seekable.length > 0
        ? el.seekable.end(el.seekable.length - 1)
        : 0
    let edge = serverEdge
    // Clamp to client buffer head if server has gone further — seeking
    // past seekable.end leaves nothing to play.
    if (seekableEnd > 0 && (edge <= 0 || edge > seekableEnd)) {
      edge = seekableEnd
    }
    if (edge <= 0 && isFinite(el.duration) && el.duration > 0) {
      edge = el.duration
    }
    // Back off from the buffer head so the element has audio to play
    // through the typical HLS / icecast segment fetch window. Seeking
    // to exactly the leading edge gives 0s of buffer and the audio
    // element stalls (looks like "audio paused"). HLS segments are
    // usually 6-10s; 2s of buffer is enough to ride through normal
    // segment-boundary fetches, and the user still reads as "live".
    const LIVE_BACKOFF = 2.0
    if (edge > LIVE_BACKOFF) edge -= LIVE_BACKOFF
    if (edge > 0) {
      // Mark this seek as Live-initiated so the 'seeked' listener
      // doesn't un-anchor us. Set the anchor BEFORE seeking so the
      // event handler sees the suppression flag before it fires.
      suppressNextSeekRef.current = true
      setLiveAnchored(true)
      seek(edge)
      play()
    }
  }, [loaded?.kind, result])

  // language from verbose_json if available
  const language =
    result?.format === 'verbose_json' ? result.body.language || 'en' : 'en'

  // Active strategy for the footer — derived from the active modality's
  // settings. The footer prefers the resolved value from the server
  // result when present and falls back to this.
  const modalitySettings = useSettings((st) => st[st.mode])
  const activeStrategy =
    modalitySettings.engine === 'websocket'
      ? 'streaming'
      : modalitySettings.strategyOverride === 'auto'
        ? 'offline'
        : modalitySettings.strategyOverride

  return (
    <div className="app">
      <div className="glows" aria-hidden />
      <Header />

      <main className="shell">
        <div className="shell__main">
          <HeroRow
            loaded={loaded}
            peaks={peaks}
            state={state}
            language={language}
            result={result}
            hasSegments={segments.length > 0}
            onPrevSegment={seekPrevSegment}
            onNextSegment={seekNextSegment}
            onLiveEdge={seekLiveEdge}
            atLiveEdge={atLiveEdge}
          />
          {error && <div className="error">{error}</div>}
          <TranscriptSection
            result={result}
            filename={loaded?.title ?? 'transcript'}
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

      <FooterRail
        loaded={loaded}
        result={result}
        ttfs={ttfs}
        activeStrategy={activeStrategy}
        live={live}
      />

      <div
        ref={audioMountRef}
        style={{ position: 'absolute', width: 0, height: 0, overflow: 'hidden' }}
      />
    </div>
  )
}
