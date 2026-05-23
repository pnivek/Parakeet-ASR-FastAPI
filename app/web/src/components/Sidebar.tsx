import { useCallback, useEffect, useRef, useState } from 'react'
import {
  postTranscription,
  postTranscriptionUrl,
  streamFileViaWS,
  type TranscriptionResponse,
  connectLiveWS,
  type LiveWSHandle,
} from '../lib/api'
import { useSettings, type CommonModality, type MicModality, type Modality } from '../lib/settings'
import { MIC_FORMAT_HINT, MIC_SAMPLE_RATE, MIC_MIME_TYPE, useMic } from '../lib/mic'
import { formatBytes } from '../lib/format'
import type {
  Engine,
  Strategy,
  StrategyOverride,
  WhisperSegment,
  Word,
  WSMessage,
} from '../lib/types'
import type { LoadedAudio } from '../lib/download'
import { OptionList } from './OptionList'

/** Alias retained so App.tsx and HeroRow keep their existing type imports. */
export type InputMode = Modality

interface Props {
  mode: InputMode
  onModeChange: (m: InputMode) => void
  onResult: (loaded: LoadedAudio, result: TranscriptionResponse) => void
  onPartial: (result: TranscriptionResponse) => void
  onError: (message: string) => void
  onBusyChange: (busy: boolean) => void
  onSessionStart?: (opts?: { resetHero?: boolean }) => void
  onAudioReady?: (loaded: LoadedAudio) => void
  onClearSource?: () => void
  onPeaks?: (peaks: number[], cumulative: boolean) => void
  onPartialSegment?: (segment: WhisperSegment | null, words: Word[]) => void
}

const STRATEGY_OVERRIDES: { id: StrategyOverride; label: string; hint: string }[] = [
  { id: 'auto', label: 'Auto', hint: 'Server picks Full / Split-full based on file size.' },
  { id: 'full', label: 'Full pass', hint: 'Force a single pass. Errors if it would OOM.' },
  { id: 'split_full', label: 'Split-full', hint: 'Force sequential full passes over slices, stitched at seams.' },
]

/** Per-modality tooltips for the Engine picker — explains what each engine
 * does in the context of its modality. */
const ENGINE_TOOLTIPS: Record<Modality, Record<Engine, string>> = {
  file: {
    rest: 'HTTP POST the file. Server returns one response.',
    websocket: "Stream the file's bytes over WS. Watch segments arrive during the upload.",
  },
  url: {
    rest: 'Server downloads the URL and transcribes the complete file.',
    websocket: 'Server pipes the URL through ffmpeg live. Works for HLS, icecast, RTSP, m3u8.',
  },
  mic: {
    rest: 'Record locally, upload as a file when you stop.',
    websocket: 'Live transcription. Partials appear as you speak.',
  },
}

/** Wire-level `?strategy=` param the backend expects, derived from the
 * (engine, strategyOverride) pair. */
function restStrategyParam(override: StrategyOverride): Strategy {
  return override === 'auto' ? 'offline' : (override as Strategy)
}

const TABS: { id: Modality; label: string }[] = [
  { id: 'file', label: 'Upload' },
  { id: 'url', label: 'URL' },
  { id: 'mic', label: 'Record' },
]

export function Sidebar({
  mode,
  onModeChange,
  onResult,
  onPartial,
  onError,
  onBusyChange,
  onSessionStart,
  onAudioReady,
  onClearSource,
  onPeaks,
  onPartialSegment,
}: Props) {
  const s = useSettings()
  const m = s[mode] as CommonModality | MicModality
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
  // Mic-record staged blob — kept separate from `pickedFile`.
  const [micBlob, setMicBlob] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const onFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return
    const file = files[0]
    setPickedFile(file)
    onAudioReady?.({
      kind: 'file',
      title: file.name.replace(/\.[^.]+$/, ''),
      source: `${formatBytes(file.size)} · ${file.type || 'audio'}`,
      file,
    })
  }

  // URL state
  const [urlInput, setUrlInput] = useState('https://')

  // Mic / streaming state
  const wsRef = useRef<LiveWSHandle | null>(null)
  const restAbortRef = useRef<AbortController | null>(null)
  const segmentsRef = useRef<WhisperSegment[]>([])
  const wordsRef = useRef<Word[]>([])
  const chunksRef = useRef<Blob[]>([])
  const recordStartRef = useRef<number>(0)
  const [recordElapsed, setRecordElapsed] = useState(0)

  // VAD + HPF config block. Pulled from the active modality — all three
  // modalities carry these fields on CommonModality, but the server only
  // honors them on WS-streaming paths (handle_streaming_pcm /
  // handle_streaming_url). REST paths bypass ffmpeg + VAD entirely.
  const vadConfig = () => ({
    vad_enabled: m.vadEnabled,
    vad_threshold: m.vadThreshold,
    vad_consecutive: m.vadConsecutive,
    vad_hangover_ms: m.vadHangoverMs,
    vad_pad_min_gap_ms: m.vadPadMinGapMs,
    vad_pad_duration_ms: m.vadPadDurationMs,
    hpf_hz: m.hpfHz,
  })

  // Partial-message throttle — see the original Sidebar's notes; same logic.
  const PARTIAL_THROTTLE_MS = 250
  const lastPartialAtRef = useRef(0)
  const partialTimerRef = useRef<number | null>(null)
  const buildPartialPayload = useCallback((): TranscriptionResponse => {
    const wall = (Date.now() - recordStartRef.current) / 1000
    const segs = segmentsRef.current
    const words = wordsRef.current
    return {
      format: 'verbose_json',
      body: {
        task: 'transcribe',
        language: 'en',
        duration: segs[segs.length - 1]?.end ?? 0,
        text: segs.map((x) => x.text).join(' ').trim(),
        segments: segs,
        words: words.length > 0 ? words : undefined,
        strategy: 'streaming',
        transcription_time_seconds: wall,
      },
    }
  }, [])
  const schedulePartial = useCallback(() => {
    const now = Date.now()
    const since = now - lastPartialAtRef.current
    if (since >= PARTIAL_THROTTLE_MS) {
      lastPartialAtRef.current = now
      onPartial(buildPartialPayload())
      return
    }
    if (partialTimerRef.current === null) {
      partialTimerRef.current = window.setTimeout(() => {
        partialTimerRef.current = null
        lastPartialAtRef.current = Date.now()
        onPartial(buildPartialPayload())
      }, PARTIAL_THROTTLE_MS - since)
    }
  }, [onPartial, buildPartialPayload])
  const cancelPendingPartial = useCallback(() => {
    if (partialTimerRef.current !== null) {
      clearTimeout(partialTimerRef.current)
      partialTimerRef.current = null
    }
    lastPartialAtRef.current = 0
  }, [])

  // WS message dispatch — same as before. `final_transcription` aliases a
  // "live recording" loaded source for mic; for file/url streaming the
  // caller has already set the audio source via onAudioReady.
  const handleMessage = useCallback(
    (msg: WSMessage, options?: { staged?: LoadedAudio }) => {
      switch (msg.type) {
        case 'segments_batch': {
          segmentsRef.current = [...segmentsRef.current, ...msg.segments]
          if (msg.words && msg.words.length > 0) {
            wordsRef.current = [...wordsRef.current, ...msg.words]
          }
          onPartialSegment?.(null, [])
          schedulePartial()
          break
        }
        case 'partial_segment':
          onPartialSegment?.(msg.segment, msg.words ?? [])
          break
        case 'final_transcription': {
          cancelPendingPartial()
          onPartialSegment?.(null, [])
          let loaded: LoadedAudio
          if (options?.staged) {
            loaded = options.staged
          } else {
            // Mic-streaming default — build a Blob from the recorded chunks.
            const blob = new Blob(chunksRef.current, { type: MIC_MIME_TYPE })
            const file = new File([blob], `mic-${Date.now()}.webm`, { type: MIC_MIME_TYPE })
            loaded = {
              kind: 'file',
              title: `live recording — ${new Date().toLocaleTimeString()}`,
              source: 'live mic',
              file,
            }
          }
          onResult(loaded, {
            format: 'verbose_json',
            body: {
              task: 'transcribe',
              language: msg.language,
              duration: msg.duration,
              text: msg.text,
              segments: msg.segments,
              words: msg.words,
              strategy: msg.strategy,
              transcription_time_seconds: msg.transcription_time,
              csv_content: msg.csv_content,
              srt_content: msg.srt_content,
            },
          })
          wsRef.current?.abort()
          wsRef.current = null
          break
        }
        case 'error':
          onError(msg.error)
          break
      }
    },
    [onResult, onError, schedulePartial, cancelPendingPartial, onPartialSegment],
  )

  const mic = useMic({
    onChunk: (blob) => {
      chunksRef.current.push(blob)
      wsRef.current?.sendBinary(blob)
    },
    onStop: () => {
      if (wsRef.current) {
        wsRef.current.finish()
        return
      }
      if (chunksRef.current.length === 0) return
      const blob = new Blob(chunksRef.current, { type: MIC_MIME_TYPE })
      const file = new File([blob], `recording-${Date.now()}.webm`, { type: MIC_MIME_TYPE })
      setMicBlob(file)
    },
    onError: (err) => {
      onError(err.message)
      wsRef.current?.abort()
      wsRef.current = null
    },
    noiseSuppression: mode === 'mic' ? s.mic.noiseSuppression : true,
    onPeakSample: (peak) => onPeaks?.([peak], false),
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

  // Pre-listen the mic in mic+WebSocket mode so the meter reflects real
  // speech before the user clicks Transcribe. Skips on plain-HTTP origins
  // where getUserMedia is undefined (LAN deployment).
  const micEngine = s.mic.engine
  useEffect(() => {
    const micAvailable =
      typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia
    if (!micAvailable) return
    const wantListen = mode === 'mic' && micEngine === 'websocket'
    if (wantListen && mic.state === 'idle') {
      void mic.listen()
    } else if (!wantListen && mic.state === 'listening') {
      mic.stopListening()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, micEngine, mic.state])

  useEffect(() => {
    return () => mic.stopListening()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // On page unload (refresh / tab close / navigation), close any open
  // WS and abort any in-flight REST. Without this, the server's
  // watch_ws_disconnect doesn't see the close until uvicorn's ping
  // timeout fires (up to 5 min), so ffmpeg keeps pulling a URL stream
  // and the model lock stays held. `pagehide` fires reliably on real
  // unloads (including iOS Safari, which suppresses `beforeunload`).
  useEffect(() => {
    const handler = () => {
      if (wsRef.current) {
        try {
          wsRef.current.abort()
        } catch {
          /* socket already gone */
        }
        wsRef.current = null
      }
      if (restAbortRef.current) {
        try {
          restAbortRef.current.abort()
        } catch {
          /* already aborted */
        }
        restAbortRef.current = null
      }
    }
    window.addEventListener('pagehide', handler)
    window.addEventListener('beforeunload', handler)
    return () => {
      window.removeEventListener('pagehide', handler)
      window.removeEventListener('beforeunload', handler)
    }
  }, [])

  const startMic = useCallback(async () => {
    onSessionStart?.({ resetHero: true })
    cancelPendingPartial()
    segmentsRef.current = []
    wordsRef.current = []
    chunksRef.current = []
    recordStartRef.current = Date.now()
    setRecordElapsed(0)
    if (s.mic.engine === 'rest') {
      // Record-locally path — useMic accumulates chunks; the staged blob
      // routes through REST upload when the user clicks Transcribe.
      wsRef.current = null
      setMicBlob(null)
      await mic.start()
      return
    }
    // WebSocket path — open the live socket then start the mic.
    const liveWs: LiveWSHandle = connectLiveWS(
      {
        sample_rate: MIC_SAMPLE_RATE,
        channels: 1,
        bytes_per_sample: 2,
        format: MIC_FORMAT_HINT,
        strategy: 'streaming',
        live_latency: s.mic.liveLatency,
        long_audio_threshold: s.mic.longAudioThreshold ?? undefined,
        ...vadConfig(),
      },
      {
        onMessage: (msg) => handleMessage(msg),
        onError: () => onError('WebSocket error — connection failed.'),
        onClose: (code, reason) => {
          if (code !== 1000 && code !== 1005) {
            onError(`WebSocket closed: code=${code} reason=${reason || 'no reason'}`)
          }
          if (wsRef.current === liveWs) wsRef.current = null
        },
      },
    )
    wsRef.current = liveWs
    await mic.start()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s, handleMessage, mic, onError, onSessionStart, cancelPendingPartial])

  const transcribe = useCallback(async () => {
    if (mode === 'file') {
      if (!pickedFile) return
      onSessionStart?.()
      const fileModality = s.file
      if (fileModality.engine === 'websocket') {
        cancelPendingPartial()
        segmentsRef.current = []
        wordsRef.current = []
        chunksRef.current = []
        recordStartRef.current = Date.now()
        const loaded: LoadedAudio = {
          kind: 'file',
          title: pickedFile.name.replace(/\.[^.]+$/, ''),
          source: `${formatBytes(pickedFile.size)} · ${pickedFile.type || 'audio'}`,
          file: pickedFile,
        }
        onAudioReady?.(loaded)
        setBusyAll(true)
        try {
          const fileExt = pickedFile.name.split('.').pop()?.toLowerCase() || 'wav'
          const fileWs: LiveWSHandle = streamFileViaWS(
            pickedFile,
            {
              sample_rate: 16000,
              channels: 1,
              bytes_per_sample: 2,
              format: fileExt,
              strategy: 'streaming',
              // Default false (offline-like preset, ~4× higher throughput
              // than 10-2-2 on a long file). User can flip on via Advanced
              // if they want partials sooner on a slow / huge upload.
              live_latency: fileModality.liveLatency,
              long_audio_threshold: fileModality.longAudioThreshold ?? undefined,
              ...vadConfig(),
            },
            {
              onMessage: (msg) => handleMessage(msg, { staged: loaded }),
              onError: () => onError('WebSocket error during file streaming.'),
              onClose: (code, reason) => {
                if (code !== 1000 && code !== 1005) {
                  onError(`WebSocket closed: code=${code} reason=${reason || 'no reason'}`)
                }
                if (wsRef.current === fileWs) wsRef.current = null
                setBusyAll(false)
              },
            },
          )
          wsRef.current = fileWs
        } catch (e) {
          onError(e instanceof Error ? e.message : String(e))
          setBusyAll(false)
        }
        return
      }
      // REST upload
      setBusyAll(true)
      setProgress(0)
      const ac = new AbortController()
      restAbortRef.current = ac
      try {
        const r = await postTranscription(
          pickedFile,
          {
            // verbose_json + both granularities are hardcoded — the transcript
            // view always needs the full body, and the download flyout derives
            // every other format from it.
            response_format: 'verbose_json',
            timestamp_granularities: ['segment', 'word'],
            strategy: restStrategyParam(fileModality.strategyOverride),
            long_audio_threshold: fileModality.longAudioThreshold ?? undefined,
          },
          { onProgress: (loaded, total) => setProgress(loaded / total), signal: ac.signal },
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
        if (e instanceof DOMException && e.name === 'AbortError') return
        onError(e instanceof Error ? e.message : String(e))
      } finally {
        if (restAbortRef.current === ac) restAbortRef.current = null
        setBusyAll(false)
        setProgress(null)
      }
    } else if (mode === 'url') {
      if (!urlInput.trim() || urlInput === 'https://') return
      onSessionStart?.({ resetHero: true })
      const urlModality = s.url
      if (urlModality.engine === 'websocket') {
        // URL-over-WS: server opens ffmpeg `-i <url>` and streams partials.
        // Browser plays the URL natively in parallel so the user can scrub
        // the live stream while transcription arrives. No waveform peaks
        // (live stream — peaks decode would need the full bytes).
        cancelPendingPartial()
        segmentsRef.current = []
        wordsRef.current = []
        chunksRef.current = []
        recordStartRef.current = Date.now()
        let staged: LoadedAudio
        try {
          const u = new URL(urlInput.trim())
          staged = { kind: 'url', title: u.pathname.split('/').filter(Boolean).pop() || u.host, source: 'live stream', url: urlInput.trim() }
        } catch {
          staged = { kind: 'url', title: urlInput.trim(), source: 'live stream', url: urlInput.trim() }
        }
        // Wire the audio element to the URL immediately so the user can
        // play the stream while transcription is in flight.
        onAudioReady?.(staged)
        setBusyAll(true)
        try {
          const urlWs: LiveWSHandle = connectLiveWS(
            {
              sample_rate: 16000,
              channels: 1,
              bytes_per_sample: 2,
              format: 'url',
              url: urlInput.trim(),
              strategy: 'streaming',
              // Defaults true for URL — realtime source benefits from
              // 10-2-2 preset (~6 s lag, partial_segment messages emit).
              live_latency: urlModality.liveLatency,
              long_audio_threshold: urlModality.longAudioThreshold ?? undefined,
              // Disable VAD for URL streams — vadConfig() returns vad_enabled
              // false for non-mic modalities. Server otherwise defaults VAD on,
              // which gates network audio too aggressively.
              ...vadConfig(),
            },
            {
              onMessage: (msg) => handleMessage(msg, { staged }),
              onError: () => onError('WebSocket error during URL stream.'),
              onClose: (code, reason) => {
                if (code !== 1000 && code !== 1005) {
                  onError(`WebSocket closed: code=${code} reason=${reason || 'no reason'}`)
                }
                if (wsRef.current === urlWs) wsRef.current = null
                setBusyAll(false)
              },
            },
          )
          wsRef.current = urlWs
        } catch (e) {
          onError(e instanceof Error ? e.message : String(e))
          setBusyAll(false)
        }
        return
      }
      // REST URL — wire the audio element to the URL immediately so the
      // user can play / scrub while the server fetches + transcribes.
      let urlName = urlInput
      try {
        const u = new URL(urlInput.trim())
        urlName = u.pathname.split('/').filter(Boolean).pop() || u.host
      } catch {
        /* keep input */
      }
      const stagedUrl: LoadedAudio = {
        kind: 'url',
        title: urlName,
        source: 'via URL',
        url: urlInput.trim(),
      }
      onAudioReady?.(stagedUrl)
      setBusyAll(true)
      const ac = new AbortController()
      restAbortRef.current = ac
      try {
        const r = await postTranscriptionUrl(
          urlInput.trim(),
          {
            response_format: 'verbose_json',
            timestamp_granularities: ['segment', 'word'],
            strategy: restStrategyParam(urlModality.strategyOverride),
            long_audio_threshold: urlModality.longAudioThreshold ?? undefined,
          },
          { signal: ac.signal },
        )
        onResult(stagedUrl, r)
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') return
        onError(e instanceof Error ? e.message : String(e))
      } finally {
        if (restAbortRef.current === ac) restAbortRef.current = null
        setBusyAll(false)
      }
    } else if (mode === 'mic') {
      if (recording) {
        mic.stop()
        return
      }
      // Record-then-upload (mic + REST), with a staged blob to send.
      if (s.mic.engine === 'rest' && micBlob) {
        onSessionStart?.({ resetHero: true })
        setBusyAll(true)
        setProgress(0)
        const ac = new AbortController()
        restAbortRef.current = ac
        try {
          const r = await postTranscription(
            micBlob,
            {
              response_format: 'verbose_json',
              timestamp_granularities: ['segment', 'word'],
              strategy: restStrategyParam(s.mic.strategyOverride),
              long_audio_threshold: s.mic.longAudioThreshold ?? undefined,
            },
            { onProgress: (loaded, total) => setProgress(loaded / total), signal: ac.signal },
          )
          onResult(
            {
              kind: 'file',
              title: `recording — ${new Date().toLocaleTimeString()}`,
              source: 'mic · recorded',
              file: micBlob,
            },
            r,
          )
        } catch (e) {
          if (e instanceof DOMException && e.name === 'AbortError') return
          onError(e instanceof Error ? e.message : String(e))
        } finally {
          if (restAbortRef.current === ac) restAbortRef.current = null
          setBusyAll(false)
          setProgress(null)
        }
        return
      }
      startMic().catch((e) => onError(e instanceof Error ? e.message : String(e)))
    }
  }, [
    mode,
    pickedFile,
    micBlob,
    urlInput,
    s,
    recording,
    mic,
    startMic,
    onResult,
    onError,
    setBusyAll,
    onSessionStart,
    onAudioReady,
    cancelPendingPartial,
    handleMessage,
  ])

  const cancelInFlight = useCallback(() => {
    if (wsRef.current && mode !== 'mic') {
      try {
        wsRef.current.abort()
      } catch (e) {
        console.warn('cancelInFlight: WS abort failed', e)
      }
      wsRef.current = null
    }
    if (restAbortRef.current) {
      try {
        restAbortRef.current.abort()
      } catch (e) {
        console.warn('cancelInFlight: REST abort failed', e)
      }
      restAbortRef.current = null
    }
    cancelPendingPartial()
    setBusyAll(false)
    setProgress(null)
  }, [mode, setBusyAll, cancelPendingPartial])

  const transcribeLabel = (() => {
    if (busy) return mode === 'mic' ? 'Working' : 'Stop'
    if (mode === 'mic') {
      if (recording) return 'Stop'
      if (s.mic.engine === 'websocket') return 'Transcribe'
      if (micBlob) return 'Transcribe'
      return 'Record'
    }
    return 'Transcribe'
  })()
  const transcribeIsAccent = busy || recording
  const transcribeShowsIcon = transcribeLabel === 'Transcribe'
  const onCommitClick = () => {
    if (busy) {
      cancelInFlight()
      return
    }
    if (mode === 'mic' && recording) {
      mic.stop()
      return
    }
    void transcribe()
  }
  const transcribeDisabled = (() => {
    if (busy) return mode === 'mic'
    if (mode === 'file') return !pickedFile
    if (mode === 'url') return !urlInput.trim() || urlInput === 'https://'
    if (mode === 'mic') return micBusy
    return false
  })()

  // Reference `m` once so the underlying setting still hydrates the
  // active modality's view; the rest of the Output/Timestamps logic
  // was retired with the format consolidation.
  void m

  return (
    <div className="sb">
      {/* Modality tabs */}
      <div className="sb__tabs-row">
        <div className="ma-tabs">
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              className={mode === id ? 'ma-tab ma-tab--active' : 'ma-tab'}
              onClick={() => onModeChange(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="sb__content">
        {mode === 'file' && (
          <UploadPane
            modality={s.file}
            update={(patch) => s.update('file', patch)}
            pickedFile={pickedFile}
            setPickedFile={setPickedFile}
            onClearSource={onClearSource}
            dragOver={dragOver}
            setDragOver={setDragOver}
            fileInputRef={fileInputRef}
            onFiles={onFiles}
          />
        )}
        {mode === 'url' && (
          <URLPane
            modality={s.url}
            update={(patch) => s.update('url', patch)}
            urlInput={urlInput}
            setUrlInput={setUrlInput}
          />
        )}
        {mode === 'mic' && (
          <RecordPane
            modality={s.mic}
            update={(patch) => s.update('mic', patch)}
            mic={mic}
            recording={recording}
            recordElapsed={recordElapsed}
            micBlob={micBlob}
            setMicBlob={setMicBlob}
          />
        )}
      </div>

      {progress !== null && (
        <div className="sb__progress">
          <div className="sb__progress-bar" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
      )}

      <div className="sb__commit">
        <button
          type="button"
          className={transcribeIsAccent ? 'ma-pill ma-pill--active' : 'ma-pill'}
          onClick={onCommitClick}
          disabled={transcribeDisabled}
        >
          {transcribeShowsIcon && (
            <svg
              viewBox="0 0 24 24"
              width="13"
              height="13"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
              <path d="M14 3v5h5" />
              <line x1="8.5" y1="12.5" x2="15" y2="12.5" />
              <line x1="8.5" y1="16" x2="14" y2="16" />
            </svg>
          )}
          {transcribeLabel}
        </button>
      </div>
      {mic.state === 'error' && mic.error && <div className="error">{mic.error}</div>}
    </div>
  )
}

// ── Shared section label ──────────────────────────────────────────
function SBLabel({ children, top }: { children: React.ReactNode; top?: number }) {
  return (
    <div className="label-eyebrow-row" style={{ marginTop: top ?? 0 }}>
      <span className="label-eyebrow">{children}</span>
      <span className="rule-extend" />
    </div>
  )
}

// ── Shared engine picker ──────────────────────────────────────────
function EnginePicker({
  modality,
  engine,
  setEngine,
}: {
  modality: Modality
  engine: Engine
  setEngine: (e: Engine) => void
}) {
  return (
    <OptionList<Engine>
      value={engine}
      options={[
        { id: 'rest', label: 'REST', disabledReason: ENGINE_TOOLTIPS[modality].rest },
        { id: 'websocket', label: 'WebSocket', disabledReason: ENGINE_TOOLTIPS[modality].websocket },
      ]}
      onChange={setEngine}
    />
  )
}

// ── Shared engine + strategy block ───────────────────────────────
// (Output/Timestamps controls were retired — verbose_json + segment+word
// are now hardcoded, with format choice deferred to the download flyout.)
function EngineStrategy({
  modality,
  m,
  update,
  /** When true, render the Strategy section even with engine=websocket,
   * disabling each option (with a tooltip). Lets the Record tab keep the
   * Strategy slot visible across engine switches so users see the option
   * exists rather than having it disappear. */
  alwaysShowStrategy = false,
}: {
  modality: Modality
  m: CommonModality
  update: (patch: Partial<CommonModality>) => void
  alwaysShowStrategy?: boolean
}) {
  const isRest = m.engine === 'rest'
  const showStrategy = isRest || alwaysShowStrategy
  return (
    <>
      <SBLabel top={22}>Engine</SBLabel>
      <EnginePicker
        modality={modality}
        engine={m.engine}
        setEngine={(e) => update({ engine: e })}
      />
      {showStrategy && (
        <>
          <SBLabel top={22}>Strategy</SBLabel>
          <OptionList<StrategyOverride>
            value={m.strategyOverride}
            options={STRATEGY_OVERRIDES.map((o) => ({
              id: o.id,
              label: o.label,
              disabled: !isRest,
              disabledReason: isRest
                ? o.hint
                : 'Strategy override only applies to the REST engine.',
            }))}
            onChange={(v) => update({ strategyOverride: v })}
          />
        </>
      )}
    </>
  )
}

// ── Shared WS-streaming knob block ───────────────────────────────
// Surfaces in the Advanced section whenever engine=websocket. Server
// only honors these on the WS-streaming dispatch path (the chunked
// engine reading from ffmpeg). REST + WS-accumulate (full/split_full)
// bypass ffmpeg entirely, so we hide the knobs there.
function WSStreamingKnobs({
  modality,
  update,
}: {
  modality: CommonModality
  update: (patch: Partial<CommonModality>) => void
}) {
  return (
    <>
      <ToggleKv
        k="Low-latency mode"
        hint="live_latency — 10-2-2 preset for faster partials at lower throughput"
        v={modality.liveLatency}
        onSet={(v) => update({ liveLatency: v })}
      />
      <NumKv
        k="High-pass filter"
        hint="hpf_hz — ffmpeg highpass cutoff in Hz; 0 disables"
        v={modality.hpfHz}
        placeholder={100}
        suffix="Hz"
        onSet={(v) => update({ hpfHz: v ?? 100 })}
      />

      <div className="sb__adv-divider" />

      <ToggleKv
        k="Voice detection"
        hint="vad_enabled — gates silence between ffmpeg and the engine"
        v={modality.vadEnabled}
        onSet={(v) => update({ vadEnabled: v })}
      />
      <NumKv
        k="Speech threshold"
        hint="vad_threshold — Silero probability cutoff 0..1; higher = pickier"
        v={modality.vadThreshold}
        placeholder={0.5}
        step={0.05}
        onSet={(v) => update({ vadThreshold: v ?? 0.5 })}
      />
      <NumKv
        k="Onset frames"
        hint="vad_consecutive — sustained speech frames to flip silent→speech"
        v={modality.vadConsecutive}
        placeholder={3}
        onSet={(v) => update({ vadConsecutive: v ?? 3 })}
      />
      <NumKv
        k="Hangover"
        hint="vad_hangover_ms — keep forwarding this long after the last loud frame"
        v={modality.vadHangoverMs}
        placeholder={500}
        suffix="ms"
        onSet={(v) => update({ vadHangoverMs: v ?? 500 })}
      />
      <NumKv
        k="Padding min gap"
        hint="vad_pad_min_gap_ms — onset padding fires only when prior silence exceeded this"
        v={modality.vadPadMinGapMs}
        placeholder={400}
        suffix="ms"
        onSet={(v) => update({ vadPadMinGapMs: v ?? 400 })}
      />
      <NumKv
        k="Onset padding"
        hint="vad_pad_duration_ms — low-noise padding injected at speech onset"
        v={modality.vadPadDurationMs}
        placeholder={0}
        suffix="ms"
        onSet={(v) => update({ vadPadDurationMs: v ?? 0 })}
      />
    </>
  )
}

// ── Upload pane ──────────────────────────────────────────────────
function UploadPane({
  modality,
  update,
  pickedFile,
  setPickedFile,
  onClearSource,
  dragOver,
  setDragOver,
  fileInputRef,
  onFiles,
}: {
  modality: CommonModality
  update: (patch: Partial<CommonModality>) => void
  pickedFile: File | null
  setPickedFile: (f: File | null) => void
  onClearSource?: () => void
  dragOver: boolean
  setDragOver: (b: boolean) => void
  fileInputRef: React.RefObject<HTMLInputElement | null>
  onFiles: (files: FileList | null) => void
}) {
  return (
    <div>
      <div
        className={dragOver ? 'sb__file-card sb__file-card--drag' : 'sb__file-card'}
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
        <div className="sb__file-card__icon">
          <svg
            viewBox="0 0 24 24"
            width="16"
            height="16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden
          >
            <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
            <path d="M14 3v5h5" />
          </svg>
        </div>
        <div className="sb__file-card__body">
          {pickedFile ? (
            <>
              <div className="sb__file-card__name">{pickedFile.name}</div>
              <div className="sb__file-card__meta">
                {formatBytes(pickedFile.size)} · {pickedFile.type || 'audio'}
              </div>
            </>
          ) : (
            <>
              <div className="sb__file-card__name sb__file-card__name--empty">
                Drop a file or browse
              </div>
              <div className="sb__file-card__meta">No file selected</div>
            </>
          )}
        </div>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,video/*"
        hidden
        onChange={(e) => onFiles(e.target.files)}
      />
      <div className="sb__file-actions">
        <button
          type="button"
          className="ma-pill ma-pill--grow"
          onClick={() => fileInputRef.current?.click()}
        >
          Browse…
        </button>
        {pickedFile && (
          <button
            type="button"
            className="ma-pill"
            aria-label="Remove file"
            onClick={(e) => {
              e.stopPropagation()
              setPickedFile(null)
              if (fileInputRef.current) fileInputRef.current.value = ''
              onClearSource?.()
            }}
          >
            <svg
              viewBox="0 0 24 24"
              width="11"
              height="11"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              aria-hidden
            >
              <path d="M19 6L6 19M6 6l13 13" />
            </svg>
          </button>
        )}
      </div>
      <div className="sb__file-formats">wav · mp3 · flac · m4a · ogg · webm</div>

      <EngineStrategy modality="file" m={modality} update={update} alwaysShowStrategy />

      <SBLabel top={22}>Advanced</SBLabel>
      <div className="sb__adv-list">
        {modality.engine === 'rest' && (
          <NumKv
            k="Long-audio threshold"
            hint="long_audio_threshold — switches to long-audio model settings above this duration"
            v={modality.longAudioThreshold}
            placeholder={480}
            suffix="s"
            onSet={(v) => update({ longAudioThreshold: v })}
          />
        )}
        {modality.engine === 'websocket' && (
          <WSStreamingKnobs modality={modality} update={update} />
        )}
      </div>
    </div>
  )
}

// ── URL pane ─────────────────────────────────────────────────────
function URLPane({
  modality,
  update,
  urlInput,
  setUrlInput,
}: {
  modality: CommonModality
  update: (patch: Partial<CommonModality>) => void
  urlInput: string
  setUrlInput: (v: string) => void
}) {
  const urlValid = /^https?:\/\/\S+/i.test(urlInput.trim()) && urlInput.trim() !== 'https://'
  return (
    <div>
      <div className="sb__url-field">
        <svg
          viewBox="0 0 24 24"
          width="14"
          height="14"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <path d="M10 13a5 5 0 0 0 7.07 0l3.18-3.18a5 5 0 0 0-7.07-7.07L11.34 5" />
          <path d="M14 11a5 5 0 0 0-7.07 0L3.75 14.18a5 5 0 0 0 7.07 7.07L12.66 19" />
        </svg>
        <input
          type="url"
          value={urlInput}
          onChange={(e) => setUrlInput(e.target.value)}
          className="sb__url-input"
          spellCheck={false}
          autoComplete="off"
          placeholder="https://…"
        />
      </div>
      <div className="sb__url-status">
        <span className={urlValid ? 'sb__url-dot sb__url-dot--on' : 'sb__url-dot'} />
        <span className="sb__url-status-label">
          {urlValid ? 'Ready to fetch' : 'Enter direct link to media'}
        </span>
      </div>

      <EngineStrategy modality="url" m={modality} update={update} alwaysShowStrategy />

      <SBLabel top={22}>Advanced</SBLabel>
      <div className="sb__adv-list">
        {modality.engine === 'rest' && (
          <NumKv
            k="Long-audio threshold"
            hint="long_audio_threshold — switches to long-audio model settings above this duration"
            v={modality.longAudioThreshold}
            placeholder={480}
            suffix="s"
            onSet={(v) => update({ longAudioThreshold: v })}
          />
        )}
        {modality.engine === 'websocket' && (
          <WSStreamingKnobs modality={modality} update={update} />
        )}
      </div>
    </div>
  )
}

// ── Record pane ──────────────────────────────────────────────────
type MicReturn = ReturnType<typeof useMic>
function RecordPane({
  modality,
  update,
  mic,
  recording,
  recordElapsed,
  micBlob,
  setMicBlob,
}: {
  modality: MicModality
  update: (patch: Partial<MicModality>) => void
  mic: MicReturn
  recording: boolean
  recordElapsed: number
  micBlob: File | null
  setMicBlob: (f: File | null) => void
}) {
  const isWs = modality.engine === 'websocket'
  return (
    <div>
      <div className="mic-card">
        <MicCardBars
          getAnalyser={mic.getAnalyser}
          active={recording || isWs}
        />
        <div className="mic-card__row">
          <span className="mic-card__status">
            {recording ? 'Recording' : isWs ? 'Listening' : 'Standby'}
          </span>
          <span
            className={
              recording || (isWs && !recording)
                ? 'mic-card__timer mic-card__timer--rec'
                : 'mic-card__timer'
            }
          >
            {isWs && !recording ? '——' : formatMicTimer(recordElapsed)}
          </span>
        </div>
      </div>

      {!recording && micBlob && !isWs && (
        <div className="sb__file-card" style={{ marginTop: 10 }}>
          <div className="sb__file-card__icon">
            <svg
              viewBox="0 0 24 24"
              width="16"
              height="16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
              <path d="M14 3v5h5" />
            </svg>
          </div>
          <div className="sb__file-card__body">
            <div className="sb__file-card__name">{micBlob.name}</div>
            <div className="sb__file-card__meta">{formatBytes(micBlob.size)} · WebM Opus</div>
          </div>
          <button
            type="button"
            className="ma-pill ma-pill--sm"
            aria-label="Discard recording"
            onClick={(e) => {
              e.stopPropagation()
              setMicBlob(null)
            }}
          >
            ✕
          </button>
        </div>
      )}

      <div className="mic-card__hint">16 kHz · Mono · WebM Opus</div>

      <EngineStrategy
        modality="mic"
        m={modality}
        update={update}
        alwaysShowStrategy
      />

      <SBLabel top={22}>Advanced</SBLabel>
      <div className="sb__adv-list">
        {/* Browser getUserMedia constraint — applies to mic capture
            regardless of engine (Live or Record). The only knob that's
            meaningful on Mic+REST. */}
        <ToggleKv
          k="Noise suppression"
          hint="noise_suppression — browser getUserMedia noise suppression"
          v={modality.noiseSuppression}
          onSet={(v) => update({ noiseSuppression: v })}
        />
        {!isWs && (
          <NumKv
            k="Long-audio threshold"
            hint="long_audio_threshold — switches to long-audio model settings above this duration"
            v={modality.longAudioThreshold}
            placeholder={480}
            suffix="s"
            onSet={(v) => update({ longAudioThreshold: v })}
          />
        )}
        {isWs && (
          <>
            <div className="sb__adv-divider" />
            <WSStreamingKnobs modality={modality} update={update} />
          </>
        )}
      </div>
    </div>
  )
}

const MIC_BARS = 28

function MicCardBars({
  getAnalyser,
  active,
}: {
  getAnalyser: () => AnalyserNode | null
  active: boolean
}) {
  const [bands, setBands] = useState<number[]>(() => new Array(MIC_BARS).fill(0))
  const dispRef = useRef<Float32Array>(new Float32Array(MIC_BARS))

  useEffect(() => {
    if (!active) {
      dispRef.current.fill(0)
      setBands(new Array(MIC_BARS).fill(0))
      return
    }
    let raf = 0
    let freq: Uint8Array | null = null
    // Cap to ~30 Hz — sampling + setBands(Array.from(disp)) was the
    // dominant per-frame cost in this loop, and 30 Hz is visually
    // indistinguishable from 60 here (the bars already smooth via the
    // exponential ema below).
    let skip = false
    const loop = () => {
      skip = !skip
      if (skip) {
        raf = requestAnimationFrame(loop)
        return
      }
      const analyser = getAnalyser()
      if (analyser) {
        if (!freq || freq.length !== analyser.frequencyBinCount) {
          freq = new Uint8Array(analyser.frequencyBinCount)
        }
        analyser.getByteFrequencyData(freq as unknown as Uint8Array<ArrayBuffer>)
        const minBin = 2
        const maxBin = Math.min(freq.length - 1, 300)
        const ratio = maxBin / minBin
        const disp = dispRef.current
        const raw = new Float32Array(MIC_BARS)
        for (let b = 0; b < MIC_BARS; b++) {
          const lo = Math.floor(minBin * Math.pow(ratio, b / MIC_BARS))
          const hi = Math.max(lo + 1, Math.floor(minBin * Math.pow(ratio, (b + 1) / MIC_BARS)))
          let sum = 0
          let n = 0
          for (let k = lo; k < hi && k < freq.length; k++) {
            sum += freq[k]
            n++
          }
          const avg = n > 0 ? sum / n / 255 : 0
          raw[b] = Math.pow(avg, 0.7)
        }
        for (let b = 0; b < MIC_BARS; b++) {
          const prev = raw[b - 1] ?? raw[b]
          const next = raw[b + 1] ?? raw[b]
          const target = (prev + 2 * raw[b] + next) / 4
          const k = target > disp[b] ? 0.35 : 0.12
          disp[b] += (target - disp[b]) * k
        }
        setBands(Array.from(disp))
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [active, getAnalyser])

  return (
    <div className="mic-card__bars">
      {bands.map((v, i) => {
        const h = active ? Math.max(4, Math.min(96, v * 92 + 4)) : 4
        return (
          <span
            key={i}
            className={active ? 'mic-card__bar mic-card__bar--on' : 'mic-card__bar'}
            style={{ height: `${h}%`, opacity: active ? 0.55 + 0.45 * v : 1 }}
          />
        )
      })}
    </div>
  )
}

function formatMicTimer(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  const t = Math.floor((seconds * 10) % 10)
  return `${m}:${String(s).padStart(2, '0')}.${t}`
}

function NumKv({
  k,
  hint,
  v,
  placeholder,
  suffix,
  step,
  onSet,
}: {
  k: string
  hint?: string
  v: number | null
  placeholder: number
  suffix?: string
  step?: number
  onSet: (v: number | null) => void
}) {
  return (
    <div className="ma-kv">
      <span className="ma-kv__k" title={hint}>
        {k}
      </span>
      <span className="ma-kv__v">
        <input
          type="number"
          className="ma-kv__input"
          value={v ?? ''}
          placeholder={`${placeholder}`}
          step={step ?? 1}
          spellCheck={false}
          onChange={(e) => onSet(e.target.value === '' ? null : Number(e.target.value))}
        />
        <span className="ma-kv__unit">{suffix ?? ''}</span>
      </span>
    </div>
  )
}

function ToggleKv({
  k,
  hint,
  v,
  onSet,
}: {
  k: string
  hint?: string
  v: boolean
  onSet: (v: boolean) => void
}) {
  return (
    <div className="ma-kv">
      <span className="ma-kv__k" title={hint}>
        {k}
      </span>
      <span className="ma-kv__v" style={{ justifyContent: 'flex-start' }}>
        <button
          type="button"
          className={v ? 'ma-switch ma-switch--on' : 'ma-switch'}
          onClick={() => onSet(!v)}
          aria-pressed={v}
        >
          <span className="ma-switch__label ma-switch__label--off">Off</span>
          <span className="ma-switch__label ma-switch__label--on">On</span>
          <span className="ma-switch__knob" />
        </button>
      </span>
    </div>
  )
}
