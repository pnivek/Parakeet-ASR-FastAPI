import type { TranscriptionResponse } from '../lib/api'
import type { LoadedAudio } from '../lib/download'
import { useDuration } from '../lib/playback'

interface Props {
  loaded: LoadedAudio | null
  result: TranscriptionResponse | null
  /** Time-to-first-segment in seconds (click → first segment), or null. */
  ttfs: number | null
  /** The currently-selected engine — shown until a result reports the
   * resolved one, so STRATEGY is never blank. */
  activeStrategy: string
  /** True while partials are arriving (mic+WS or URL+WS). Drives RTFx
   * into the live counter-based formula instead of the offline
   * duration/asr_time. */
  live: boolean
}

/** Map the server's resolved-strategy enum to a short, human-readable
 * label for the footer chip. Unknown values pass through verbatim. */
const STRATEGY_LABEL: Record<string, string> = {
  full: 'Full pass',
  split_full: 'Split-full',
  streaming: 'Streaming',
  // `offline` shouldn't appear in a result (it always resolves to FULL or
  // SPLIT_FULL), but show it as-is if it does.
  offline: 'offline',
}

/**
 * 68 px sticky footer. Currently-loaded file on the left, real metrics on
 * the right (resolved strategy, ASR time, RTFx, time-to-first-segment,
 * segments, words). When no result yet, the metric values show "—".
 */
export function FooterRail({
  loaded,
  result,
  ttfs,
  activeStrategy,
  live,
}: Props) {
  const dur = useDuration()
  const verbose = result?.format === 'verbose_json' ? result.body : null
  const asr = verbose?.transcription_time_seconds
  // Offline ratio: how fast did the GPU finish vs the source's duration.
  // For live sessions this turns into 19×-ish ("we spent 5% of the time
  // compute-bound, 95% waiting on audio") — useless as a live metric, so
  // we hand it to the tooltip instead.
  const offlineRtfx = asr && asr > 0 ? verbose!.duration / asr : null
  const offlineLabel = offlineRtfx != null ? offlineRtfx.toFixed(1) + '×' : '—'
  let rtfxLabel = offlineLabel
  // Tooltip starts with offline-headroom info (when available) and
  // appends raw counters in live mode for diagnostic transparency —
  // the cap at min(1, committed/received) is hiding the true
  // lag/overshoot otherwise.
  const tipParts: string[] = []
  // "Live" formula is only meaningful for true realtime sources (mic
  // recording in progress, broadcast URL stream). File uploads — even
  // when streamed via WebSocket — process the source faster than
  // realtime, and the offline duration/asr_time metric correctly
  // shows the "X× faster" ratio. Without this gate the live formula
  // caps at 1.00× LIVE which hides the file's real throughput.
  const useLiveFormula = live && loaded?.kind !== 'file'
  if (offlineRtfx != null && useLiveFormula) {
    tipParts.push(`headroom: ${offlineRtfx.toFixed(1)}× (asr_time vs duration)`)
  }
  if (useLiveFormula) {
    // Live ratio expressed in **speech-seconds** so silence/music drops
    // out symmetrically. Server tracks Silero-detected speech in
    // (a) `speech_received_s` (cumulative wall-clock speech ingested)
    // and (b) `speech_committed_s` (sum of committed segment spans
    // post-translation). Capped at 1.0 because committed segments can
    // span more wall-time than VAD classified as speech (brief mid-
    // sentence pauses below threshold).
    const received = verbose?.speech_received_s ?? 0
    const committed = verbose?.speech_committed_s ?? 0
    const audio = verbose?.audio_received_s ?? 0
    // Warm-up guard: until we have a meaningful denominator the ratio
    // is too noisy to display. Also covers "Silero not loaded" (counters
    // never arrive → both stay at 0 → show —).
    if (received >= 1.0) {
      const rawRatio = committed / received
      const ratio = Math.min(1.0, rawRatio)
      const tag = ratio >= 0.9 ? 'LIVE' : 'falling behind'
      rtfxLabel = `${ratio.toFixed(2)}× ${tag}`
      tipParts.push(
        `received ${received.toFixed(1)}s speech / ${audio.toFixed(1)}s audio`,
      )
      tipParts.push(
        `committed ${committed.toFixed(1)}s` +
          (rawRatio > 1 ? ` (raw ratio ${rawRatio.toFixed(2)}× capped at 1.00×)` : ''),
      )
    } else {
      rtfxLabel = '—'
      tipParts.push(
        `warming up — need ≥ 1.0s speech (have ${received.toFixed(1)}s)`,
      )
    }
  }
  const rtfxTip = tipParts.join('\n')
  const segs = verbose ? verbose.segments.length : '—'
  const words = verbose?.words?.length ?? '—'
  const asrLabel = asr ? `${asr.toFixed(2)}s` : '—'
  const ttfsLabel = ttfs != null ? `${ttfs.toFixed(2)}s` : '—'
  // Resolved strategy from the result wins; otherwise show the active
  // engine so STRATEGY is always populated (live mic, pre-result, etc).
  const rawStrategy = verbose?.strategy || activeStrategy
  const strategy = STRATEGY_LABEL[rawStrategy] ?? rawStrategy

  return (
    <div className="footer">
      <div className="footer__l">
        <span className="footer__l-label">SOURCE</span>
        {loaded ? (
          <div className="footer__file">
            <span className="footer__file-name">{loaded.title}</span>
            <span className="footer__file-meta">
              {loaded.source}
              {dur > 0 && <> · {dur.toFixed(2)}s</>}
            </span>
          </div>
        ) : (
          <span className="footer__file-meta">no audio loaded</span>
        )}
      </div>
      <div className="footer__r">
        <Metric label="STRATEGY" v={strategy} />
        <Metric label="ASR" v={asrLabel} />
        <Metric label="RTFx" v={rtfxLabel} hot title={rtfxTip || undefined} />
        <Metric label="TTFS" v={ttfsLabel} />
        <Metric label="SEGMENTS" v={segs} />
        <Metric label="WORDS" v={words} />
      </div>
    </div>
  )
}

function Metric({
  label,
  v,
  hot,
  title,
}: {
  label: string
  v: string | number
  hot?: boolean
  title?: string
}) {
  return (
    <div className="footer__metric" title={title}>
      <span className="footer__metric-l">{label}</span>
      <span className={hot ? 'footer__metric-v footer__metric-v--hot' : 'footer__metric-v'}>
        {v}
      </span>
    </div>
  )
}
