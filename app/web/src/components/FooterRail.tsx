import { useEffect, useState } from 'react'
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
   * into the live-ratio formula instead of the offline duration/asr_time. */
  live: boolean
  /** performance.now() at the start of the active session. Lets us compute
   * wall-elapsed for the live RTFx ratio without re-reading state every
   * tick. Pass 0 when no session is active. */
  sessionStartMs: number
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
  sessionStartMs,
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
  // Live ratio: how well are we keeping up with realtime? 1.0 = on pace,
  // <1 = falling behind, >1 = burst-eating backlog. Ticked once a second
  // by an interval that captures `performance.now()` outside render — the
  // value is stored in state so the render pass stays pure. We only ever
  // call setWallElapsed from the async interval callback (not from the
  // effect body) so the lint rule doesn't flag a sync-in-effect setState.
  const [wallElapsed, setWallElapsed] = useState(0)
  useEffect(() => {
    if (!live || sessionStartMs <= 0) return
    const id = setInterval(() => {
      setWallElapsed((performance.now() - sessionStartMs) / 1000)
    }, 1000)
    return () => clearInterval(id)
  }, [live, sessionStartMs])
  let rtfxLabel = offlineLabel
  let rtfxTip = ''
  if (live && sessionStartMs > 0) {
    const audioReceived = verbose?.duration ?? 0
    if (wallElapsed > 0 && audioReceived > 0) {
      const ratio = audioReceived / wallElapsed
      // Anything ≥ 0.95 reads as "keeping up" in practice (we're bounded
      // by chunk emission cadence, not GPU). Below that label it.
      const tag = ratio >= 0.95 ? 'LIVE' : 'falling behind'
      rtfxLabel = `${ratio.toFixed(2)}× ${tag}`
      rtfxTip =
        offlineRtfx != null
          ? `headroom: ${offlineRtfx.toFixed(1)}× (asr_time vs duration)`
          : ''
    } else {
      rtfxLabel = '—'
    }
  }
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
