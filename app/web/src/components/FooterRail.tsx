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
export function FooterRail({ loaded, result, ttfs, activeStrategy }: Props) {
  const dur = useDuration()
  const verbose = result?.format === 'verbose_json' ? result.body : null
  const asr = verbose?.transcription_time_seconds
  const rtfx = asr && asr > 0 ? (verbose!.duration / asr).toFixed(1) + '×' : '—'
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
        <Metric label="RTFx" v={rtfx} hot />
        <Metric label="TTFS" v={ttfsLabel} />
        <Metric label="SEGMENTS" v={segs} />
        <Metric label="WORDS" v={words} />
      </div>
    </div>
  )
}

function Metric({ label, v, hot }: { label: string; v: string | number; hot?: boolean }) {
  return (
    <div className="footer__metric">
      <span className="footer__metric-l">{label}</span>
      <span className={hot ? 'footer__metric-v footer__metric-v--hot' : 'footer__metric-v'}>
        {v}
      </span>
    </div>
  )
}
