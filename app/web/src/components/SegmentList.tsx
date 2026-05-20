import { Fragment, useState } from 'react'
import type { WhisperSegment } from '../lib/types'

const NS_TIP =
  'no_speech_prob has no equivalent in the Parakeet TDT decoder — Whisper computes it from a dedicated `<|nospeech|>` encoder token, which Parakeet does not emit.'
const AVG_TIP =
  'avg_logprob is unavailable on FULL_GRAPH-mode CUDA-graph decode (NeMo 2.7.3). Re-deploy with USE_CUDA_GRAPHS=false to populate it at ~2× slower decode.'

function fmt(t: number): string {
  const m = Math.floor(t / 60)
  const s = (t - m * 60).toFixed(2).padStart(5, '0')
  return `${String(m).padStart(2, '0')}:${s}`
}

interface Props {
  segments: WhisperSegment[]
  /** Current audio playback time, used to highlight the playing segment. */
  currentTime?: number
  /** Callback when a segment row is clicked — usually to seek the audio. */
  onSeek?: (t: number) => void
}

/**
 * Whisper-style segments table. Click a row → seek audio. Each row expands
 * to show the per-segment Whisper metadata fields (tokens, logprob,
 * compression ratio, no_speech_prob). Honest tooltips on null fields explain
 * why they're null instead of fabricating values.
 */
export function SegmentList({ segments, currentTime, onSeek }: Props) {
  const [openId, setOpenId] = useState<number | null>(null)

  if (segments.length === 0) {
    return <p className="muted">No segments.</p>
  }

  return (
    <table className="segments">
      <thead>
        <tr>
          <th style={{ width: '8ch' }}>Start</th>
          <th style={{ width: '8ch' }}>End</th>
          <th>Text</th>
        </tr>
      </thead>
      <tbody>
        {segments.map((seg) => {
          const playing =
            currentTime !== undefined && currentTime >= seg.start && currentTime < seg.end
          const open = openId === seg.id
          return (
            <Fragment key={seg.id}>
              <tr
                className={`segments__row ${playing ? 'segments__row--playing' : ''}`}
                onClick={() => {
                  setOpenId(open ? null : seg.id)
                  onSeek?.(seg.start)
                }}
              >
                <td className="tabular">{fmt(seg.start)}</td>
                <td className="tabular">{fmt(seg.end)}</td>
                <td>{seg.text}</td>
              </tr>
              {open && (
                <tr className="segments__detail">
                  <td colSpan={3}>
                    <dl>
                      <dt>tokens</dt>
                      <dd>
                        {seg.tokens.length} ids · first 8:{' '}
                        <code>[{seg.tokens.slice(0, 8).join(', ')}{seg.tokens.length > 8 ? ', …' : ''}]</code>
                      </dd>
                      <dt>temperature</dt>
                      <dd>{seg.temperature}</dd>
                      <dt>compression_ratio</dt>
                      <dd>{seg.compression_ratio}</dd>
                      <dt>avg_logprob</dt>
                      <dd>
                        {seg.avg_logprob === null ? (
                          <span className="null" title={AVG_TIP}>
                            null
                          </span>
                        ) : (
                          seg.avg_logprob
                        )}
                      </dd>
                      <dt>no_speech_prob</dt>
                      <dd>
                        <span className="null" title={NS_TIP}>
                          {seg.no_speech_prob === null ? 'null' : seg.no_speech_prob}
                        </span>
                      </dd>
                    </dl>
                  </td>
                </tr>
              )}
            </Fragment>
          )
        })}
      </tbody>
    </table>
  )
}
