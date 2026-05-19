import type { VerboseJsonResponse } from '../../lib/types'
import { SegmentList } from '../SegmentList'

interface Props {
  body: VerboseJsonResponse
  /** Current audio time, drives row highlighting + word timeline (Phase 4). */
  currentTime?: number
  onSeek?: (t: number) => void
}

export function VerboseJsonView({ body, currentTime, onSeek }: Props) {
  return (
    <div className="output">
      <header className="output__header">
        <h3>Transcription</h3>
        <dl className="output__meta">
          <dt>strategy</dt><dd>{body.strategy}</dd>
          <dt>duration</dt><dd>{body.duration.toFixed(2)} s</dd>
          <dt>ASR time</dt><dd>{body.transcription_time_seconds.toFixed(2)} s</dd>
          <dt>RTFx</dt>
          <dd>
            {body.transcription_time_seconds > 0
              ? `${(body.duration / body.transcription_time_seconds).toFixed(1)}×`
              : '—'}
          </dd>
          <dt>segments</dt><dd>{body.segments.length}</dd>
          {body.words && <><dt>words</dt><dd>{body.words.length}</dd></>}
        </dl>
      </header>

      {body.text && (
        <details className="output__plain" open>
          <summary>Plain text</summary>
          <p>{body.text}</p>
        </details>
      )}

      <SegmentList segments={body.segments} currentTime={currentTime} onSeek={onSeek} />

      <details>
        <summary>Raw response ({body.segments.length} segments, {Object.keys(body).length} keys)</summary>
        <pre className="output__raw">{JSON.stringify(body, null, 2)}</pre>
      </details>
    </div>
  )
}
