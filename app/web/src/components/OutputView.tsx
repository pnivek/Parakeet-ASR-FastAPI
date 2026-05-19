import type { TranscriptionResponse } from '../lib/api'
import { JsonView } from './outputs/JsonView'
import { SrtView } from './outputs/SrtView'
import { TextView } from './outputs/TextView'
import { VerboseJsonView } from './outputs/VerboseJsonView'
import { downloadResult, extForFormat } from '../lib/download'

interface Props {
  result: TranscriptionResponse
  filename: string
  currentTime?: number
}

const DownloadIcon = () => (
  <svg viewBox="0 0 24 24" width={13} height={13} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M12 4v12M12 16l-4-4M12 16l4-4" />
    <path d="M4 20h16" />
  </svg>
)

/**
 * Top-level output card with header (format pill + Download) and the
 * format-driven inner view.
 */
export function OutputView({ result, filename, currentTime }: Props) {
  const ext = extForFormat(result.format)
  return (
    <section className="glass output">
      <div className="output__head">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span className="output__label">RESPONSE</span>
          <span className="output__format-pill">{result.format}</span>
        </div>
        <button type="button" className="export-btn" onClick={() => downloadResult(result, filename)}>
          <DownloadIcon />
          Download
          <span className="export-btn__ext">.{ext}</span>
        </button>
      </div>
      <div className="output__body">
        {result.format === 'json' && <JsonView body={result.body} />}
        {result.format === 'verbose_json' && (
          <VerboseJsonView body={result.body} currentTime={currentTime} />
        )}
        {result.format === 'text' && <TextView body={result.body} />}
        {result.format === 'srt' && <SrtView body={result.body} variant="srt" />}
        {result.format === 'vtt' && <SrtView body={result.body} variant="vtt" />}
      </div>
    </section>
  )
}
