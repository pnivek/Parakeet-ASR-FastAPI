import type { TranscriptionResponse } from '../lib/api'
import { JsonView } from './outputs/JsonView'
import { SrtView } from './outputs/SrtView'
import { TextView } from './outputs/TextView'
import { VerboseJsonView } from './outputs/VerboseJsonView'

interface Props {
  result: TranscriptionResponse
  currentTime?: number
  onSeek?: (t: number) => void
}

/** Switches the result view based on the format the server returned. */
export function OutputView({ result, currentTime, onSeek }: Props) {
  switch (result.format) {
    case 'json':
      return <JsonView body={result.body} />
    case 'verbose_json':
      return <VerboseJsonView body={result.body} currentTime={currentTime} onSeek={onSeek} />
    case 'text':
      return <TextView body={result.body} />
    case 'srt':
      return <SrtView body={result.body} variant="srt" />
    case 'vtt':
      return <SrtView body={result.body} variant="vtt" />
  }
}
