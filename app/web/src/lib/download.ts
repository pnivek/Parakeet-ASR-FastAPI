import type { TranscriptionResponse } from './api'

/** Map a response_format to the extension the user expects on the file. */
export function extForFormat(fmt: TranscriptionResponse['format']): string {
  switch (fmt) {
    case 'json':
    case 'verbose_json':
      return 'json'
    case 'text':
      return 'txt'
    case 'srt':
      return 'srt'
    case 'vtt':
      return 'vtt'
  }
}

/** Serialize a result back to a string the user can download. */
export function serializeResult(result: TranscriptionResponse): { content: string; mime: string } {
  switch (result.format) {
    case 'json':
    case 'verbose_json':
      return { content: JSON.stringify(result.body, null, 2), mime: 'application/json' }
    case 'srt':
      return { content: result.body, mime: 'application/x-subrip' }
    case 'vtt':
      return { content: result.body, mime: 'text/vtt' }
    case 'text':
      return { content: result.body, mime: 'text/plain' }
  }
}

/** Trigger a browser download for the given result, using the given base name. */
export function downloadResult(result: TranscriptionResponse, baseName: string) {
  const { content, mime } = serializeResult(result)
  const ext = extForFormat(result.format)
  const safe = baseName.replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '_') || 'transcript'
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${safe}.${ext}`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
