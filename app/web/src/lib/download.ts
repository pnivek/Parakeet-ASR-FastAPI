import type { TranscriptionResponse } from './api'

/**
 * The currently-loaded audio. Two shapes: an uploaded File (we own the
 * blob), or a URL ingested via the server (we don't have local bytes,
 * just a label and origin URL).
 */
export type LoadedAudio =
  | { kind: 'file'; title: string; source: string; file: File }
  | { kind: 'url'; title: string; source: string; url: string }

/** Map response_format to the extension the user expects. */
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

/** Serialize a result body for download. */
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

/** Trigger a browser download of the transcription result. */
export function downloadResult(result: TranscriptionResponse, baseName: string) {
  const { content, mime } = serializeResult(result)
  const ext = extForFormat(result.format)
  const safe = baseName.replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '_') || 'transcript'
  triggerDownload(new Blob([content], { type: mime }), `${safe}.${ext}`)
}

/** Download the original audio (only works for file-uploaded audio). */
export function downloadAudio(loaded: LoadedAudio) {
  if (loaded.kind === 'file') {
    triggerDownload(loaded.file, loaded.file.name)
  } else {
    window.open(loaded.url, '_blank', 'noopener')
  }
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
