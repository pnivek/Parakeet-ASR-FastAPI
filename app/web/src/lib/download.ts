import type { TranscriptionResponse } from './api'
import type { VerboseJsonResponse } from './types'

/**
 * The currently-loaded audio. Two shapes: an uploaded File (we own the
 * blob), or a URL ingested via the server (we don't have local bytes,
 * just a label and origin URL).
 */
export type LoadedAudio =
  | { kind: 'file'; title: string; source: string; file: File }
  | { kind: 'url'; title: string; source: string; url: string }

/** Target format the user chose at *download time* — independent of the
 * request format. We always request verbose_json, so any of these can be
 * generated from a single response. */
export type DownloadFormat = 'json' | 'text' | 'srt' | 'vtt' | 'csv'

const EXT: Record<DownloadFormat, string> = {
  json: 'json',
  text: 'txt',
  srt: 'srt',
  vtt: 'vtt',
  csv: 'csv',
}

const MIME: Record<DownloadFormat, string> = {
  json: 'application/json',
  text: 'text/plain',
  srt: 'application/x-subrip',
  vtt: 'text/vtt',
  csv: 'text/csv',
}

/** WebVTT serializer — fallback for the (transitional) case where a
 * server response didn't include `vtt_content`. Mirrors the server's
 * `_segments_to_vtt` so output is byte-identical. */
function segmentsToVtt(body: VerboseJsonResponse): string {
  const fmt = (t: number) => {
    const h = Math.floor(t / 3600)
    const m = Math.floor((t % 3600) / 60)
    const s = t % 60
    const pad2 = (n: number) => String(n).padStart(2, '0')
    const sec = s.toFixed(3).padStart(6, '0')
    return `${pad2(h)}:${pad2(m)}:${sec}`
  }
  const lines = ['WEBVTT', '']
  for (const s of body.segments) {
    lines.push(`${fmt(s.start)} --> ${fmt(s.end)}`)
    lines.push((s.text || '').trim())
    lines.push('')
  }
  return lines.join('\n')
}

/** Materialize a verbose_json result as the chosen download format.
 * Falls back gracefully when the server hasn't shipped `vtt_content`
 * yet (computed client-side from segments). */
function contentFor(body: VerboseJsonResponse, format: DownloadFormat): string {
  switch (format) {
    case 'json':
      return JSON.stringify(body, null, 2)
    case 'text':
      return body.text
    case 'srt':
      return body.srt_content ?? ''
    case 'vtt':
      return body.vtt_content ?? segmentsToVtt(body)
    case 'csv':
      return body.csv_content ?? ''
  }
}

/** Trigger a browser download of the result in `format`, picking the
 * extension + mime automatically. No-ops if the result isn't a
 * verbose_json payload (shouldn't happen — we always request verbose). */
export function downloadAs(
  result: TranscriptionResponse,
  format: DownloadFormat,
  baseName: string,
) {
  if (result.format !== 'verbose_json') return
  const content = contentFor(result.body, format)
  const safe =
    baseName.replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '_') || 'transcript'
  triggerDownload(
    new Blob([content], { type: MIME[format] }),
    `${safe}.${EXT[format]}`,
  )
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
