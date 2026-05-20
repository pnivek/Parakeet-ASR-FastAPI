interface Props {
  body: string
  /** "srt" or "vtt" — drives the download filename + heading. */
  variant: 'srt' | 'vtt'
}

/**
 * SubRip / WebVTT view. The server returns the file content as a plain text
 * body; we just render it and offer a download. The grayed timestamp lines
 * are highlighted with a slightly different color so the structure is readable.
 */
export function SrtView({ body, variant }: Props) {
  const download = () => {
    const blob = new Blob([body], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `transcript.${variant}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  return (
    <div className="output">
      <header className="output__header">
        <h3>{variant.toUpperCase()} subtitle</h3>
        <button type="button" onClick={download}>
          Download .{variant}
        </button>
      </header>
      <pre className="output__subtitle">{body}</pre>
    </div>
  )
}
