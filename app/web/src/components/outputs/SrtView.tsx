interface Props {
  body: string
  /** "srt" or "vtt" — used for label semantics; download is on the parent. */
  variant: 'srt' | 'vtt'
}

export function SrtView({ body }: Props) {
  return <pre className="code">{body}</pre>
}
