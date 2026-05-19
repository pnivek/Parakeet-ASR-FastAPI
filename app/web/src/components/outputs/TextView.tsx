interface Props {
  body: string
}

export function TextView({ body }: Props) {
  return (
    <div className="output">
      <h3>Transcript (plain text)</h3>
      <pre className="output__textbody">{body}</pre>
      <button
        type="button"
        onClick={() => navigator.clipboard.writeText(body)}
        className="output__copy"
      >
        Copy to clipboard
      </button>
    </div>
  )
}
