import type { CompactJsonResponse } from '../../lib/types'

export function JsonView({ body }: { body: CompactJsonResponse }) {
  return (
    <div className="output">
      <h3>{body.text ? 'Transcript' : 'No text returned'}</h3>
      <p className="output__plaintext">{body.text}</p>
      <details>
        <summary>Raw JSON</summary>
        <pre className="output__raw">{JSON.stringify(body, null, 2)}</pre>
      </details>
    </div>
  )
}
