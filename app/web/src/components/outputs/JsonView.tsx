import type { CompactJsonResponse } from '../../lib/types'

export function JsonView({ body }: { body: CompactJsonResponse }) {
  return <pre className="code">{JSON.stringify(body, null, 2)}</pre>
}
