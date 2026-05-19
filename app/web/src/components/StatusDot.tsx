export type StreamState = 'idle' | 'streaming' | 'done' | 'error'

const LABEL: Record<StreamState, string> = {
  idle: 'IDLE',
  streaming: 'STREAMING',
  done: 'COMPLETE',
  error: 'ERROR',
}

export function StatusDot({ state }: { state: StreamState }) {
  return (
    <span className={`statusdot statusdot--${state}`}>
      <span className="statusdot__dot" />
      {LABEL[state]}
    </span>
  )
}
