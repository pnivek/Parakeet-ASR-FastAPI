interface Props {
  body: string
}

export function TextView({ body }: Props) {
  return <div className="plain-text" style={{ color: 'var(--fg)' }}>{body}</div>
}
