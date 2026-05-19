import { useState, useMemo, Fragment } from 'react'
import type { VerboseJsonResponse, WhisperSegment } from '../../lib/types'
import { seek } from '../../lib/playback'
import { formatTime } from '../../lib/format'

interface Props {
  body: VerboseJsonResponse
  currentTime?: number
  onSeek?: (t: number) => void
}

type Tab = 'text' | 'timeline' | 'segments' | 'raw'
const TABS: Tab[] = ['text', 'timeline', 'segments', 'raw']

const NULL_TIPS: Record<string, string> = {
  avg_logprob:
    'Not exposed on FULL_GRAPH-mode CUDA-graph decode (NeMo 2.7.3). Set USE_CUDA_GRAPHS=false to populate at ~2× slower decode.',
  no_speech_prob:
    'No equivalent in Parakeet TDT — Whisper derives this from a dedicated `<|nospeech|>` encoder token that Parakeet does not emit.',
}

export function VerboseJsonView({ body, currentTime = 0 }: Props) {
  const [tab, setTab] = useState<Tab>('text')

  const activeSegIdx = useMemo(
    () => body.segments.findIndex((s) => currentTime >= s.start && currentTime <= s.end),
    [body.segments, currentTime],
  )
  const activeWordIdx = useMemo(() => {
    if (!body.words) return -1
    return body.words.findIndex((w) => currentTime >= w.start && currentTime <= w.end)
  }, [body.words, currentTime])

  const rtfx =
    body.transcription_time_seconds > 0
      ? (body.duration / body.transcription_time_seconds).toFixed(1)
      : '—'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="meta-row">
        <MetaCell label="STRATEGY" value={body.strategy} />
        <MetaCell label="DURATION" value={`${body.duration.toFixed(2)}s`} mono />
        <MetaCell label="ASR TIME" value={`${body.transcription_time_seconds.toFixed(2)}s`} mono />
        <MetaCell label="RTFx" value={`${rtfx}×`} mono hot />
        <MetaCell label="SEGMENTS" value={body.segments.length} mono />
        <MetaCell label="WORDS" value={body.words?.length ?? 0} mono />
      </div>

      <div className="subtab-row">
        <div className="subtabs" role="tablist">
          {TABS.map((v) => (
            <button
              key={v}
              type="button"
              role="tab"
              aria-selected={tab === v}
              className={tab === v ? 'subtabs__btn subtabs__btn--active' : 'subtabs__btn'}
              onClick={() => setTab(v)}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      {tab === 'text' && (
        <PlainText body={body} activeWordIdx={activeWordIdx} />
      )}
      {tab === 'timeline' && <WordTimelineChips body={body} activeWordIdx={activeWordIdx} />}
      {tab === 'segments' && <SegmentsTable segments={body.segments} activeIdx={activeSegIdx} />}
      {tab === 'raw' && <RawJson body={body} />}
    </div>
  )
}

function MetaCell({
  label,
  value,
  mono,
  hot,
}: {
  label: string
  value: string | number
  mono?: boolean
  hot?: boolean
}) {
  return (
    <div className={`meta-cell${hot ? ' meta-cell--hot' : ''}${mono ? ' meta-cell--mono' : ''}`}>
      <div className="meta-cell__l">{label}</div>
      <div className="meta-cell__v">{value}</div>
    </div>
  )
}

function PlainText({ body, activeWordIdx }: { body: VerboseJsonResponse; activeWordIdx: number }) {
  if (body.words && body.words.length > 0) {
    return (
      <div className="plain-text">
        {body.words.map((w, i) => {
          const cls =
            i === activeWordIdx ? 'plain-text__word plain-text__word--active' : 'plain-text__word'
          return (
            <Fragment key={`${i}-${w.start}`}>
              <span className={cls} onClick={() => seek(w.start)}>
                {w.word}
              </span>
              {i < body.words!.length - 1 && ' '}
            </Fragment>
          )
        })}
      </div>
    )
  }
  return <div className="plain-text" style={{ color: 'var(--fg)' }}>{body.text}</div>
}

function WordTimelineChips({
  body,
  activeWordIdx,
}: {
  body: VerboseJsonResponse
  activeWordIdx: number
}) {
  if (!body.words || body.words.length === 0) {
    return (
      <div className="empty">
        No word-level timestamps in this response — enable <span className="mono">word</span> in
        Timestamp granularities to see them.
      </div>
    )
  }
  return (
    <div className="words">
      {body.words.map((w, i) => {
        const active = i === activeWordIdx
        return (
          <button
            key={`${i}-${w.start}`}
            type="button"
            className={active ? 'word-chip word-chip--active' : 'word-chip'}
            onClick={() => seek(w.start)}
            title={`${w.start.toFixed(2)}s — ${w.end.toFixed(2)}s`}
          >
            <span className="word-chip__word">{w.word}</span>
            <span className="word-chip__t">{w.start.toFixed(2)}</span>
          </button>
        )
      })}
    </div>
  )
}

function SegmentsTable({
  segments,
  activeIdx,
}: {
  segments: WhisperSegment[]
  activeIdx: number
}) {
  const [openId, setOpenId] = useState<number | null>(null)
  if (segments.length === 0) return <div className="empty">No segments.</div>
  return (
    <div className="segs">
      {segments.map((s, i) => {
        const active = i === activeIdx
        const open = openId === s.id
        return (
          <div key={s.id} className={active ? 'seg seg--active' : 'seg'}>
            <div
              className="seg__head"
              onClick={() => {
                setOpenId(open ? null : s.id)
                seek(s.start)
              }}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  setOpenId(open ? null : s.id)
                  seek(s.start)
                }
              }}
            >
              <div className="seg__time">
                {formatTime(s.start)}
                <br />
                <span className="seg__time-end">→ {formatTime(s.end)}</span>
              </div>
              <div className="seg__text">{s.text}</div>
              <ChevIcon up={open} />
            </div>
            {open && (
              <div className="seg__detail">
                <KvCell k="id" v={s.id} />
                <KvCell k="seek" v={s.seek} />
                <KvCell k="temperature" v={s.temperature} />
                <KvCell k="avg_logprob" v={s.avg_logprob} />
                <KvCell k="compression_ratio" v={s.compression_ratio} />
                <KvCell k="no_speech_prob" v={s.no_speech_prob} />
                <div className="seg__tokens">
                  <div className="seg__tokens-l">TOKENS ({s.tokens.length})</div>
                  <div className="seg__tokens-grid">
                    {s.tokens.slice(0, 64).map((t, idx) => (
                      <span key={idx} className="tok">
                        {t}
                      </span>
                    ))}
                    {s.tokens.length > 64 && (
                      <span className="tok" style={{ color: 'var(--muted)' }}>
                        +{s.tokens.length - 64}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function KvCell({ k, v }: { k: string; v: unknown }) {
  const isNull = v === null || v === undefined
  const tip = isNull ? NULL_TIPS[k] : undefined
  return (
    <div className={isNull ? 'kv kv--null' : 'kv'} title={tip}>
      <span className="kv__k">{k}</span>
      <span className="kv__v">
        {isNull ? (
          <>
            <span className="kv__null-dot" />
            null
          </>
        ) : (
          String(v)
        )}
      </span>
    </div>
  )
}

function RawJson({ body }: { body: VerboseJsonResponse }) {
  const text = JSON.stringify(body, null, 2)
  // Lightweight syntax color via regex.
  const html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"(\w+)":/g, '<span style="color:var(--accent)">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span style="color:oklch(0.85 0.1 60)">"$1"</span>')
    .replace(/: (-?\d+\.?\d*)/g, ': <span style="color:oklch(0.78 0.13 145)">$1</span>')
    .replace(/: (null|true|false)/g, ': <span style="color:oklch(0.72 0.12 30); font-style:italic;">$1</span>')
  return <pre className="code" dangerouslySetInnerHTML={{ __html: html }} />
}

function ChevIcon({ up }: { up: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={14}
      height={14}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ transform: up ? 'rotate(180deg)' : 'none', transition: 'transform .15s' }}
      aria-hidden
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  )
}
