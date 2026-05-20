import { useMemo, useState, Fragment } from 'react'
import type { TranscriptionResponse } from '../lib/api'
import type { VerboseJsonResponse, WhisperSegment } from '../lib/types'
import { play, seek } from '../lib/playback'
import { formatTime } from '../lib/format'

interface Props {
  result: TranscriptionResponse | null
  filename: string
  currentTime: number
}

type View = 'text' | 'segments' | 'words' | 'raw'
const VIEWS: View[] = ['text', 'segments', 'words', 'raw']

const NULL_TIPS: Record<string, string> = {
  avg_logprob:
    'Not exposed on FULL_GRAPH-mode CUDA-graph decode (NeMo 2.7.3). Set USE_CUDA_GRAPHS=false to populate.',
  no_speech_prob:
    'No equivalent in Parakeet TDT — Whisper derives this from a `<|nospeech|>` token the model does not emit.',
  id: 'Not populated for now.',
  seek: 'Not populated for now.',
}

export function TranscriptSection({ result, filename, currentTime }: Props) {
  const [view, setView] = useState<View>('text')

  if (!result) {
    return (
      <section className="transcript">
        <div className="transcript__head">
          <div className="transcript__head-l">
            <span className="label-eyebrow">TRANSCRIPT</span>
          </div>
        </div>
        <div className="empty">Pick a source on the right to begin.</div>
      </section>
    )
  }

  const isVerbose = result.format === 'verbose_json'

  return (
    <section className="transcript">
      <div className="transcript__head">
        <div className="transcript__head-l">
          <span className="label-eyebrow">TRANSCRIPT</span>
          <span className="format-pill">{result.format}</span>
        </div>
        {isVerbose && (
          <div className="ma-segmented">
            {VIEWS.map((v) => (
              <button
                key={v}
                type="button"
                className={view === v ? 'ma-pill ma-pill--active' : 'ma-pill'}
                onClick={() => setView(v)}
              >
                {v}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="transcript__body">
        {result.format === 'json' && (
          <Code text={JSON.stringify(result.body, null, 2)} filename={filename} />
        )}
        {result.format === 'text' && <div className="editorial-body">{result.body}</div>}
        {result.format === 'srt' && <Code text={result.body} filename={filename} />}
        {result.format === 'vtt' && <Code text={result.body} filename={filename} />}
        {isVerbose && <VerboseBody body={result.body} view={view} currentTime={currentTime} filename={filename} />}
      </div>
    </section>
  )
}

function VerboseBody({
  body,
  view,
  currentTime,
  filename,
}: {
  body: VerboseJsonResponse
  view: View
  currentTime: number
  filename: string
}) {
  const activeWordIdx = useMemo(() => {
    if (!body.words) return -1
    return body.words.findIndex((w) => currentTime >= w.start && currentTime <= w.end)
  }, [body.words, currentTime])
  const activeSegIdx = useMemo(
    () => body.segments.findIndex((s) => currentTime >= s.start && currentTime <= s.end),
    [body.segments, currentTime],
  )

  if (view === 'text') return <PlainText body={body} currentTime={currentTime} activeWordIdx={activeWordIdx} />
  if (view === 'segments') return <SegmentRows segments={body.segments} activeIdx={activeSegIdx} />
  if (view === 'words') return <WordsGrid body={body} activeIdx={activeWordIdx} />
  return <Code text={JSON.stringify(body, null, 2)} filename={filename} syntaxColor />
}

// ── Plain text view with reveal animation ─────────────────────────
const REVEAL_LEAD = 0.18 // start fading in this much before w.start (s)
const REVEAL_FADE = 0.36 // fade duration (s)

function PlainText({
  body,
  currentTime,
  activeWordIdx,
}: {
  body: VerboseJsonResponse
  currentTime: number
  activeWordIdx: number
}) {
  if (body.words && body.words.length > 0) {
    return (
      <div className="editorial-body">
        {body.words.map((w, i) => {
          const reveal = w.start - REVEAL_LEAD
          const dt = currentTime - reveal
          const raw = dt <= 0 ? 0 : dt >= REVEAL_FADE ? 1 : dt / REVEAL_FADE
          const eased = raw * raw * (3 - 2 * raw) // smoothstep
          const active = i === activeWordIdx
          const past = i < activeWordIdx
          const color = active ? 'var(--accent)' : past ? 'var(--fg)' : 'var(--muted)'
          return (
            <Fragment key={`${i}-${w.start}`}>
              <span
                className="editorial-word"
                onClick={() => {
                  seek(w.start)
                  play()
                }}
                style={{
                  color,
                  opacity: eased,
                  filter: eased < 1 ? `blur(${(1 - eased) * 2.2}px)` : 'none',
                  transform: eased < 1 ? `translateY(${(1 - eased) * 3}px)` : 'none',
                  willChange: eased < 1 ? 'opacity, filter, transform' : 'auto',
                }}
              >
                {w.word}
                {active && <span className="editorial-word__under" />}
              </span>
              {i < body.words!.length - 1 && ' '}
            </Fragment>
          )
        })}
      </div>
    )
  }
  return <div className="editorial-body">{body.text}</div>
}

// ── Segments with single-line timestamps + compact metadata ─────
function SegmentRows({ segments, activeIdx }: { segments: WhisperSegment[]; activeIdx: number }) {
  const [openId, setOpenId] = useState<number | null>(null)
  if (segments.length === 0) return <div className="empty">No segments.</div>
  const goTo = (start: number, id: number) => {
    setOpenId((cur) => (cur === id ? null : id))
    seek(start)
    play()
  }
  return (
    <div className="segs-list">
      {segments.map((s, i) => {
        const active = i === activeIdx
        const open = openId === s.id
        return (
          <div
            key={s.id}
            className={active ? 'segs-row segs-row--active' : 'segs-row'}
            onClick={() => goTo(s.start, s.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                goTo(s.start, s.id)
              }
            }}
          >
            <span className="segs-row__t">
              {formatTime(s.start)} <span className="segs-row__t-end">→ {formatTime(s.end)}</span>
            </span>
            <div>
              <div className="segs-row__text">{s.text}</div>
              {open && (
                <div className="segs-detail" onClick={(e) => e.stopPropagation()}>
                  <Kv k="id" v={s.id} />
                  <Kv k="seek" v={s.seek} />
                  <Kv k="temperature" v={s.temperature} />
                  <Kv k="avg_logprob" v={s.avg_logprob} />
                  <Kv k="compression_ratio" v={s.compression_ratio} />
                  <Kv k="no_speech_prob" v={s.no_speech_prob} />
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function Kv({ k, v }: { k: string; v: unknown }) {
  const isNull = v === null || v === undefined
  const tip = isNull ? NULL_TIPS[k] : undefined
  return (
    <div className={isNull ? 'kv kv--null' : 'kv'} title={tip}>
      <span className="kv__k">{k}</span>
      <span className="kv__v">{isNull ? 'null' : String(v)}</span>
    </div>
  )
}

function WordsGrid({ body, activeIdx }: { body: VerboseJsonResponse; activeIdx: number }) {
  if (!body.words || body.words.length === 0) {
    return (
      <div className="empty">
        No word-level timestamps — enable the <span className="mono">word</span> granularity.
      </div>
    )
  }
  return (
    <div className="words-grid">
      {body.words.map((w, i) => (
        <button
          key={`${i}-${w.start}`}
          type="button"
          className={i === activeIdx ? 'words-cell words-cell--active' : 'words-cell'}
          onClick={() => {
            seek(w.start)
            play()
          }}
        >
          <span className="words-cell__word">{w.word}</span>
          <span className="words-cell__t num">
            {w.start.toFixed(2)} → {w.end.toFixed(2)}
          </span>
        </button>
      ))}
    </div>
  )
}

// ── Code block with Copy + Expand controls ─────────────────────────
const CopyIcon = () => (
  <svg viewBox="0 0 24 24" width={10} height={10} fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="9" y="9" width="11" height="11" rx="1.5" />
    <path d="M5 15V5a1 1 0 0 1 1-1h10" />
  </svg>
)
const CheckIcon = () => (
  <svg viewBox="0 0 24 24" width={10} height={10} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M5 12l5 5L20 7" />
  </svg>
)
const ExpandIcon = () => (
  <svg viewBox="0 0 24 24" width={10} height={10} fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
  </svg>
)
const CloseIcon = () => (
  <svg viewBox="0 0 24 24" width={10} height={10} fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" />
  </svg>
)

function Code({
  text,
  filename: _filename,
  syntaxColor = false,
}: {
  text: string
  filename: string
  syntaxColor?: boolean
}) {
  const [copied, setCopied] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const html = useMemo(() => {
    if (!syntaxColor) return null
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"(\w+)":/g, '<span style="color:var(--accent)">"$1"</span>:')
      .replace(/: "([^"]*)"/g, ': <span style="color:oklch(0.78 0.07 80)">"$1"</span>')
      .replace(/: (-?\d+\.?\d*)/g, ': <span style="color:oklch(0.78 0.06 140)">$1</span>')
      .replace(/: (null|true|false)/g, ': <span style="color:oklch(0.72 0.06 30); font-style:italic;">$1</span>')
  }, [text, syntaxColor])

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      /* noop — older browsers / file://; user can still see + select */
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 1400)
  }

  const controls = (
    <div className="code-controls">
      <button type="button" className="code-btn" onClick={onCopy}>
        {copied ? (
          <>
            <CheckIcon /> Copied
          </>
        ) : (
          <>
            <CopyIcon /> Copy
          </>
        )}
      </button>
      <button type="button" className="code-btn" onClick={() => setExpanded((e) => !e)}>
        {expanded ? (
          <>
            <CloseIcon /> Close
          </>
        ) : (
          <>
            <ExpandIcon /> Expand
          </>
        )}
      </button>
    </div>
  )

  const block = syntaxColor && html ? (
    <pre className="code" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <pre className="code">{text}</pre>
  )

  if (expanded) {
    return (
      <div className="code-overlay">
        <div className="code-wrap" style={{ flex: 1, minHeight: 0 }}>
          {controls}
          {block}
        </div>
      </div>
    )
  }

  return (
    <div className="code-wrap">
      {controls}
      {block}
    </div>
  )
}
