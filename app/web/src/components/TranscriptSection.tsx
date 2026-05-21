import { memo, useMemo, useState } from 'react'
import type { TranscriptionResponse } from '../lib/api'
import type { VerboseJsonResponse, WhisperSegment, Word } from '../lib/types'
import { play, seek } from '../lib/playback'
import { formatTime } from '../lib/format'

interface Props {
  result: TranscriptionResponse | null
  filename: string
  currentTime: number
  /** True while partials are still arriving (live mic / progressive WS). Drives
   * the smoothstep fade-in reveal. For a finalized result we just colour
   * words by playback position — no fade. */
  live: boolean
  /** Map of segment.id → performance.now() at first observation. Drives the
   * fade-in of newly arrived segments during live streaming. */
  segmentArrivals: Map<number, number>
  /** Map of word-index → arrival ms. Drives the text view's word reveal. */
  wordArrivals: Map<number, number>
  /** Engine's in-flight sentence buffer — uncommitted tokens streamed
   * by the server between actual .!? commits. Rendered as dimmed text
   * at the end of the transcript so the user sees words appear as the
   * model decodes them, without us mutating the model's outputs. */
  partialSegment?: WhisperSegment | null
  partialWords?: Word[]
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

export function TranscriptSection({
  result,
  filename,
  currentTime,
  live,
  segmentArrivals,
  wordArrivals,
  partialSegment,
  partialWords,
}: Props) {
  const [view, setView] = useState<View>('text')

  // No committed transcript yet — but if the engine is already
  // streaming an in-flight partial, render that. Otherwise show the
  // empty state.
  if (!result) {
    if (!partialSegment) {
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
    const partialOnlyBody: VerboseJsonResponse = {
      task: 'transcribe',
      language: 'en',
      duration: partialSegment.end,
      text: '',
      segments: [],
      words: [],
      strategy: 'progressive',
      transcription_time_seconds: 0,
    }
    return (
      <section className="transcript">
        <div className="transcript__head">
          <div className="transcript__head-l">
            <span className="label-eyebrow">TRANSCRIPT</span>
            <span className="format-pill">verbose_json</span>
          </div>
        </div>
        <div className="transcript__body">
          <VerboseBody
            body={partialOnlyBody}
            view={view}
            currentTime={currentTime}
            filename={filename}
            live={live}
            segmentArrivals={segmentArrivals}
            wordArrivals={wordArrivals}
            partialSegment={partialSegment}
            partialWords={partialWords ?? []}
          />
        </div>
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
        {isVerbose && (
          <VerboseBody
            body={result.body}
            view={view}
            currentTime={currentTime}
            filename={filename}
            live={live}
            segmentArrivals={segmentArrivals}
            wordArrivals={wordArrivals}
            partialSegment={partialSegment ?? null}
            partialWords={partialWords ?? []}
          />
        )}
      </div>
    </section>
  )
}

function VerboseBody({
  body,
  view,
  currentTime,
  filename,
  live,
  segmentArrivals,
  wordArrivals,
  partialSegment,
  partialWords,
}: {
  body: VerboseJsonResponse
  view: View
  currentTime: number
  filename: string
  live: boolean
  segmentArrivals: Map<number, number>
  wordArrivals: Map<number, number>
  partialSegment: WhisperSegment | null
  partialWords: Word[]
}) {
  const activeWordIdx = useMemo(() => {
    if (!body.words) return -1
    return body.words.findIndex((w) => currentTime >= w.start && currentTime <= w.end)
  }, [body.words, currentTime])
  const activeSegIdx = useMemo(
    () => body.segments.findIndex((s) => currentTime >= s.start && currentTime <= s.end),
    [body.segments, currentTime],
  )

  if (view === 'text')
    return (
      <PlainText
        body={body}
        currentTime={currentTime}
        activeWordIdx={activeWordIdx}
        live={live}
        wordArrivals={wordArrivals}
        partialWords={partialWords}
      />
    )
  if (view === 'segments')
    return (
      <SegmentRows
        segments={body.segments}
        activeIdx={activeSegIdx}
        live={live}
        segmentArrivals={segmentArrivals}
        partialSegment={partialSegment}
      />
    )
  if (view === 'words')
    return <WordsGrid body={body} activeIdx={activeWordIdx} partialWords={partialWords} />
  return <Code text={JSON.stringify(body, null, 2)} filename={filename} syntaxColor />
}

// ── Plain text view with reveal animation ─────────────────────────
// Animation strategy: each new word is mounted with a `.word-reveal`
// class that runs a 380ms CSS keyframe (App.css → @keyframes wordReveal).
// The browser drives the animation in the compositor — React doesn't
// re-render per frame. Result: zero per-frame React work during a
// live stream, regardless of how many words have accumulated.
//
// Once a word is mounted, subsequent re-renders keep its className
// stable (same arrival stamp → same class), so the animation does
// NOT replay. Stable keys (`${i}-${w.start}`) also keep React from
// unmounting + remounting elements as new partials arrive.

const Word = memo(function Word({
  word,
  start,
  active,
  past,
  reveal,
  isLast,
}: {
  word: string
  start: number
  active: boolean
  past: boolean
  reveal: boolean
  isLast: boolean
}) {
  const color = active ? 'var(--accent)' : past ? 'var(--fg)' : 'var(--muted)'
  return (
    <>
      <span
        className={reveal ? 'editorial-word word-reveal' : 'editorial-word'}
        onClick={() => {
          seek(start)
          play()
        }}
        style={{ color }}
      >
        {word}
        {active && <span className="editorial-word__under" />}
      </span>
      {!isLast && ' '}
    </>
  )
})

function PlainText({
  body,
  currentTime: _currentTime,
  activeWordIdx,
  live,
  wordArrivals,
  partialWords,
}: {
  body: VerboseJsonResponse
  currentTime: number
  activeWordIdx: number
  /** True only while transcripts are arriving mid-stream — drives the
   * arrival-based fade-in. False for finalized results. */
  live: boolean
  /** Per-word-index arrival timestamps for the live reveal. The CSS
   * fade runs on first mount; we only need to know whether a word has
   * an arrival stamp so we apply the class for ones that should fade. */
  wordArrivals: Map<number, number>
  /** In-flight words from the engine's uncommitted sentence buffer.
   * Rendered after the committed words with a dimmed style. Replaced
   * by real committed words when the sentence terminates. */
  partialWords: Word[]
}) {
  const committed = body.words ?? []
  if (committed.length === 0 && partialWords.length === 0) {
    return <div className="editorial-body">{body.text}</div>
  }
  const total = committed.length
  const lastCommittedIsActuallyLast = partialWords.length === 0
  return (
    <div className="editorial-body">
      {committed.map((w, i) => {
        const active = i === activeWordIdx
        const past = i < activeWordIdx
        const isLast = i === total - 1 && lastCommittedIsActuallyLast
        return (
          <Word
            key={`${i}-${w.start}`}
            word={w.word}
            start={w.start}
            active={active}
            past={past}
            reveal={live && wordArrivals.has(i)}
            isLast={isLast}
          />
        )
      })}
      {partialWords.length > 0 && (
        <span className="editorial-partial">
          {/* Separating space when there are committed words before. */}
          {committed.length > 0 && ' '}
          {partialWords.map((w, i) => (
            <span
              key={`partial-${i}-${w.start}`}
              className="editorial-word editorial-word--partial"
              onClick={() => {
                seek(w.start)
                play()
              }}
            >
              {w.word}
              {i < partialWords.length - 1 && ' '}
            </span>
          ))}
        </span>
      )}
    </div>
  )
}

// ── Segments with single-line timestamps + compact metadata ─────
// Same animation strategy as words: each new row mounts with a CSS
// class that drives a one-shot keyframe; React doesn't re-render per
// frame.

function SegmentRows({
  segments,
  activeIdx,
  live,
  segmentArrivals,
  partialSegment,
}: {
  segments: WhisperSegment[]
  activeIdx: number
  live: boolean
  segmentArrivals: Map<number, number>
  partialSegment: WhisperSegment | null
}) {
  const [openId, setOpenId] = useState<number | null>(null)

  if (segments.length === 0 && !partialSegment) {
    return (
      <div className="empty">{live ? 'Listening…' : 'No segments.'}</div>
    )
  }

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
        const reveal = live && segmentArrivals.has(s.id)
        const cls = [
          active ? 'segs-row segs-row--active' : 'segs-row',
          reveal ? 'seg-reveal' : '',
        ]
          .filter(Boolean)
          .join(' ')
        return (
          <div
            key={s.id}
            className={cls}
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
      {partialSegment && (
        <div
          className="segs-row segs-row--partial"
          onClick={() => {
            seek(partialSegment.start)
            play()
          }}
        >
          <span className="segs-row__t">
            {formatTime(partialSegment.start)}{' '}
            <span className="segs-row__t-end">→ …</span>
          </span>
          <div>
            <div className="segs-row__text segs-row__text--partial">{partialSegment.text}</div>
          </div>
        </div>
      )}
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

function WordsGrid({
  body,
  activeIdx,
  partialWords,
}: {
  body: VerboseJsonResponse
  activeIdx: number
  partialWords: Word[]
}) {
  const committed = body.words ?? []
  if (committed.length === 0 && partialWords.length === 0) {
    return (
      <div className="empty">
        No word-level timestamps — enable the <span className="mono">word</span> granularity.
      </div>
    )
  }
  return (
    <div className="words-grid">
      {committed.map((w, i) => (
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
      {partialWords.map((w, i) => (
        <button
          key={`partial-${i}-${w.start}`}
          type="button"
          className="words-cell words-cell--partial"
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
