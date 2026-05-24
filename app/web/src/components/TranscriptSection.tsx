import { memo, useEffect, useMemo, useRef, useState } from 'react'
import type { TranscriptionResponse } from '../lib/api'
import type { VerboseJsonResponse, WhisperSegment, Word } from '../lib/types'
import { play, seek, useCurrentTime } from '../lib/playback'
import { formatTime } from '../lib/format'

interface Props {
  result: TranscriptionResponse | null
  filename: string
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

/**
 * Wrapped in React.memo so peaks-driven re-renders in App.tsx (~15Hz
 * during URL playback) don't cascade into the transcript subtree.
 * Currents-time-driven highlight tracking happens INSIDE the section
 * (VerboseBody self-subscribes) so we don't need currentTime as a
 * prop — that was the channel that defeated memo before.
 */
export const TranscriptSection = memo(function TranscriptSection({
  result,
  filename,
  live,
  segmentArrivals,
  wordArrivals,
  partialSegment,
  partialWords,
}: Props) {
  const [view, setView] = useState<View>('text')
  const bodyRef = useRef<HTMLDivElement | null>(null)
  // `stuck` = pinned to the bottom (auto-follow). Scrolling up breaks
  // the pin; scrolling back to the bottom (or the Follow button) restores
  // it. `lastTopRef` lets us tell a user scroll-up from our own writes:
  // auto-follow only ever scrolls DOWN, so any decrease in scrollTop is
  // the user.
  const [stuck, setStuck] = useState(true)
  const lastTopRef = useRef(0)
  const followBottom = () => {
    const el = bodyRef.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }

  // Re-pin whenever a fresh live stream begins.
  useEffect(() => {
    if (live) setStuck(true)
  }, [live])

  // Follow new content while streaming + pinned. One smooth scroll per
  // update (~4×/s) toward the *real* bottom — so it can't fall behind,
  // and there's no per-frame reflow to jank the render.
  useEffect(() => {
    if (live && stuck) followBottom()
  }, [live, stuck, result, partialSegment, partialWords, view])

  const onBodyScroll = () => {
    const el = bodyRef.current
    if (!el) return
    const cur = el.scrollTop
    const dist = el.scrollHeight - cur - el.clientHeight
    if (cur < lastTopRef.current - 6) {
      setStuck(false) // user dragged up
    } else if (dist < 40) {
      setStuck(true) // back at the bottom
    }
    lastTopRef.current = cur
  }
  const snapToBottom = () => {
    setStuck(true)
    followBottom()
  }

  // Empty state — nothing committed and nothing streaming yet.
  if (!result && !partialSegment) {
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

  // Verbose body: a committed verbose result, or a synthetic body when
  // only an in-flight partial exists (so the view tabs are usable from
  // the very first streamed token).
  const verboseBody: VerboseJsonResponse | null =
    result?.format === 'verbose_json'
      ? result.body
      : !result && partialSegment
        ? {
            task: 'transcribe',
            language: 'en',
            duration: partialSegment.end,
            text: '',
            segments: [],
            words: [],
            strategy: 'streaming',
            transcription_time_seconds: 0,
          }
        : null

  const formatLabel = result ? result.format : 'verbose_json'

  return (
    <section className="transcript">
      <div className="transcript__head">
        <div className="transcript__head-l">
          <span className="label-eyebrow">TRANSCRIPT</span>
          <span className="format-pill">{formatLabel}</span>
        </div>
        {verboseBody && (
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
      <div className="transcript__body" ref={bodyRef} onScroll={onBodyScroll}>
        {result?.format === 'json' && (
          <Code text={JSON.stringify(result.body, null, 2)} filename={filename} />
        )}
        {result?.format === 'text' && <div className="editorial-body">{result.body}</div>}
        {result?.format === 'srt' && <Code text={result.body} filename={filename} />}
        {result?.format === 'vtt' && <Code text={result.body} filename={filename} />}
        {verboseBody && (
          <VerboseBody
            body={verboseBody}
            view={view}
            filename={filename}
            live={live}
            segmentArrivals={segmentArrivals}
            wordArrivals={wordArrivals}
            partialSegment={partialSegment ?? null}
            partialWords={partialWords ?? []}
          />
        )}
      </div>
      {live && !stuck && (
        <button type="button" className="transcript__follow" onClick={snapToBottom}>
          <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M12 5v14M5 12l7 7 7-7" />
          </svg>
          Follow
        </button>
      )}
    </section>
  )
})

function VerboseBody({
  body,
  view,
  filename,
  live,
  segmentArrivals,
  wordArrivals,
  partialSegment,
  partialWords,
}: {
  body: VerboseJsonResponse
  view: View
  filename: string
  live: boolean
  segmentArrivals: Map<number, number>
  wordArrivals: Map<number, number>
  partialSegment: WhisperSegment | null
  partialWords: Word[]
}) {
  // Text view doesn't need currentTime at all — ActiveWordTracker
  // subscribes itself + drives the underline imperatively. Bypass the
  // 60Hz useCurrentTime subscription so the heavy editorial-body
  // subtree doesn't reconcile every tick (peaks + canvas redraws then
  // share the main thread without contention → no waveform stutter).
  if (view === 'text')
    return (
      <PlainText
        body={body}
        live={live}
        wordArrivals={wordArrivals}
        partialWords={partialWords}
      />
    )
  if (view === 'segments')
    return (
      <SegmentsView
        segments={body.segments}
        live={live}
        segmentArrivals={segmentArrivals}
        partialSegment={partialSegment}
      />
    )
  if (view === 'words')
    return <WordsView body={body} partialWords={partialWords} />
  return <Code text={JSON.stringify(body, null, 2)} filename={filename} syntaxColor />
}

// Thin wrappers that subscribe to currentTime + compute activeIdx, so
// the work happens at the view level rather than VerboseBody — keeps
// the text-view path completely off the per-tick re-render cascade.

function SegmentsView({
  segments,
  live,
  segmentArrivals,
  partialSegment,
}: {
  segments: WhisperSegment[]
  live: boolean
  segmentArrivals: Map<number, number>
  partialSegment: WhisperSegment | null
}) {
  const t = useCurrentTime()
  const activeIdx = useMemo(
    () => segments.findIndex((s) => t >= s.start && t <= s.end),
    [segments, t],
  )
  // Partial segment is "active" when current time has reached its
  // start. End may not be authoritative (engine is still extending it),
  // so don't gate on t <= partial.end — once playback enters the
  // in-flight sentence, light it up.
  const partialActive =
    partialSegment !== null && t >= partialSegment.start
  return (
    <SegmentRows
      segments={segments}
      activeIdx={activeIdx}
      live={live}
      segmentArrivals={segmentArrivals}
      partialSegment={partialSegment}
      partialActive={partialActive}
    />
  )
}

function WordsView({
  body,
  partialWords,
}: {
  body: VerboseJsonResponse
  partialWords: Word[]
}) {
  const t = useCurrentTime()
  const activeIdx = useMemo(() => {
    const committed = body.words ?? []
    if (committed.length === 0 && partialWords.length === 0) return -1
    let found = -1
    if (committed.length > 0 && t >= committed[0].start) {
      let lo = 0,
        hi = committed.length - 1
      while (lo <= hi) {
        const mid = (lo + hi) >> 1
        if (committed[mid].start <= t) {
          found = mid
          lo = mid + 1
        } else {
          hi = mid - 1
        }
      }
    }
    for (let i = 0; i < partialWords.length; i++) {
      if (partialWords[i].start <= t) {
        found = committed.length + i
      } else {
        break
      }
    }
    return found
  }, [body.words, partialWords, t])
  return <WordsGrid body={body} activeIdx={activeIdx} partialWords={partialWords} />
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

/**
 * Per-word render. Two states:
 *   - committed: solid `--fg` (white). Drives no per-word highlight or
 *     underline — playback position is tracked by a single moving
 *     underline element in PlainText below, not by recoloring words.
 *   - partial:  `--muted` (grey) via the `--partial` class.
 *
 * The data-word-idx attribute lets PlainText's underline effect look
 * up the active word's DOM rect without ref management (which would
 * defeat memo).
 */
const Word = memo(function Word({
  idx,
  word,
  start,
  reveal,
  partial,
  isLast,
}: {
  idx: number
  word: string
  start: number
  reveal: boolean
  partial: boolean
  isLast: boolean
}) {
  const classes = ['editorial-word']
  if (partial) classes.push('editorial-word--partial')
  if (reveal) classes.push('word-reveal')
  return (
    <>
      <span
        className={classes.join(' ')}
        data-word-idx={idx}
        onClick={() => {
          seek(start)
          play()
        }}
      >
        {word}
      </span>
      {!isLast && ' '}
    </>
  )
})

/** Memoization fence: re-render the word list only when its identity
 * actually changes — not on every 60Hz currentTime tick. The combined
 * (committed + partial) list is keyed by length and the last partial's
 * start time, which moves whenever new content lands. */
const WordList = memo(function WordList({
  body,
  live,
  wordArrivals,
  partialWords,
}: {
  body: VerboseJsonResponse
  live: boolean
  wordArrivals: Map<number, number>
  partialWords: Word[]
}) {
  const committed = body.words ?? []
  const committedCount = committed.length
  const total = committedCount + partialWords.length
  return (
    <>
      {Array.from({ length: total }).map((_unused, i) => {
        const isPartial = i >= committedCount
        const w = isPartial ? partialWords[i - committedCount] : committed[i]
        const isLast = i === total - 1
        return (
          <Word
            key={i}
            idx={i}
            word={w.word}
            start={w.start}
            partial={isPartial}
            reveal={!isPartial && live && wordArrivals.has(i)}
            isLast={isLast}
          />
        )
      })}
    </>
  )
})

function PlainText({
  body,
  live,
  wordArrivals,
  partialWords,
}: {
  body: VerboseJsonResponse
  /** True only while transcripts are arriving mid-stream — drives the
   * arrival-based fade-in. False for finalized results. */
  live: boolean
  wordArrivals: Map<number, number>
  partialWords: Word[]
}) {
  const committed = body.words ?? []
  const containerRef = useRef<HTMLDivElement | null>(null)

  if (committed.length === 0 && partialWords.length === 0) {
    return <div className="editorial-body">{body.text}</div>
  }

  return (
    <div className="editorial-body" ref={containerRef} style={{ position: 'relative' }}>
      <WordList
        body={body}
        live={live}
        wordArrivals={wordArrivals}
        partialWords={partialWords}
      />
      <ActiveWordTracker
        committed={body.words ?? []}
        partialWords={partialWords}
        containerRef={containerRef}
      />
    </div>
  )
}

/** Sibling component that drives the playhead underline.
 *
 * Subscribes to `useCurrentTime()` ITSELF so the parent (PlainText /
 * WordList) doesn't reconcile on every 60Hz tick. Updates the
 * absolutely-positioned underline `<div>` via a ref-driven imperative
 * style write rather than React state — the only DOM mutation per tick
 * is the four style fields on a single element. */
function ActiveWordTracker({
  committed,
  partialWords,
  containerRef,
}: {
  committed: Word[]
  partialWords: Word[]
  containerRef: React.RefObject<HTMLDivElement | null>
}) {
  const t = useCurrentTime()
  const lineRef = useRef<HTMLSpanElement | null>(null)
  const lastIdxRef = useRef<number>(-1)

  // Binary search over COMMITTED first, then linear walk through PARTIAL
  // for the in-flight sentence. data-word-idx in WordList runs
  // 0..committedCount+partialCount-1, so a partial hit yields index
  // committed.length + partialIdx — same DOM node the cursor can land on.
  // Partial words may shuffle/extend as the engine decodes; the cursor
  // tracks the latest model output. Acceptable trade-off for letting the
  // underline live on preview text instead of dying at the last commit.
  const activeIdx = useMemo(() => {
    if (committed.length === 0 && partialWords.length === 0) return -1
    // First: committed binary search.
    let found = -1
    if (committed.length > 0 && t >= committed[0].start) {
      let lo = 0,
        hi = committed.length - 1
      while (lo <= hi) {
        const mid = (lo + hi) >> 1
        if (committed[mid].start <= t) {
          found = mid
          lo = mid + 1
        } else {
          hi = mid - 1
        }
      }
    }
    // Then: linear walk through partial (usually <20 words) for any
    // newer hit. Partial words always come AFTER committed in audio time.
    for (let i = 0; i < partialWords.length; i++) {
      if (partialWords[i].start <= t) {
        found = committed.length + i
      } else {
        break
      }
    }
    return found
  }, [committed, partialWords, t])

  useEffect(() => {
    const container = containerRef.current
    const line = lineRef.current
    if (!container || !line) return
    if (activeIdx < 0) {
      line.style.opacity = '0'
      return
    }
    // The effect only fires when activeIdx changes (deps below) — by
    // construction the underline only needs to move at word boundaries
    // (a few Hz at most), not every 60Hz currentTime tick. Dropping `t`
    // from deps removes the per-frame effect-runner overhead that was
    // showing up as waveform stutter in the text view.
    if (activeIdx === lastIdxRef.current) return
    lastIdxRef.current = activeIdx
    const wordEl = container.querySelector<HTMLSpanElement>(
      `[data-word-idx="${activeIdx}"]`,
    )
    if (!wordEl) return
    const cRect = container.getBoundingClientRect()
    const wRect = wordEl.getBoundingClientRect()
    line.style.opacity = '1'
    // Sit the underline a hair below the word baseline rather than the
    // bounding-box bottom — feels tucked to the descender rather than
    // floating in space.
    line.style.transform = `translate(${wRect.left - cRect.left}px, ${
      wRect.bottom - cRect.top - 2
    }px)`
    line.style.width = `${wRect.width}px`
  }, [activeIdx, containerRef])

  // Re-measure on window resize — line wraps shift word positions, and
  // the imperative style cache (`lastIdxRef`) would otherwise stick at
  // the pre-resize coordinates.
  useEffect(() => {
    const onResize = () => {
      lastIdxRef.current = -1 // force a re-measure on the next tick
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  return (
    <span
      ref={lineRef}
      aria-hidden
      className="text-playhead"
      style={{
        position: 'absolute',
        left: 0,
        top: 0,
        width: 0,
        height: 2,
        background: 'var(--accent)',
        borderRadius: 2,
        pointerEvents: 'none',
        opacity: 0,
        transform: 'translate(0,0)',
        transition:
          'transform 220ms cubic-bezier(0.22,0.61,0.36,1), width 220ms cubic-bezier(0.22,0.61,0.36,1), opacity 220ms ease-out',
      }}
    />
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
  partialActive = false,
}: {
  segments: WhisperSegment[]
  activeIdx: number
  live: boolean
  segmentArrivals: Map<number, number>
  partialSegment: WhisperSegment | null
  /** When true, the partial-segment row gets `--active` styling so the
   * playhead's active highlight extends into in-flight preview text. */
  partialActive?: boolean
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

  // Combined list. The partial's id is `next_seg_id` — the same id the
  // committed segment will use when the sentence terminates — so when
  // partial → committed happens React reuses the same <div>; only the
  // className and content update, and the CSS transition on .segs-row
  // smoothly animates the styling change.
  type Row = { seg: WhisperSegment; partial: boolean; activeIdx: number }
  const rows: Row[] = segments.map((s, i) => ({ seg: s, partial: false, activeIdx: i }))
  if (partialSegment) {
    rows.push({ seg: partialSegment, partial: true, activeIdx: -1 })
  }

  return (
    <div className="segs-list">
      {rows.map(({ seg: s, partial, activeIdx: i }) => {
        const active = partial ? partialActive : i === activeIdx
        const open = !partial && openId === s.id
        const reveal = !partial && live && segmentArrivals.has(s.id)
        const cls = [
          'segs-row',
          active ? 'segs-row--active' : '',
          partial ? 'segs-row--partial' : '',
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
              {formatTime(s.start)}{' '}
              <span className="segs-row__t-end">
                → {partial ? '…' : formatTime(s.end)}
              </span>
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
  // Combined list with stable index keys so a partial-to-committed
  // transition reuses the same <button>; only its className flips,
  // CSS transition handles the smooth visual change.
  const committedCount = committed.length
  const total = committedCount + partialWords.length
  return (
    <div className="words-grid">
      {Array.from({ length: total }).map((_unused, i) => {
        const isPartial = i >= committedCount
        const w = isPartial ? partialWords[i - committedCount] : committed[i]
        const classes = ['words-cell']
        if (!isPartial && i === activeIdx) classes.push('words-cell--active')
        if (isPartial) classes.push('words-cell--partial')
        return (
          <button
            key={i}
            type="button"
            className={classes.join(' ')}
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
        )
      })}
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
