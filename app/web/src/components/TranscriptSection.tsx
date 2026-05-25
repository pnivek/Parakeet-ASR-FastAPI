import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
  partialSegment,
  partialWords,
}: Props) {
  const [view, setView] = useState<View>('text')
  // Callback ref so LazyChunks downstream can react to the scroll
  // container becoming available (IntersectionObserver needs its root).
  // Doubles as the bodyRef the auto-follow + scroll handlers need.
  const [bodyEl, setBodyEl] = useState<HTMLDivElement | null>(null)
  const bodyRef = (el: HTMLDivElement | null) => setBodyEl(el)
  // `stuck` = pinned to the bottom (auto-follow). Scrolling up breaks
  // the pin; scrolling back to the bottom (or the Follow button) restores
  // it. `lastTopRef` lets us tell a user scroll-up from our own writes:
  // auto-follow only ever scrolls DOWN, so any decrease in scrollTop is
  // the user.
  const [stuck, setStuck] = useState(true)
  const lastTopRef = useRef(0)
  const followBottom = () => {
    const el = bodyEl
    if (!el) return
    // Always instant. Smooth scroll on a tall, fast-growing scroller
    // queues per-frame layout work that interleaves with React commits
    // — the result is bouncy + stuttery. Snap-scroll is one paint per
    // call and is what the user explicitly asked for.
    el.scrollTo({ top: el.scrollHeight, behavior: 'auto' })
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
    const el = bodyEl
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

  return (
    <section className="transcript">
      <div className="transcript__head">
        <div className="transcript__head-l">
          <span className="label-eyebrow">TRANSCRIPT</span>
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
            partialSegment={partialSegment ?? null}
            partialWords={partialWords ?? []}
            scrollRoot={bodyEl}
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
  partialSegment,
  partialWords,
  scrollRoot,
}: {
  body: VerboseJsonResponse
  view: View
  filename: string
  live: boolean
  segmentArrivals: Map<number, number>
  partialSegment: WhisperSegment | null
  partialWords: Word[]
  scrollRoot: HTMLElement | null
}) {
  // Text view is now pure-text-node + Range-API; it self-subscribes
  // to useCurrentTime inside CursorOverlay. The other views (segments,
  // words) wrap their own subscribers below.
  if (view === 'text')
    return (
      <PlainText
        body={body}
        live={live}
        partialSegment={partialSegment}
        partialWords={partialWords}
        scrollRoot={scrollRoot}
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
  return (
    <SegmentRows
      segments={segments}
      activeIdx={activeIdx}
      live={live}
      segmentArrivals={segmentArrivals}
      partialSegment={partialSegment}
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
  // Interval-containment across committed + partial — same rule as
  // ActiveWordTracker. WordsGrid renders partial cells immediately
  // after committed at index committed.length + partialIdx.
  const activeIdx = useMemo(() => {
    const committed = body.words ?? []
    if (committed.length === 0 && partialWords.length === 0) return -1
    let found = -1
    if (committed.length > 0 && t >= committed[0].start) {
      let lo = 0,
        hi = committed.length - 1,
        candidate = -1
      while (lo <= hi) {
        const mid = (lo + hi) >> 1
        if (committed[mid].start <= t) {
          candidate = mid
          lo = mid + 1
        } else {
          hi = mid - 1
        }
      }
      if (candidate >= 0) {
        const lookback = Math.max(0, candidate - 3)
        found = candidate
        for (let i = candidate; i >= lookback; i--) {
          if (committed[i].end >= t) {
            found = i
            break
          }
        }
      }
    }
    let partialHit = -1
    for (let i = 0; i < partialWords.length; i++) {
      if (partialWords[i].start <= t) partialHit = i
      else break
    }
    if (partialHit >= 0) {
      let pi = partialHit
      const lookbackP = Math.max(0, partialHit - 3)
      for (let i = partialHit; i >= lookbackP; i--) {
        if (partialWords[i].end >= t) {
          pi = i
          break
        }
      }
      found = committed.length + pi
    }
    return found
  }, [body.words, partialWords, t])
  return <WordsGrid body={body} activeIdx={activeIdx} partialWords={partialWords} />
}

// ── Text view: pure text node + cursor overlay ────────────────────
//
// The whole committed transcript is rendered as a SINGLE text node
// inside one <span>; the partial in-flight sentence is a second
// <span>. No per-word DOM, no per-chunk DOM, no IntersectionObserver.
// DOM size is O(1) regardless of session length — five elements total.
//
// Cursor + click work via the Range API on the text nodes:
//   - Cursor: word index → character offset → Range → getBoundingClientRect.
//   - Click: caretRangeFromPoint(x, y) → character offset → binary
//     search wordCharOffsets → word → seek.
//
// Browser handles inline-text layout once for the whole transcript
// (extremely well-optimized); scroll is pure compositor work; React
// only reconciles when new committed words actually land. The Chunk
// /IO/Word machinery this replaced was generating dozens of React
// re-renders per frame during auto-scroll which caused the bouncy
// auto-scroll + Follow-button jank.

/** Cross-engine wrapper around caretRangeFromPoint / caretPositionFromPoint
 * (Chromium/Safari vs Firefox respectively). Returns the text node + the
 * character offset within it at the given client (x, y). Null if the
 * point isn't on text. */
function caretFromPoint(
  x: number,
  y: number,
): { node: Node; offset: number } | null {
  type Doc = Document & {
    caretRangeFromPoint?: (x: number, y: number) => Range | null
    caretPositionFromPoint?: (
      x: number,
      y: number,
    ) => { offsetNode: Node; offset: number } | null
  }
  const doc = document as Doc
  if (doc.caretRangeFromPoint) {
    const r = doc.caretRangeFromPoint(x, y)
    if (!r) return null
    return { node: r.startContainer, offset: r.startOffset }
  }
  if (doc.caretPositionFromPoint) {
    const p = doc.caretPositionFromPoint(x, y)
    if (!p) return null
    return { node: p.offsetNode, offset: p.offset }
  }
  return null
}

/** Concatenate words into their text representation and return the
 * character offset of each word's start in that concatenation.
 *
 *   words = [{word:'Hi'}, {word:'there'}, {word:'!'}]
 *   text = "Hi there !"
 *   offsets = [0, 3, 9]
 *
 * Words are joined by single spaces — same convention as the engine's
 * `seg.text`. Cursor + click math both work against these offsets. */
function buildTextAndOffsets(words: Word[]): { text: string; offsets: number[] } {
  const offsets: number[] = new Array(words.length)
  let pos = 0
  const parts: string[] = new Array(words.length)
  for (let i = 0; i < words.length; i++) {
    offsets[i] = pos
    parts[i] = words[i].word
    pos += words[i].word.length + 1 // +1 for the joining space
  }
  return { text: parts.join(' '), offsets }
}

/** Apply a CSS Highlight to a range covering newly-committed text, then
 * remove after `MS` so the area falls back to the default committed
 * (white) color. Color rule lives in `App.css` under
 * `::highlight(just-committed)`. Browser support check is defensive —
 * Highlight API is in Chromium 105+, WebKit 17.2+; falls back to no-op
 * (the text just appears committed immediately) on older browsers. */
const JUST_COMMITTED_MS = 320
function flashJustCommitted(textNode: Node, start: number, end: number): void {
  type CSSWithHighlights = typeof CSS & { highlights?: { get: (name: string) => unknown; set: (name: string, h: unknown) => void } }
  type WithHighlight = Window & { Highlight?: new (...ranges: Range[]) => { add: (r: Range) => void; delete: (r: Range) => void } }
  const cssH = (CSS as CSSWithHighlights).highlights
  const HighlightCtor = (window as unknown as WithHighlight).Highlight
  if (!cssH || !HighlightCtor) return
  let highlight = cssH.get('just-committed') as { add: (r: Range) => void; delete: (r: Range) => void } | undefined
  if (!highlight) {
    highlight = new HighlightCtor()
    cssH.set('just-committed', highlight)
  }
  const range = document.createRange()
  try {
    range.setStart(textNode, start)
    range.setEnd(textNode, end)
  } catch {
    return // text node was replaced before we got here; bail.
  }
  highlight.add(range)
  setTimeout(() => {
    try {
      highlight!.delete(range)
    } catch {
      /* range/highlight gone; nothing to do */
    }
  }, JUST_COMMITTED_MS)
}

/** Pure text-node renderer. Two spans (committed + partial), one click
 * handler, one underline overlay. Scales O(1) in DOM with session
 * length. */
function PlainText({
  body,
  live,
  partialSegment,
  partialWords,
  scrollRoot,
}: {
  body: VerboseJsonResponse
  /** True only while transcripts are arriving mid-stream. Used by the
   * just-committed flash logic (don't bother on finalized results). */
  live: boolean
  partialSegment: WhisperSegment | null
  partialWords: Word[]
  /** Kept for prop signature compat with VerboseBody — not used in
   * pure text mode (no IntersectionObserver per chunk). */
  scrollRoot: HTMLElement | null
}) {
  void scrollRoot
  void partialSegment

  const committed = body.words ?? []
  const containerRef = useRef<HTMLDivElement | null>(null)
  const committedSpanRef = useRef<HTMLSpanElement | null>(null)
  const partialSpanRef = useRef<HTMLSpanElement | null>(null)

  // Text + offset arrays for committed and partial. Recompute when
  // committed length OR partial words change. Cheap even at 27k words —
  // a single linear walk per recompute (~ms).
  const { text: committedText, offsets: committedOffsets } = useMemo(
    () => buildTextAndOffsets(committed),
    [committed],
  )
  const { text: partialText, offsets: partialOffsets } = useMemo(
    () => buildTextAndOffsets(partialWords),
    [partialWords],
  )

  // Fade newly-committed text from grey → default. Tracks committed
  // length across renders; on growth, applies a brief CSS Highlight to
  // the new character range so it stays muted for ~320ms before
  // snapping back to default. Only runs in live mode (finalized
  // results don't need the indicator).
  const prevCommittedLenRef = useRef<number>(0)
  useEffect(() => {
    const prev = prevCommittedLenRef.current
    const curr = committed.length
    prevCommittedLenRef.current = curr
    if (!live) return
    if (curr <= prev) return // no new commit
    const span = committedSpanRef.current
    const textNode = span?.firstChild
    if (!textNode || textNode.nodeType !== Node.TEXT_NODE) return
    const startChar = committedOffsets[prev] ?? 0
    const lastIdx = curr - 1
    const endChar =
      committedOffsets[lastIdx] + committed[lastIdx].word.length
    flashJustCommitted(textNode, startChar, endChar)
  }, [committed, committedOffsets, live])

  // Click handler: map (x, y) → text caret → character offset → word
  // index → seek. One handler for the whole editorial body.
  const onClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const caret = caretFromPoint(e.clientX, e.clientY)
      if (!caret) return
      const node = caret.node
      const offset = caret.offset
      const committedSpan = committedSpanRef.current
      const partialSpan = partialSpanRef.current
      let inCommitted = false
      let inPartial = false
      if (committedSpan && node === committedSpan.firstChild) inCommitted = true
      else if (partialSpan && node === partialSpan.firstChild) inPartial = true
      if (inCommitted) {
        // Binary search committedOffsets for the largest entry ≤ offset.
        let lo = 0,
          hi = committedOffsets.length - 1,
          found = -1
        while (lo <= hi) {
          const mid = (lo + hi) >> 1
          if (committedOffsets[mid] <= offset) {
            found = mid
            lo = mid + 1
          } else {
            hi = mid - 1
          }
        }
        if (found >= 0) {
          seek(committed[found].start)
          play()
        }
      } else if (inPartial) {
        let lo = 0,
          hi = partialOffsets.length - 1,
          found = -1
        while (lo <= hi) {
          const mid = (lo + hi) >> 1
          if (partialOffsets[mid] <= offset) {
            found = mid
            lo = mid + 1
          } else {
            hi = mid - 1
          }
        }
        if (found >= 0) {
          seek(partialWords[found].start)
          play()
        }
      }
    },
    [committed, committedOffsets, partialWords, partialOffsets],
  )

  if (committed.length === 0 && partialWords.length === 0) {
    return <div className="editorial-body">{body.text}</div>
  }

  // Both spans always render so the spans have stable DOM identity
  // across renders — React updates their text content in place rather
  // than mounting/unmounting them.
  return (
    <div
      className="editorial-body"
      ref={containerRef}
      onClick={onClick}
      style={{ position: 'relative', cursor: 'text' }}
    >
      <span ref={committedSpanRef}>
        {committedText}
        {committedText && partialText ? ' ' : ''}
      </span>
      <span ref={partialSpanRef} className="editorial-text--partial">
        {partialText}
      </span>
      <CursorOverlay
        committed={committed}
        committedOffsets={committedOffsets}
        partialWords={partialWords}
        partialOffsets={partialOffsets}
        committedSpanRef={committedSpanRef}
        partialSpanRef={partialSpanRef}
        containerRef={containerRef}
      />
    </div>
  )
}

/** Plays the role of the old `ActiveWordTracker`. Subscribes to
 * `useCurrentTime()`, computes the active word index, and imperatively
 * positions an absolutely-positioned underline span via the Range API
 * (instead of looking up per-word elements by ID).
 *
 * Same monotonic-forward guard + seek-back reset as the old tracker. */
function CursorOverlay({
  committed,
  committedOffsets,
  partialWords,
  partialOffsets,
  committedSpanRef,
  partialSpanRef,
  containerRef,
}: {
  committed: Word[]
  committedOffsets: number[]
  partialWords: Word[]
  partialOffsets: number[]
  committedSpanRef: React.RefObject<HTMLSpanElement | null>
  partialSpanRef: React.RefObject<HTMLSpanElement | null>
  containerRef: React.RefObject<HTMLDivElement | null>
}) {
  const t = useCurrentTime()
  const lineRef = useRef<HTMLSpanElement | null>(null)
  const lastIdxRef = useRef<number>(-1)
  const lastTRef = useRef<number>(0)

  // Interval-containment across committed + partial — same selection
  // rule as the previous ActiveWordTracker. Indices are "global":
  // 0..committed.length-1 = committed, committed.length..= partial.
  const activeIdx = useMemo(() => {
    if (committed.length === 0 && partialWords.length === 0) return -1
    let found = -1
    if (committed.length > 0 && t >= committed[0].start) {
      let lo = 0,
        hi = committed.length - 1,
        candidate = -1
      while (lo <= hi) {
        const mid = (lo + hi) >> 1
        if (committed[mid].start <= t) {
          candidate = mid
          lo = mid + 1
        } else {
          hi = mid - 1
        }
      }
      if (candidate >= 0) {
        const lookback = Math.max(0, candidate - 3)
        found = candidate
        for (let i = candidate; i >= lookback; i--) {
          if (committed[i].end >= t) {
            found = i
            break
          }
        }
      }
    }
    let partialHit = -1
    for (let i = 0; i < partialWords.length; i++) {
      if (partialWords[i].start <= t) partialHit = i
      else break
    }
    if (partialHit >= 0) {
      let pi = partialHit
      const lookbackP = Math.max(0, partialHit - 3)
      for (let i = partialHit; i >= lookbackP; i--) {
        if (partialWords[i].end >= t) {
          pi = i
          break
        }
      }
      found = committed.length + pi
    }
    return found
  }, [committed, partialWords, t])

  // Detect user seek-back via t-delta; reset monotonic guard so the
  // cursor can re-anchor backward.
  useEffect(() => {
    if (t < lastTRef.current - 0.5) {
      lastIdxRef.current = -1
    }
    lastTRef.current = t
  }, [t])

  useEffect(() => {
    const container = containerRef.current
    const line = lineRef.current
    if (!container || !line) return
    if (activeIdx < 0) {
      if (lastIdxRef.current < 0) line.style.opacity = '0'
      return
    }
    if (activeIdx < lastIdxRef.current) return
    if (activeIdx === lastIdxRef.current) return
    lastIdxRef.current = activeIdx

    const inCommitted = activeIdx < committed.length
    const span = inCommitted ? committedSpanRef.current : partialSpanRef.current
    const textNode = span?.firstChild
    if (!textNode || textNode.nodeType !== Node.TEXT_NODE) return

    let charStart: number
    let wordLen: number
    if (inCommitted) {
      charStart = committedOffsets[activeIdx]
      wordLen = committed[activeIdx].word.length
    } else {
      const pi = activeIdx - committed.length
      if (pi >= partialWords.length) return
      charStart = partialOffsets[pi]
      wordLen = partialWords[pi].word.length
    }

    const range = document.createRange()
    try {
      range.setStart(textNode, charStart)
      range.setEnd(textNode, charStart + wordLen)
    } catch {
      return
    }
    const wRect = range.getBoundingClientRect()
    const cRect = container.getBoundingClientRect()
    line.style.opacity = '1'
    line.style.transform = `translate(${wRect.left - cRect.left}px, ${
      wRect.bottom - cRect.top - 2
    }px)`
    line.style.width = `${wRect.width}px`
  }, [
    activeIdx,
    committed,
    committedOffsets,
    partialWords,
    partialOffsets,
    committedSpanRef,
    partialSpanRef,
    containerRef,
  ])

  // Resize handler — wrap shifts invalidate the cached underline position.
  useEffect(() => {
    const onResize = () => {
      lastIdxRef.current = -1
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
        // Partial rows keep their --partial dimming regardless of where
        // the playhead is — the active highlight only marks committed
        // rows so preview text stays visually "tentative."
        const active = !partial && i === activeIdx
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
