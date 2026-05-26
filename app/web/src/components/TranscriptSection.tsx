import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { TranscriptionResponse } from '../lib/api'
import type { VerboseJsonResponse, WhisperSegment, Word } from '../lib/types'
import { play, seek, useCurrentTime } from '../lib/playback'
import { formatTime } from '../lib/format'

interface Props {
  result: TranscriptionResponse | null
  filename: string
  /** True while partials are still arriving (live mic / progressive WS).
   * Gates the segment-arrival fade animation + the auto-scroll-to-bottom
   * behavior. False once the result is finalized. */
  live: boolean
  /** Map of segment.id → performance.now() at first observation. Drives
   * the .seg-reveal fade-in on newly-arrived segments in the segments
   * view. */
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
  const [bodyEl, setBodyEl] = useState<HTMLDivElement | null>(null)
  const bodyRef = (el: HTMLDivElement | null) => setBodyEl(el)
  // Bottom sentinel — a 1px invisible div placed after the transcript
  // content. IntersectionObserver watches its visibility against the
  // body element as root; that's the only signal we use for both
  // "user is pinned to bottom" and "should we auto-scroll new content".
  //
  // This replaces a scroll-event-based `stuck` state machine that flashed
  // the Follow button for one frame on every auto-scroll: `scrollTo`
  // fires a scroll event before layout settles for the just-appended
  // partial text, the `dist = scrollHeight - scrollTop - clientHeight`
  // calc reads > 40 momentarily, and the button toggles on/off in a
  // single render. IO is layout-driven and async — it can't race with
  // its own writes. Same pattern use-stick-to-bottom + react-virtuoso use.
  const sentinelRef = useRef<HTMLDivElement | null>(null)
  // `atBottomRef` is the source of truth for the auto-scroll decision
  // (read synchronously inside the content-update useLayoutEffect).
  // `atBottomState` mirrors it for the Follow-button conditional render.
  // Initialised true so the first batch of content auto-scrolls.
  const atBottomRef = useRef(true)
  const [atBottomState, setAtBottomState] = useState(true)

  useEffect(() => {
    if (!bodyEl) return
    const sentinel = sentinelRef.current
    if (!sentinel) return
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries[0]?.isIntersecting ?? false
        atBottomRef.current = visible
        setAtBottomState(visible)
      },
      { root: bodyEl, threshold: 0 },
    )
    obs.observe(sentinel)
    return () => obs.disconnect()
  }, [bodyEl])

  // Force re-pin to bottom when a new live session starts. Otherwise
  // a "scrolled up" state carried over from a previous session would
  // suppress auto-follow until the user manually scrolled back.
  useEffect(() => {
    if (!live) return
    atBottomRef.current = true
    setAtBottomState(true)
    sentinelRef.current?.scrollIntoView({ block: 'end' })
  }, [live])

  // Auto-scroll: keyed on content changes. `useLayoutEffect` so we
  // read atBottomRef + write scrollIntoView before the browser paints
  // the new content, eliminating the flash of un-anchored bottom-edge
  // before the IO callback fires.
  useLayoutEffect(() => {
    if (!live) return
    if (!atBottomRef.current) return
    sentinelRef.current?.scrollIntoView({ block: 'end' })
  }, [live, result, partialSegment, partialWords, view])

  const snapToBottom = () => {
    sentinelRef.current?.scrollIntoView({ block: 'end' })
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
      <div className="transcript__body" ref={bodyRef}>
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
          />
        )}
        <div ref={sentinelRef} aria-hidden style={{ height: 1 }} />
      </div>
      <button
        type="button"
        className="transcript__follow"
        onClick={snapToBottom}
        aria-hidden={!live || atBottomState}
        style={{
          opacity: live && !atBottomState ? 1 : 0,
          pointerEvents: live && !atBottomState ? 'auto' : 'none',
        }}
      >
        <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M12 5v14M5 12l7 7 7-7" />
        </svg>
        Follow
      </button>
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
}: {
  body: VerboseJsonResponse
  view: View
  filename: string
  live: boolean
  segmentArrivals: Map<number, number>
  partialSegment: WhisperSegment | null
  partialWords: Word[]
}) {
  // Text view is now pure-text-node + Range-API; it self-subscribes
  // to useCurrentTime inside CursorOverlay. The other views (segments,
  // words) wrap their own subscribers below.
  if (view === 'text')
    return <PlainText body={body} partialWords={partialWords} />
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
  // Interval-containment across committed + partial. WordsGrid renders
  // partial cells immediately after committed at index
  // committed.length + partialIdx.
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
// Committed + partial words are joined and rendered as a SINGLE text
// node inside one <span>. The partial tail is coloured grey via a
// CSS Highlight Range; on commit the Range shrinks in place. DOM is
// O(1) regardless of session length — five elements total.
//
// Cursor + click work via the Range API on the text node:
//   - Cursor: word index → character offset → Range → getBoundingClientRect.
//   - Click: caretRangeFromPoint(x, y) → character offset → binary
//     search the offsets array → word → seek.

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

/** Pure text-node renderer. ONE text span carries the entire transcript
 * (committed words + in-flight partial words joined by spaces). The
 * partial tail is coloured grey via the CSS Highlight API — a Range
 * positioned over the partial char range, with `::highlight(partial)`
 * setting the color in CSS. When a chunk commits, only the highlight
 * range shrinks — the text node content at the formerly-partial
 * positions doesn't change, so there's zero layout shift and the
 * transition is literally "characters change color" without any
 * content/height jitter to bounce the auto-scroll. */
function PlainText({
  body,
  partialWords,
}: {
  body: VerboseJsonResponse
  partialWords: Word[]
}) {
  const committed = body.words ?? []
  const containerRef = useRef<HTMLDivElement | null>(null)
  const textSpanRef = useRef<HTMLSpanElement | null>(null)

  // Content-fingerprint cache for the heavy text+offsets build.
  // `body.words` and `partialWords` come in as FRESH array references
  // on every parent render — Sidebar rebuilds the verbose body on
  // every WS message even when the underlying word list hasn't
  // grown. A naive `useMemo([committed, partialWords])` would
  // re-compute (and re-string-identify) on every WS message, which
  // is what triggered the flash: the highlight useEffect saw a new
  // `text` reference and tore down + recreated the CSS Highlight.
  //
  // The fingerprint here is `committed.length` (committed is
  // append-only — a length match means identical content) plus a
  // cheap O(partial) join of partial words (partials are small — at
  // most a few dozen tokens). When the fingerprint matches the last
  // render, we return the cached text/offsets — same string
  // identity, so the highlight effect's `text` dep doesn't trip and
  // the Highlight is left alone.
  const partialJoin = partialWords.map((w) => w.word).join(' ')
  const cacheRef = useRef<{
    committedLen: number
    partialJoin: string
    text: string
    offsets: number[]
    partialStartChar: number
  } | null>(null)
  let cache = cacheRef.current
  if (
    !cache ||
    cache.committedLen !== committed.length ||
    cache.partialJoin !== partialJoin
  ) {
    const total = committed.length + partialWords.length
    const offsets: number[] = new Array(total)
    const parts: string[] = new Array(total)
    let pos = 0
    for (let i = 0; i < committed.length; i++) {
      offsets[i] = pos
      parts[i] = committed[i].word
      pos += committed[i].word.length + 1
    }
    const partialStart = pos
    for (let i = 0; i < partialWords.length; i++) {
      offsets[committed.length + i] = pos
      parts[committed.length + i] = partialWords[i].word
      pos += partialWords[i].word.length + 1
    }
    cache = {
      committedLen: committed.length,
      partialJoin,
      text: parts.join(' '),
      offsets,
      partialStartChar: partialStart,
    }
    cacheRef.current = cache
  }
  const { text, offsets, partialStartChar } = cache

  // Position the 'partial' CSS Highlight over the partial char range.
  // Stable Highlight + Range refs: the Highlight object is created
  // exactly once (idempotent — first render that has the API and a
  // valid text node), and the Range is mutated in place via
  // setStart/setEnd. This is the critical difference from the
  // previous implementation which did `cssH.set('partial', new
  // Highlight(range))` on every effect run — that recreate visibly
  // flashed the partial region on every WS message because the
  // browser tore down + repainted the entire highlighted area.
  //
  // Highlight API is in Chromium 105+ / WebKit 17.2+; on older
  // engines the partial text just renders in default colour (no JS
  // error — graceful degradation).
  //
  // useLayoutEffect (not useEffect) so the Range reposition lands
  // BEFORE the browser paints the just-mutated text node. With
  // useEffect the order was: React mutates `<span>` text → browser
  // paints with stale Highlight position (new partial chars show up
  // default/white) → effect runs → second paint with correct
  // Highlight. On fast commits (file+WS) the intermediate frame was
  // visible as a "grey text disappearing then reappearing as white"
  // flash. useLayoutEffect collapses both into one paint.
  const highlightRef = useRef<unknown>(null)
  const rangeRef = useRef<Range | null>(null)
  useLayoutEffect(() => {
    type HighlightLike = { size?: number }
    type CSSWithHighlights = typeof CSS & {
      highlights?: { set: (name: string, h: unknown) => void }
    }
    type WithHighlight = Window & {
      Highlight?: new (...ranges: Range[]) => HighlightLike
    }
    const cssH = (CSS as CSSWithHighlights).highlights
    const HighlightCtor = (window as unknown as WithHighlight).Highlight
    if (!cssH || !HighlightCtor) return
    const span = textSpanRef.current
    const textNode = span?.firstChild
    if (!textNode || textNode.nodeType !== Node.TEXT_NODE) return

    let r = rangeRef.current
    if (!r) {
      r = document.createRange()
      rangeRef.current = r
    }
    // When partialWords is empty, collapse the Range to a 0-length
    // span at position 0. Browser renders no highlighted area, but
    // the Highlight + Range stay alive so the next non-empty partial
    // doesn't pay the recreate cost. Try/catch protects against
    // setStart/setEnd throwing if the text node was replaced
    // mid-paint (rare; bail and let the next render retry).
    const start = partialWords.length === 0 ? 0 : partialStartChar
    const end = partialWords.length === 0 ? 0 : text.length
    try {
      r.setStart(textNode, start)
      r.setEnd(textNode, end)
    } catch {
      return
    }
    if (!highlightRef.current) {
      const h = new HighlightCtor(r)
      highlightRef.current = h
      cssH.set('partial', h)
    }
  }, [text, partialStartChar, partialWords.length])

  // Click handler: caret → char offset → word index → seek. Binary
  // search over the unified offsets array (committed + partial in one).
  const onClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const caret = caretFromPoint(e.clientX, e.clientY)
      if (!caret) return
      const span = textSpanRef.current
      if (!span || caret.node !== span.firstChild) return
      const offset = caret.offset
      let lo = 0,
        hi = offsets.length - 1,
        found = -1
      while (lo <= hi) {
        const mid = (lo + hi) >> 1
        if (offsets[mid] <= offset) {
          found = mid
          lo = mid + 1
        } else {
          hi = mid - 1
        }
      }
      if (found < 0) return
      const word =
        found < committed.length
          ? committed[found]
          : partialWords[found - committed.length]
      seek(word.start)
      play()
    },
    [committed, partialWords, offsets],
  )

  if (committed.length === 0 && partialWords.length === 0) {
    return <div className="editorial-body">{body.text}</div>
  }

  return (
    <div
      className="editorial-body"
      ref={containerRef}
      onClick={onClick}
      style={{ position: 'relative', cursor: 'text' }}
    >
      <span ref={textSpanRef}>{text}</span>
      <CursorOverlay
        committed={committed}
        partialWords={partialWords}
        offsets={offsets}
        textSpanRef={textSpanRef}
        containerRef={containerRef}
      />
    </div>
  )
}

/** Subscribes to `useCurrentTime()`, computes the active word index,
 * and imperatively positions an absolutely-positioned underline span
 * via the Range API over the single text node. */
function CursorOverlay({
  committed,
  partialWords,
  offsets,
  textSpanRef,
  containerRef,
}: {
  committed: Word[]
  partialWords: Word[]
  offsets: number[]
  textSpanRef: React.RefObject<HTMLSpanElement | null>
  containerRef: React.RefObject<HTMLDivElement | null>
}) {
  const t = useCurrentTime()
  const lineRef = useRef<HTMLSpanElement | null>(null)
  const lastIdxRef = useRef<number>(-1)
  const lastTRef = useRef<number>(0)

  // Interval-containment across committed + partial. Indices are
  // "global": 0..committed.length-1 = committed, committed.length..=
  // partial.
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

    const span = textSpanRef.current
    const textNode = span?.firstChild
    if (!textNode || textNode.nodeType !== Node.TEXT_NODE) return

    const charStart = offsets[activeIdx]
    const word =
      activeIdx < committed.length
        ? committed[activeIdx]
        : partialWords[activeIdx - committed.length]
    if (!word) return
    const wordLen = word.word.length

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
  }, [activeIdx, committed, partialWords, offsets, textSpanRef, containerRef])

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
