import { useMemo, useState, Fragment } from 'react'
import type { TranscriptionResponse } from '../lib/api'
import type { VerboseJsonResponse, WhisperSegment } from '../lib/types'
import { seek } from '../lib/playback'
import { formatTime } from '../lib/format'
import { downloadResult } from '../lib/download'

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
        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          {isVerbose && (
            <div className="view-nav">
              {VIEWS.map((v) => (
                <button
                  key={v}
                  type="button"
                  className={view === v ? 'view-nav__btn view-nav__btn--active' : 'view-nav__btn'}
                  onClick={() => setView(v)}
                >
                  {v}
                </button>
              ))}
            </div>
          )}
          <button
            type="button"
            className="transcript__download"
            onClick={() => downloadResult(result, filename || 'transcript')}
          >
            Download
          </button>
        </div>
      </div>
      <div className="transcript__body">
        {result.format === 'json' && <Code text={JSON.stringify(result.body, null, 2)} />}
        {result.format === 'text' && (
          <div className="editorial-body">{result.body}</div>
        )}
        {result.format === 'srt' && <Code text={result.body} />}
        {result.format === 'vtt' && <Code text={result.body} />}
        {isVerbose && (
          <VerboseBody body={result.body} view={view} currentTime={currentTime} />
        )}
      </div>
    </section>
  )
}

function VerboseBody({
  body,
  view,
  currentTime,
}: {
  body: VerboseJsonResponse
  view: View
  currentTime: number
}) {
  const activeWordIdx = useMemo(() => {
    if (!body.words) return -1
    return body.words.findIndex((w) => currentTime >= w.start && currentTime <= w.end)
  }, [body.words, currentTime])
  const activeSegIdx = useMemo(
    () => body.segments.findIndex((s) => currentTime >= s.start && currentTime <= s.end),
    [body.segments, currentTime],
  )

  const rtfx =
    body.transcription_time_seconds > 0
      ? (body.duration / body.transcription_time_seconds).toFixed(1)
      : '—'

  return (
    <>
      <div className="meta-row">
        <Meta label="STRATEGY" value={body.strategy} />
        <Meta label="DURATION" value={`${body.duration.toFixed(2)}s`} />
        <Meta label="ASR TIME" value={`${body.transcription_time_seconds.toFixed(2)}s`} />
        <Meta label="RTFx" value={`${rtfx}×`} hot />
        <Meta label="SEGMENTS" value={body.segments.length} />
        <Meta label="WORDS" value={body.words?.length ?? 0} />
      </div>
      {view === 'text' && <PlainText body={body} activeWordIdx={activeWordIdx} />}
      {view === 'segments' && <SegmentRows segments={body.segments} activeIdx={activeSegIdx} />}
      {view === 'words' && <WordsGrid body={body} activeIdx={activeWordIdx} />}
      {view === 'raw' && <Code text={JSON.stringify(body, null, 2)} />}
    </>
  )
}

function Meta({ label, value, hot }: { label: string; value: string | number; hot?: boolean }) {
  return (
    <div className={hot ? 'meta-cell meta-cell--hot' : 'meta-cell'}>
      <span className="meta-cell__l">{label}</span>
      <span className="meta-cell__v">{value}</span>
    </div>
  )
}

function PlainText({ body, activeWordIdx }: { body: VerboseJsonResponse; activeWordIdx: number }) {
  if (body.words && body.words.length > 0) {
    return (
      <div className="editorial-body">
        {body.words.map((w, i) => {
          let cls = 'editorial-word'
          if (i === activeWordIdx) cls += ' editorial-word--active'
          else if (i > activeWordIdx + 6 && activeWordIdx >= 0) cls += ' editorial-word--upcoming'
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
  return <div className="editorial-body">{body.text}</div>
}

function SegmentRows({ segments, activeIdx }: { segments: WhisperSegment[]; activeIdx: number }) {
  const [openId, setOpenId] = useState<number | null>(null)
  if (segments.length === 0) return <div className="empty">No segments.</div>
  return (
    <div className="segs-list">
      {segments.map((s, i) => {
        const active = i === activeIdx
        const open = openId === s.id
        return (
          <Fragment key={s.id}>
            <div
              className={active ? 'segs-row segs-row--active' : 'segs-row'}
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
              <div className="segs-row__time num">
                {formatTime(s.start)}
                <br />
                <span className="segs-row__time-end">→ {formatTime(s.end)}</span>
              </div>
              <div className="segs-row__text">{s.text}</div>
            </div>
            {open && (
              <div className="segs-detail">
                <Kv k="id" v={s.id} />
                <Kv k="seek" v={s.seek} />
                <Kv k="temperature" v={s.temperature} />
                <Kv k="avg_logprob" v={s.avg_logprob} />
                <Kv k="compression_ratio" v={s.compression_ratio} />
                <Kv k="no_speech_prob" v={s.no_speech_prob} />
              </div>
            )}
          </Fragment>
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
          onClick={() => seek(w.start)}
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

function Code({ text }: { text: string }) {
  return <pre className="code">{text}</pre>
}
