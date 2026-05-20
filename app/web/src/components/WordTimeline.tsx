import type { Word } from '../lib/types'
import { seek } from '../lib/playback'

interface Props {
  words: Word[]
  /** Current audio playback time, drives the active-word highlight. */
  currentTime?: number
}

/**
 * Word-by-word inline display. Each word is a clickable chip; clicking
 * seeks the audio. The word containing `currentTime` is highlighted so
 * users see the playhead position synchronously.
 */
export function WordTimeline({ words, currentTime }: Props) {
  if (words.length === 0) return null
  return (
    <div className="words">
      {words.map((w, i) => {
        const isActive =
          currentTime !== undefined && currentTime >= w.start && currentTime < w.end
        return (
          <button
            type="button"
            key={`${i}-${w.start}`}
            className={`words__word ${isActive ? 'words__word--active' : ''}`}
            onClick={() => seek(w.start)}
            title={`${w.start.toFixed(2)}s – ${w.end.toFixed(2)}s`}
          >
            {w.word}
          </button>
        )
      })}
    </div>
  )
}
