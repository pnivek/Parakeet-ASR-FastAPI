/**
 * Decode an audio File (or Blob) and downsample to a fixed-length array of
 * absolute-value peaks in [0..1] for waveform rendering.
 *
 * Uses Web Audio's decodeAudioData. We never have to play through this
 * AudioContext — it's only used to decode, then closed.
 */
export async function computePeaks(file: Blob, bins = 200): Promise<number[]> {
  const arrayBuf = await file.arrayBuffer()
  const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  const ctx = new Ctx()
  try {
    const audio = await ctx.decodeAudioData(arrayBuf.slice(0))
    const ch = audio.getChannelData(0)
    const samplesPerBin = Math.max(1, Math.floor(ch.length / bins))
    const peaks = new Array<number>(bins).fill(0)
    let maxPeak = 0
    for (let b = 0; b < bins; b++) {
      let peak = 0
      const start = b * samplesPerBin
      const end = Math.min(ch.length, start + samplesPerBin)
      for (let i = start; i < end; i++) {
        const v = Math.abs(ch[i])
        if (v > peak) peak = v
      }
      peaks[b] = peak
      if (peak > maxPeak) maxPeak = peak
    }
    // Normalize so the loudest bin = 1.0; keeps quiet clips visible.
    if (maxPeak > 0) {
      for (let i = 0; i < peaks.length; i++) peaks[i] = peaks[i] / maxPeak
    }
    return peaks
  } finally {
    if (ctx.state !== 'closed') ctx.close().catch(() => {})
  }
}
