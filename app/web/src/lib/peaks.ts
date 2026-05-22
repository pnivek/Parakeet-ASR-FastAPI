/**
 * Decode an audio File/Blob and downsample to a fixed-length peaks array
 * in [0..1], for waveform rendering.
 *
 * Two paths:
 *
 *  - WAV (PCM int16/int24/float32 in RIFF): parse the header, then read
 *    220 tiny windows out of the file via Blob.slice(). We never load
 *    the whole audio — even a 3-hour, 350 MB recording only reads a few
 *    MB total. No decodeAudioData, no AudioBuffer allocation. Works at
 *    arbitrary size.
 *
 *  - Anything else (mp3/m4a/opus/webm): decodeAudioData on the whole
 *    blob, then iterate the channel data with a stride so peak compute
 *    stays bounded regardless of duration. Capped at 200 MB compressed
 *    input — decoded PCM for longer files can OOM the tab.
 */

/** Files larger than this skip the decodeAudioData path. WAV bypasses it. */
const COMPRESSED_DECODE_CAP_BYTES = 200 * 1024 * 1024

export async function computePeaks(file: Blob, bins = 220): Promise<number[]> {
  // Sniff the first 12 bytes for the RIFF/WAVE (or RIFX) magic so the
  // fast path runs for ANY wav — regardless of filename or MIME type.
  // This is what makes long uploads (multi-hour WAVs that blow past the
  // decodeAudioData cap) still render a waveform.
  let isWav = false
  try {
    const sig = new Uint8Array(await file.slice(0, 12).arrayBuffer())
    const riff =
      (sig[0] === 0x52 && sig[1] === 0x49 && sig[2] === 0x46 && sig[3] === 0x46) || // "RIFF"
      (sig[0] === 0x52 && sig[1] === 0x49 && sig[2] === 0x46 && sig[3] === 0x58) //   "RIFX"
    const wave = sig[8] === 0x57 && sig[9] === 0x41 && sig[10] === 0x56 && sig[11] === 0x45 // "WAVE"
    isWav = riff && wave
  } catch {
    // ignore — fall through to extension/MIME heuristics
  }
  if (!isWav) {
    const name = (file as File).name || ''
    isWav =
      name.toLowerCase().endsWith('.wav') ||
      file.type === 'audio/wav' ||
      file.type === 'audio/x-wav'
  }

  if (isWav) {
    try {
      return await fastWavPeaks(file, bins)
    } catch (e) {
      // Header parse failed (unusual format, or mislabelled). Fall
      // through to the decode path.
      console.warn('fastWavPeaks failed, falling back to decodeAudioData:', e)
    }
  }

  if (file.size > COMPRESSED_DECODE_CAP_BYTES) {
    throw new Error(
      `Peaks decode skipped: ${(file.size / 1024 / 1024).toFixed(0)} MB > ${COMPRESSED_DECODE_CAP_BYTES / 1024 / 1024} MB cap.`,
    )
  }
  return await decodedPeaks(file, bins)
}

/**
 * Read a WAV's duration in seconds straight from the header — no
 * decode, no <audio> element. Returns null if it's not a parseable
 * WAV. Used as a duration fallback in the hero meta for very long
 * uploads where the <audio> element is slow (or fails) to report
 * `duration`.
 */
export async function readWavDuration(file: Blob): Promise<number | null> {
  try {
    const headerSlice = await file.slice(0, Math.min(64 * 1024, file.size)).arrayBuffer()
    const dv = new DataView(headerSlice)
    const td = new TextDecoder('ascii')
    const riff = td.decode(new Uint8Array(headerSlice, 0, 4))
    if (riff !== 'RIFF' && riff !== 'RIFX') return null
    const isLE = riff === 'RIFF'
    if (td.decode(new Uint8Array(headerSlice, 8, 4)) !== 'WAVE') return null
    let sampleRate = 0
    let channels = 0
    let bitsPerSample = 0
    let dataSize = 0
    let dataOffset = 0
    let cur = 12
    while (cur + 8 <= headerSlice.byteLength) {
      const id = td.decode(new Uint8Array(headerSlice, cur, 4))
      const size = dv.getUint32(cur + 4, isLE)
      if (id === 'fmt ') {
        channels = dv.getUint16(cur + 10, isLE)
        sampleRate = dv.getUint32(cur + 12, isLE)
        bitsPerSample = dv.getUint16(cur + 22, isLE)
      } else if (id === 'data') {
        dataOffset = cur + 8
        dataSize = size
        break
      }
      cur += 8 + size + (size & 1)
    }
    if (!sampleRate || !channels || !bitsPerSample) return null
    const bytesPerFrame = (bitsPerSample / 8) * channels
    const usable =
      dataSize > 0 && dataOffset + dataSize <= file.size
        ? dataSize
        : Math.max(0, file.size - dataOffset)
    const frames = Math.floor(usable / bytesPerFrame)
    const seconds = frames / sampleRate
    return isFinite(seconds) && seconds > 0 ? seconds : null
  } catch {
    return null
  }
}

/**
 * Direct WAV reader. Parses the RIFF chunks to find the data chunk + PCM
 * format, then takes 220 strided peeks across the file via Blob.slice().
 * Memory + CPU is O(bins), not O(file size).
 */
export async function fastWavPeaks(file: Blob, bins = 220): Promise<number[]> {
  // The 'fmt ' + 'data' chunks usually live in the first 1 KB but some
  // tools dump large LIST/INFO/JUNK metadata first; grab 64 KB to be
  // safe before giving up and falling back to decodeAudioData.
  const headerSlice = await file.slice(0, Math.min(64 * 1024, file.size)).arrayBuffer()
  const dv = new DataView(headerSlice)
  const td = new TextDecoder('ascii')

  const riff = td.decode(new Uint8Array(headerSlice, 0, 4))
  if (riff !== 'RIFF' && riff !== 'RIFX') throw new Error('Not a RIFF file')
  const isLE = riff === 'RIFF' // RIFX is big-endian (rare)
  const form = td.decode(new Uint8Array(headerSlice, 8, 4))
  if (form !== 'WAVE') throw new Error('Not a WAV (form ' + form + ')')

  let formatCode = 0
  let channels = 0
  let bitsPerSample = 0
  let dataOffset = 0
  let dataSize = 0

  // Walk chunks. RIFF chunk size for the data chunk can be wrong for
  // files >4 GB or unknown-length streams — defend against that below.
  let cur = 12
  while (cur + 8 <= headerSlice.byteLength) {
    const chunkId = td.decode(new Uint8Array(headerSlice, cur, 4))
    const chunkSize = dv.getUint32(cur + 4, isLE)
    if (chunkId === 'fmt ') {
      formatCode = dv.getUint16(cur + 8, isLE)
      channels = dv.getUint16(cur + 10, isLE)
      // sampleRate = dv.getUint32(cur + 12, isLE)
      bitsPerSample = dv.getUint16(cur + 22, isLE)
    } else if (chunkId === 'data') {
      dataOffset = cur + 8
      dataSize = chunkSize
      break
    }
    cur += 8 + chunkSize + (chunkSize & 1) // word-align
  }
  if (!dataOffset) throw new Error('No data chunk found in first 16 KB')
  if (formatCode !== 1 && formatCode !== 3) {
    // 1 = PCM, 3 = IEEE float. Anything else (μ-law, A-law, extensible)
    // we don't handle here.
    throw new Error(`Unsupported WAV format code ${formatCode}`)
  }
  if (bitsPerSample !== 16 && bitsPerSample !== 24 && bitsPerSample !== 32) {
    throw new Error(`Unsupported bits/sample ${bitsPerSample}`)
  }
  if (!channels) throw new Error('Missing channel count')

  const bytesPerFrame = (bitsPerSample / 8) * channels
  // Some WAVs write 0xFFFFFFFF for streamed content — fall back to file
  // size for the bound in that case.
  const usableDataSize =
    dataSize > 0 && dataOffset + dataSize <= file.size
      ? dataSize
      : Math.max(0, file.size - dataOffset)
  const totalFrames = Math.floor(usableDataSize / bytesPerFrame)
  const framesPerBin = Math.floor(totalFrames / bins)
  if (framesPerBin === 0) return new Array(bins).fill(0)

  // CRITICAL: only read a small representative WINDOW at the start of
  // each bin — NOT the whole bin. Reading the entire bin range means
  // reading the whole file (220 bins × framesPerBin = totalFrames),
  // which for a multi-hour WAV is hundreds of MB across 220 parallel
  // arrayBuffer() allocations → OOM/hang and no waveform. A 2048-frame
  // window per bin keeps total reads to ~900 KB regardless of length.
  const WINDOW_FRAMES = 2048
  const readBin = async (b: number): Promise<number> => {
    const frameStart = b * framesPerBin
    const framesAvail = Math.min(framesPerBin, totalFrames - frameStart)
    const framesToRead = Math.min(WINDOW_FRAMES, framesAvail)
    if (framesToRead <= 0) return 0
    const byteStart = dataOffset + frameStart * bytesPerFrame
    const byteEnd = byteStart + framesToRead * bytesPerFrame
    const buf = await file.slice(byteStart, byteEnd).arrayBuffer()
    const view = new DataView(buf)
    let peak = 0
    if (formatCode === 1 && bitsPerSample === 16) {
      // 16-bit PCM via Int16Array. Channel 0 only (step by channels).
      const samples = new Int16Array(buf)
      for (let i = 0; i < samples.length; i += channels) {
        const v = Math.abs(samples[i])
        if (v > peak) peak = v
      }
      return peak / 32768
    }
    if (formatCode === 1 && bitsPerSample === 24) {
      // 24-bit little-endian PCM, channel 0 only.
      for (let f = 0; f < framesToRead; f++) {
        const o = f * bytesPerFrame
        if (o + 3 > buf.byteLength) break
        const lo = view.getUint8(o)
        const mid = view.getUint8(o + 1)
        const hi = view.getInt8(o + 2) // sign-extends
        const v = Math.abs((hi << 16) | (mid << 8) | lo)
        if (v > peak) peak = v
      }
      return peak / (1 << 23)
    }
    if (formatCode === 3 && bitsPerSample === 32) {
      // IEEE float32, channel 0 only.
      const samples = new Float32Array(buf)
      for (let i = 0; i < samples.length; i += channels) {
        const v = Math.abs(samples[i])
        if (v > peak) peak = v
      }
      return peak
    }
    if (formatCode === 1 && bitsPerSample === 32) {
      // 32-bit PCM (rare). Int32Array.
      const samples = new Int32Array(buf)
      for (let i = 0; i < samples.length; i += channels) {
        const v = Math.abs(samples[i])
        if (v > peak) peak = v
      }
      return peak / 2147483648
    }
    return 0
  }

  // Issue all bin reads in parallel — Blob.slice() reads are typically
  // cheap and the browser pipelines them.
  const peaks = await Promise.all(
    Array.from({ length: bins }, (_, b) => readBin(b)),
  )
  let maxPeak = 0
  for (const p of peaks) if (p > maxPeak) maxPeak = p
  if (maxPeak > 0) {
    for (let i = 0; i < peaks.length; i++) peaks[i] = peaks[i] / maxPeak
  }
  return peaks
}

/**
 * Decode + bin via Web Audio API. Strided peak iteration keeps the
 * compute O(bins * 256) regardless of duration, so even a 1-hour file
 * binnable in <100ms after decode. The decode itself runs off-thread.
 */
async function decodedPeaks(file: Blob, bins: number): Promise<number[]> {
  const arrayBuf = await file.arrayBuffer()
  const Ctx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  const ctx = new Ctx()
  try {
    const audio = await ctx.decodeAudioData(arrayBuf.slice(0))
    const ch = audio.getChannelData(0)
    const samplesPerBin = Math.max(1, Math.floor(ch.length / bins))
    const stride = Math.max(1, Math.floor(samplesPerBin / 256))
    const peaks = new Array<number>(bins).fill(0)
    let maxPeak = 0
    for (let b = 0; b < bins; b++) {
      let peak = 0
      const start = b * samplesPerBin
      const end = Math.min(ch.length, start + samplesPerBin)
      for (let i = start; i < end; i += stride) {
        const v = Math.abs(ch[i])
        if (v > peak) peak = v
      }
      peaks[b] = peak
      if (peak > maxPeak) maxPeak = peak
    }
    if (maxPeak > 0) {
      for (let i = 0; i < peaks.length; i++) peaks[i] = peaks[i] / maxPeak
    }
    return peaks
  } finally {
    if (ctx.state !== 'closed') ctx.close().catch(() => {})
  }
}
