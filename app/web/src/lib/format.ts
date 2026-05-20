/** Format seconds as M:SS for position counters / time labels. */
export function formatTime(sec: number): string {
  if (!isFinite(sec) || sec < 0) return '0:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
    .toString()
    .padStart(2, '0')
  return `${m}:${s}`
}

/** Format bytes as a human-readable size. */
export function formatBytes(n: number): string {
  if (!isFinite(n) || n <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = n
  let u = 0
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024
    u++
  }
  return `${v.toFixed(v < 10 && u > 0 ? 1 : 0)} ${units[u]}`
}
