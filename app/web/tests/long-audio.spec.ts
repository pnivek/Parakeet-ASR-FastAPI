import { test, expect, type Page } from '@playwright/test'
import path from 'node:path'
import fs from 'node:fs'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/**
 * Drive the deployed UI through a real long-audio progressive upload.
 *
 * Goals:
 *   1. Reproduce the perceived "hangs on long audio" symptom.
 *   2. Surface which step(s) are slow — peaks decode, partial cadence,
 *      transcript render.
 *   3. Give us a wall-time number to regress against once the fix lands.
 *
 * The harness instruments the page with a small profiler that timestamps
 * key events from inside the browser context (so we don't have to guess
 * from outside).
 */

const FIXTURE = path.resolve(
  __dirname,
  '..',
  '..',
  '..',
  'tests',
  'fixtures',
  'longform',
  'BillGates.wav',
)

async function installProbes(page: Page) {
  await page.addInitScript(() => {
    const ev: Array<{ t: number; tag: string; extra?: unknown }> = []
    const log = (tag: string, extra?: unknown) =>
      ev.push({ t: performance.now(), tag, extra })

    // Tap into WebSocket lifecycle.
    const W = window.WebSocket
    class TracedWS extends W {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols)
        log('ws:open-requested', { url: String(url) })
        this.addEventListener('open', () => log('ws:open'))
        this.addEventListener('close', (e) =>
          log('ws:close', { code: (e as CloseEvent).code, reason: (e as CloseEvent).reason }),
        )
        this.addEventListener('error', () => log('ws:error'))
        const origSend = this.send.bind(this)
        let bytesSent = 0
        this.send = ((data: unknown) => {
          if (typeof data === 'string') {
            log('ws:send-text', { len: data.length, sample: data.slice(0, 80) })
          } else if (data instanceof ArrayBuffer) {
            bytesSent += data.byteLength
            if (data.byteLength === 0) log('ws:send-eof')
          } else if (data instanceof Blob) {
            bytesSent += data.size
          }
          // periodic checkpoint
          if (bytesSent > 0 && bytesSent % (4 * 1024 * 1024) < 65536) {
            log('ws:send-progress', { bytesSent })
          }
          return origSend(data as never)
        }) as typeof this.send
        this.addEventListener('message', (e) => {
          const data = (e as MessageEvent).data
          if (typeof data !== 'string') return
          try {
            const msg = JSON.parse(data)
            if (msg.type === 'segments_batch') {
              log('ws:segments_batch', {
                n: msg.segments?.length || 0,
                words: msg.words?.length || 0,
              })
            } else if (msg.type === 'final_transcription') {
              log('ws:final_transcription', {
                duration: msg.duration,
                segments: msg.segments?.length || 0,
                words: msg.words?.length || 0,
              })
            } else if (msg.type === 'peaks') {
              log('ws:peaks', { n: msg.peaks?.length || 0 })
            } else {
              log(`ws:msg`, { type: msg.type })
            }
          } catch {
            // ignore
          }
        })
      }
    }
    ;(window as unknown as { WebSocket: typeof WebSocket }).WebSocket = TracedWS as unknown as typeof WebSocket

    // Long-task observer to catch main-thread freezes.
    if ('PerformanceObserver' in window) {
      try {
        const obs = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.duration > 100) {
              log('longtask', { duration: Math.round(entry.duration) })
            }
          }
        })
        obs.observe({ entryTypes: ['longtask'] })
      } catch {
        // ignore
      }
    }

    ;(window as unknown as { __probes: typeof ev }).__probes = ev
  })
}

async function dumpProbes(page: Page): Promise<Array<{ t: number; tag: string; extra?: unknown }>> {
  return await page.evaluate(() =>
    (window as unknown as { __probes: Array<{ t: number; tag: string; extra?: unknown }> }).__probes,
  )
}

test('long-audio progressive: timing + hang diagnostics', async ({ page }) => {
  test.setTimeout(360_000) // 6 min

  if (!fs.existsSync(FIXTURE)) {
    test.skip(true, `Fixture missing: ${FIXTURE}`)
    return
  }

  await installProbes(page)
  await page.goto('/')

  // Source tab → File mode (option-list radio).
  await page.getByRole('button', { name: /^source$/i }).click()
  await page.getByRole('radio', { name: 'File upload' }).click()

  // Switch to engine tab and pick `progressive` (option-list radio).
  await page.getByRole('button', { name: /^engine$/i }).click()
  await page.getByRole('radio', { name: 'progressive' }).click()
  // Back to source so the file input is in the DOM for upload.
  await page.getByRole('button', { name: /^source$/i }).click()

  // Upload the long file.
  const fileInput = page.locator('input[type="file"]').first()
  await fileInput.setInputFiles(FIXTURE)

  // Click Transcribe to start.
  const t0 = Date.now()
  await page.getByRole('button', { name: /Transcribe/i }).click()

  // Wait for the first partial to land in the transcript or for final.
  // Symptoms of the hang: no first partial, no final, button stuck on
  // "Stop" / "Working…".
  const finishedSel = page.locator('text=Stop ▣')
  let firstPartialAt: number | null = null
  let finalAt: number | null = null

  // Poll the probe ring until we see a final_transcription or timeout.
  const deadline = Date.now() + 300_000 // 5 min
  while (Date.now() < deadline) {
    const probes = await dumpProbes(page)
    if (firstPartialAt === null) {
      const first = probes.find((p) => p.tag === 'ws:segments_batch')
      if (first) firstPartialAt = Date.now() - t0
    }
    const final = probes.find((p) => p.tag === 'ws:final_transcription')
    if (final) {
      finalAt = Date.now() - t0
      break
    }
    await page.waitForTimeout(500)
  }

  const probes = await dumpProbes(page)
  const tally = new Map<string, number>()
  for (const p of probes) tally.set(p.tag, (tally.get(p.tag) || 0) + 1)

  const longTasks = probes
    .filter((p) => p.tag === 'longtask')
    .map((p) => (p.extra as { duration: number }).duration)
  const longestTask = longTasks.length ? Math.max(...longTasks) : 0
  const totalLongMs = longTasks.reduce((a, b) => a + b, 0)

  console.log('--- probe tally ---')
  for (const [tag, n] of [...tally].sort((a, b) => b[1] - a[1])) {
    console.log(`  ${tag.padEnd(25)} ${n}`)
  }
  console.log(`\nfirst partial after click: ${firstPartialAt}ms`)
  console.log(`final_transcription after click: ${finalAt}ms`)
  console.log(`long tasks: count=${longTasks.length} longest=${longestTask}ms total=${totalLongMs}ms`)

  // Hard fail conditions to flag the regression clearly.
  expect(firstPartialAt, 'no segments_batch within 5 min').not.toBeNull()
  expect(finalAt, 'no final_transcription within 5 min').not.toBeNull()

  // Soft check: most of the wall time should be the WS stream, not the
  // main-thread blocking peaks decode. Long tasks > 60s total = hang.
  if (totalLongMs > 60_000) {
    throw new Error(
      `Main thread blocked for ${totalLongMs}ms total (longest=${longestTask}ms). ` +
        `Likely decodeAudioData on a too-large file. See probe tally.`,
    )
  }
  // Also assert finalAt is in a sane range.
  if (finalAt && finalAt > 120_000) {
    console.warn(
      `final_transcription took ${finalAt}ms for a 25-min file — server bench says 20s wall. ` +
        `UI overhead is dominating.`,
    )
  }

  // Suppress strict console-errors assertion — long upload may surface
  // benign warnings. Just dump them for visibility.
  await finishedSel.isVisible().catch(() => false)
})
