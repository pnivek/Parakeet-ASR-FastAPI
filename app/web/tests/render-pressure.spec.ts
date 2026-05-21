import { test, expect } from '@playwright/test'
import path from 'node:path'
import fs from 'node:fs'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
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

/**
 * Regression gate for the "Stop flushes everything" mic / streaming bug.
 *
 * Symptom: TranscriptSection used to run a rAF tick during `live=true`
 * that called setNow(...) 60x/s. That saturated the main thread, kept
 * React from flushing partials, and partials only "caught up" after
 * Stop killed the rAF. We now drive reveals via CSS keyframes — zero
 * per-frame React work.
 *
 * This test pushes the heaviest plausible stream (a 25-min file via
 * progressive WS), counts main-thread long tasks during streaming, and
 * fails if the cumulative long-task time exceeds a modest budget.
 */
test('streaming long file does not saturate the main thread', async ({ page }) => {
  test.setTimeout(180_000)

  if (!fs.existsSync(FIXTURE)) {
    test.skip(true, `Fixture missing: ${FIXTURE}`)
    return
  }

  await page.addInitScript(() => {
    const longTasks: { duration: number; t: number }[] = []
    if ('PerformanceObserver' in window) {
      try {
        const obs = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            // Anything > 50ms is a frame drop on a 60Hz target.
            longTasks.push({ duration: entry.duration, t: performance.now() })
          }
        })
        obs.observe({ entryTypes: ['longtask'] })
      } catch {
        // ignore
      }
    }
    ;(window as unknown as { __longTasks: typeof longTasks }).__longTasks = longTasks
  })

  await page.goto('/')
  await page.getByRole('button', { name: /^source$/i }).click()
  await page.getByRole('radio', { name: 'File upload' }).click()
  await page.getByRole('button', { name: /^engine$/i }).click()
  await page.getByRole('radio', { name: 'progressive' }).click()
  await page.getByRole('button', { name: /^source$/i }).click()
  await page.locator('input[type="file"]').first().setInputFiles(FIXTURE)
  await page.getByRole('button', { name: /Transcribe/i }).click()

  // Wait until final or 120s. We expect ~22s.
  const deadline = Date.now() + 120_000
  let done = false
  while (Date.now() < deadline) {
    done = await page.evaluate(() => {
      const el = document.querySelector('.editorial-body')
      return !!el && el.textContent !== null && el.textContent.length > 1000
    })
    if (done) break
    await page.waitForTimeout(500)
  }

  const longTasks = await page.evaluate(() =>
    (window as unknown as { __longTasks: { duration: number; t: number }[] }).__longTasks,
  )
  const total = longTasks.reduce((a, b) => a + b.duration, 0)
  const longest = longTasks.length ? Math.max(...longTasks.map((t) => t.duration)) : 0
  console.log(`long tasks: count=${longTasks.length} longest=${longest.toFixed(0)}ms total=${total.toFixed(0)}ms`)

  expect(done, 'stream finished within 120s').toBe(true)
  // Budget: under the old code we routinely saw >2s cumulative long-task
  // time during a 22s stream. After the CSS-driven reveal we should be
  // comfortably under 1s, leaving 22x the budget for runtime variance.
  expect(total, 'cumulative main-thread blocking').toBeLessThan(2000)
})
