import { test, expect } from '@playwright/test'

/**
 * Smoke test: load the deployed Maison v6 UI, walk the three sidebar
 * tabs + the three input modes, verify the Engine gate, and capture
 * screenshots.
 */
test('Maison v6 UI — boots, tabs switch, modes switch, engine gate enforced', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text())
  })
  page.on('pageerror', (e) => {
    consoleErrors.push(`pageerror: ${e.message}`)
  })

  await page.goto('/')

  // Title + brand
  await expect(page).toHaveTitle(/Parakeet/)
  await expect(page.getByText('Parakeet', { exact: true })).toBeVisible()
  await expect(page.getByText('playground')).toBeVisible()

  // Retired chrome must be gone.
  await expect(page.getByText(/v5\s*[·•]\s*maison/i)).toHaveCount(0)
  await expect(page.getByText(/^Pricing$/i)).toHaveCount(0)
  await expect(page.getByText(/Get API key/i)).toHaveCount(0)

  // Sidebar uses three tabs now: source / output / engine
  const sourceTab = page.getByRole('button', { name: /^source$/i })
  const outputTab = page.getByRole('button', { name: /^output$/i })
  const engineTab = page.getByRole('button', { name: /^engine$/i })
  await expect(sourceTab).toBeVisible()
  await expect(outputTab).toBeVisible()
  await expect(engineTab).toBeVisible()

  // Source tab — Input list defaults to file → file card shown.
  await sourceTab.click()
  await expect(page.getByText(/Drop a file or browse/i)).toBeVisible()

  // Switch to URL mode via the Input option list.
  await page.getByRole('radio', { name: 'URL' }).click()
  await expect(page.getByPlaceholder('https://…')).toBeVisible()

  // Switch to mic mode — the Capture sub-list (Live / Recording) shows.
  await page.getByRole('radio', { name: 'Microphone' }).click()
  await expect(page.getByRole('radio', { name: /Live transcription/i })).toBeVisible()
  await expect(page.getByRole('radio', { name: /^Recording$/i })).toBeVisible()

  // Output tab — Format option list visible (current = verbose_json),
  // Timestamps are ma-check rows.
  await outputTab.click()
  await expect(page.getByRole('radio', { name: 'verbose_json' })).toBeVisible()
  await expect(page.getByRole('button', { name: /^Segment$/i })).toBeVisible()
  await expect(page.getByRole('button', { name: /^Word$/i })).toBeVisible()

  // Engine tab — Engine picker (Offline / Streaming) + Strategy override.
  await engineTab.click()
  await expect(page.getByRole('radio', { name: 'Offline' })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Streaming' })).toBeVisible()
  // The override is visible (Engine defaults to Offline → it's shown).
  await expect(page.getByRole('radio', { name: 'Auto' })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Full pass' })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Split-full' })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Chunked' })).toBeVisible()

  // Engine gate: file allows both engines.
  await sourceTab.click()
  await page.getByRole('radio', { name: 'File upload' }).click()
  await engineTab.click()
  await expect(page.getByRole('radio', { name: 'Offline' })).toBeEnabled()
  await expect(page.getByRole('radio', { name: 'Streaming' })).toBeEnabled()

  // Engine gate: URL locks Streaming out (REST upload only).
  await sourceTab.click()
  await page.getByRole('radio', { name: 'URL' }).click()
  await engineTab.click()
  await expect(page.getByRole('radio', { name: 'Streaming' })).toBeDisabled()

  // Footer rail metric labels present.
  for (const label of ['STRATEGY', 'ASR', 'RTFx', 'TTFS', 'SEGMENTS', 'WORDS']) {
    await expect(page.getByText(label, { exact: true })).toBeVisible()
  }

  // Transcribe pill pinned at the bottom of the sidebar.
  await expect(page.getByRole('button', { name: /Transcribe/i })).toBeVisible()

  // Snapshots.
  await page.screenshot({ path: 'tests/screenshots/maison-source.png', fullPage: true })
  await outputTab.click()
  await page.screenshot({ path: 'tests/screenshots/maison-output.png', fullPage: true })
  await engineTab.click()
  await page.screenshot({ path: 'tests/screenshots/maison-engine.png', fullPage: true })

  expect(consoleErrors, `client console errors:\n${consoleErrors.join('\n')}`).toEqual([])
})
