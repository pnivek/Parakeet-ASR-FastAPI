import { test, expect } from '@playwright/test'

/**
 * Smoke test: load the deployed Maison v6 UI, walk the three modality
 * tabs, verify the per-tab engine + strategy + output picker, and
 * capture screenshots.
 */
test('Maison v6 UI — boots, modality tabs switch, per-tab engine picker works', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text())
  })
  page.on('pageerror', (e) => {
    consoleErrors.push(`pageerror: ${e.message}`)
  })

  await page.goto('/')

  await expect(page).toHaveTitle(/Parakeet/)
  await expect(page.getByText('Parakeet', { exact: true })).toBeVisible()
  await expect(page.getByText('playground')).toBeVisible()

  // Retired chrome must be gone.
  await expect(page.getByText(/v5\s*[·•]\s*maison/i)).toHaveCount(0)
  await expect(page.getByText(/^Pricing$/i)).toHaveCount(0)
  await expect(page.getByText(/Get API key/i)).toHaveCount(0)

  // Sidebar now uses three modality tabs: Upload / URL / Record.
  const uploadTab = page.getByRole('button', { name: /^Upload$/ })
  const urlTab = page.getByRole('button', { name: /^URL$/ })
  const recordTab = page.getByRole('button', { name: /^Record$/ })
  await expect(uploadTab).toBeVisible()
  await expect(urlTab).toBeVisible()
  await expect(recordTab).toBeVisible()

  // Upload tab — file card + Engine picker (REST / WebSocket) + Strategy.
  await uploadTab.click()
  await expect(page.getByText(/Drop a file or browse/i)).toBeVisible()
  await expect(page.getByRole('radio', { name: 'REST' })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'WebSocket' })).toBeVisible()
  // REST default → Strategy override visible (no Chunked).
  await expect(page.getByRole('radio', { name: 'Auto' })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Full pass' })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Split-full' })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Chunked' })).toHaveCount(0)
  // Output/Timestamps controls retired — format choice now lives in
  // the hero download flyout (verbose_json is always requested).
  await expect(page.getByRole('radio', { name: 'verbose_json' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: /^Segment$/i })).toHaveCount(0)
  await expect(page.getByRole('button', { name: /^Word$/i })).toHaveCount(0)

  // Hero transport row — prev/next segment, play, rewind, sync. Live
  // (URL-only) and Download (result-only) are conditional and not
  // asserted here.
  await expect(page.getByRole('button', { name: /Previous segment/i })).toBeVisible()
  await expect(page.getByRole('button', { name: /Next segment/i })).toBeVisible()
  await expect(page.getByRole('button', { name: /^Play$/ })).toBeVisible()
  await expect(page.getByRole('button', { name: /Rewind/i })).toBeVisible()
  await expect(page.getByRole('button', { name: /^Sync$/ })).toBeVisible()

  // URL tab — URL input + Engine picker.
  await urlTab.click()
  await expect(page.getByPlaceholder('https://…')).toBeVisible()
  await expect(page.getByRole('radio', { name: 'REST' })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'WebSocket' })).toBeVisible()

  // Record tab — mic meter + Engine picker. The old Live/Recording sub-list
  // is gone (the Engine picker replaces it).
  await recordTab.click()
  await expect(page.getByRole('radio', { name: 'REST' })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'WebSocket' })).toBeVisible()
  await expect(page.getByRole('radio', { name: /Live transcription/i })).toHaveCount(0)

  // Footer rail metric labels present.
  for (const label of ['STRATEGY', 'ASR', 'RTFx', 'TTFS', 'SEGMENTS', 'WORDS']) {
    await expect(page.getByText(label, { exact: true })).toBeVisible()
  }

  // Commit button — scope to the sidebar's commit row so we don't collide
  // with the "Record" modality tab in the tab strip above.
  await expect(
    page.locator('.sb__commit').getByRole('button', { name: /Transcribe|Record/ }),
  ).toBeVisible()

  // Snapshots.
  await uploadTab.click()
  await page.screenshot({ path: 'tests/screenshots/maison-upload.png', fullPage: true })
  await urlTab.click()
  await page.screenshot({ path: 'tests/screenshots/maison-url.png', fullPage: true })
  await recordTab.click()
  await page.screenshot({ path: 'tests/screenshots/maison-record.png', fullPage: true })

  expect(consoleErrors, `client console errors:\n${consoleErrors.join('\n')}`).toEqual([])
})
