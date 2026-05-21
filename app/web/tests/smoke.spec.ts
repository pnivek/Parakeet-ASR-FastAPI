import { test, expect } from '@playwright/test'

/**
 * Smoke test: load the deployed Maison v6 UI, walk the three sidebar
 * tabs + the three input modes, verify the strategy gate, and capture
 * screenshots.
 */
test('Maison v6 UI — boots, tabs switch, modes switch, strategy gate enforced', async ({ page }) => {
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

  // Source tab — Input dropdown defaults to file → drop zone shown
  await sourceTab.click()
  await expect(page.getByText(/Drop file/i)).toBeVisible()

  // Switch to URL mode via the Input dropdown
  await page.getByLabel('Input source').click()
  await page.getByRole('option', { name: 'URL' }).click()
  await expect(page.getByPlaceholder('https://…')).toBeVisible()

  // Switch to mic mode
  await page.getByLabel('Input source').click()
  await page.getByRole('option', { name: 'live microphone' }).click()
  await expect(page.locator('.mic__status')).toBeVisible()

  // Output tab — Format dropdown shows current value (verbose_json); the
  // listbox isn't open by default. Timestamps are now ma-check rows.
  await outputTab.click()
  await expect(page.getByLabel('Response format')).toBeVisible()
  await expect(page.getByLabel('Response format')).toHaveText(/verbose_json/i)
  await expect(page.getByRole('button', { name: /^Segment$/i })).toBeVisible()
  await expect(page.getByRole('button', { name: /^Word$/i })).toBeVisible()

  // Engine tab — Strategy dropdown shows current value
  await engineTab.click()
  await expect(page.getByLabel('Strategy')).toBeVisible()

  // Strategy gate: open the dropdown and confirm the progressive option
  // is enabled for file mode and disabled for URL mode.
  await sourceTab.click()
  await page.getByLabel('Input source').click()
  await page.getByRole('option', { name: 'file upload' }).click()
  await engineTab.click()
  await page.getByLabel('Strategy').click()
  await expect(page.getByRole('option', { name: 'progressive' })).toBeEnabled()
  // Close the menu.
  await page.keyboard.press('Escape')

  await sourceTab.click()
  await page.getByLabel('Input source').click()
  await page.getByRole('option', { name: 'URL' }).click()
  await engineTab.click()
  await page.getByLabel('Strategy').click()
  await expect(page.getByRole('option', { name: 'progressive' })).toBeDisabled()
  await page.keyboard.press('Escape')

  // Footer rail metric labels present.
  for (const label of ['STRATEGY', 'ASR', 'RTFx', 'SEGMENTS', 'WORDS']) {
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
