import { test, expect } from '@playwright/test'

/**
 * Smoke test: load the deployed Maison UI, walk through the three input
 * modes, verify the strategy gate, and capture screenshots.
 */
test('Maison UI — boots, modes switch, strategy gate enforced', async ({ page }) => {
  // Capture all console messages so we surface client errors in the report.
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

  // The retired "v5 maison" chip should NOT appear; nor the Pricing link.
  await expect(page.getByText(/v5\s*[·•]\s*maison/i)).toHaveCount(0)
  await expect(page.getByText(/^Pricing$/i)).toHaveCount(0)

  // Sidebar sections are visible.
  for (const label of ['Input', 'Format', 'Strategy', 'Timestamps', 'Advanced']) {
    await expect(page.getByText(label, { exact: true })).toBeVisible()
  }

  // Default mode = File. Switch through all three and assert the right
  // mode-specific UI shows.
  await page.getByRole('button', { name: /^File$/ }).click()
  await expect(page.getByText(/Drop file/i)).toBeVisible()

  await page.getByRole('button', { name: /^URL$/ }).click()
  await expect(page.getByPlaceholder('https://…')).toBeVisible()

  await page.getByRole('button', { name: /^Live mic$/ }).click()
  // mic status text appears
  await expect(page.locator('.mic__status')).toBeVisible()

  // Strategy gate: progressive should be enabled while in mic mode.
  await page.getByRole('button', { name: /^progressive$/ }).click()
  // (we just need to confirm it's clickable; settings persist)

  // Switch back to File — progressive should be disabled / fallback engaged.
  await page.getByRole('button', { name: /^File$/ }).click()
  const progressiveBtn = page.getByRole('button', { name: /^progressive$/ })
  await expect(progressiveBtn).toBeDisabled()

  // Footer rail metric labels are present.
  for (const label of ['STRATEGY', 'ASR', 'RTFx', 'SEGMENTS', 'WORDS']) {
    await expect(page.getByText(label, { exact: true })).toBeVisible()
  }

  // Snapshot the final state for the artefact folder.
  await page.screenshot({ path: 'tests/screenshots/maison-file.png', fullPage: true })

  // Visit mic mode one more time and snapshot — useful for diffing the
  // "secure context required" message when running over HTTP.
  await page.getByRole('button', { name: /^Live mic$/ }).click()
  await page.getByRole('button', { name: /Record|Stop/ }).click().catch(() => {})
  await page.waitForTimeout(400)
  await page.screenshot({ path: 'tests/screenshots/maison-mic.png', fullPage: true })

  expect(consoleErrors, `client console errors:\n${consoleErrors.join('\n')}`).toEqual([])
})
