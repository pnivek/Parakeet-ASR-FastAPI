import { defineConfig, devices } from '@playwright/test'

/**
 * Playwright config for smoke-testing the deployed Maison UI.
 *
 * BASE_URL defaults to the staging deployment on Sparkly. Override with
 * `BASE_URL=http://localhost:5173` to point at a local dev server.
 */
export default defineConfig({
  testDir: './tests',
  reporter: 'list',
  use: {
    baseURL: process.env.BASE_URL || 'http://192.168.0.172:8777',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    viewport: { width: 1440, height: 900 },
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
})
