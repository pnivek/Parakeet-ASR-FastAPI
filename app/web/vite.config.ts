import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Build the SPA into ../static so FastAPI's existing StaticFiles mount
// (app/main.py serves /static and /) picks it up unchanged.
//
// In dev, proxy API + WS to the backend. Default target is localhost:8777
// (matches PORT in app/.env); override with PARAKEET_URL=http://host:port.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.PARAKEET_URL || 'http://localhost:8777'

  return {
    plugins: [react()],
    build: {
      outDir: '../static',
      emptyOutDir: true,
    },
    server: {
      port: 5173,
      proxy: {
        '/v1/audio/transcriptions': {
          target: apiTarget,
          changeOrigin: true,
          ws: true,
        },
        '/v1': { target: apiTarget, changeOrigin: true },
        '/health': { target: apiTarget, changeOrigin: true },
        '/readyz': { target: apiTarget, changeOrigin: true },
      },
    },
  }
})
