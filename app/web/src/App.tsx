import { useEffect, useState } from 'react'
import './App.css'

type HealthResponse = {
  model_loaded: boolean
  model_name: string
  config: Record<string, unknown>
  decoding_strategy?: string
  device?: string
  dtype?: string
  status?: string
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/health')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setHealth)
      .catch((e) => setError(String(e)))
  }, [])

  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 720, margin: '4rem auto', padding: '0 1rem' }}>
      <h1 style={{ margin: 0 }}>Parakeet ASR</h1>
      <p style={{ color: '#666', marginTop: '0.25rem' }}>
        React + Vite scaffolding. Phases 2+ will replace this placeholder.
      </p>

      <section style={{ marginTop: '2rem', padding: '1rem 1.25rem', background: '#f6f7f9', borderRadius: 8 }}>
        <h2 style={{ margin: 0, fontSize: '1.1rem' }}>Backend health</h2>
        {error && <p style={{ color: 'crimson' }}>Failed to reach /health: {error}</p>}
        {!error && !health && <p>Loading…</p>}
        {health && (
          <pre style={{ overflow: 'auto', margin: '0.5rem 0 0', fontSize: 12 }}>
            {JSON.stringify(health, null, 2)}
          </pre>
        )}
      </section>
    </main>
  )
}

export default App
