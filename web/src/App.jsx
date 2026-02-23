import React, { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [overlayUrl, setOverlayUrl] = useState(null)
  const onPick = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setResult(null)
    setOverlayUrl(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }
  const onSubmit = async () => {
    if (!file) return
    setLoading(true)
    setResult(null)
    setOverlayUrl(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd })
      const data = await res.json()
      setResult(data)
      if (data.gradcam_overlay_path) setOverlayUrl(`${API_BASE}${data.gradcam_overlay_path}`)
    } catch (e) {
      setResult({ error: 'Request failed' })
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="page">
      <header className="nav">
        <div className="brand">🌿 Plant AI</div>
        <div className="meta">Backend: {API_BASE}</div>
      </header>
      <main className="content">
        <div className="card">
          <h2>Upload a leaf image</h2>
          <input className="file" type="file" accept="image/*" onChange={onPick} />
          {preview && <img alt="preview" className="preview" src={preview} />}
          <button className="primary" disabled={!file || loading} onClick={onSubmit}>
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
        {result && (
          <div className="card">
            <h2>Prediction</h2>
            <pre className="code">{JSON.stringify(result, null, 2)}</pre>
            {overlayUrl && <img alt="gradcam" className="overlay" src={overlayUrl} />}
          </div>
        )}
      </main>
      <footer className="foot">© {new Date().getFullYear()} Plant AI</footer>
    </div>
  )
}
