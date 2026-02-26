import React, { useEffect, useMemo, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
const GITHUB = 'https://github.com/neeraj214/Plant-AI-'

function useHover() {
  const [hover, setHover] = useState(false)
  const bind = {
    onDragEnter: (e) => { e.preventDefault(); setHover(true) },
    onDragOver: (e) => { e.preventDefault(); setHover(true) },
    onDragLeave: () => setHover(false),
    onDrop: (e) => { e.preventDefault(); setHover(false) }
  }
  return [hover, bind]
}

function RiskBadge({ confidence }) {
  const level = confidence >= 0.75 ? 'high' : confidence >= 0.4 ? 'mid' : 'low'
  const text = level === 'high' ? 'High Risk' : level === 'mid' ? 'Medium Risk' : 'Low Risk'
  return <span className={`badge ${level}`}>{text}</span>
}

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [overlayUrl, setOverlayUrl] = useState(null)
  const [alpha, setAlpha] = useState(0.5)
  const [backend, setBackend] = useState(null)
  const [inferMs, setInferMs] = useState(null)
  const [hover, bindHover] = useHover()
  const inputRef = useRef(null)

  useEffect(() => {
    fetch(`${API_BASE}/health`).then(r => r.json()).then(d => setBackend(d.backend)).catch(() => {})
  }, [])

  const onPick = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setResult(null)
    setOverlayUrl(null)
    setInferMs(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  const onDrop = (e) => {
    if (!e.dataTransfer.files?.length) return
    const f = e.dataTransfer.files[0]
    setFile(f)
    setResult(null)
    setOverlayUrl(null)
    setInferMs(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  const onSubmit = async () => {
    if (!file) return
    setLoading(true)
    setResult(null)
    setOverlayUrl(null)
    setInferMs(null)
    const fd = new FormData()
    fd.append('file', file)
    const t0 = performance.now()
    try {
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd })
      const data = await res.json()
      setResult(data)
      const t1 = performance.now()
      setInferMs(Math.round(t1 - t0))
      if (data.gradcam_overlay_path) setOverlayUrl(`${API_BASE}${data.gradcam_overlay_path}`)
    } catch {
      setResult({ error: 'Request failed' })
    } finally {
      setLoading(false)
    }
  }

  const conf = useMemo(() => {
    if (!result || typeof result.confidence !== 'number') return 0
    return Math.max(0, Math.min(1, result.confidence))
  }, [result])

  return (
    <div className="page">
      <header className="nav">
        <div className="brand">
          <div>🌱 Plant AI</div>
          <div className="tag">AI‑Powered Disease Detection with Explainability</div>
        </div>
        <div className="links">
          <a href={GITHUB} target="_blank" rel="noreferrer">GitHub</a>
          <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">API Docs</a>
          <button className="toggle" onClick={() => document.body.classList.toggle('light')}>Theme</button>
        </div>
      </header>

      <main className="content">
        <section className="hero">
          <div className="card">
            <div
              className={`upload ${hover ? 'hover' : ''}`}
              {...bindHover}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
            >
              <h2>Upload Leaf Image</h2>
              <p>JPG, JPEG, PNG</p>
              <input ref={inputRef} className="file" type="file" accept="image/*" onChange={onPick} hidden />
            </div>
            <div style={{ display: 'flex', gap: 12, marginTop: 12 }}>
              <button className="primary" disabled={!file || loading} onClick={onSubmit}>
                {loading ? 'Analyzing…' : 'Analyze'}
              </button>
              <div className="row">Backend: {backend || '…'} | {API_BASE}</div>
            </div>
          </div>

          {preview && (
            <div className="card">
              <div className="grid2">
                <div>
                  <div className="label">Input</div>
                  <div className="frame">
                    <img alt="input" className="img" src={preview} />
                  </div>
                </div>
                <div>
                  <div className="label">AI Attention Map</div>
                  <div className="frame">
                    <img alt="input-b" className="img" src={preview} />
                    {overlayUrl && <img alt="gradcam" className="overlay" src={overlayUrl} style={{ opacity: alpha }} />}
                  </div>
                  <input className="slider" type="range" min={0} max={1} step={0.01} value={alpha} onChange={e => setAlpha(parseFloat(e.target.value))} />
                </div>
              </div>
            </div>
          )}

          {result && (
            <>
              <div className="summary">
                <div className="disease">{result.class_name || result.class_index || 'Unknown'}</div>
                <div className="row">
                  <div>Confidence: {(conf * 100).toFixed(1)}%</div>
                  <RiskBadge confidence={conf} />
                </div>
                <div className="bar">
                  <div className="fill" style={{ width: `${conf * 100}%` }} />
                </div>
              </div>

              <details className="panel">
                <summary>Model & Explainability Details</summary>
                <div className="row">Backend: {backend || '…'}</div>
                <div className="row">Model: EfficientNetV2</div>
                <div className="row">Inference time: {inferMs != null ? `${inferMs} ms` : '…'}</div>
                {Array.isArray(result.top3) && (
                  <div className="row">Top‑3: {result.top3.join(', ')}</div>
                )}
                <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
                  <button className="primary" onClick={() => window.print()}>Download AI Report</button>
                </div>
              </details>
            </>
          )}
        </section>
      </main>

      <footer className="foot">
        Powered by FastAPI + PyTorch • Model: EfficientNetV2 • © {new Date().getFullYear()} Neeraj Negi
      </footer>
    </div>
  )
}
