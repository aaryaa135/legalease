import React, { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from './utils/api.js'
import SourceCard from './components/SourceCard.jsx'
import StatsBar from './components/StatsBar.jsx'
import HydePanel from './components/HydePanel.jsx'
import UploadPanel from './components/UploadPanel.jsx'
import DocumentsPanel from './components/DocumentsPanel.jsx'

// ── Quick prompts ──────────────────────────────────────────────────────────────
const QUICK_PROMPTS = [
  'What are my rights if I am arrested without a warrant?',
  'How do I file an RTI application and what is the time limit?',
  'What constitutes a valid contract under Indian law?',
  'What are the penalties for cybercrime under the IT Act?',
  'How can I file a consumer complaint under the Consumer Protection Act?',
  'What is the difference between bailable and non-bailable offences?',
]

// ── Styles ─────────────────────────────────────────────────────────────────────
const s = {
  app: {
    display: 'flex',
    height: '100vh',
    overflow: 'hidden',
  },

  // Sidebar
  sidebar: {
    width: 300,
    minWidth: 300,
    background: 'var(--bg2)',
    borderRight: '1px solid var(--border)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  sidebarHeader: {
    padding: '22px 20px 14px',
    borderBottom: '1px solid var(--border)',
    flexShrink: 0,
  },
  logo: {
    fontFamily: 'var(--serif)',
    fontSize: 22,
    color: 'var(--gold2)',
    fontStyle: 'italic',
    letterSpacing: '-0.01em',
  },
  logoSub: {
    fontFamily: 'var(--mono)',
    fontSize: 10,
    color: 'var(--text3)',
    marginTop: 2,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
  },
  sidebarBody: {
    flex: 1,
    overflowY: 'auto',
    padding: '0 16px',
  },
  sidebarDivider: {
    height: 1,
    background: 'var(--border)',
    margin: '4px 0',
  },

  // Chat area
  chat: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    background: 'var(--bg)',
  },
  chatHeader: {
    padding: '16px 28px',
    borderBottom: '1px solid var(--border)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexShrink: 0,
  },
  chatTitle: {
    fontFamily: 'var(--serif)',
    fontSize: 16,
    color: 'var(--text2)',
    fontStyle: 'italic',
  },
  statusDot: (ok) => ({
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: ok ? 'var(--green)' : 'var(--text3)',
    display: 'inline-block',
    marginRight: 6,
  }),
  statusText: {
    fontFamily: 'var(--mono)',
    fontSize: 11,
    color: 'var(--text3)',
  },

  // Messages
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '28px 32px',
    display: 'flex',
    flexDirection: 'column',
    gap: 28,
  },
  empty: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 32,
    padding: '40px 20px',
  },
  emptyTitle: {
    fontFamily: 'var(--serif)',
    fontSize: 36,
    color: 'var(--gold2)',
    fontStyle: 'italic',
    textAlign: 'center',
  },
  emptySubtitle: {
    fontFamily: 'var(--mono)',
    fontSize: 11,
    color: 'var(--text3)',
    textTransform: 'uppercase',
    letterSpacing: '0.12em',
    textAlign: 'center',
  },
  quickGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
    gap: 10,
    width: '100%',
    maxWidth: 800,
  },
  quickBtn: {
    background: 'var(--bg2)',
    border: '1px solid var(--border)',
    borderRadius: 10,
    padding: '12px 15px',
    color: 'var(--text2)',
    fontSize: 13,
    fontFamily: 'var(--sans)',
    cursor: 'pointer',
    textAlign: 'left',
    lineHeight: 1.45,
    transition: 'border-color 0.15s, color 0.15s',
  },

  // Message bubbles
  msgWrap: (role) => ({
    display: 'flex',
    flexDirection: 'column',
    alignItems: role === 'user' ? 'flex-end' : 'flex-start',
    gap: 10,
    maxWidth: '100%',
  }),
  bubble: (role) => ({
    maxWidth: role === 'user' ? 540 : '100%',
    padding: role === 'user' ? '10px 15px' : '0',
    background: role === 'user' ? 'var(--bg3)' : 'transparent',
    border: role === 'user' ? '1px solid var(--border)' : 'none',
    borderRadius: role === 'user' ? 12 : 0,
    color: 'var(--text)',
    fontSize: 14,
    lineHeight: 1.65,
  }),
  sourcesLabel: {
    fontFamily: 'var(--mono)',
    fontSize: 10,
    color: 'var(--text3)',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    marginTop: 8,
    marginBottom: 6,
  },
  sourcesGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
    gap: 8,
  },
  thinking: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    fontFamily: 'var(--mono)',
    fontSize: 12,
    color: 'var(--text3)',
  },

  // Input
  inputRow: {
    padding: '16px 24px 20px',
    borderTop: '1px solid var(--border)',
    flexShrink: 0,
    background: 'var(--bg)',
  },
  inputWrap: {
    display: 'flex',
    gap: 10,
    background: 'var(--bg2)',
    border: '1px solid var(--border)',
    borderRadius: 12,
    padding: '4px 4px 4px 16px',
    transition: 'border-color 0.15s',
  },
  textarea: {
    flex: 1,
    background: 'transparent',
    border: 'none',
    outline: 'none',
    color: 'var(--text)',
    fontSize: 14,
    fontFamily: 'var(--sans)',
    lineHeight: 1.55,
    resize: 'none',
    padding: '8px 0',
    minHeight: 22,
    maxHeight: 120,
  },
  sendBtn: (active) => ({
    padding: '8px 18px',
    borderRadius: 9,
    border: 'none',
    background: active ? 'var(--gold)' : 'var(--bg3)',
    color: active ? '#0c0d0f' : 'var(--text3)',
    fontFamily: 'var(--mono)',
    fontSize: 12,
    fontWeight: 500,
    cursor: active ? 'pointer' : 'not-allowed',
    transition: 'background 0.15s, color 0.15s',
    flexShrink: 0,
    alignSelf: 'flex-end',
    marginBottom: 4,
  }),
  hydeToggle: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    padding: '0 8px',
    fontFamily: 'var(--mono)',
    fontSize: 11,
    color: 'var(--text3)',
    cursor: 'pointer',
    userSelect: 'none',
    flexShrink: 0,
    alignSelf: 'center',
  },
  toggleKnob: (on) => ({
    width: 28,
    height: 16,
    borderRadius: 8,
    background: on ? 'var(--gold)' : 'var(--bg3)',
    border: `1px solid ${on ? 'var(--gold)' : 'var(--border2)'}`,
    position: 'relative',
    transition: 'background 0.15s',
  }),
  knobDot: (on) => ({
    position: 'absolute',
    top: 2,
    left: on ? 13 : 2,
    width: 10,
    height: 10,
    borderRadius: '50%',
    background: on ? '#0c0d0f' : 'var(--text3)',
    transition: 'left 0.15s',
  }),
}

// ── Typing dots ────────────────────────────────────────────────────────────────
function ThinkingDots() {
  return (
    <div style={s.thinking}>
      <style>{`
        @keyframes blink { 0%,80%,100%{opacity:0.2} 40%{opacity:1} }
        .d { width:5px; height:5px; border-radius:50%; background:var(--gold); display:inline-block; animation: blink 1.2s infinite; }
        .d:nth-child(2){animation-delay:.2s}
        .d:nth-child(3){animation-delay:.4s}
      `}</style>
      <span className="d" /><span className="d" /><span className="d" />
      <span style={{ marginLeft: 4 }}>Searching legal documents…</span>
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [hydeOn, setHydeOn] = useState(true)
  const [serverOk, setServerOk] = useState(null)
  const [refreshDocs, setRefreshDocs] = useState(0)
  const bottomRef = useRef()
  const textareaRef = useRef()

  // Health check on mount
  useEffect(() => {
    api.health()
      .then(() => setServerOk(true))
      .catch(() => setServerOk(false))
  }, [])

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = useCallback(async (question) => {
    const q = (question || input).trim()
    if (!q || loading) return
    setInput('')
    setMessages((m) => [...m, { role: 'user', text: q }])
    setLoading(true)
    try {
      const res = await api.query(q, { hyde_enabled: hydeOn })
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          text: res.answer,
          sources: res.sources,
          stats: res.retrieval_stats,
          hydeDoc: res.hypothetical_doc,
        },
      ])
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: `**Error:** ${e.message}`, sources: [], stats: null },
      ])
    } finally {
      setLoading(false)
    }
  }, [input, loading, hydeOn])

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const autoResize = (e) => {
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
  }

  return (
    <div style={s.app}>
      {/* ── Sidebar ── */}
      <aside style={s.sidebar}>
        <div style={s.sidebarHeader}>
          <div style={s.logo}>LegalEase</div>
          <div style={s.logoSub}>Indian Law · RAG Assistant</div>
          <div style={{ marginTop: 10, display: 'flex', alignItems: 'center' }}>
            <span style={s.statusDot(serverOk)} />
            <span style={s.statusText}>
              {serverOk === null ? 'connecting…' : serverOk ? 'backend connected' : 'backend offline'}
            </span>
          </div>
        </div>

        <div style={s.sidebarBody}>
          <UploadPanel onUploaded={() => setRefreshDocs((n) => n + 1)} />
          <div style={s.sidebarDivider} />
          <DocumentsPanel refreshTrigger={refreshDocs} />
        </div>
      </aside>

      {/* ── Main chat ── */}
      <main style={s.chat}>
        <div style={s.chatHeader}>
          <span style={s.chatTitle}>Ask anything about Indian law</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Pipeline: HyDE → ChromaDB → Cohere Rerank → GPT-4o-mini
            </span>
          </div>
        </div>

        {/* Messages */}
        <div style={s.messages}>
          {messages.length === 0 ? (
            <div style={s.empty}>
              <div>
                <div style={s.emptyTitle}>⚖️ LegalEase</div>
                <div style={{ ...s.emptySubtitle, marginTop: 8 }}>Indian law · RAG · Section citations</div>
              </div>
              <div style={s.quickGrid}>
                {QUICK_PROMPTS.map((p) => (
                  <button
                    key={p}
                    style={s.quickBtn}
                    onClick={() => send(p)}
                    onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--border2)'; e.currentTarget.style.color = 'var(--text)' }}
                    onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text2)' }}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} style={s.msgWrap(msg.role)}>
                <div style={s.bubble(msg.role)}>
                  {msg.role === 'user' ? (
                    msg.text
                  ) : (
                    <div className="answer-md">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                    </div>
                  )}
                </div>

                {msg.role === 'assistant' && (
                  <>
                    {msg.stats && <StatsBar stats={msg.stats} />}
                    {msg.hydeDoc && <HydePanel hydeDoc={msg.hydeDoc} />}
                    {msg.sources?.length > 0 && (
                      <>
                        <div style={s.sourcesLabel}>
                          {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''} · reranked by Cohere
                        </div>
                        <div style={s.sourcesGrid}>
                          {msg.sources.map((src, j) => (
                            <SourceCard key={j} source={src} index={j} />
                          ))}
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            ))
          )}

          {loading && (
            <div style={s.msgWrap('assistant')}>
              <ThinkingDots />
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input bar */}
        <div style={s.inputRow}>
          <div
            style={s.inputWrap}
            onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--border2)')}
            onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--border)')}
          >
            <textarea
              ref={textareaRef}
              style={s.textarea}
              placeholder="Ask a legal question… (Enter to send, Shift+Enter for newline)"
              value={input}
              onChange={(e) => { setInput(e.target.value); autoResize(e) }}
              onKeyDown={handleKey}
              rows={1}
            />
            <label
              style={s.hydeToggle}
              title="HyDE: generates a hypothetical answer first for better retrieval"
            >
              <div style={s.toggleKnob(hydeOn)} onClick={() => setHydeOn(!hydeOn)}>
                <div style={s.knobDot(hydeOn)} />
              </div>
              HyDE
            </label>
            <button
              style={s.sendBtn(!!input.trim() && !loading)}
              onClick={() => send()}
              disabled={!input.trim() || loading}
            >
              {loading ? '…' : 'Ask →'}
            </button>
          </div>
          <div style={{ marginTop: 7, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', textAlign: 'center' }}>
            LegalEase answers from indexed documents only · Always consult a qualified lawyer for legal advice
          </div>
        </div>
      </main>
    </div>
  )
}
