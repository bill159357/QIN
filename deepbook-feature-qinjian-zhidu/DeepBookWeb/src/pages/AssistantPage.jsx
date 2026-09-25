import { useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const ASSISTANT_NAME = '简衡'
const INITIAL_MESSAGE = {
  id: 'welcome',
  role: 'assistant',
  content:
    '我是“简衡”，负责“秦简智读——基于大模型的秦简数字化与智能服务平台”内的秦简知识问答、释读思路梳理与平台使用指导。你可以直接问我秦简内容、OCR 复核、疑难字判断，或让我们一起结合平台资料做分析。',
}

const STARTER_PROMPTS = [
  '秦简里常见的纪年表达有什么特点？',
  '结合平台能力，给我一个秦简 OCR 复核流程。',
  '“甲渠”在简牍材料里通常对应什么历史语境？',
  '如果 OCR 结果里有残缺字和异体字，应该怎么判断？',
]

const normalizeContextArray = (value) => (Array.isArray(value) ? value : [])

function AssistantPage({ apiBase }) {
  const [messages, setMessages] = useState([INITIAL_MESSAGE])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [includePlatformContext, setIncludePlatformContext] = useState(true)
  const [lastContext, setLastContext] = useState({ documents: [], gallery_items: [] })
  const [lastModel, setLastModel] = useState('')

  const conversationMessages = useMemo(
    () =>
      messages
        .filter((item) => item.role === 'user' || item.role === 'assistant')
        .map(({ role, content }) => ({ role, content })),
    [messages],
  )

  const sendMessage = async (contentOverride = '') => {
    const nextContent = String(contentOverride || input).trim()
    if (!nextContent || loading) return

    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: nextContent,
    }
    const nextConversation = [...conversationMessages, { role: 'user', content: nextContent }]

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setError('')
    setLoading(true)

    try {
      const response = await fetch(`${apiBase}/assistant/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: nextConversation,
          include_platform_context: includePlatformContext,
        }),
      })

      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || '秦简问答暂时不可用，请稍后重试。')
      }

      const assistantMessage = data.assistant || {}
      const assistantContent = String(assistantMessage.content || '').trim()
      if (!assistantContent) {
        throw new Error('秦简问答返回了空内容。')
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: assistantContent,
        },
      ])
      setLastContext(data.context || { documents: [], gallery_items: [] })
      setLastModel(String(data.model || ''))
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '秦简问答请求失败。')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    await sendMessage()
  }

  const handleComposerKeyDown = async (event) => {
    if (event.key !== 'Enter' || event.shiftKey) return
    event.preventDefault()
    await sendMessage()
  }

  const resetConversation = () => {
    setMessages([INITIAL_MESSAGE])
    setInput('')
    setError('')
    setLastContext({ documents: [], gallery_items: [] })
    setLastModel('')
  }

  const documentContext = normalizeContextArray(lastContext?.documents)
  const galleryContext = normalizeContextArray(lastContext?.gallery_items)

  return (
    <section className="assistant-page-grid">
      <div className="panel assistant-chat-panel">
        <div className="assistant-panel-header">
          <div>
            <h2>{ASSISTANT_NAME} · 秦简问答</h2>
            <p>以秦简研究与平台释读场景为核心，可结合平台资料回答知识问题与使用问题。</p>
          </div>
          <div className="assistant-panel-actions">
            <label className="assistant-context-toggle">
              <input
                type="checkbox"
                checked={includePlatformContext}
                onChange={(event) => setIncludePlatformContext(event.target.checked)}
              />
              <span>结合平台资料</span>
            </label>
            <button className="button ghost small" type="button" onClick={resetConversation}>
              清空会话
            </button>
          </div>
        </div>

        <div className="assistant-thread">
          {messages.map((message) => (
            <article
              key={message.id}
              className={`assistant-bubble assistant-bubble-${message.role}`}
            >
              <div className="assistant-bubble-meta">
                {message.role === 'assistant' ? ASSISTANT_NAME : '你'}
              </div>
              <div className="assistant-bubble-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
              </div>
            </article>
          ))}

          {loading && (
            <article className="assistant-bubble assistant-bubble-assistant assistant-bubble-loading">
              <div className="assistant-bubble-meta">{ASSISTANT_NAME}</div>
              <div className="assistant-bubble-content">正在整理平台资料并生成回答...</div>
            </article>
          )}
        </div>

        {error && <div className="error assistant-error">{error}</div>}

        <form className="assistant-composer" onSubmit={handleSubmit}>
          <textarea
            className="assistant-composer-input"
            rows={4}
            placeholder="向简衡提问：例如请解释一段秦简释文、帮我分析疑难字，或询问平台如何复核 OCR 结果。"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleComposerKeyDown}
          />
          <div className="assistant-composer-footer">
            <div className="assistant-composer-tip">`Enter` 发送，`Shift + Enter` 换行</div>
            <button className="button primary" type="submit" disabled={loading || !input.trim()}>
              {loading ? '发送中...' : '发送问题'}
            </button>
          </div>
        </form>
      </div>

      <aside className="assistant-sidebar">
        <section className="panel assistant-side-panel">
          <div className="assistant-side-title">人格设定</div>
          <div className="assistant-persona-card">
            <div className="assistant-persona-name">{ASSISTANT_NAME}</div>
            <p>
              偏学术型、重证据、重释读边界。优先回答秦简、简牍数字化、OCR
              复核和平台使用问题；遇到残缺、异文、分歧结论时，会明确说明不确定性。
            </p>
            {lastModel && <div className="assistant-model-tag">当前模型：{lastModel}</div>}
          </div>
        </section>

        <section className="panel assistant-side-panel">
          <div className="assistant-side-title">快捷提问</div>
          <div className="assistant-starter-list">
            {STARTER_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                className="assistant-starter"
                type="button"
                onClick={() => sendMessage(prompt)}
                disabled={loading}
              >
                {prompt}
              </button>
            ))}
          </div>
        </section>

        <section className="panel assistant-side-panel">
          <div className="assistant-side-title">本轮参考平台资料</div>
          {documentContext.length === 0 && galleryContext.length === 0 ? (
            <div className="assistant-context-empty">
              发送问题后，这里会展示本轮回答引用到的平台文档片段和简牍样例。
            </div>
          ) : (
            <div className="assistant-context-groups">
              {documentContext.length > 0 && (
                <div className="assistant-context-block">
                  <div className="assistant-context-label">全文检索文档</div>
                  {documentContext.map((item) => (
                    <article key={`doc-${item.doc_id}`} className="assistant-context-card">
                      <div className="assistant-context-title">{item.title || '未命名文档'}</div>
                      <div className="assistant-context-snippet">{item.snippet || '无摘要'}</div>
                    </article>
                  ))}
                </div>
              )}

              {galleryContext.length > 0 && (
                <div className="assistant-context-block">
                  <div className="assistant-context-label">简牍藏馆样例</div>
                  {galleryContext.map((item) => (
                    <article key={`gallery-${item.id}`} className="assistant-context-card">
                      <div className="assistant-context-title">
                        {item.title || `简牍 ${item.id}`}
                      </div>
                      <div className="assistant-context-snippet">
                        释文：{item.reference_text || '暂无释文'}
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
      </aside>
    </section>
  )
}

export default AssistantPage
