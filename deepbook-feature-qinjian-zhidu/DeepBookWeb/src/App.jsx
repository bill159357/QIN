import { useEffect, useMemo, useRef, useState } from 'react'
import 'katex/dist/katex.min.css'
import './App.css'
import Modal from './components/Modal'
import AssistantPage from './pages/AssistantPage'
import GalleryPage from './pages/GalleryPage'
import HistoryPage from './pages/HistoryPage'
import HomePage from './pages/HomePage'
import OcrPage from './pages/OcrPage'
import SearchPage from './pages/SearchPage'

const API_BASE = import.meta.env.VITE_API_BASE || '/api'
const DEFAULT_PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
const PLATFORM_NAME = '秦简智读——基于大模型的秦简数字化与智能服务平台'
const HOME_SAMPLE_STORAGE_KEY = 'deepbook.home_samples'
const HOME_SAMPLE_NAME_STORAGE_KEY = 'deepbook.home_sample_names'
const MAX_HOME_SAMPLES = 6
const MAX_HOME_SAMPLE_NAME_LENGTH = 60

const NAV_ITEMS = [
  {
    id: 'home',
    label: '首页',
    desc: '样例演示与用户指引',
    hero: `从样例演示到识别入口，快速熟悉${PLATFORM_NAME}的完整使用流程。`,
    badge: '样例演示 · 使用指引',
  },
  {
    id: 'gallery',
    label: '简牍藏馆',
    desc: '瀑布流浏览秦简图像',
    hero: '浏览评测集秦简样例，点击详情查看检测框、参考释文与图像细节。',
    badge: '瀑布流浏览 · 详情复核',
  },
  {
    id: 'assistant',
    label: '秦简问答',
    desc: '秦简知识与平台问答',
    hero: '围绕秦简知识与平台能力发起问答，结合简牍藏馆、识别结果与全文检索给出专业回答。',
    badge: '秦简问答 · 平台联动',
  },
  {
    id: 'ocr',
    label: '秦简识别',
    desc: '上传文件与识别结果',
    hero: '支持 DeepSeek-OCR-2、PaddleOCR-VL-1.5 与简牍整页识别的结构化输出。',
    badge: '批量处理 · 进度可视化',
  },
  {
    id: 'search',
    label: '全文检索',
    desc: '跨文档搜索与查看',
    hero: '按关键词回查历史识别内容，定位结果并继续复核、导出。',
    badge: '关键词回查 · 结果定位',
  },
  {
    id: 'history',
    label: '历史任务',
    desc: '任务记录与复查',
    hero: '查看批量识别任务状态、结果样例与导出记录，支持快速复查。',
    badge: '任务记录 · 结果复查',
  },
]

const NAV_ITEMS_BY_ID = Object.fromEntries(NAV_ITEMS.map((item) => [item.id, item]))

const PRIMARY_NAV_IDS = ['home', 'gallery', 'assistant']

const GROUPED_NAV = {
  id: 'workspace',
  label: '数字整理',
  desc: '秦简识别 / 全文检索 / 历史任务',
  pageIds: ['ocr', 'search', 'history'],
}

const EXHIBIT_BACKDROP_IMAGES = [
  '/newback/back1.jpg',
  '/newback/back2.jpg',
  '/newback/back3.png',
]

const normalizeHomeSampleIds = (value) => {
  if (!Array.isArray(value)) return []
  const seen = new Set()
  const normalized = []
  value.forEach((item) => {
    const text = String(item || '').trim()
    if (!text || seen.has(text)) return
    seen.add(text)
    normalized.push(text)
  })
  return normalized.slice(0, MAX_HOME_SAMPLES)
}

const readHomeSampleIdsFromStorage = () => {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(HOME_SAMPLE_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return normalizeHomeSampleIds(parsed)
  } catch {
    return []
  }
}

const normalizeHomeSampleNames = (value) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const next = {}
  Object.entries(value).forEach(([rawJobId, rawName]) => {
    const jobId = String(rawJobId || '').trim()
    const sampleName = String(rawName || '')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, MAX_HOME_SAMPLE_NAME_LENGTH)
    if (!jobId || !sampleName) return
    next[jobId] = sampleName
  })
  return next
}

const readHomeSampleNamesFromStorage = () => {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(HOME_SAMPLE_NAME_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    return normalizeHomeSampleNames(parsed)
  } catch {
    return {}
  }
}

const filterHomeSampleNamesByIds = (value, jobIds) => {
  const validIds = new Set(normalizeHomeSampleIds(jobIds))
  const names = normalizeHomeSampleNames(value)
  const next = {}
  Object.entries(names).forEach(([jobId, sampleName]) => {
    if (validIds.has(jobId)) {
      next[jobId] = sampleName
    }
  })
  return next
}

const resolveInitialPage = () => {
  const openPage = new URL(window.location.href).searchParams.get('openPage')
  if (NAV_ITEMS.some((item) => item.id === openPage)) {
    return openPage
  }
  return 'home'
}

const resolveInitialFocusJobId = () =>
  new URL(window.location.href).searchParams.get('openJobId') || ''

function App() {
  const [activePage, setActivePage] = useState(resolveInitialPage)
  const [groupNavOpen, setGroupNavOpen] = useState(false)
  const [toasts, setToasts] = useState([])
  const [completionPromptJobs, setCompletionPromptJobs] = useState([])
  const [focusJobId, setFocusJobId] = useState(resolveInitialFocusJobId)
  const [homeSampleJobIds, setHomeSampleJobIds] = useState(() =>
    readHomeSampleIdsFromStorage(),
  )
  const [homeSampleNames, setHomeSampleNames] = useState(() =>
    filterHomeSampleNamesByIds(readHomeSampleNamesFromStorage(), homeSampleJobIds),
  )
  const notifiedRef = useRef(new Set())
  const initializedRef = useRef(false)
  const groupNavRef = useRef(null)

  const pageMeta = useMemo(
    () => NAV_ITEMS.find((item) => item.id === activePage) || NAV_ITEMS[0],
    [activePage],
  )
  const groupedPageMeta = useMemo(
    () =>
      GROUPED_NAV.pageIds
        .map((pageId) => NAV_ITEMS_BY_ID[pageId])
        .find((item) => item?.id === activePage) || null,
    [activePage],
  )
  const isGroupedPageActive = GROUPED_NAV.pageIds.includes(activePage)
  const pageTitle = pageMeta?.label || '秦简识别'
  const pageDescription =
    pageMeta?.hero || '支持 DeepSeek-OCR-2 与 PaddleOCR-VL-1.5 的扫描文档识别与结构化输出。'
  const pageBadge = pageMeta?.badge || '批量处理 · 进度可视化'
  const currentCompletionJobId = completionPromptJobs[0] || ''

  useEffect(() => {
    if (!groupNavOpen) return

    const handlePointerDown = (event) => {
      if (groupNavRef.current?.contains(event.target)) return
      setGroupNavOpen(false)
    }

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        setGroupNavOpen(false)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [groupNavOpen])

  useEffect(() => {
    const url = new URL(window.location.href)
    const hasLaunchParams =
      url.searchParams.has('openPage') || url.searchParams.has('openJobId')
    if (!hasLaunchParams) return
    url.searchParams.delete('openPage')
    url.searchParams.delete('openJobId')
    const nextUrl = `${url.pathname}${url.search}${url.hash}`
    window.history.replaceState(null, '', nextUrl)
  }, [])

  useEffect(() => {
    const pushToast = (message, type = 'info', duration = 4000) => {
      if (!message) return
      const id = `${Date.now()}-${Math.random()}`
      setToasts((prev) => [...prev, { id, message, type }])
      window.setTimeout(() => {
        setToasts((prev) => prev.filter((item) => item.id !== id))
      }, duration)
    }

    const handler = (event) => {
      const detail = event.detail || {}
      const message = detail.message || ''
      if (!message) return
      const type = detail.type || 'info'
      const duration = Number(detail.duration) || 4000
      pushToast(message, type, duration)
    }

    window.addEventListener('app:toast', handler)
    return () => {
      window.removeEventListener('app:toast', handler)
    }
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      window.localStorage.setItem(
        HOME_SAMPLE_STORAGE_KEY,
        JSON.stringify(normalizeHomeSampleIds(homeSampleJobIds)),
      )
    } catch {
      // ignore local storage errors
    }
  }, [homeSampleJobIds])

  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      window.localStorage.setItem(
        HOME_SAMPLE_NAME_STORAGE_KEY,
        JSON.stringify(filterHomeSampleNamesByIds(homeSampleNames, homeSampleJobIds)),
      )
    } catch {
      // ignore local storage errors
    }
  }, [homeSampleNames, homeSampleJobIds])

  useEffect(() => {
    let alive = true
    let timerId

    const pollJobs = async () => {
      try {
        if (!alive) return
        const response = await fetch(`${API_BASE}/ocr/jobs`)
        if (!alive || !response.ok) return
        const data = await response.json()
        if (!alive) return
        const jobs = data.jobs || []
        if (!initializedRef.current) {
          jobs.forEach((job) => {
            if (job?.id && job.status === 'succeeded') {
              notifiedRef.current.add(job.id)
            }
          })
          initializedRef.current = true
          return
        }
        jobs.forEach((job) => {
          if (!job?.id) return
          if (job.status === 'succeeded' && !notifiedRef.current.has(job.id)) {
            notifiedRef.current.add(job.id)
            setCompletionPromptJobs((prev) =>
              prev.includes(job.id) ? prev : [...prev, job.id],
            )
          }
        })
      } catch {
        // ignore polling errors
      }
    }

    pollJobs()
    timerId = window.setInterval(pollJobs, 4000)

    return () => {
      alive = false
      if (timerId) clearInterval(timerId)
    }
  }, [])

  const consumeCompletionPrompt = () => {
    setCompletionPromptJobs((prev) => prev.slice(1))
  }

  const handleCompletionLater = () => {
    if (!currentCompletionJobId) return
    consumeCompletionPrompt()
    window.dispatchEvent(
      new CustomEvent('app:toast', {
        detail: {
          message: '识别已完成，你可以稍后在“历史任务”页面查看结果。',
          type: 'info',
        },
      }),
    )
  }

  const handleCompletionView = () => {
    if (!currentCompletionJobId) return
    const nextUrl = new URL(window.location.href)
    nextUrl.searchParams.set('openPage', 'ocr')
    nextUrl.searchParams.set('openJobId', currentCompletionJobId)
    const openedWindow = window.open(nextUrl.toString(), '_blank')
    if (openedWindow) {
      try {
        openedWindow.opener = null
      } catch {
        // ignore cross-window access errors
      }
    } else {
      setActivePage('ocr')
      setFocusJobId(currentCompletionJobId)
    }
    consumeCompletionPrompt()
  }

  const toggleHomeSampleJob = (jobId) => {
    const normalizedJobId = String(jobId || '').trim()
    if (!normalizedJobId) return
    setHomeSampleJobIds((prev) => {
      const nextJobIds = prev.includes(normalizedJobId)
        ? prev.filter((id) => id !== normalizedJobId)
        : [normalizedJobId, ...prev].slice(0, MAX_HOME_SAMPLES)
      setHomeSampleNames((prevNames) =>
        filterHomeSampleNamesByIds(prevNames, nextJobIds),
      )
      return nextJobIds
    })
  }

  const renameHomeSampleJob = (jobId, sampleName) => {
    const normalizedJobId = String(jobId || '').trim()
    if (!normalizedJobId) return
    const normalizedName = String(sampleName || '')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, MAX_HOME_SAMPLE_NAME_LENGTH)
    setHomeSampleNames((prev) => {
      if (!normalizedName) {
        if (!prev[normalizedJobId]) return prev
        const next = { ...prev }
        delete next[normalizedJobId]
        return next
      }
      if (prev[normalizedJobId] === normalizedName) return prev
      return { ...prev, [normalizedJobId]: normalizedName }
    })
  }

  const handlePageChange = (pageId) => {
    setActivePage(pageId)
    setGroupNavOpen(false)
  }

  return (
    <div className="app-scene">
      <div className="exhibit-backdrop" aria-hidden="true">
        <div className="exhibit-backdrop-overlay" />
        {EXHIBIT_BACKDROP_IMAGES.map((src, index) => (
          <figure
            key={src}
            className={`exhibit-backdrop-frame exhibit-backdrop-frame-${index + 1}`}
          >
            <img src={src} alt="" loading={index === 0 ? 'eager' : 'lazy'} />
          </figure>
        ))}
        <div className="exhibit-backdrop-wash" />
      </div>
      <div className="app-shell">
        {toasts.length > 0 && (
          <div className="toast-container">
            {toasts.map((toast) => (
              <div key={toast.id} className={`toast toast-${toast.type}`}>
                {toast.message}
              </div>
            ))}
          </div>
        )}
        <Modal
          open={Boolean(currentCompletionJobId)}
          onClose={handleCompletionLater}
          title="识别完成"
          className="completion-prompt-modal"
          footer={
            <div className="modal-actions">
              <button className="button ghost" onClick={handleCompletionLater} type="button">
                暂不查看
              </button>
              <button className="button primary" onClick={handleCompletionView} type="button">
                查看详情
              </button>
            </div>
          }
        >
          <div className="completion-body">
            <div className="completion-message">
              识别任务 <span className="completion-job-id">{currentCompletionJobId.slice(0, 8)}</span>{' '}
              已完成
            </div>
            <div className="completion-sub">是否立即查看识别结果？</div>
          </div>
        </Modal>
        <header className="app-header">
          <div className="app-header-main">
            <div className="brand app-header-brand">
              <div className="brand-mark" aria-hidden="true">
                秦
              </div>
              <div className="app-header-brand-copy">
                <div className="brand-kicker">简牍文脉 · 数智重光</div>
                <div className="brand-title">
                  <span>秦简智读</span>
                  <span>基于大模型的秦简数字化与智能服务平台</span>
                </div>
              </div>
            </div>

            <div className="app-header-nav-bar">
              <nav className="header-nav" aria-label="主导航">
                {PRIMARY_NAV_IDS.map((pageId) => {
                  const item = NAV_ITEMS_BY_ID[pageId]
                  return (
                    <button
                      key={item.id}
                      className={`header-nav-item ${activePage === item.id ? 'active' : ''}`}
                      onClick={() => handlePageChange(item.id)}
                      type="button"
                    >
                      <span className="header-nav-item-label">{item.label}</span>
                      <span className="header-nav-item-desc">{item.desc}</span>
                    </button>
                  )
                })}

                <div
                  ref={groupNavRef}
                  className={`header-nav-group ${groupNavOpen ? 'open' : ''} ${isGroupedPageActive ? 'active' : ''}`}
                >
                  <button
                    className={`header-nav-item header-nav-group-trigger ${isGroupedPageActive ? 'active' : ''}`}
                    onClick={() => setGroupNavOpen((prev) => !prev)}
                    type="button"
                    aria-expanded={groupNavOpen}
                    aria-haspopup="menu"
                  >
                    <span className="header-nav-item-label">{GROUPED_NAV.label}</span>
                    <span className="header-nav-item-desc">
                      {isGroupedPageActive && groupedPageMeta
                        ? `${groupedPageMeta.label} · ${groupedPageMeta.desc}`
                        : GROUPED_NAV.desc}
                    </span>
                    <span className="header-nav-group-caret" aria-hidden="true" />
                  </button>

                  <div className="header-nav-group-menu" role="menu">
                    {GROUPED_NAV.pageIds.map((pageId) => {
                      const item = NAV_ITEMS_BY_ID[pageId]
                      return (
                        <button
                          key={item.id}
                          className={`header-subnav-item ${activePage === item.id ? 'active' : ''}`}
                          onClick={() => handlePageChange(item.id)}
                          type="button"
                          role="menuitem"
                        >
                          <span className="header-subnav-label">{item.label}</span>
                          <span className="header-subnav-desc">{item.desc}</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              </nav>
            </div>
          </div>
        </header>

        <main className="app-main">
          <header className="hero">
            <div className="hero-copy">
              <div className="hero-kicker">数字展陈式工作台</div>
              <h1>{pageTitle}</h1>
              <p>{pageDescription}</p>
            </div>
          </header>

          {activePage === 'home' && (
            <HomePage
              apiBase={API_BASE}
              sampleJobIds={homeSampleJobIds}
              sampleNames={homeSampleNames}
              onStartRecognition={() => handlePageChange('ocr')}
              onOpenHistory={() => handlePageChange('history')}
              onToggleSample={toggleHomeSampleJob}
              onRenameSample={renameHomeSampleJob}
            />
          )}
          {activePage === 'gallery' && <GalleryPage apiBase={API_BASE} />}
          {activePage === 'assistant' && <AssistantPage apiBase={API_BASE} />}
          {activePage === 'ocr' && (
            <OcrPage
              apiBase={API_BASE}
              defaultPrompt={DEFAULT_PROMPT}
              focusJobId={focusJobId}
              onFocusJobHandled={() => setFocusJobId('')}
            />
          )}
          {activePage === 'search' && <SearchPage apiBase={API_BASE} />}
          {activePage === 'history' && (
            <HistoryPage
              apiBase={API_BASE}
              featuredSampleJobIds={homeSampleJobIds}
              onToggleHomeSampleJob={toggleHomeSampleJob}
            />
          )}
        </main>
      </div>
    </div>
  )
}

export default App
