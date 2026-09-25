import { useEffect, useMemo, useState } from 'react'
import Modal from '../components/Modal'
import ResultsPanel from '../components/ResultsPanel'

const STATUS_TEXT = {
  pending: '排队中',
  queued: '队列中',
  running: '处理中',
  pausing: '暂停中',
  succeeded: '已完成',
  failed: '失败',
  paused: '已暂停',
  canceled: '已取消',
}

const GUIDE_STEPS = [
  {
    title: '第 1 步：准备文件',
    detail: '支持图片（PNG/JPG/JPEG/WebP/BMP/GIF）和 PDF，扫描件请尽量保证清晰、方向正确。',
  },
  {
    title: '第 2 步：选择模型',
    detail:
      '通用文档优先使用 DeepSeek-OCR-2；复杂版面、图表与细粒度解析建议使用 PaddleOCR-VL-1.5。',
  },
  {
    title: '第 3 步：开始识别',
    detail: '在秦简识别页上传文件后点击“开始识别”，任务会进入队列并实时反馈状态与进度。',
  },
  {
    title: '第 4 步：复查与导出',
    detail: '识别完成后可查看 Markdown/纯文本/检测框，支持按页或整体导出 MD、DOCX、PDF。',
  },
]

const PLATFORM_PILLARS = [
  {
    title: '智能识读',
    detail: '支持图片与 PDF 入卷，自动进入 OCR 任务流并持续反馈进度。',
  },
  {
    title: '结果复核',
    detail: '可查看 Markdown、纯文本、检测框与页级结果，便于快速校核。',
  },
  {
    title: '知识联动',
    detail: '识别结果可继续进入全文检索与秦简问答，形成研究闭环。',
  },
]

const UPLOAD_TIPS = [
  {
    title: '支持格式',
    value: 'PDF、PNG、JPG、JPEG、WEBP、BMP、GIF',
  },
  {
    title: '推荐做法',
    value: '多页文档优先 PDF；截图类场景可直接粘贴上传（Ctrl+V）。',
  },
  {
    title: '质量建议',
    value: '避免模糊、强反光、严重倾斜，能明显提升识别准确率。',
  },
]

const HERO_SCROLL_IMAGES = [
  '/backgroud/ScreenShot_2026-03-11_161206_574.png',
  '/backgroud/ScreenShot_2026-03-11_161332_948.png',
]

const HERO_GUIDE_TAGS = ['识别复核', '全文检索', '秦简问答']

const resolveModelLabel = (value) => {
  const text = String(value || '').trim()
  if (!text) return 'DeepSeek-OCR-2'
  const normalized = text.toLowerCase().replace(/_/g, '-')
  if (
    normalized === 'paddleocr-vl-1.5' ||
    normalized === 'paddleocr-vl' ||
    normalized === 'paddleocr' ||
    normalized === 'paddle'
  ) {
    return 'PaddleOCR-VL-1.5'
  }
  if (
    normalized === 'deepseek-ocr-2' ||
    normalized === 'deepseek ocr 2' ||
    normalized === 'deepseek'
  ) {
    return 'DeepSeek-OCR-2'
  }
  if (
    normalized === 'deepjiandu-ocr-v1' ||
    normalized === 'deepjiandu-ocr' ||
    normalized === 'deepjiandu'
  ) {
    return 'DeepJiandu-OCR-v1'
  }
  return text
}

const formatUploadedFiles = (files) => {
  if (!Array.isArray(files) || files.length === 0) return '-'
  if (files.length <= 2) return files.join('、')
  return `${files.slice(0, 2).join('、')} 等 ${files.length} 个文件`
}

const formatDate = (value) => {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

const toPreviewText = (value, maxChars = 160) => {
  const plain = String(value || '')
    .replace(/<[^>]*>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  if (!plain) return ''
  if (plain.length <= maxChars) return plain
  return `${plain.slice(0, maxChars - 1)}…`
}

function HomePage({
  apiBase,
  sampleJobIds,
  sampleNames = {},
  onStartRecognition,
  onOpenHistory,
  onToggleSample,
  onRenameSample = () => {},
}) {
  const [sampleItems, setSampleItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [detailOpen, setDetailOpen] = useState(false)
  const [detailJob, setDetailJob] = useState(null)
  const [detailResults, setDetailResults] = useState([])
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState('')

  useEffect(() => {
    let alive = true
    const controller = new AbortController()

    const loadSamples = async () => {
      const ids = Array.isArray(sampleJobIds) ? sampleJobIds.filter(Boolean) : []
      if (!ids.length) {
        setSampleItems([])
        setError('')
        setLoading(false)
        return
      }
      setLoading(true)
      setError('')
      try {
        const tasks = ids.map(async (jobId) => {
          const normalizedJobId = String(jobId || '').trim()
          if (!normalizedJobId) return null
          try {
            const jobResp = await fetch(`${apiBase}/ocr/jobs/${normalizedJobId}`, {
              signal: controller.signal,
            })
            if (jobResp.status === 404) {
              return {
                id: normalizedJobId,
                missing: true,
              }
            }
            if (!jobResp.ok) {
              throw new Error('加载任务失败')
            }
            const job = await jobResp.json()
            let previewText = ''
            try {
              const resultResp = await fetch(`${apiBase}/ocr/jobs/${normalizedJobId}/results`, {
                signal: controller.signal,
              })
              if (resultResp.ok) {
                const resultData = await resultResp.json()
                const first = (resultData.results || [])[0] || {}
                previewText = toPreviewText(
                  first.edited_text || first.text || first.raw_text || first.edited_markdown || first.markdown,
                )
              }
            } catch {
              // ignore result loading errors and fallback to job-level text
            }
            if (!previewText) {
              previewText = toPreviewText(job.text_excerpt || '')
            }
            return {
              ...job,
              id: normalizedJobId,
              sample_preview: previewText,
              ocr_model: resolveModelLabel(job.ocr_model),
            }
          } catch (err) {
            if (err?.name === 'AbortError') return null
            return {
              id: normalizedJobId,
              failed: true,
            }
          }
        })
        const results = await Promise.all(tasks)
        if (!alive) return
        setSampleItems(results.filter(Boolean))
      } catch (err) {
        if (!alive || err?.name === 'AbortError') return
        setError(err instanceof Error ? err.message : '样例加载失败')
      } finally {
        if (alive) setLoading(false)
      }
    }

    loadSamples()
    return () => {
      alive = false
      controller.abort()
    }
  }, [apiBase, sampleJobIds])

  const sampleCount = useMemo(() => sampleItems.filter((item) => !item.missing).length, [sampleItems])
  const heroMetrics = useMemo(
    () => [
      {
        value: String(sampleCount).padStart(2, '0'),
        label: '首页样例',
      },
      {
        value: '识别 / 检索 / 问答',
        label: '核心链路',
      },
      {
        value: 'MD · DOCX · PDF',
        label: '结果导出',
      },
    ],
    [sampleCount],
  )

  const resolveSampleName = (item) => {
    const jobId = String(item?.id || '').trim()
    if (!jobId) return '未命名样例'
    const customName = String(sampleNames?.[jobId] || '').trim()
    if (customName) return customName
    const firstFile = Array.isArray(item?.files) && item.files.length > 0 ? String(item.files[0] || '').trim() : ''
    if (firstFile) return firstFile
    return `任务 ${jobId.slice(0, 8)}`
  }

  const renameSample = (item) => {
    const jobId = String(item?.id || '').trim()
    if (!jobId) return
    const currentName = String(sampleNames?.[jobId] || '').trim()
    const initialName = currentName || resolveSampleName(item)
    const nextName = window.prompt('设置样例名称（留空可恢复默认）', initialName)
    if (nextName === null) return
    onRenameSample(jobId, nextName)
  }

  const handleItemUpdated = (prevItem, updated) => {
    if (!updated) return
    setDetailResults((prev) =>
      prev.map((item) => {
        if (item.doc_id && updated.doc_id && item.doc_id === updated.doc_id) {
          return { ...item, ...updated }
        }
        if (
          !item.doc_id &&
          prevItem?.output_dir &&
          item.output_dir === prevItem.output_dir
        ) {
          return { ...item, ...updated }
        }
        return item
      }),
    )
  }

  const buildInputUrl = (fileName, id = detailJob?.id) => {
    if (!id || !fileName) return ''
    return `${apiBase}/ocr/jobs/${id}/inputs/${encodeURI(fileName)}`
  }

  const renderInputPreview = (fileName, id) => {
    const url = buildInputUrl(fileName, id)
    if (!url) return null
    const lower = fileName.toLowerCase()
    if (lower.endsWith('.pdf')) {
      return (
        <div className="input-card" key={fileName}>
          <div className="input-name">{fileName}</div>
          <iframe className="input-frame" src={url} title={fileName} />
          <a className="input-link" href={url} target="_blank" rel="noreferrer">
            新窗口打开
          </a>
        </div>
      )
    }
    if (/\.(png|jpg|jpeg|gif|bmp|webp)$/i.test(lower)) {
      return (
        <div className="input-card" key={fileName}>
          <div className="input-name">{fileName}</div>
          <img className="input-image" src={url} alt={fileName} />
        </div>
      )
    }
    return (
      <div className="input-card" key={fileName}>
        <div className="input-name">{fileName}</div>
        <a className="input-link" href={url} target="_blank" rel="noreferrer">
          下载文件
        </a>
      </div>
    )
  }

  const openDetail = async (jobId) => {
    const normalizedJobId = String(jobId || '').trim()
    if (!normalizedJobId) return
    setDetailOpen(true)
    setDetailLoading(true)
    setDetailError('')
    setDetailResults([])
    try {
      const jobResp = await fetch(`${apiBase}/ocr/jobs/${normalizedJobId}`)
      if (!jobResp.ok) {
        throw new Error('加载任务失败')
      }
      const jobData = await jobResp.json()
      setDetailJob(jobData)

      const resultResp = await fetch(`${apiBase}/ocr/jobs/${normalizedJobId}/results`)
      if (!resultResp.ok) {
        throw new Error('加载结果失败')
      }
      const resultData = await resultResp.json()
      setDetailResults(resultData.results || [])
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : '加载结果失败')
    } finally {
      setDetailLoading(false)
    }
  }

  const closeDetail = () => {
    setDetailOpen(false)
    setDetailJob(null)
    setDetailResults([])
    setDetailError('')
  }

  return (
    <>
      <section className="panel home-hero-panel">
        <div className="home-hero-stage">
          <div className="home-hero-copy">
            <div className="home-hero-prelude">
              <span className="home-hero-prelude-label">平台导览</span>
              {HERO_GUIDE_TAGS.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
            <div className="home-hero-kicker">数智护简 · 文脉重光</div>
            <h2>秦简不再遥远</h2>
            <div className="home-hero-quote">让图像入卷，让释文可检，让知识可问。</div>
            <p>
              以更具展陈感的首屏串联上传识别、结果复核、全文检索与知识问答，让平台首页从工具入口升级为数字简牍展台。
            </p>
            <div className="home-hero-actions">
              <button className="button primary" type="button" onClick={onStartRecognition}>
                去秦简识别
              </button>
              <button className="button ghost" type="button" onClick={onOpenHistory}>
                去历史任务选样例
              </button>
            </div>
          </div>
          <div className="home-hero-aside">
            <div className="home-hero-ribbon">
              <span>简牍藏馆</span>
              <span>智能释读</span>
              <span>学术联动</span>
            </div>
            <div className="home-hero-scrolls">
              {HERO_SCROLL_IMAGES.map((src, index) => (
                <figure
                  key={src}
                  className={`home-hero-scroll-card home-hero-scroll-card-${index + 1}`}
                >
                  <img src={src} alt="" loading="lazy" />
                </figure>
              ))}
              <div className="home-hero-seal">简</div>
            </div>
            <div className="home-hero-metric-grid">
              {heroMetrics.map((item) => (
                <div key={item.label} className="home-hero-metric">
                  <strong>{item.value}</strong>
                  <span>{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="home-capability-grid">
          {PLATFORM_PILLARS.map((item) => (
            <article key={item.title} className="home-capability-card">
              <div className="home-capability-title">{item.title}</div>
              <p>{item.detail}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="section-heading">
          <span className="section-kicker">四步入卷</span>
          <div>
            <h2>用户指引</h2>
            <p>从准备文件到复核导出，首页直接给出完整使用路径。</p>
          </div>
        </div>
        <div className="home-guide-grid">
          {GUIDE_STEPS.map((step) => (
            <article key={step.title} className="home-guide-card">
              <h3>{step.title}</h3>
              <p>{step.detail}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="section-heading">
          <span className="section-kicker">上传前检查</span>
          <div>
            <h2>上传建议</h2>
            <p>保证图像质量和文档形态，能显著提升整页识别与文本结构化效果。</p>
          </div>
        </div>
        <div className="home-upload-tip-grid">
          {UPLOAD_TIPS.map((item) => (
            <div key={item.title} className="home-upload-tip">
              <div className="home-upload-tip-title">{item.title}</div>
              <div className="home-upload-tip-value">{item.value}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="home-sample-header section-heading">
          <span className="section-kicker">样例展陈</span>
          <div>
            <h2>样例演示</h2>
            <p>当前已配置 {sampleCount} 个样例。可在“历史任务”中把任务设为首页样例。</p>
          </div>
        </div>
        {error && <div className="error">{error}</div>}
        {loading && <div className="search-loading">样例加载中...</div>}
        {!loading && sampleItems.length === 0 && (
          <div className="search-empty">
            还没有首页样例。请到“历史任务”点击“设为首页样例”后返回查看。
          </div>
        )}
        {!loading && sampleItems.length > 0 && (
          <div className="home-sample-grid">
            {sampleItems.map((item) => (
              <article key={item.id} className="home-sample-card">
                <div className="home-sample-meta">
                  <div>
                    <div className="home-sample-title">{resolveSampleName(item)}</div>
                    <div className="home-sample-sub">
                      任务 {String(item.id || '').slice(0, 8)} · 创建于 {formatDate(item.created_at)}
                    </div>
                  </div>
                  {!item.missing && (
                    <span className={`status status-${item.status || 'queued'}`}>
                      {STATUS_TEXT[item.status] || item.status || '未知'}
                    </span>
                  )}
                </div>
                {item.missing ? (
                  <div className="home-sample-missing">该任务已不存在，请移除该样例。</div>
                ) : (
                  <>
                    <div className="home-sample-detail">
                      <div>上传文件：{formatUploadedFiles(item.files)}</div>
                      <div>模型：{item.ocr_model}</div>
                      <div>页数：{item.page_count ?? item.results_count ?? item.results?.length ?? 0} 页</div>
                    </div>
                    <div className="home-sample-preview">
                      纯文本预览：{item.sample_preview || '暂无可展示内容'}
                    </div>
                  </>
                )}
                <div className="home-sample-actions">
                  {!item.missing && (
                    <button
                      className="button ghost small"
                      type="button"
                      onClick={() => openDetail(item.id)}
                    >
                      查看任务详情
                    </button>
                  )}
                  {!item.missing && (
                    <button
                      className="button ghost small"
                      type="button"
                      onClick={() => renameSample(item)}
                    >
                      设置名称
                    </button>
                  )}
                  <button
                    className="button danger small"
                    type="button"
                    onClick={() => onToggleSample(item.id)}
                  >
                    移除样例
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <Modal
        open={detailOpen}
        onClose={closeDetail}
        title={detailJob ? `任务详情 · ${detailJob.id}` : '任务详情'}
      >
        {detailJob?.files?.length > 0 && (
          <div className="input-section">
            <div className="input-title">上传文件</div>
            <div className="input-grid">
              {detailJob.files.map((fileName) =>
                renderInputPreview(fileName, detailJob.id),
              )}
            </div>
          </div>
        )}
        {detailJob && (
          <div className="detail-header">
            <div>
              <div className="detail-title">
                状态：{STATUS_TEXT[detailJob.status] || detailJob.status}
              </div>
              <div className="detail-sub">
                模型 {resolveModelLabel(detailJob.ocr_model)} · 进度{' '}
                {detailJob.progress || 0}% · 文件数 {detailJob.files?.length || 0}
              </div>
            </div>
          </div>
        )}
        {detailJob && (
          <>
            <div className="progress">
              <div
                className="progress-bar"
                style={{ width: `${detailJob.progress || 0}%` }}
              />
            </div>
            <div className="progress-meta">进度 {detailJob.progress || 0}%</div>
          </>
        )}
        {detailError && <div className="error">{detailError}</div>}
        {detailLoading && <div className="search-loading">结果加载中...</div>}
        {!detailLoading && detailResults.length > 0 && (
          <ResultsPanel
            title=""
            items={detailResults}
            apiBase={apiBase}
            fallbackJobId={detailJob?.id}
            emptyText=""
            onItemUpdated={handleItemUpdated}
          />
        )}
        {!detailLoading && detailResults.length === 0 && !detailError && (
          <div className="empty-state">暂无识别结果。</div>
        )}
      </Modal>
    </>
  )
}

export default HomePage
