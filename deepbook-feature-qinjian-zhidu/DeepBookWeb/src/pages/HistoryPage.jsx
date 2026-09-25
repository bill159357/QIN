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

const ACTIVE_DETAIL_STATUSES = new Set([
  'pending',
  'queued',
  'running',
  'pausing',
  'paused',
])

const buildResultSignature = (item = {}) => {
  const cropped = Array.isArray(item.cropped_images) ? item.cropped_images.join('|') : ''
  const outputs = Array.isArray(item.output_files) ? item.output_files.join('|') : ''
  return [
    item.doc_id || '',
    item.output_dir || '',
    item.file || item.file_name || '',
    item.edited_markdown || '',
    item.markdown || '',
    item.edited_text || '',
    item.text || '',
    item.raw_text || '',
    item.boxes_image || '',
    item.output_dir_rel || '',
    outputs,
    cropped,
  ].join('@@')
}

const areResultListsEqual = (prevList = [], nextList = []) => {
  if (prevList === nextList) return true
  if (!Array.isArray(prevList) || !Array.isArray(nextList)) return false
  if (prevList.length !== nextList.length) return false
  for (let i = 0; i < prevList.length; i += 1) {
    if (buildResultSignature(prevList[i]) !== buildResultSignature(nextList[i])) {
      return false
    }
  }
  return true
}

const isSameDetailJob = (prev, next) => {
  if (!prev || !next) return false
  const prevFiles = Array.isArray(prev.files) ? prev.files : []
  const nextFiles = Array.isArray(next.files) ? next.files : []
  if (prevFiles.length !== nextFiles.length) return false
  for (let i = 0; i < prevFiles.length; i += 1) {
    if (prevFiles[i] !== nextFiles[i]) return false
  }
  return (
    prev.id === next.id &&
    prev.status === next.status &&
    prev.progress === next.progress &&
    (prev.error || '') === (next.error || '') &&
    (prev.updated_at || '') === (next.updated_at || '') &&
    (prev.results_count ?? 0) === (next.results_count ?? 0)
  )
}

const PAGE_SIZE_OPTIONS = [10, 20, 50]

const STATUS_OPTIONS = [
  { value: 'all', label: '全部状态' },
  { value: 'queued', label: STATUS_TEXT.queued },
  { value: 'running', label: STATUS_TEXT.running },
  { value: 'pausing', label: STATUS_TEXT.pausing },
  { value: 'paused', label: STATUS_TEXT.paused },
  { value: 'succeeded', label: STATUS_TEXT.succeeded },
  { value: 'failed', label: STATUS_TEXT.failed },
  { value: 'canceled', label: STATUS_TEXT.canceled },
]

const MODEL_OPTIONS = [
  { value: 'all', label: '全部模型' },
  { value: 'DeepSeek-OCR-2', label: 'DeepSeek-OCR-2' },
  { value: 'PaddleOCR-VL-1.5', label: 'PaddleOCR-VL-1.5' },
  { value: 'DeepJiandu-OCR-v1', label: 'DeepJiandu-OCR-v1' },
]

const formatDate = (value) => {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

const formatUploadedFiles = (files) => {
  if (!Array.isArray(files) || files.length === 0) return '-'
  if (files.length <= 2) return files.join('、')
  return `${files.slice(0, 2).join('、')} 等 ${files.length} 个文件`
}

const resolveOcrModelLabel = (value) => {
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

function HistoryPage({
  apiBase,
  featuredSampleJobIds = [],
  onToggleHomeSampleJob = () => {},
}) {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [keyword, setKeyword] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [modelFilter, setModelFilter] = useState('all')
  const [createdFrom, setCreatedFrom] = useState('')
  const [createdTo, setCreatedTo] = useState('')
  const [pageSize, setPageSize] = useState(20)
  const [currentPage, setCurrentPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [totalPages, setTotalPages] = useState(0)
  const [historyRan, setHistoryRan] = useState(false)
  const [detailOpen, setDetailOpen] = useState(false)
  const [detailJob, setDetailJob] = useState(null)
  const [detailResults, setDetailResults] = useState([])
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState('')
  const [detailActionLoading, setDetailActionLoading] = useState(false)
  const [deletingJobId, setDeletingJobId] = useState('')
  const featuredSampleSet = useMemo(
    () => new Set(Array.isArray(featuredSampleJobIds) ? featuredSampleJobIds : []),
    [featuredSampleJobIds],
  )

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

  const loadJobs = async ({
    targetPage = currentPage,
    targetPageSize = pageSize,
    filters,
  } = {}) => {
    const activeFilters = filters || {
      keyword,
      statusFilter,
      modelFilter,
      createdFrom,
      createdTo,
    }
    setError('')
    setLoading(true)
    setHistoryRan(true)
    try {
      const params = new URLSearchParams()
      const normalizedKeyword = (activeFilters.keyword || '').trim()
      if (normalizedKeyword) params.set('q', normalizedKeyword)
      if (activeFilters.statusFilter && activeFilters.statusFilter !== 'all') {
        params.set('status', activeFilters.statusFilter)
      }
      if (activeFilters.modelFilter && activeFilters.modelFilter !== 'all') {
        params.set('ocr_model', activeFilters.modelFilter)
      }
      if (activeFilters.createdFrom) params.set('created_from', activeFilters.createdFrom)
      if (activeFilters.createdTo) params.set('created_to', activeFilters.createdTo)
      params.set('page', String(targetPage))
      params.set('page_size', String(targetPageSize))

      const response = await fetch(`${apiBase}/ocr/jobs?${params.toString()}`)
      if (!response.ok) {
        throw new Error('加载任务失败')
      }
      const data = await response.json()
      const jobList = data.jobs || []
      const resolvedTotal = Number(data.total)
      const resolvedPage = Number(data.page) || targetPage
      const computedTotal = Number.isFinite(resolvedTotal) ? resolvedTotal : jobList.length
      const resolvedTotalPages =
        Number(data.total_pages) ||
        Math.ceil(computedTotal / Math.max(1, targetPageSize))

      setJobs(jobList)
      setTotal(computedTotal)
      setCurrentPage(resolvedPage)
      setTotalPages(resolvedTotalPages)
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载任务失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadJobs({ targetPage: 1 })
  }, [apiBase])

  const submitFilters = async (event) => {
    event.preventDefault()
    await loadJobs({ targetPage: 1 })
  }

  const resetFilters = async () => {
    const nextFilters = {
      keyword: '',
      statusFilter: 'all',
      modelFilter: 'all',
      createdFrom: '',
      createdTo: '',
    }
    setKeyword('')
    setStatusFilter('all')
    setModelFilter('all')
    setCreatedFrom('')
    setCreatedTo('')
    setPageSize(20)
    await loadJobs({
      targetPage: 1,
      targetPageSize: 20,
      filters: nextFilters,
    })
  }

  const fetchDetail = async (jobId) => {
    const jobResp = await fetch(`${apiBase}/ocr/jobs/${jobId}`)
    if (!jobResp.ok) {
      throw new Error('加载任务失败')
    }
    const jobData = await jobResp.json()
    setDetailJob((prev) => (isSameDetailJob(prev, jobData) ? prev : jobData))

    const resultResp = await fetch(`${apiBase}/ocr/jobs/${jobId}/results`)
    if (!resultResp.ok) {
      throw new Error('加载结果失败')
    }
    const resultData = await resultResp.json()
    const nextResults = resultData.results || []
    setDetailResults((prev) =>
      areResultListsEqual(prev, nextResults) ? prev : nextResults,
    )
  }

  const openDetail = async (job) => {
    if (!job?.id) return
    setDetailJob(job)
    setDetailResults([])
    setDetailError('')
    setDetailLoading(true)
    setDetailOpen(true)
    try {
      await fetchDetail(job.id)
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : '加载结果失败')
    } finally {
      setDetailLoading(false)
    }
  }

  useEffect(() => {
    if (!detailOpen || !detailJob?.id) return undefined
    const source = new EventSource(`${apiBase}/ocr/jobs/${detailJob.id}/stream`)
    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.type === 'snapshot') {
          const snapshotJob = payload.data?.job
          const snapshotResults = payload.data?.results || []
          if (snapshotJob) {
            setDetailJob((prev) =>
              isSameDetailJob(prev, snapshotJob) ? prev : snapshotJob,
            )
          }
          setDetailResults((prev) =>
            areResultListsEqual(prev, snapshotResults) ? prev : snapshotResults,
          )
        } else if (payload.type === 'result') {
          const result = payload.data?.result
          if (result) {
            setDetailResults((prev) => {
              const exists = prev.some(
                (item) => item.output_dir === result.output_dir,
              )
              return exists ? prev : [...prev, result]
            })
          }
          const progress = payload.data?.progress
          const status = payload.data?.status
          if (progress !== undefined || status) {
            setDetailJob((prev) =>
              prev
                ? {
                    ...prev,
                    progress: progress ?? prev.progress,
                    status: status ?? prev.status,
                  }
                : prev,
            )
          }
        } else if (payload.type === 'status') {
          const status = payload.data?.status
          const progress = payload.data?.progress
          const error = payload.data?.error
          if (status || progress !== undefined || error) {
            setDetailJob((prev) =>
              prev
                ? {
                    ...prev,
                    status: status ?? prev.status,
                    progress: progress ?? prev.progress,
                    error: error ?? prev.error,
                  }
                : prev,
            )
          }
        }
      } catch {
        // ignore malformed events
      }
    }
    source.onerror = () => {
      source.close()
    }
    return () => source.close()
  }, [apiBase, detailOpen, detailJob?.id])

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

  const closeDetail = () => {
    setDetailOpen(false)
    setDetailJob(null)
    setDetailResults([])
    setDetailError('')
  }

  useEffect(() => {
    if (!detailOpen || !detailJob?.id) return
    if (!ACTIVE_DETAIL_STATUSES.has(detailJob.status)) return
    let alive = true
    let timerId

    const poll = async () => {
      if (!alive) return
      try {
        await fetchDetail(detailJob.id)
      } catch (err) {
        if (alive) {
          setDetailError(err instanceof Error ? err.message : '加载结果失败')
        }
      }
    }

    poll()
    timerId = setInterval(poll, 2000)

    return () => {
      alive = false
      if (timerId) clearInterval(timerId)
    }
  }, [apiBase, detailOpen, detailJob?.id, detailJob?.status])

  const updateDetailJob = async (action) => {
    if (!detailJob?.id) return
    setDetailError('')
    setDetailActionLoading(true)
    try {
      const response = await fetch(`${apiBase}/ocr/jobs/${detailJob.id}/${action}`, {
        method: 'POST',
      })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || '操作失败')
      }
      const data = await response.json()
      setDetailJob(data)
      loadJobs()
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : '操作失败')
    } finally {
      setDetailActionLoading(false)
    }
  }

  const cancelDetailJob = async () => {
    if (!detailJob?.id) return
    setDetailError('')
    setDetailActionLoading(true)
    try {
      const response = await fetch(`${apiBase}/ocr/jobs/${detailJob.id}/cancel`, {
        method: 'POST',
      })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || '移除任务失败')
      }
      const data = await response.json()
      setDetailJob(data)
      loadJobs()
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : '移除任务失败')
    } finally {
      setDetailActionLoading(false)
    }
  }

  const deleteHistoryJob = async (job) => {
    if (!job?.id) return
    const confirmed = window.confirm(
      `确认删除任务 ${job.id.slice(0, 8)} 吗？删除后将移除历史记录和关联文件，且不可恢复。`,
    )
    if (!confirmed) return

    setError('')
    setDeletingJobId(job.id)
    try {
      const response = await fetch(`${apiBase}/ocr/jobs/${job.id}`, {
        method: 'DELETE',
      })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || '删除任务失败')
      }
      if (detailOpen && detailJob?.id === job.id) {
        closeDetail()
      }
      if (featuredSampleSet.has(job.id)) {
        onToggleHomeSampleJob(job.id)
      }
      await loadJobs({ targetPage: currentPage })
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除任务失败')
    } finally {
      setDeletingJobId('')
    }
  }

  return (
    <>
      <section className="panel">
        <div className="history-header">
          <h2>历史任务</h2>
          <button
            className="button ghost small"
            onClick={() => loadJobs({ targetPage: currentPage })}
            type="button"
          >
            刷新列表
          </button>
        </div>
        <form className="history-filter-form" onSubmit={submitFilters}>
          <div className="search-bar">
            <input
              className="search-input"
              type="text"
              placeholder="任务 ID 或文件名"
              value={keyword}
              onChange={(event) => setKeyword(event.target.value)}
            />
            <button className="button primary" type="submit" disabled={loading}>
              {loading ? '检索中...' : '搜索'}
            </button>
            <button className="button ghost" type="button" onClick={resetFilters}>
              重置条件
            </button>
          </div>
          <div className="filter-grid history-filter-grid">
            <label className="field">
              <span>任务状态</span>
              <select
                className="select-input"
                value={statusFilter}
                onChange={(event) => setStatusFilter(event.target.value)}
              >
                {STATUS_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>识别模型</span>
              <select
                className="select-input"
                value={modelFilter}
                onChange={(event) => setModelFilter(event.target.value)}
              >
                {MODEL_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>开始日期</span>
              <input
                type="date"
                value={createdFrom}
                onChange={(event) => setCreatedFrom(event.target.value)}
              />
            </label>
            <label className="field">
              <span>结束日期</span>
              <input
                type="date"
                value={createdTo}
                onChange={(event) => setCreatedTo(event.target.value)}
              />
            </label>
            <label className="field">
              <span>每页条数</span>
              <select
                className="select-input"
                value={pageSize}
                onChange={(event) => {
                  const nextSize = Number(event.target.value) || 20
                  setPageSize(nextSize)
                  if (historyRan) {
                    loadJobs({ targetPage: 1, targetPageSize: nextSize })
                  }
                }}
              >
                {PAGE_SIZE_OPTIONS.map((size) => (
                  <option key={size} value={size}>
                    {size} 条/页
                  </option>
                ))}
              </select>
            </label>
          </div>
        </form>
        {error && <div className="error">{error}</div>}
        {historyRan && !loading && (
          <div className="result-summary">
            共 {total} 个任务，第 {currentPage} / {Math.max(totalPages, 1)} 页
          </div>
        )}
        {loading && <div className="search-loading">任务加载中...</div>}
        {!loading && jobs.length === 0 && (
          <div className="search-empty">当前条件下暂无任务。</div>
        )}
        {jobs.length > 0 && (
          <div className="job-list">
            {jobs.map((job) => (
              <div key={job.id} className="job-card">
                <div className="job-meta">
                  <div>
                    <div className="job-title">任务 {job.id.slice(0, 8)}</div>
                    <div className="job-sub">创建于 {formatDate(job.created_at)}</div>
                  </div>
                  <span className={`status status-${job.status}`}>
                    {STATUS_TEXT[job.status] || job.status}
                  </span>
                </div>
                <div className="job-details">
                  <div className="job-file-line">上传文件：{formatUploadedFiles(job.files)}</div>
                  <div>页数：{job.page_count ?? job.results_count ?? job.results?.length ?? 0} 页</div>
                  <div>模型：{resolveOcrModelLabel(job.ocr_model)}</div>
                  <div>进度：{job.progress || 0}%</div>
                </div>
                <div className="job-snippet">
                  <span className="job-snippet-text">
                    纯文本预览：{job.text_excerpt || '暂无文本摘要'}
                  </span>
                </div>
                <div className="job-actions">
                  <button
                    className={`button ${
                      featuredSampleSet.has(job.id) ? 'ghost' : 'primary'
                    } small`}
                    onClick={() => onToggleHomeSampleJob(job.id)}
                    disabled={deletingJobId === job.id}
                    type="button"
                  >
                    {featuredSampleSet.has(job.id) ? '取消首页样例' : '设为首页样例'}
                  </button>
                  <button
                    className="button ghost small"
                    onClick={() => openDetail(job)}
                    disabled={deletingJobId === job.id}
                    type="button"
                  >
                    查看详情
                  </button>
                  <button
                    className="button danger small"
                    onClick={() => deleteHistoryJob(job)}
                    disabled={deletingJobId === job.id}
                    type="button"
                  >
                    {deletingJobId === job.id ? '删除中...' : '删除任务'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
        {historyRan && total > 0 && (
          <div className="pagination">
            <button
              className="button ghost small"
              type="button"
              onClick={() => loadJobs({ targetPage: currentPage - 1 })}
              disabled={loading || currentPage <= 1}
            >
              上一页
            </button>
            <div className="pagination-info">
              第 {currentPage} / {Math.max(totalPages, 1)} 页
            </div>
            <button
              className="button ghost small"
              type="button"
              onClick={() => loadJobs({ targetPage: currentPage + 1 })}
              disabled={loading || currentPage >= totalPages}
            >
              下一页
            </button>
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
                模型 {resolveOcrModelLabel(detailJob.ocr_model)} · 进度{' '}
                {detailJob.progress || 0}% · 文件数 {detailJob.files?.length || 0}
              </div>
            </div>
            <div className="detail-actions">
              {detailJob.status === 'running' && (
                <button
                  className="button ghost small"
                  onClick={() => updateDetailJob('pause')}
                  disabled={detailActionLoading}
                  type="button"
                >
                  {detailActionLoading ? '处理中...' : '暂停任务'}
                </button>
              )}
              {detailJob.status === 'paused' && (
                <button
                  className="button ghost small"
                  onClick={() => updateDetailJob('resume')}
                  disabled={detailActionLoading}
                  type="button"
                >
                  {detailActionLoading ? '处理中...' : '继续任务'}
                </button>
              )}
              {(detailJob.status === 'queued' || detailJob.status === 'paused') && (
                <button
                  className="button danger small"
                  onClick={cancelDetailJob}
                  disabled={detailActionLoading}
                  type="button"
                >
                  {detailActionLoading ? '处理中...' : '移出队列'}
                </button>
              )}
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

export default HistoryPage
