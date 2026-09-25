import { useEffect, useState } from 'react'
import Modal from './Modal'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'

const VIEW_OPTIONS = [
  { id: 'markdown', label: 'Markdown 预览' },
  { id: 'text', label: '纯文本' },
  { id: 'boxes', label: '检测框' },
  { id: 'cropped', label: '裁剪图' },
  { id: 'raw', label: '原始文本' },
]

const EDIT_OPTIONS = [
  { id: 'markdown', label: 'Markdown' },
  { id: 'text', label: '纯文本' },
]

const EXPORT_FORMATS = [
  { id: 'md', label: 'MD' },
  { id: 'txt', label: 'TXT' },
  { id: 'docx', label: 'DOCX' },
  { id: 'pdf', label: 'PDF' },
]

const OCR_MODEL_PADDLE = 'PaddleOCR-VL-1.5'

const isPaddleModelValue = (value = '') => {
  const normalized = String(value || '')
    .trim()
    .toLowerCase()
    .replace(/_/g, '-')
  return (
    normalized === 'paddleocr-vl-1.5' ||
    normalized === 'paddleocr-vl' ||
    normalized === 'paddleocr' ||
    normalized === 'paddle'
  )
}

const normalizeMathDelimiters = (markdown = '') => {
  if (!markdown) return ''
  let output = markdown
  output = output.replace(/\\\[((?:.|\n)*?)\\\]/g, (_, content) => `$$${content}$$`)
  output = output.replace(/\\\(((?:.|\n)*?)\\\)/g, (_, content) => `$${content}$`)
  output = output.replace(/(^|[^\\])\$\$([\s\S]*?)\$\$/g, (_, prefix, content) => {
    const normalized = (content || '').trim()
    return `${prefix}$$${normalized || content}$$`
  })
  output = output.replace(/(^|[^\\])\$([^$\n]+?)\$/g, (_, prefix, content) => {
    const normalized = (content || '').trim()
    if (!normalized || normalized === content) return `${prefix}$${content}$`
    const seemsMath = /\\[A-Za-z]+|[\^_{}=+\-*/<>]/.test(normalized)
    if (!seemsMath) return `${prefix}$${content}$`
    return `${prefix}$${normalized}$`
  })
  return output
}

const preserveLiteralImageToken = (markdown = '') => {
  if (!markdown) return ''
  return markdown
    .replace(/<\s*\/\s*image\s*>/gi, '&lt;/image&gt;')
    .replace(/<\s*image\s*>/gi, '&lt;image&gt;')
}

const escapeHtml = (value = '') =>
  String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const isStandaloneImageLine = (line = '') => {
  const text = (line || '').trim()
  if (!text) return false
  if (/^!\[[^\]]*]\([^)]+\)(?:\s*\{[^}]*\}\s*)*$/i.test(text)) return true
  if (/^<img\b[^>]*\/?>$/i.test(text)) return true
  if (/^<div\b[^>]*>\s*<img\b[^>]*\/?>\s*<\/div>$/i.test(text)) return true
  return false
}

const isCenteredHtmlBlock = (line = '') =>
  /^<div\b[^>]*text-align\s*:\s*center[^>]*>[\s\S]*<\/div>$/i.test((line || '').trim())

const isLikelyFigureCaptionLine = (line = '') => {
  const text = (line || '').trim()
  if (!text || text.length > 220) return false
  if (/^(#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s*|`{3,})/.test(text)) return false
  if (/<img\b|<table\b|<tr\b|<td\b|!\[[^\]]*]\([^)]+\)/i.test(text)) return false
  if (/^\([A-Za-z0-9ivxIVX]+\)\s+\S+/.test(text)) return true
  if (/^(?:fig(?:ure)?\.?\s*\d+|图\s*\d+|表\s*\d+)/i.test(text)) return true
  return false
}

const centerPaddleFigureCaptions = (markdown = '') => {
  if (!markdown) return ''
  const lines = markdown.replace(/\r\n/g, '\n').split('\n')
  for (let i = 0; i < lines.length; i += 1) {
    if (!isStandaloneImageLine(lines[i])) continue
    for (let j = i + 1; j < lines.length; j += 1) {
      const raw = lines[j] ?? ''
      const trimmed = raw.trim()
      if (!trimmed) continue
      if (isCenteredHtmlBlock(trimmed)) continue
      if (!isLikelyFigureCaptionLine(trimmed)) break
      lines[j] = `<div style="text-align: center;">${escapeHtml(trimmed)}</div>`
    }
  }
  return lines.join('\n')
}

function ResultsPanel({
  title,
  items,
  apiBase,
  fallbackJobId,
  emptyText,
  onItemUpdated,
}) {
  const [viewByKey, setViewByKey] = useState({})
  const [copyStatus, setCopyStatus] = useState({})
  const [editByKey, setEditByKey] = useState({})
  const [draftByKey, setDraftByKey] = useState({})
  const [saveStatus, setSaveStatus] = useState({})
  const [saveLoading, setSaveLoading] = useState({})
  const [exportLoading, setExportLoading] = useState({})
  const [localOverrides, setLocalOverrides] = useState({})
  const [batchExportOpen, setBatchExportOpen] = useState(false)
  const [batchSelectedDocIds, setBatchSelectedDocIds] = useState([])
  const [batchExportLoading, setBatchExportLoading] = useState({})
  const [batchExportStatus, setBatchExportStatus] = useState('')

  const getJobIdForItem = (item) => item?.job_id || fallbackJobId

  const mergeItem = (item) => {
    if (item?.doc_id && localOverrides[item.doc_id]) {
      return { ...item, ...localOverrides[item.doc_id] }
    }
    return item
  }

  const buildFileUrl = (relPath, item) => {
    const resolvedJobId = getJobIdForItem(item)
    if (!resolvedJobId || !relPath) return ''
    return `${apiBase}/ocr/jobs/${resolvedJobId}/files/${encodeURI(relPath)}`
  }

  const resolveRelPath = (relPath, item) => {
    if (!relPath) return ''
    let cleaned = relPath.replace(/^\.?\//, '').replace(/\\/g, '/')
    cleaned = cleaned.replace(/^(?:(?:images|imgs)\/)+/i, 'images/')
    const base = item?.output_dir_rel ? item.output_dir_rel.replace(/\\/g, '/') : ''
    return base ? `${base}/${cleaned}` : cleaned
  }

  const getMarkdownContent = (item) =>
    item?.edited_markdown ?? item?.markdown ?? ''
  const getTextContent = (item) => item?.edited_text ?? item?.text ?? ''

  const getCopyText = (item, view) => {
    if (view === 'markdown') {
      const content = getMarkdownContent(item)
      return isPaddleModelValue(item?.ocr_model)
        ? centerPaddleFigureCaptions(content)
        : content
    }
    if (view === 'text') return getTextContent(item)
    if (view === 'boxes') {
      return item.boxes_image ? buildFileUrl(item.boxes_image, item) : ''
    }
    if (view === 'cropped') {
      if (!item.cropped_images || item.cropped_images.length === 0) return ''
      return item.cropped_images.map((path) => buildFileUrl(path, item)).join('\n')
    }
    if (view === 'raw') {
      return item.raw_text || (item.raw ? JSON.stringify(item.raw, null, 2) : '')
    }
    return ''
  }

  const resolvePublicApiBase = () => {
    if (typeof window === 'undefined') return ''
    if (/^https?:\/\//i.test(apiBase)) return apiBase.replace(/\/+$/, '')
    if (apiBase.startsWith('/')) {
      return `${window.location.origin}${apiBase}`.replace(/\/+$/, '')
    }
    return `${window.location.origin}/${apiBase}`.replace(/\/+$/, '')
  }

  const getExportExtension = (format) => {
    if (format === 'markdown' || format === 'md') return 'md'
    if (format === 'text' || format === 'txt') return 'txt'
    return format
  }

  const getExportButtonLabel = (format) => {
    const ext = getExportExtension(format).toUpperCase()
    return `导出 ${ext}`
  }

  const buildExportUrl = (item, format) => {
    if (!item?.doc_id) return ''
    const params = new URLSearchParams({ format })
    if (format === 'md' || format === 'markdown') {
      const publicApiBase = resolvePublicApiBase()
      if (publicApiBase) {
        params.set('asset_base', publicApiBase.replace(/\/+$/, ''))
      }
    }
    return `${apiBase}/documents/${item.doc_id}/export?${params.toString()}`
  }

  const buildJobExportUrl = (jobId, format, docIds = []) => {
    if (!jobId) return ''
    const params = new URLSearchParams({ format })
    if (docIds.length > 0) {
      params.set('doc_ids', docIds.join(','))
    }
    if (format === 'md' || format === 'markdown') {
      const publicApiBase = resolvePublicApiBase()
      if (publicApiBase) {
        params.set('asset_base', publicApiBase)
      }
    }
    return `${apiBase}/ocr/jobs/${jobId}/export?${params.toString()}`
  }

  const sanitizeFilename = (value) =>
    value.replace(/[\\/:*?"<>|]+/g, '_').replace(/\s+/g, ' ').trim()

  const getFilenameFromHeader = (contentDisposition) => {
    if (!contentDisposition) return ''
    const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i)
    if (utf8Match?.[1]) {
      try {
        return decodeURIComponent(utf8Match[1])
      } catch {
        return utf8Match[1]
      }
    }
    const match = contentDisposition.match(/filename="?([^";]+)"?/i)
    return match?.[1] || ''
  }

  const exportableItems = (items || [])
    .map((item, index) => {
      const resolved = mergeItem(item)
      return {
        ...resolved,
        _index: index + 1,
        _label: resolved.file || resolved.title || `第 ${index + 1} 页`,
      }
    })
    .filter((item) => Boolean(item.doc_id))

  const batchJobId =
    exportableItems.find((item) => item.job_id)?.job_id || fallbackJobId || ''
  const canBatchExport = exportableItems.length > 1 && Boolean(batchJobId)

  const areIdListsEqual = (left, right) => {
    if (left === right) return true
    if (!Array.isArray(left) || !Array.isArray(right)) return false
    if (left.length !== right.length) return false
    for (let i = 0; i < left.length; i += 1) {
      if (left[i] !== right[i]) return false
    }
    return true
  }

  useEffect(() => {
    if (!batchExportOpen) return
    const validIds = new Set(exportableItems.map((item) => item.doc_id))
    const allIds = exportableItems.map((item) => item.doc_id)
    setBatchSelectedDocIds((prev) => {
      const filtered = prev.filter((id) => validIds.has(id))
      const next = filtered.length > 0 ? filtered : allIds
      return areIdListsEqual(prev, next) ? prev : next
    })
  }, [batchExportOpen, exportableItems])

  const downloadResponseToFile = async (response, fallbackBase, format) => {
    const blob = await response.blob()
    const headerName = response.headers.get('Content-Disposition')
    const headerFilename = getFilenameFromHeader(headerName || '')
    const ext = getExportExtension(format)
    const filename = sanitizeFilename(headerFilename || `${fallbackBase}.${ext}`)
    const objectUrl = window.URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = objectUrl
    anchor.download = filename
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    window.URL.revokeObjectURL(objectUrl)
  }

  const handleExport = async (item, viewKey, format) => {
    if (!item?.doc_id) {
      setSaveStatus((prev) => ({ ...prev, [viewKey]: '文档尚未入库，无法导出。' }))
      setTimeout(() => {
        setSaveStatus((prev) => ({ ...prev, [viewKey]: '' }))
      }, 1800)
      return
    }
    const exportKey = `${viewKey}-${format}`
    const url = buildExportUrl(item, format)
    if (!url) return
    try {
      setExportLoading((prev) => ({ ...prev, [exportKey]: true }))
      const response = await fetch(url)
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || '导出失败')
      }
      const fallbackBase = sanitizeFilename(
        item.file || item.title || `document_${item.doc_id}`,
      )
      await downloadResponseToFile(response, fallbackBase, format)
      setSaveStatus((prev) => ({ ...prev, [viewKey]: '已开始下载' }))
      setTimeout(() => {
        setSaveStatus((prev) => ({ ...prev, [viewKey]: '' }))
      }, 1500)
    } catch (err) {
      setSaveStatus((prev) => ({
        ...prev,
        [viewKey]: err instanceof Error ? err.message : '导出失败',
      }))
    } finally {
      setExportLoading((prev) => ({ ...prev, [exportKey]: false }))
    }
  }

  const openBatchExport = () => {
    const allDocIds = exportableItems.map((item) => item.doc_id)
    setBatchSelectedDocIds(allDocIds)
    setBatchExportStatus('')
    setBatchExportOpen(true)
  }

  const toggleBatchDoc = (docId) => {
    setBatchSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId],
    )
  }

  const selectAllBatchDocs = () => {
    setBatchSelectedDocIds(exportableItems.map((item) => item.doc_id))
  }

  const clearBatchDocs = () => {
    setBatchSelectedDocIds([])
  }

  const handleBatchExport = async (
    format,
    { selectedOnly = false, closeAfterExport = false } = {},
  ) => {
    if (!batchJobId) {
      setBatchExportStatus('未找到任务编号，无法整体导出。')
      return
    }

    const allDocIds = exportableItems.map((item) => item.doc_id)
    const selectedSet = new Set(batchSelectedDocIds)
    const targetDocIds = selectedOnly
      ? allDocIds.filter((docId) => selectedSet.has(docId))
      : allDocIds
    if (!targetDocIds.length) {
      setBatchExportStatus('请至少选择一页后再导出。')
      return
    }

    const loadingKey = `${format}-${selectedOnly ? 'selected' : 'all'}`
    const shouldPassDocIds = selectedOnly || targetDocIds.length !== allDocIds.length
    const url = buildJobExportUrl(batchJobId, format, shouldPassDocIds ? targetDocIds : [])
    if (!url) return

    try {
      setBatchExportLoading((prev) => ({ ...prev, [loadingKey]: true }))
      const response = await fetch(url)
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || '整体导出失败')
      }
      const fallbackBase = sanitizeFilename(
        `job_${batchJobId.slice(0, 8)}_${targetDocIds.length}pages`,
      )
      await downloadResponseToFile(response, fallbackBase, format)
      setBatchExportStatus('已开始下载')
      if (closeAfterExport) {
        setBatchExportOpen(false)
      }
      setTimeout(() => {
        setBatchExportStatus('')
      }, 1500)
    } catch (err) {
      setBatchExportStatus(err instanceof Error ? err.message : '整体导出失败')
    } finally {
      setBatchExportLoading((prev) => ({ ...prev, [loadingKey]: false }))
    }
  }

  const startEdit = (viewKey, item, currentView) => {
    if (!item?.doc_id) {
      setSaveStatus((prev) => ({ ...prev, [viewKey]: '文档尚未入库，无法编辑。' }))
      setTimeout(() => {
        setSaveStatus((prev) => ({ ...prev, [viewKey]: '' }))
      }, 1800)
      return
    }
    const defaultMode =
      currentView === 'text' || currentView === 'markdown' ? currentView : 'markdown'
    const isPaddle = isPaddleModelValue(item?.ocr_model)
    const initialMarkdown = getMarkdownContent(item)
    const markdownDraft = isPaddle
      ? centerPaddleFigureCaptions(initialMarkdown)
      : initialMarkdown
    setDraftByKey((prev) => ({
      ...prev,
      [viewKey]: {
        mode: defaultMode,
        markdown: markdownDraft,
        text: getTextContent(item),
      },
    }))
    setEditByKey((prev) => ({ ...prev, [viewKey]: true }))
  }

  const cancelEdit = (viewKey) => {
    setEditByKey((prev) => ({ ...prev, [viewKey]: false }))
    setDraftByKey((prev) => {
      const next = { ...prev }
      delete next[viewKey]
      return next
    })
    setSaveStatus((prev) => ({ ...prev, [viewKey]: '' }))
  }

  const updateDraft = (viewKey, field, value) => {
    setDraftByKey((prev) => ({
      ...prev,
      [viewKey]: {
        ...prev[viewKey],
        [field]: value,
      },
    }))
  }

  const saveEdit = async (item, viewKey) => {
    const draft = draftByKey[viewKey]
    if (!item?.doc_id || !draft) return
    const isPaddle = isPaddleModelValue(item?.ocr_model)
    const nextEditedMarkdown = isPaddle
      ? centerPaddleFigureCaptions(draft.markdown)
      : draft.markdown
    const payload =
      draft.mode === 'text'
        ? { edited_text: draft.text }
        : { edited_markdown: nextEditedMarkdown }
    try {
      setSaveLoading((prev) => ({ ...prev, [viewKey]: true }))
      const response = await fetch(`${apiBase}/documents/${item.doc_id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || '保存失败')
      }
      const data = await response.json()
      const updated = data.document
      const normalizedUpdated =
        updated && item?.ocr_model && !updated.ocr_model
          ? { ...updated, ocr_model: item.ocr_model }
          : updated
      if (updated) {
        if (onItemUpdated) {
          onItemUpdated(item, normalizedUpdated)
        } else if (updated.doc_id) {
          setLocalOverrides((prev) => ({
            ...prev,
            [updated.doc_id]: normalizedUpdated,
          }))
        }
      }
      setSaveStatus((prev) => ({ ...prev, [viewKey]: '已保存' }))
      setEditByKey((prev) => ({ ...prev, [viewKey]: false }))
      setDraftByKey((prev) => {
        const next = { ...prev }
        delete next[viewKey]
        return next
      })
      setTimeout(() => {
        setSaveStatus((prev) => ({ ...prev, [viewKey]: '' }))
      }, 1500)
    } catch (err) {
      setSaveStatus((prev) => ({
        ...prev,
        [viewKey]: err instanceof Error ? err.message : '保存失败',
      }))
    } finally {
      setSaveLoading((prev) => ({ ...prev, [viewKey]: false }))
    }
  }

  const handleCopy = async (item, viewKey, view) => {
    const text = getCopyText(item, view)
    if (!text) {
      setCopyStatus((prev) => ({ ...prev, [viewKey]: '无可复制内容' }))
      setTimeout(() => {
        setCopyStatus((prev) => ({ ...prev, [viewKey]: '' }))
      }, 1500)
      return
    }
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
      } else {
        const textarea = document.createElement('textarea')
        textarea.value = text
        textarea.style.position = 'fixed'
        textarea.style.opacity = '0'
        document.body.appendChild(textarea)
        textarea.focus()
        textarea.select()
        document.execCommand('copy')
        document.body.removeChild(textarea)
      }
      setCopyStatus((prev) => ({ ...prev, [viewKey]: '已复制' }))
    } catch {
      setCopyStatus((prev) => ({ ...prev, [viewKey]: '复制失败' }))
    } finally {
      setTimeout(() => {
        setCopyStatus((prev) => ({ ...prev, [viewKey]: '' }))
      }, 1500)
    }
  }

  const renderResultView = (item, view) => {
    if (view === 'markdown') {
      const content = getMarkdownContent(item)
      if (!content) {
        return <div className="empty-state">未检测到 markdown 输出。</div>
      }
      const isPaddle =
        item?.ocr_model === OCR_MODEL_PADDLE ||
        isPaddleModelValue(item?.ocr_model)
      const normalizedContent = preserveLiteralImageToken(
        normalizeMathDelimiters(content),
      )
      const displayContent = isPaddle
        ? centerPaddleFigureCaptions(normalizedContent)
        : normalizedContent
      return (
        <div className="markdown-preview">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeRaw, rehypeKatex]}
            components={{
              a: ({ href = '', children, ...props }) => (
                <a href={href} target="_blank" rel="noreferrer" {...props}>
                  {children}
                </a>
              ),
              img: ({ src = '', alt = '', ...props }) => {
                const relPath = resolveRelPath(src, item)
                const url = buildFileUrl(relPath, item)
                const paddleClassName = isPaddle
                  ? 'markdown-image paddle-markdown-image'
                  : 'markdown-image'
                const mergedClassName = [props.className, paddleClassName]
                  .filter(Boolean)
                  .join(' ')
                return (
                  <img
                    src={url}
                    alt={alt || 'markdown image'}
                    {...props}
                    className={mergedClassName}
                  />
                )
              },
            }}
          >
            {displayContent}
          </ReactMarkdown>
        </div>
      )
    }

    if (view === 'text') {
      const content = getTextContent(item)
      return (
        <pre className="result-body">{content || '未检测到纯文本输出。'}</pre>
      )
    }

    if (view === 'boxes') {
      if (!item.boxes_image) {
        return <div className="empty-state">未检测到检测框标注图输出。</div>
      }
      return (
        <div className="image-view">
          <img src={buildFileUrl(item.boxes_image, item)} alt="检测框标注预览" />
        </div>
      )
    }

    if (view === 'cropped') {
      if (!item.cropped_images || item.cropped_images.length === 0) {
        return <div className="empty-state">未检测到裁剪图片。</div>
      }
      return (
        <div className="image-grid">
          {item.cropped_images.map((path) => (
            <img key={path} src={buildFileUrl(path, item)} alt="裁剪图" />
          ))}
        </div>
      )
    }

    if (view === 'raw') {
      return (
        <pre className="result-body">
          {item.raw_text || JSON.stringify(item.raw || {}, null, 2)}
        </pre>
      )
    }

    return null
  }

  if (!items || items.length === 0) {
    if (!emptyText) return null
    return (
      <section className="panel">
        {title && <h2>{title}</h2>}
        <div className="empty-state">{emptyText}</div>
      </section>
    )
  }

  return (
    <>
      <section className="panel">
        {title && <h2>{title}</h2>}
        {canBatchExport && (
          <div className="batch-export-bar">
            <div className="batch-export-info">
              共 {exportableItems.length} 页，可整体导出全部页或按页导出。
            </div>
            <div className="batch-export-actions">
              {EXPORT_FORMATS.map((fmt) => {
                const loadingKey = `${fmt.id}-all`
                return (
                  <button
                    key={fmt.id}
                    className="button ghost small"
                    type="button"
                    onClick={() => handleBatchExport(fmt.id, { selectedOnly: false })}
                    disabled={batchExportLoading[loadingKey]}
                  >
                    {batchExportLoading[loadingKey]
                      ? '导出中...'
                      : `全部${getExportButtonLabel(fmt.id)}`}
                  </button>
                )
              })}
              <button
                className="button primary small"
                type="button"
                onClick={openBatchExport}
              >
                选择页导出
              </button>
              {batchExportStatus && <span className="copy-status">{batchExportStatus}</span>}
            </div>
          </div>
        )}
        <div className="results">
          {items.map((item, index) => {
            const resolvedItem = mergeItem(item)
            const baseKey = item.doc_id || item.output_dir || item.file || index
            const viewKey = `${title || 'results'}-${baseKey}-${index}`
            const current = viewByKey[viewKey] || 'markdown'
            const editing = editByKey[viewKey]
            const draft = draftByKey[viewKey]
            const canEdit = Boolean(resolvedItem.doc_id)
            return (
              <article key={viewKey} className="result-card">
                <div className="result-header">
                  <div className="result-meta">
                    <span className="result-name">
                      {resolvedItem.file || resolvedItem.title || '未命名文件'}
                    </span>
                    <span className="result-sub">
                      {resolvedItem.output_files?.length || 0} 个输出文件
                    </span>
                  </div>
                  <div className="result-header-actions">
                    {EXPORT_FORMATS.map((fmt) => {
                      const exportKey = `${viewKey}-${fmt.id}`
                      return (
                        <button
                          key={fmt.id}
                          className="button ghost small"
                          type="button"
                          onClick={() => handleExport(resolvedItem, viewKey, fmt.id)}
                          disabled={exportLoading[exportKey]}
                          title={resolvedItem.doc_id ? '' : '文档尚未入库，无法导出。'}
                        >
                          {exportLoading[exportKey]
                            ? '导出中...'
                            : getExportButtonLabel(fmt.id)}
                        </button>
                      )
                    })}
                  </div>
                </div>
                <div className="view-toolbar">
                  <div className="view-tabs">
                    {VIEW_OPTIONS.map((option) => (
                      <button
                        key={option.id}
                        className={`view-tab ${current === option.id ? 'active' : ''}`}
                        type="button"
                        onClick={() =>
                          setViewByKey((prev) => ({ ...prev, [viewKey]: option.id }))
                        }
                        disabled={editing}
                        title={editing ? '当前页正在编辑中，请先保存或取消。' : ''}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                  <div className="result-actions">
                    {current === 'markdown' || current === 'text' ? (
                      <button
                        className="button ghost small"
                        onClick={() => startEdit(viewKey, resolvedItem, current)}
                        type="button"
                        disabled={!canEdit || editing}
                        title={canEdit ? '' : '文档尚未入库，无法编辑。'}
                      >
                        {editing ? '编辑中' : '编辑'}
                      </button>
                    ) : null}
                    <button
                      className="button ghost small"
                      onClick={() => handleCopy(resolvedItem, viewKey, current)}
                      type="button"
                    >
                      复制当前视图
                    </button>
                    {(copyStatus[viewKey] || saveStatus[viewKey]) && (
                      <span className="copy-status">
                        {copyStatus[viewKey] || saveStatus[viewKey]}
                      </span>
                    )}
                  </div>
                </div>
                {editing && draft ? (
                  <div className="edit-panel">
                    <div className="edit-tabs">
                      {EDIT_OPTIONS.map((option) => (
                        <button
                          key={option.id}
                          className={`view-tab ${
                            draft.mode === option.id ? 'active' : ''
                          }`}
                          onClick={() =>
                            updateDraft(viewKey, 'mode', option.id)
                          }
                          type="button"
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                    <textarea
                      className="edit-textarea"
                      value={draft.mode === 'text' ? draft.text : draft.markdown}
                      onChange={(event) =>
                        updateDraft(
                          viewKey,
                          draft.mode === 'text' ? 'text' : 'markdown',
                          event.target.value,
                        )
                      }
                    />
                    <div className="edit-actions">
                      <button
                        className="button ghost small"
                        onClick={() => cancelEdit(viewKey)}
                        type="button"
                        disabled={saveLoading[viewKey]}
                      >
                        取消
                      </button>
                      <button
                        className="button primary small"
                        onClick={() => saveEdit(resolvedItem, viewKey)}
                        type="button"
                        disabled={saveLoading[viewKey]}
                      >
                        {saveLoading[viewKey] ? '保存中...' : '保存'}
                      </button>
                    </div>
                  </div>
                ) : (
                  renderResultView(resolvedItem, current)
                )}
              </article>
            )
          })}
        </div>
      </section>

      <Modal
        open={batchExportOpen}
        onClose={() => setBatchExportOpen(false)}
        title="选择页导出"
        footer={
          <div className="modal-actions batch-export-footer">
            <button
              className="button ghost"
              type="button"
              onClick={() => setBatchExportOpen(false)}
            >
              取消
            </button>
            {EXPORT_FORMATS.map((fmt) => {
              const loadingKey = `${fmt.id}-selected`
              return (
                <button
                  key={fmt.id}
                  className="button primary small"
                  type="button"
                  onClick={() =>
                    handleBatchExport(fmt.id, {
                      selectedOnly: true,
                      closeAfterExport: true,
                    })
                  }
                  disabled={batchExportLoading[loadingKey] || batchSelectedDocIds.length === 0}
                >
                  {batchExportLoading[loadingKey]
                    ? '导出中...'
                    : `导出所选 ${fmt.label}`}
                </button>
              )
            })}
          </div>
        }
      >
        <div className="batch-select-toolbar">
          <span>
            已选择 {batchSelectedDocIds.length} / {exportableItems.length} 页
          </span>
          <div className="batch-select-actions">
            <button className="button ghost small" type="button" onClick={selectAllBatchDocs}>
              全选
            </button>
            <button className="button ghost small" type="button" onClick={clearBatchDocs}>
              清空
            </button>
          </div>
        </div>
        <div className="batch-select-list">
          {exportableItems.map((item) => (
            <label key={item.doc_id} className="batch-select-item">
              <input
                type="checkbox"
                checked={batchSelectedDocIds.includes(item.doc_id)}
                onChange={() => toggleBatchDoc(item.doc_id)}
              />
              <span>
                第 {item._index} 页 · {item._label}
              </span>
            </label>
          ))}
        </div>
        {batchExportStatus && <div className="copy-status">{batchExportStatus}</div>}
      </Modal>
    </>
  )
}

export default ResultsPanel
