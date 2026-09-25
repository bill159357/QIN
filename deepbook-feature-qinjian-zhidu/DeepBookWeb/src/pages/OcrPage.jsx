import { useCallback, useEffect, useRef, useState } from 'react'
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

const OCR_MODEL_PADDLE = 'PaddleOCR-VL-1.5'
const OCR_MODEL_JIANDU = 'DeepJiandu-OCR-v1'
const DEFAULT_OCR_MODEL = OCR_MODEL_JIANDU
const DEFAULT_OCR_MODEL_STORAGE_KEY = 'deepbook.default_ocr_model'
const OCR_MODEL_OPTIONS = [
  { value: 'DeepSeek-OCR-2', label: 'DeepSeek-OCR-2' },
  { value: OCR_MODEL_PADDLE, label: OCR_MODEL_PADDLE },
  { value: OCR_MODEL_JIANDU, label: OCR_MODEL_JIANDU },
]

const PADDLE_AUX_LABEL_OPTIONS = [
  { key: 'header', label: '页眉' },
  { key: 'header_image', label: '页眉图片' },
  { key: 'footer', label: '页脚' },
  { key: 'footer_image', label: '页脚图片' },
  { key: 'number', label: '页码' },
  { key: 'footnote', label: '脚注' },
  { key: 'aside_text', label: '旁注文本' },
]

const DEFAULT_PADDLE_OPTIONS = {
  auxContent: {
    header: false,
    header_image: false,
    footer: false,
    footer_image: false,
    number: false,
    footnote: true,
    aside_text: false,
  },
  useDocOrientationClassify: false,
  useDocUnwarping: false,
  useLayoutDetection: true,
  useChartRecognition: false,
  layoutThreshold: 0.5,
  layoutUnclipRatio: 1.0,
  layoutMergeBboxesMode: 'large',
  mergeTables: true,
  relevelTitles: true,
  layoutShapeMode: 'auto',
  promptLabel: 'ocr',
  repetitionPenalty: 1,
  temperature: 0,
  topP: 1,
  minPixels: 147384,
  maxPixels: 2822400,
  showFormulaNumber: false,
  prettifyMarkdown: false,
  visualize: true,
  layoutNms: true,
  restructurePages: false,
}

const DEFAULT_JIANDU_OPTIONS = {
  scoreThresh: 0.3,
  topkDetect: 300,
  nmsIou: 0.35,
  minBoxSize: 8,
  maxAspect: 1.8,
  recTopK: 5,
}

const resolveOcrModel = (value) =>
  OCR_MODEL_OPTIONS.some((item) => item.value === value) ? value : DEFAULT_OCR_MODEL

const readDefaultOcrModelFromStorage = () => {
  if (typeof window === 'undefined') return DEFAULT_OCR_MODEL
  try {
    const stored = window.localStorage.getItem(DEFAULT_OCR_MODEL_STORAGE_KEY) || ''
    return resolveOcrModel(stored)
  } catch {
    return DEFAULT_OCR_MODEL
  }
}

function OcrPage({ apiBase, defaultPrompt, focusJobId, onFocusJobHandled }) {
  const [files, setFiles] = useState([])
  const [prompt, setPrompt] = useState(defaultPrompt)
  const [baseSize, setBaseSize] = useState(1024)
  const [imageSize, setImageSize] = useState(768)
  const [cropMode, setCropMode] = useState(true)
  const [defaultOcrModel, setDefaultOcrModel] = useState(() =>
    readDefaultOcrModelFromStorage(),
  )
  const [ocrModel, setOcrModel] = useState(() => readDefaultOcrModelFromStorage())
  const [paddleOptions, setPaddleOptions] = useState(DEFAULT_PADDLE_OPTIONS)
  const [jianduOptions, setJianduOptions] = useState(DEFAULT_JIANDU_OPTIONS)
  const [jobId, setJobId] = useState('')
  const [jobInfo, setJobInfo] = useState(null)
  const [results, setResults] = useState([])
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [actionState, setActionState] = useState({ jobId: '', action: '' })
  const [queueInfo, setQueueInfo] = useState({
    current_job: null,
    running_jobs: [],
    queue: [],
  })
  const [queueLoading, setQueueLoading] = useState(false)
  const [queueError, setQueueError] = useState('')
  const [detailOpen, setDetailOpen] = useState(false)
  const resultsRef = useRef(new Set())
  const fileInputRef = useRef(null)
  const [uploadDetailOpen, setUploadDetailOpen] = useState(false)
  const [selectedUploadFile, setSelectedUploadFile] = useState(null)
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState('')
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [dragActive, setDragActive] = useState(false)

  const handleItemUpdated = (prevItem, updated) => {
    if (!updated) return
    setResults((prev) =>
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

  const getLocalFileKey = (file) =>
    `${file.name}::${file.size}::${file.lastModified}`

  const formatFileSize = (size = 0) => {
    if (!Number.isFinite(size) || size <= 0) return '0 B'
    if (size < 1024) return `${size} B`
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
    return `${(size / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatUploadedFiles = (inputFiles) => {
    if (!Array.isArray(inputFiles) || inputFiles.length === 0) return '无'
    if (inputFiles.length <= 2) return inputFiles.join('、')
    return `${inputFiles.slice(0, 2).join('、')} 等 ${inputFiles.length} 个文件`
  }

  const resolvePageCount = (job) => {
    if (!job) return 0
    const explicit = Number(job.page_count)
    if (Number.isFinite(explicit) && explicit > 0) return explicit
    const fromResults = Number(job.results_count)
    if (Number.isFinite(fromResults) && fromResults > 0) return fromResults
    const filesLen = Array.isArray(job.files) ? job.files.length : 0
    return filesLen
  }

  const isImageFile = (file) => {
    const name = file?.name?.toLowerCase() || ''
    return (
      Boolean(file?.type?.startsWith('image/')) ||
      /\.(png|jpg|jpeg|gif|bmp|webp)$/i.test(name)
    )
  }

  const isPdfFile = (file) => {
    const name = file?.name?.toLowerCase() || ''
    return file?.type === 'application/pdf' || name.endsWith('.pdf')
  }

  const pickSupportedFiles = (incomingFiles) => {
    const supported = incomingFiles.filter(
      (file) => isImageFile(file) || isPdfFile(file),
    )
    const skipped = incomingFiles.length - supported.length
    if (skipped > 0) {
      setError(`已忽略 ${skipped} 个不支持的文件（仅支持图片或 PDF）。`)
    } else {
      setError('')
    }
    return supported
  }

  const getClipboardFileExtension = (mimeType = '') => {
    const normalized = mimeType.toLowerCase()
    if (normalized === 'image/png') return 'png'
    if (normalized === 'image/jpeg') return 'jpg'
    if (normalized === 'image/webp') return 'webp'
    if (normalized === 'image/gif') return 'gif'
    if (normalized === 'image/bmp') return 'bmp'
    if (normalized === 'application/pdf') return 'pdf'
    if (normalized.startsWith('image/')) {
      return normalized.replace('image/', '') || 'png'
    }
    return 'bin'
  }

  const normalizeClipboardFile = (rawFile, index = 0) => {
    if (!rawFile) return null
    if (rawFile instanceof File && rawFile.name) {
      return rawFile
    }
    const fallbackExt = getClipboardFileExtension(rawFile.type)
    const fallbackName = `clipboard_${Date.now()}_${index}.${fallbackExt}`
    return new File([rawFile], fallbackName, {
      type: rawFile.type || 'application/octet-stream',
      lastModified: Date.now(),
    })
  }

  const appendFiles = (incomingFiles) => {
    if (!incomingFiles.length) return
    setFiles((prev) => {
      const map = new Map()
      prev.forEach((file) => map.set(getLocalFileKey(file), file))
      incomingFiles.forEach((file) => map.set(getLocalFileKey(file), file))
      return Array.from(map.values())
    })
  }

  const openUploadDetail = (file) => {
    if (!file) return
    setSelectedUploadFile(file)
    setUploadDetailOpen(true)
  }

  const handleFileInputChange = (event) => {
    const incoming = pickSupportedFiles(Array.from(event.target.files || []))
    if (!incoming.length) {
      event.target.value = ''
      return
    }
    appendFiles(incoming)
    event.target.value = ''
  }

  const openFilePicker = () => {
    fileInputRef.current?.click()
  }

  const handleDropzoneDragEnter = (event) => {
    event.preventDefault()
    setDragActive(true)
  }

  const handleDropzoneDragOver = (event) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
    if (!dragActive) {
      setDragActive(true)
    }
  }

  const handleDropzoneDragLeave = (event) => {
    event.preventDefault()
    const nextTarget = event.relatedTarget
    if (nextTarget && event.currentTarget.contains(nextTarget)) return
    setDragActive(false)
  }

  const handleDropzoneDrop = (event) => {
    event.preventDefault()
    setDragActive(false)
    const incoming = pickSupportedFiles(Array.from(event.dataTransfer.files || []))
    if (!incoming.length) return
    appendFiles(incoming)
  }

  useEffect(() => {
    const isEditableTarget = (target) => {
      if (!(target instanceof HTMLElement)) return false
      const tagName = (target.tagName || '').toLowerCase()
      if (tagName === 'input' || tagName === 'textarea' || tagName === 'select') {
        return true
      }
      if (target.isContentEditable) return true
      return Boolean(
        target.closest(
          'input, textarea, select, [contenteditable=""], [contenteditable="true"]',
        ),
      )
    }

    const handlePaste = (event) => {
      if (isEditableTarget(event.target)) return

      const clipboardData = event.clipboardData
      if (!clipboardData) return

      const rawClipboardFiles = []
      const items = Array.from(clipboardData.items || [])
      if (items.length > 0) {
        items.forEach((item) => {
          if (item.kind !== 'file') return
          const file = item.getAsFile()
          if (file) rawClipboardFiles.push(file)
        })
      } else {
        rawClipboardFiles.push(...Array.from(clipboardData.files || []))
      }

      if (!rawClipboardFiles.length) return

      const normalizedFiles = rawClipboardFiles
        .map((file, index) => normalizeClipboardFile(file, index))
        .filter(Boolean)
      if (!normalizedFiles.length) return

      const supported = pickSupportedFiles(normalizedFiles)
      if (!supported.length) return

      event.preventDefault()
      appendFiles(supported)
      setError('')
    }

    window.addEventListener('paste', handlePaste)
    return () => window.removeEventListener('paste', handlePaste)
  }, [])

  const removeSelectedFile = (targetFile) => {
    const targetKey = getLocalFileKey(targetFile)
    const removingCurrentSelection =
      selectedUploadFile &&
      getLocalFileKey(selectedUploadFile) === targetKey
    setFiles((prev) => prev.filter((file) => getLocalFileKey(file) !== targetKey))
    if (removingCurrentSelection) {
      setSelectedUploadFile(null)
      setUploadDetailOpen(false)
      return
    }
    setSelectedUploadFile((prev) => {
      if (!prev) return prev
      return getLocalFileKey(prev) === targetKey ? null : prev
    })
  }

  const clearSelectedFiles = () => {
    setFiles([])
    setSelectedUploadFile(null)
    setUploadDetailOpen(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  useEffect(() => {
    if (!selectedUploadFile) {
      setUploadPreviewUrl('')
      return undefined
    }
    const objectUrl = URL.createObjectURL(selectedUploadFile)
    setUploadPreviewUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [selectedUploadFile])

  const statusLabel = (status) => STATUS_TEXT[status] || status || '未知'
  const isCurrentModelDefault = resolveOcrModel(ocrModel) === resolveOcrModel(defaultOcrModel)

  const setAsDefaultOcrModel = () => {
    const nextDefault = resolveOcrModel(ocrModel)
    try {
      window.localStorage.setItem(DEFAULT_OCR_MODEL_STORAGE_KEY, nextDefault)
    } catch {
      // ignore storage write failures
    }
    setDefaultOcrModel(nextDefault)
  }

  const setPaddleOption = (key, value) => {
    setPaddleOptions((prev) => ({ ...prev, [key]: value }))
  }

  const setPaddleAuxContent = (key, enabled) => {
    setPaddleOptions((prev) => ({
      ...prev,
      auxContent: { ...prev.auxContent, [key]: enabled },
    }))
  }

  const setJianduOption = (key, value) => {
    setJianduOptions((prev) => ({ ...prev, [key]: value }))
  }

  const normalizeQueueInfo = (payload) => {
    const runningFromApi = Array.isArray(payload?.running_jobs) ? payload.running_jobs : []
    const fallbackRunning = payload?.current_job ? [payload.current_job] : []
    const running_jobs = (runningFromApi.length > 0 ? runningFromApi : fallbackRunning).filter(
      Boolean,
    )
    const current_job = payload?.current_job || running_jobs[0] || null
    return {
      current_job,
      running_jobs,
      queue: Array.isArray(payload?.queue) ? payload.queue : [],
    }
  }

  const isActionPending = (targetJobId, targetAction) =>
    Boolean(targetJobId) &&
    actionState.jobId === targetJobId &&
    actionState.action === targetAction

  const isJobBusy = (targetJobId) =>
    Boolean(targetJobId) &&
    actionState.jobId === targetJobId &&
    Boolean(actionState.action)
  const isPaddleModel = ocrModel === OCR_MODEL_PADDLE
  const isJianduModel = ocrModel === OCR_MODEL_JIANDU
  const isDeepSeekModel = !isPaddleModel && !isJianduModel
  const runningJobs = queueInfo.running_jobs?.length
    ? queueInfo.running_jobs
    : queueInfo.current_job
      ? [queueInfo.current_job]
      : []
  const primaryRunningJob = runningJobs[0] || null

  const startOcr = async () => {
    setError('')
    setResults([])

    if (!files.length) {
      setError('请先选择待识别的图片或 PDF 文件。')
      return
    }

    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))
    formData.append('ocr_model', ocrModel)
    if (isDeepSeekModel) {
      formData.append('prompt', prompt)
      formData.append('base_size', String(baseSize))
      formData.append('image_size', String(imageSize))
      formData.append('crop_mode', String(cropMode))
    } else if (isPaddleModel) {
      const ignoredLabels = PADDLE_AUX_LABEL_OPTIONS.filter(
        (item) => !paddleOptions.auxContent[item.key],
      ).map((item) => item.key)
      ignoredLabels.forEach((label) => formData.append('markdown_ignore_labels', label))
      formData.append(
        'use_doc_orientation_classify',
        String(paddleOptions.useDocOrientationClassify),
      )
      formData.append('use_doc_unwarping', String(paddleOptions.useDocUnwarping))
      formData.append('use_layout_detection', String(Boolean(paddleOptions.useLayoutDetection)))
      formData.append(
        'use_chart_recognition',
        String(Boolean(paddleOptions.useChartRecognition)),
      )
      formData.append(
        'layout_threshold',
        String(Number(paddleOptions.layoutThreshold) || 0.5),
      )
      formData.append(
        'layout_unclip_ratio',
        String(Number(paddleOptions.layoutUnclipRatio) || 1.0),
      )
      formData.append(
        'layout_merge_bboxes_mode',
        String(paddleOptions.layoutMergeBboxesMode || 'large'),
      )
      formData.append('merge_tables', String(Boolean(paddleOptions.mergeTables)))
      formData.append('relevel_titles', String(Boolean(paddleOptions.relevelTitles)))
      formData.append('layout_shape_mode', String(paddleOptions.layoutShapeMode || 'auto'))
      formData.append('prompt_label', String(paddleOptions.promptLabel || 'ocr'))
      formData.append(
        'repetition_penalty',
        String(Number(paddleOptions.repetitionPenalty) || 1),
      )
      formData.append('temperature', String(Number(paddleOptions.temperature) || 0))
      formData.append('top_p', String(Number(paddleOptions.topP) || 1))
      formData.append('min_pixels', String(Number(paddleOptions.minPixels) || 147384))
      formData.append('max_pixels', String(Number(paddleOptions.maxPixels) || 2822400))
      formData.append('show_formula_number', String(Boolean(paddleOptions.showFormulaNumber)))
      formData.append('prettify_markdown', String(Boolean(paddleOptions.prettifyMarkdown)))
      formData.append('visualize', String(Boolean(paddleOptions.visualize)))
      formData.append('layout_nms', String(Boolean(paddleOptions.layoutNms)))
      formData.append('restructure_pages', String(Boolean(paddleOptions.restructurePages)))
    } else if (isJianduModel) {
      formData.append(
        'jiandu_score_thresh',
        String(Number(jianduOptions.scoreThresh) || 0.3),
      )
      formData.append(
        'jiandu_topk_detect',
        String(Math.max(1, Number(jianduOptions.topkDetect) || 300)),
      )
      formData.append('jiandu_nms_iou', String(Number(jianduOptions.nmsIou) || 0.35))
      formData.append(
        'jiandu_min_box_size',
        String(Math.max(1, Number(jianduOptions.minBoxSize) || 8)),
      )
      formData.append(
        'jiandu_max_aspect',
        String(Math.max(1, Number(jianduOptions.maxAspect) || 1.8)),
      )
      formData.append(
        'jiandu_rec_top_k',
        String(Math.max(1, Number(jianduOptions.recTopK) || 5)),
      )
    }

    try {
      setSubmitting(true)
      clearSelectedFiles()
      const response = await fetch(`${apiBase}/ocr/jobs`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || '提交失败，请检查后端服务是否启动。')
      }
      const data = await response.json()
      setJobId(data.id)
      setJobInfo(data)
      resultsRef.current = new Set()
      loadQueue()
    } catch (err) {
      setError(err instanceof Error ? err.message : '提交失败')
    } finally {
      setSubmitting(false)
    }
  }

  const togglePause = async (nextAction, targetJobId = getActiveJobId()) => {
    const activeId = targetJobId
    if (!activeId) return
    setActionState({ jobId: activeId, action: nextAction })
    setError('')
    try {
      const response = await fetch(`${apiBase}/ocr/jobs/${activeId}/${nextAction}`, {
        method: 'POST',
      })
      if (!response.ok) {
        throw new Error('操作失败，请稍后再试。')
      }
      const data = await response.json()
      setJobId(data.id)
      setJobInfo(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : '操作失败')
    } finally {
      setActionState({ jobId: '', action: '' })
    }
  }

  const cancelJob = async (targetJobId = getActiveJobId()) => {
    const activeId = targetJobId
    if (!activeId) return
    setActionState({ jobId: activeId, action: 'cancel' })
    setError('')
    try {
      const response = await fetch(`${apiBase}/ocr/jobs/${activeId}/cancel`, {
        method: 'POST',
      })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || '移除任务失败')
      }
      const data = await response.json()
      setJobId(data.id)
      setJobInfo(data)
      loadQueue()
    } catch (err) {
      setError(err instanceof Error ? err.message : '移除任务失败')
    } finally {
      setActionState({ jobId: '', action: '' })
    }
  }

  useEffect(() => {
    if (!jobId) return
    let alive = true
    let timerId

    const poll = async () => {
      try {
        const response = await fetch(`${apiBase}/ocr/jobs/${jobId}`)
        if (!response.ok) {
          throw new Error('查询任务失败')
        }
        const data = await response.json()
        if (!alive) return
        setJobInfo(data)

        if (['running', 'pausing', 'paused', 'succeeded'].includes(data.status)) {
          const resultResp = await fetch(`${apiBase}/ocr/jobs/${jobId}/results`)
          if (resultResp.ok) {
            const resultData = await resultResp.json()
            if (alive) {
              setResults(resultData.results || [])
            }
          }
          if (data.status === 'succeeded') {
            clearInterval(timerId)
          }
        }

        if (data.status === 'failed') {
          setError(data.error || '识别失败')
          clearInterval(timerId)
        }
      } catch (err) {
        if (alive) {
          setError(err instanceof Error ? err.message : '轮询失败')
        }
      }
    }

    poll()
    timerId = setInterval(poll, 1500)

    return () => {
      alive = false
      if (timerId) clearInterval(timerId)
    }
  }, [apiBase, jobId])

  useEffect(() => {
    if (!jobId) return
    const source = new EventSource(`${apiBase}/ocr/jobs/${jobId}/stream`)
    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.type === 'snapshot') {
          const snapshotJob = payload.data?.job
          const snapshotResults = payload.data?.results || []
          if (snapshotJob) {
            setJobInfo(snapshotJob)
          }
          resultsRef.current = new Set()
          snapshotResults.forEach((item) => {
            const key = item.output_dir || item.file
            if (key) {
              resultsRef.current.add(key)
            }
          })
          setResults(snapshotResults)
        } else if (payload.type === 'result') {
          const result = payload.data?.result
          if (result) {
            const key = result.output_dir || result.file
            if (!resultsRef.current.has(key)) {
              resultsRef.current.add(key)
              setResults((prev) => [...prev, result])
            }
          }
          const progress = payload.data?.progress
          const status = payload.data?.status
          if (progress !== undefined || status) {
            setJobInfo((prev) =>
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
            setJobInfo((prev) =>
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
  }, [apiBase, jobId])

  const loadQueue = async () => {
    setQueueError('')
    setQueueLoading(true)
    try {
      const response = await fetch(`${apiBase}/ocr/queue`)
      if (!response.ok) {
        throw new Error('加载队列失败')
      }
      const data = await response.json()
      setQueueInfo(normalizeQueueInfo(data))
    } catch (err) {
      setQueueError(err instanceof Error ? err.message : '加载队列失败')
    } finally {
      setQueueLoading(false)
    }
  }

  useEffect(() => {
    let alive = true
    let timerId

    const pollQueue = async () => {
      if (!alive) return
      try {
        const response = await fetch(`${apiBase}/ocr/queue`)
        if (!response.ok) {
          throw new Error('加载队列失败')
        }
        const data = await response.json()
        if (alive) {
          setQueueInfo(normalizeQueueInfo(data))
        }
      } catch (err) {
        if (alive) {
          setQueueError(err instanceof Error ? err.message : '加载队列失败')
        }
      }
    }

    pollQueue()
    timerId = setInterval(pollQueue, 2000)

    return () => {
      alive = false
      if (timerId) clearInterval(timerId)
    }
  }, [apiBase])

  useEffect(() => {
    if (jobId) return
    if (primaryRunningJob?.id) {
      setJobId(primaryRunningJob.id)
      setJobInfo(primaryRunningJob)
    }
  }, [jobId, primaryRunningJob])

  const getActiveJobId = () => primaryRunningJob?.id || jobInfo?.id || jobId

  const openJobDetailById = useCallback(
    async (targetJobId, fallbackJob = null) => {
      if (!targetJobId) return
      resultsRef.current = new Set()
      setResults([])
      setError('')
      setJobId(targetJobId)
      setJobInfo((prev) => {
        if (prev?.id === targetJobId) return prev
        return fallbackJob || null
      })
      setDetailOpen(true)

      try {
        const response = await fetch(`${apiBase}/ocr/jobs/${targetJobId}`)
        if (!response.ok) return
        const data = await response.json()
        setJobInfo(data)

        if (['running', 'pausing', 'paused', 'succeeded'].includes(data.status)) {
          const resultResp = await fetch(`${apiBase}/ocr/jobs/${targetJobId}/results`)
          if (!resultResp.ok) return
          const resultData = await resultResp.json()
          const nextResults = resultData.results || []
          resultsRef.current = new Set()
          nextResults.forEach((item) => {
            const key = item.output_dir || item.file
            if (key) {
              resultsRef.current.add(key)
            }
          })
          setResults(nextResults)
        }
      } catch {
        // ignore detail prefetch errors
      }
    },
    [apiBase],
  )

  useEffect(() => {
    if (!focusJobId) return
    openJobDetailById(focusJobId)
    onFocusJobHandled?.()
  }, [focusJobId, onFocusJobHandled, openJobDetailById])

  const buildInputUrl = (fileName, id = getActiveJobId()) => {
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

  const updateQueueOrder = async (order) => {
    setQueueError('')
    setQueueLoading(true)
    try {
      const response = await fetch(`${apiBase}/ocr/queue/reorder`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ order }),
      })
      if (!response.ok) {
        throw new Error('调整队列失败')
      }
      const data = await response.json()
      setQueueInfo(normalizeQueueInfo(data))
    } catch (err) {
      setQueueError(err instanceof Error ? err.message : '调整队列失败')
    } finally {
      setQueueLoading(false)
    }
  }

  const moveQueueItem = (jobId, direction) => {
    const order = queueInfo.queue.map((job) => job.id)
    const index = order.indexOf(jobId)
    if (index === -1) return
    const targetIndex = direction === 'up' ? index - 1 : index + 1
    if (targetIndex < 0 || targetIndex >= order.length) return
    const newOrder = [...order]
    const [removed] = newOrder.splice(index, 1)
    newOrder.splice(targetIndex, 0, removed)
    updateQueueOrder(newOrder)
  }

  const removeQueueItem = async (jobId) => {
    setQueueError('')
    setQueueLoading(true)
    try {
      const response = await fetch(`${apiBase}/ocr/queue/${jobId}`, {
        method: 'DELETE',
      })
      if (!response.ok) {
        throw new Error('移除任务失败')
      }
      await loadQueue()
    } catch (err) {
      setQueueError(err instanceof Error ? err.message : '移除任务失败')
    } finally {
      setQueueLoading(false)
    }
  }

  const renderRunningJobCard = (currentJob, index) => {
    if (!currentJob?.id) return null
    const currentJobId = currentJob.id
    const pauseBusy =
      isActionPending(currentJobId, 'pause') || currentJob.status === 'pausing'
    const resumeBusy = isActionPending(currentJobId, 'resume')
    const cancelBusy = isActionPending(currentJobId, 'cancel')
    const busy = isJobBusy(currentJobId)
    const titlePrefix = runningJobs.length > 1 ? `执行任务 ${index + 1}` : '当前任务'
    return (
      <div key={currentJobId} className="queue-current">
        <div className="queue-current-main">
          <div className="queue-current-head">
            <div className="queue-title">
              {titlePrefix}：<span className="queue-item-id">{currentJobId.slice(0, 8)}</span>
            </div>
            <span className={`queue-status queue-status-${currentJob.status || 'queued'}`}>
              {statusLabel(currentJob.status)}
            </span>
          </div>
          <div className="queue-sub">
            文件数 {currentJob.files?.length || 0} · 页数 {resolvePageCount(currentJob)} 页 · 模型{' '}
            {resolveOcrModel(currentJob.ocr_model)} · 进度 {currentJob.progress || 0}%
          </div>
          <div className="queue-file-line">上传文件：{formatUploadedFiles(currentJob.files)}</div>
          <div className="progress">
            <div className="progress-bar" style={{ width: `${currentJob.progress || 0}%` }} />
          </div>
          <div className="progress-meta">进度 {currentJob.progress || 0}%</div>
          {currentJob.status === 'pausing' && (
            <div className="queue-tip">正在暂停中，任务会保留当前进度。</div>
          )}
          {currentJob.status === 'paused' && (
            <div className="queue-tip">任务已暂停，可继续执行或移出队列。</div>
          )}
          {currentJob.error && <div className="error">{currentJob.error}</div>}
        </div>
        <div className="queue-actions queue-actions-column">
          {(currentJob.status === 'running' || currentJob.status === 'pausing') && (
            <button
              className="button ghost small"
              onClick={() => togglePause('pause', currentJobId)}
              disabled={busy}
            >
              {pauseBusy ? '暂停中...' : '暂停任务'}
            </button>
          )}
          {(currentJob.status === 'paused' || currentJob.status === 'pausing') && (
            <button
              className="button ghost small"
              onClick={() => togglePause('resume', currentJobId)}
              disabled={busy || currentJob.status === 'pausing'}
            >
              {resumeBusy ? '继续中...' : '继续任务'}
            </button>
          )}
          {(currentJob.status === 'queued' || currentJob.status === 'paused') && (
            <button
              className="button danger small"
              onClick={() => cancelJob(currentJobId)}
              disabled={busy}
            >
              {cancelBusy ? '移除中...' : '移出队列'}
            </button>
          )}
          <button
            className="button ghost small"
            onClick={() => openJobDetailById(currentJobId, currentJob)}
            type="button"
          >
            查看详情
          </button>
        </div>
      </div>
    )
  }

  return (
    <>
      <section className="panel">
        <div className="field">
          <label htmlFor="file-input">上传文件（支持图片或 PDF，支持多选）</label>
          <input
            id="file-input"
            ref={fileInputRef}
            className="file-input-hidden"
            type="file"
            accept="image/*,.pdf"
            multiple
            onChange={handleFileInputChange}
          />
          <div
            className={`upload-dropzone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDropzoneDragEnter}
            onDragOver={handleDropzoneDragOver}
            onDragLeave={handleDropzoneDragLeave}
            onDrop={handleDropzoneDrop}
            onClick={openFilePicker}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                openFilePicker()
              }
            }}
            role="button"
            tabIndex={0}
            aria-label="拖拽或点击上传文件"
          >
            <div className="upload-dropzone-main">
              <div className="upload-dropzone-title">
                拖拽文件到此处，或点击选择文件
              </div>
              <div className="upload-dropzone-sub">
                支持 PNG / JPG / JPEG / WEBP / PDF，可多次追加上传，也支持 Ctrl+V 粘贴剪贴板文件
              </div>
            </div>
            <button
              className="button primary"
              onClick={(event) => {
                event.stopPropagation()
                openFilePicker()
              }}
              type="button"
          >
              选择文件
            </button>
          </div>
          {files.length > 0 && (
            <>
              <div className="upload-summary">
                <span>已选择 {files.length} 个文件</span>
                <div className="upload-summary-actions">
                  <span className="upload-hint upload-hint-inline">
                    再次点击“选择文件”会追加到当前列表。
                  </span>
                  <button
                    className="button ghost small"
                    onClick={clearSelectedFiles}
                    type="button"
                  >
                    清空列表
                  </button>
                </div>
              </div>
              <div className="upload-file-list">
                {files.map((file) => (
                  <div key={getLocalFileKey(file)} className="upload-file-card">
                    <button
                      className="upload-file-main upload-file-main-button"
                      onClick={() => openUploadDetail(file)}
                      type="button"
                    >
                      <div className="upload-file-name">{file.name}</div>
                      <div className="upload-file-meta">
                        大小 {formatFileSize(file.size)} · 类型 {file.type || '未知'}
                      </div>
                    </button>
                    <div className="upload-file-actions">
                      <button
                        className="button ghost small"
                        onClick={() => openUploadDetail(file)}
                        type="button"
                      >
                        查看详情
                      </button>
                      <button
                        className="button danger small"
                        onClick={() => removeSelectedFile(file)}
                        type="button"
                      >
                        移除
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="actions ocr-control-row">
          <button
            className={`button advanced-toggle-button ${
              advancedOpen ? 'active' : ''
            }`}
            onClick={() => setAdvancedOpen((prev) => !prev)}
            type="button"
            aria-expanded={advancedOpen}
            aria-controls="ocr-advanced-options"
          >
            <span>高级选项</span>
            <span className={`advanced-chevron ${advancedOpen ? 'open' : ''}`}>▾</span>
          </button>
          <button
            className="button primary ocr-start-button"
            onClick={startOcr}
            disabled={submitting}
          >
            {submitting ? '提交中...' : '开始识别'}
          </button>
        </div>
        {error && <div className="error form-error">{error}</div>}
        {advancedOpen && (
          <div id="ocr-advanced-options" className="advanced-body">
            <div className="fields advanced-fields">
              <div className="field">
                <label htmlFor="ocr-model">识别模型</label>
                <select
                  id="ocr-model"
                  value={ocrModel}
                  onChange={(event) => setOcrModel(event.target.value)}
                >
                  {OCR_MODEL_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <div className="model-default-row">
                  <span className="model-default-text">
                    默认模型：{resolveOcrModel(defaultOcrModel)}
                  </span>
                  <button
                    className="button ghost small"
                    onClick={setAsDefaultOcrModel}
                    type="button"
                    disabled={isCurrentModelDefault}
                    title={
                      isCurrentModelDefault
                        ? '当前模型已是默认模型'
                        : '将当前选择保存为默认模型'
                    }
                  >
                    {isCurrentModelDefault ? '已设为默认' : '设为默认模型'}
                  </button>
                </div>
              </div>
              {isPaddleModel ? (
                <div className="field advanced-inline-hint paddle-options-panel">
                  <div className="advanced-hint-single-line">
                    当前为 PaddleOCR-VL-1.5，DeepSeek-OCR-2 专属参数已隐藏。
                  </div>
                  <div className="paddle-option-section">
                    <div className="paddle-option-title">辅助内容解析</div>
                    <div className="paddle-option-desc">
                      模型自动识别并过滤辅助内容，开启后将恢复解析
                    </div>
                    <div className="paddle-toggle-grid">
                      {PADDLE_AUX_LABEL_OPTIONS.map((item) => (
                        <div className="field toggle paddle-toggle-field" key={item.key}>
                          <label htmlFor={`paddle-aux-${item.key}`}>{item.label}</label>
                          <div className="toggle-row">
                            <label className="switch" htmlFor={`paddle-aux-${item.key}`}>
                              <input
                                id={`paddle-aux-${item.key}`}
                                type="checkbox"
                                checked={Boolean(paddleOptions.auxContent[item.key])}
                                onChange={(event) =>
                                  setPaddleAuxContent(item.key, event.target.checked)
                                }
                              />
                              <span className="slider" />
                            </label>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="paddle-option-section paddle-option-section-divider">
                    <div className="paddle-option-title">模型参数设置</div>
                    <div className="paddle-toggle-grid">
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-use-doc-orientation-classify">图片方向矫正</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-use-doc-orientation-classify">
                            <input
                              id="paddle-use-doc-orientation-classify"
                              type="checkbox"
                              checked={Boolean(paddleOptions.useDocOrientationClassify)}
                              onChange={(event) =>
                                setPaddleOption(
                                  'useDocOrientationClassify',
                                  event.target.checked,
                                )
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-use-doc-unwarping">图片扭曲矫正</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-use-doc-unwarping">
                            <input
                              id="paddle-use-doc-unwarping"
                              type="checkbox"
                              checked={Boolean(paddleOptions.useDocUnwarping)}
                              onChange={(event) =>
                                setPaddleOption('useDocUnwarping', event.target.checked)
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-use-layout-detection">版面分析</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-use-layout-detection">
                            <input
                              id="paddle-use-layout-detection"
                              type="checkbox"
                              checked={Boolean(paddleOptions.useLayoutDetection)}
                              onChange={(event) =>
                                setPaddleOption('useLayoutDetection', event.target.checked)
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-use-chart-recognition">图表识别</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-use-chart-recognition">
                            <input
                              id="paddle-use-chart-recognition"
                              type="checkbox"
                              checked={Boolean(paddleOptions.useChartRecognition)}
                              onChange={(event) =>
                                setPaddleOption('useChartRecognition', event.target.checked)
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-merge-tables">跨页表格合并</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-merge-tables">
                            <input
                              id="paddle-merge-tables"
                              type="checkbox"
                              checked={Boolean(paddleOptions.mergeTables)}
                              onChange={(event) =>
                                setPaddleOption('mergeTables', event.target.checked)
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-relevel-titles">段落标题级别识别</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-relevel-titles">
                            <input
                              id="paddle-relevel-titles"
                              type="checkbox"
                              checked={Boolean(paddleOptions.relevelTitles)}
                              onChange={(event) =>
                                setPaddleOption('relevelTitles', event.target.checked)
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-layout-nms">NMS后处理</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-layout-nms">
                            <input
                              id="paddle-layout-nms"
                              type="checkbox"
                              checked={Boolean(paddleOptions.layoutNms)}
                              onChange={(event) =>
                                setPaddleOption('layoutNms', event.target.checked)
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-restructure-pages">重构多页结果</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-restructure-pages">
                            <input
                              id="paddle-restructure-pages"
                              type="checkbox"
                              checked={Boolean(paddleOptions.restructurePages)}
                              onChange={(event) =>
                                setPaddleOption('restructurePages', event.target.checked)
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-show-formula-number">公式编号展示</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-show-formula-number">
                            <input
                              id="paddle-show-formula-number"
                              type="checkbox"
                              checked={Boolean(paddleOptions.showFormulaNumber)}
                              onChange={(event) =>
                                setPaddleOption('showFormulaNumber', event.target.checked)
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-prettify-markdown">Markdown 美化</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-prettify-markdown">
                            <input
                              id="paddle-prettify-markdown"
                              type="checkbox"
                              checked={Boolean(paddleOptions.prettifyMarkdown)}
                              onChange={(event) =>
                                setPaddleOption('prettifyMarkdown', event.target.checked)
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                      <div className="field toggle paddle-toggle-field">
                        <label htmlFor="paddle-visualize">可视化</label>
                        <div className="toggle-row">
                          <label className="switch" htmlFor="paddle-visualize">
                            <input
                              id="paddle-visualize"
                              type="checkbox"
                              checked={Boolean(paddleOptions.visualize)}
                              onChange={(event) =>
                                setPaddleOption('visualize', event.target.checked)
                              }
                            />
                            <span className="slider" />
                          </label>
                        </div>
                      </div>
                    </div>
                    <div className="paddle-input-grid">
                      <div className="field">
                        <label htmlFor="paddle-layout-threshold">版面区域过滤强度</label>
                        <input
                          id="paddle-layout-threshold"
                          type="number"
                          step="0.05"
                          min="0"
                          max="1"
                          value={paddleOptions.layoutThreshold}
                          onChange={(event) =>
                            setPaddleOption(
                              'layoutThreshold',
                              Math.max(0, Math.min(1, Number(event.target.value) || 0.5)),
                            )
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="paddle-layout-unclip-ratio">扩张系数</label>
                        <input
                          id="paddle-layout-unclip-ratio"
                          type="number"
                          step="0.1"
                          min="0.1"
                          value={paddleOptions.layoutUnclipRatio}
                          onChange={(event) =>
                            setPaddleOption(
                              'layoutUnclipRatio',
                              Math.max(0.1, Number(event.target.value) || 1.0),
                            )
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="paddle-layout-merge-bboxes-mode">
                          版面区域检测的重叠框过滤方式
                        </label>
                        <select
                          id="paddle-layout-merge-bboxes-mode"
                          value={String(paddleOptions.layoutMergeBboxesMode || 'large')}
                          onChange={(event) =>
                            setPaddleOption('layoutMergeBboxesMode', event.target.value)
                          }
                        >
                          <option value="large">保留外部大框</option>
                          <option value="small">保留内部小框</option>
                          <option value="union">保留全部重叠框</option>
                        </select>
                      </div>
                      <div className="field">
                        <label htmlFor="paddle-layout-shape-mode">版面检测结果的几何形状</label>
                        <select
                          id="paddle-layout-shape-mode"
                          value={String(paddleOptions.layoutShapeMode || 'auto')}
                          onChange={(event) =>
                            setPaddleOption('layoutShapeMode', event.target.value)
                          }
                        >
                          <option value="rect">矩形</option>
                          <option value="quad">四边形</option>
                          <option value="poly">多边形</option>
                          <option value="auto">自动</option>
                        </select>
                      </div>
                      <div className="field">
                        <label htmlFor="paddle-prompt-label">提示词类型设置</label>
                        <select
                          id="paddle-prompt-label"
                          value={String(paddleOptions.promptLabel || 'ocr')}
                          onChange={(event) =>
                            setPaddleOption('promptLabel', event.target.value)
                          }
                        >
                          <option value="ocr">通用识别</option>
                          <option value="formula">公式识别</option>
                          <option value="table">表格识别</option>
                          <option value="chart">图表识别</option>
                        </select>
                      </div>
                      <div className="field">
                        <label htmlFor="paddle-repetition-penalty">重复抑制强度</label>
                        <input
                          id="paddle-repetition-penalty"
                          type="number"
                          step="0.1"
                          value={paddleOptions.repetitionPenalty}
                          onChange={(event) =>
                            setPaddleOption(
                              'repetitionPenalty',
                              Number(event.target.value) || 1,
                            )
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="paddle-temperature">识别稳定性</label>
                        <input
                          id="paddle-temperature"
                          type="number"
                          step="0.1"
                          value={paddleOptions.temperature}
                          onChange={(event) =>
                            setPaddleOption('temperature', Number(event.target.value) || 0)
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="paddle-top-p">结果可信范围</label>
                        <input
                          id="paddle-top-p"
                          type="number"
                          step="0.1"
                          value={paddleOptions.topP}
                          onChange={(event) =>
                            setPaddleOption('topP', Number(event.target.value) || 1)
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="paddle-min-pixels">最小图像尺寸</label>
                        <input
                          id="paddle-min-pixels"
                          type="number"
                          value={paddleOptions.minPixels}
                          onChange={(event) =>
                            setPaddleOption(
                              'minPixels',
                              Math.max(1, Number(event.target.value) || 147384),
                            )
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="paddle-max-pixels">最大图像尺寸</label>
                        <input
                          id="paddle-max-pixels"
                          type="number"
                          value={paddleOptions.maxPixels}
                          onChange={(event) =>
                            setPaddleOption(
                              'maxPixels',
                              Math.max(1, Number(event.target.value) || 2822400),
                            )
                          }
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ) : isJianduModel ? (
                <div className="field advanced-inline-hint paddle-options-panel">
                  <div className="advanced-hint-single-line">
                    当前为 DeepJiandu-OCR-v1，DeepSeek 与 Paddle 专属参数已隐藏。
                  </div>
                  <div className="paddle-option-section">
                    <div className="paddle-option-title">简牍整页识别</div>
                    <div className="paddle-option-desc">
                      使用本地检测器与识别器联动，输出检测框标注图、切字图和逐字 Top-K。
                    </div>
                    <div className="paddle-input-grid">
                      <div className="field">
                        <label htmlFor="jiandu-score-thresh">检测阈值</label>
                        <input
                          id="jiandu-score-thresh"
                          type="number"
                          step="0.01"
                          min="0.01"
                          max="0.95"
                          value={jianduOptions.scoreThresh}
                          onChange={(event) =>
                            setJianduOption(
                              'scoreThresh',
                              Math.max(
                                0.01,
                                Math.min(0.95, Number(event.target.value) || 0.3),
                              ),
                            )
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="jiandu-topk-detect">检测候选上限</label>
                        <input
                          id="jiandu-topk-detect"
                          type="number"
                          min="1"
                          max="2000"
                          value={jianduOptions.topkDetect}
                          onChange={(event) =>
                            setJianduOption(
                              'topkDetect',
                              Math.max(1, Number(event.target.value) || 300),
                            )
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="jiandu-nms-iou">NMS IoU</label>
                        <input
                          id="jiandu-nms-iou"
                          type="number"
                          step="0.05"
                          min="0"
                          max="1"
                          value={jianduOptions.nmsIou}
                          onChange={(event) =>
                            setJianduOption(
                              'nmsIou',
                              Math.max(
                                0,
                                Math.min(1, Number(event.target.value) || 0.35),
                              ),
                            )
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="jiandu-min-box-size">最小字符框尺寸</label>
                        <input
                          id="jiandu-min-box-size"
                          type="number"
                          step="1"
                          min="1"
                          value={jianduOptions.minBoxSize}
                          onChange={(event) =>
                            setJianduOption(
                              'minBoxSize',
                              Math.max(1, Number(event.target.value) || 8),
                            )
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="jiandu-max-aspect">最大宽高比</label>
                        <input
                          id="jiandu-max-aspect"
                          type="number"
                          step="0.1"
                          min="1"
                          max="6"
                          value={jianduOptions.maxAspect}
                          onChange={(event) =>
                            setJianduOption(
                              'maxAspect',
                              Math.max(1, Number(event.target.value) || 1.8),
                            )
                          }
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="jiandu-rec-top-k">识别 Top-K</label>
                        <input
                          id="jiandu-rec-top-k"
                          type="number"
                          min="1"
                          max="20"
                          value={jianduOptions.recTopK}
                          onChange={(event) =>
                            setJianduOption(
                              'recTopK',
                              Math.max(1, Number(event.target.value) || 5),
                            )
                          }
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="field advanced-inline-hint">
                    <div className="advanced-hint-single-line">
                      当前为 DeepSeek-OCR-2，PaddleOCR-VL-1.5 与 DeepJiandu 专属参数已隐藏。
                    </div>
                  </div>
                  <div className="field">
                    <label htmlFor="prompt">识别提示词（可选）</label>
                    <textarea
                      id="prompt"
                      value={prompt}
                      onChange={(event) => setPrompt(event.target.value)}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="base-size">基础尺寸</label>
                    <input
                      id="base-size"
                      type="number"
                      min="256"
                      max="2048"
                      value={baseSize}
                      onChange={(event) => setBaseSize(Number(event.target.value))}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="image-size">图像尺寸</label>
                    <input
                      id="image-size"
                      type="number"
                      min="256"
                      max="2048"
                      value={imageSize}
                      onChange={(event) => setImageSize(Number(event.target.value))}
                    />
                  </div>
                  <div className="field toggle">
                    <label htmlFor="crop-mode">裁剪模式</label>
                    <div className="toggle-row">
                      <label className="switch" htmlFor="crop-mode">
                        <input
                          id="crop-mode"
                          type="checkbox"
                          checked={cropMode}
                          onChange={(event) => setCropMode(event.target.checked)}
                        />
                        <span className="slider" />
                      </label>
                      <span className="toggle-text">{cropMode ? '启用' : '关闭'}</span>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </section>

      <Modal
        open={uploadDetailOpen}
        onClose={() => setUploadDetailOpen(false)}
        title="已上传文件详情"
        footer={
          <div className="modal-actions">
            <button
              className="button ghost"
              onClick={() => setUploadDetailOpen(false)}
              type="button"
            >
              关闭
            </button>
            {selectedUploadFile && (
              <button
                className="button danger"
                onClick={() => {
                  removeSelectedFile(selectedUploadFile)
                  setUploadDetailOpen(false)
                }}
                type="button"
              >
                移除该文件
              </button>
            )}
          </div>
        }
      >
        {selectedUploadFile ? (
          <div className="upload-detail-body">
            <div className="upload-detail-meta">
              <div className="upload-detail-meta-label">文件名</div>
              <div className="upload-detail-meta-value">{selectedUploadFile.name}</div>
              <div className="upload-detail-meta-label">文件大小</div>
              <div className="upload-detail-meta-value">
                {formatFileSize(selectedUploadFile.size)}
              </div>
              <div className="upload-detail-meta-label">文件类型</div>
              <div className="upload-detail-meta-value">
                {selectedUploadFile.type || '未知'}
              </div>
              <div className="upload-detail-meta-label">最后修改</div>
              <div className="upload-detail-meta-value">
                {new Date(selectedUploadFile.lastModified).toLocaleString()}
              </div>
            </div>

            <div className="upload-detail-preview">
              {isImageFile(selectedUploadFile) && uploadPreviewUrl && (
                <img
                  className="upload-detail-image"
                  src={uploadPreviewUrl}
                  alt={selectedUploadFile.name}
                />
              )}
              {isPdfFile(selectedUploadFile) && uploadPreviewUrl && (
                <iframe
                  className="upload-detail-frame"
                  src={uploadPreviewUrl}
                  title={selectedUploadFile.name}
                />
              )}
              {!isImageFile(selectedUploadFile) && !isPdfFile(selectedUploadFile) && (
                <div className="upload-detail-empty">
                  当前文件类型暂不支持预览，可直接提交识别。
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="upload-detail-empty">未选择文件。</div>
        )}
      </Modal>

      <section className="panel">
        <div className="queue-header">
          <h2>任务队列</h2>
          <button
            className="button ghost small"
            onClick={loadQueue}
            type="button"
            disabled={queueLoading}
          >
            {queueLoading ? '刷新中...' : '刷新'}
          </button>
        </div>
        <div className="queue-summary">
          <span className="queue-pill">执行中：{runningJobs.length} 个</span>
          <span className="queue-pill">排队任务：{queueInfo.queue.length} 个</span>
        </div>
        {queueError && <div className="error">{queueError}</div>}
        {runningJobs.map((runningJob, index) => renderRunningJobCard(runningJob, index))}
        {queueInfo.queue.length === 0 && runningJobs.length === 0 && !jobInfo && (
          <div className="queue-empty">暂无排队任务。</div>
        )}
        {queueInfo.queue.length > 0 && (
          <div className="queue-list">
            {queueInfo.queue.map((job, index) => (
              <div key={job.id} className="queue-item">
                <div className="queue-item-main">
                  <div className="queue-item-head">
                    <div className="queue-title">队列 {index + 1}</div>
                    <span className={`queue-status queue-status-${job.status || 'queued'}`}>
                      {statusLabel(job.status || 'queued')}
                    </span>
                  </div>
                  <div className="queue-sub">
                    任务 <span className="queue-item-id">{job.id.slice(0, 8)}</span> · 文件数{' '}
                    {job.files?.length || 0} · 页数 {resolvePageCount(job)} 页 · 模型{' '}
                    {resolveOcrModel(job.ocr_model)}
                  </div>
                  <div className="queue-file-line">
                    上传文件：{formatUploadedFiles(job.files)}
                  </div>
                </div>
                <div className="queue-actions queue-actions-compact">
                  <button
                    className="button ghost small"
                    onClick={() => openJobDetailById(job.id, job)}
                    type="button"
                  >
                    查看详情
                  </button>
                  <button
                    className="button ghost small"
                    onClick={() => moveQueueItem(job.id, 'up')}
                    disabled={index === 0 || queueLoading}
                    type="button"
                  >
                    上移
                  </button>
                  <button
                    className="button ghost small"
                    onClick={() => moveQueueItem(job.id, 'down')}
                    disabled={index === queueInfo.queue.length - 1 || queueLoading}
                    type="button"
                  >
                    下移
                  </button>
                  <button
                    className="button danger small"
                    onClick={() => removeQueueItem(job.id)}
                    disabled={queueLoading}
                    type="button"
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <Modal
        open={detailOpen}
        onClose={() => setDetailOpen(false)}
        title={jobInfo ? `任务详情 · ${jobInfo.id}` : '任务详情'}
      >
        {jobInfo?.files?.length > 0 && (
          <div className="input-section">
            <div className="input-title">上传文件</div>
            <div className="input-grid">
              {jobInfo.files.map((fileName) => renderInputPreview(fileName, jobInfo.id))}
            </div>
          </div>
        )}
        {jobInfo && (
          <div className="detail-header">
            <div>
              <div className="detail-title">
                状态：{STATUS_TEXT[jobInfo.status] || jobInfo.status}
              </div>
              <div className="detail-sub">
                进度 {jobInfo.progress || 0}% · 文件数 {jobInfo.files?.length || 0} · 模型{' '}
                {resolveOcrModel(jobInfo.ocr_model)}
              </div>
            </div>
            <div className="detail-actions">
              {(jobInfo.status === 'running' || jobInfo.status === 'pausing') && (
                <button
                  className="button ghost small"
                  onClick={() => togglePause('pause', jobInfo.id)}
                  disabled={isJobBusy(jobInfo.id)}
                  type="button"
                >
                  {jobInfo.status === 'pausing' || isActionPending(jobInfo.id, 'pause')
                    ? '暂停中...'
                    : '暂停任务'}
                </button>
              )}
              {(jobInfo.status === 'paused' || jobInfo.status === 'pausing') && (
                <button
                  className="button ghost small"
                  onClick={() => togglePause('resume', jobInfo.id)}
                  disabled={isJobBusy(jobInfo.id) || jobInfo.status === 'pausing'}
                  type="button"
                >
                  {isActionPending(jobInfo.id, 'resume') ? '继续中...' : '继续任务'}
                </button>
              )}
              {(jobInfo.status === 'queued' || jobInfo.status === 'paused') && (
                <button
                  className="button danger small"
                  onClick={() => cancelJob(jobInfo.id)}
                  disabled={isJobBusy(jobInfo.id)}
                  type="button"
                >
                  {isActionPending(jobInfo.id, 'cancel') ? '移除中...' : '移出队列'}
                </button>
              )}
            </div>
          </div>
        )}
        <ResultsPanel
          title=""
          items={results}
          apiBase={apiBase}
          fallbackJobId={jobId}
          emptyText="暂无识别结果。"
          onItemUpdated={handleItemUpdated}
        />
      </Modal>
    </>
  )
}

export default OcrPage
