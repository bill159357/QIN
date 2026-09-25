import { useEffect, useState } from 'react'
import Modal from '../components/Modal'

const PAGE_SIZE = 36
const LEGACY_REFERENCE_NOTE =
  'reference_text is reconstructed from labeled characters by right-to-left columns, then top-to-bottom in each column.'
const REFERENCE_NOTE_ZH =
  '参考释文由已标注字符按列重建：各列自右向左排列，每列内再按自上而下顺序生成。'

const normalizePreviewText = (value, maxChars = 44) => {
  const text = String(value || '').replace(/\s+/g, '').trim()
  if (!text) return '暂无参考释文'
  if (text.length <= maxChars) return text
  return `${text.slice(0, maxChars - 1)}…`
}

const formatCount = (value, suffix) => `${Number(value || 0)} ${suffix}`

const resolveManifestNote = (value) => {
  const text = String(value || '').trim()
  if (!text) return ''
  if (text === LEGACY_REFERENCE_NOTE) {
    return REFERENCE_NOTE_ZH
  }
  return text
}

function GalleryPage({ apiBase }) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [queryInput, setQueryInput] = useState('')
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(1)
  const [hasNext, setHasNext] = useState(false)
  const [total, setTotal] = useState(0)
  const [manifest, setManifest] = useState(null)
  const [detailOpen, setDetailOpen] = useState(false)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState('')
  const [detailItem, setDetailItem] = useState(null)
  const [showBoxes, setShowBoxes] = useState(true)

  useEffect(() => {
    let alive = true
    const controller = new AbortController()

    const loadItems = async () => {
      setLoading(true)
      setError('')
      try {
        const params = new URLSearchParams({
          page: String(page),
          page_size: String(PAGE_SIZE),
        })
        if (query) {
          params.set('q', query)
        }
        const response = await fetch(`${apiBase}/jiandu/gallery?${params.toString()}`, {
          signal: controller.signal,
        })
        if (!response.ok) {
          throw new Error(`请求失败：${response.status}`)
        }
        const data = await response.json()
        if (!alive) return
        const nextItems = Array.isArray(data.items) ? data.items : []
        setItems((prev) => {
          if (page === 1) return nextItems
          const seen = new Set(prev.map((item) => item.id))
          return [...prev, ...nextItems.filter((item) => item?.id && !seen.has(item.id))]
        })
        setHasNext(Boolean(data.has_next))
        setTotal(Number(data.total || 0))
        setManifest(data.manifest || null)
      } catch (loadError) {
        if (!alive || loadError?.name === 'AbortError') return
        setError(loadError.message || '藏馆数据加载失败，请稍后重试。')
      } finally {
        if (alive) {
          setLoading(false)
        }
      }
    }

    loadItems()
    return () => {
      alive = false
      controller.abort()
    }
  }, [apiBase, page, query])

  const handleSearchSubmit = (event) => {
    event.preventDefault()
    const nextQuery = queryInput.trim()
    setPage(1)
    setQuery(nextQuery)
  }

  const handleReset = () => {
    setQueryInput('')
    setQuery('')
    setPage(1)
  }

  const openDetail = async (item) => {
    const itemId = String(item?.id || '').trim()
    if (!itemId) return
    setDetailOpen(true)
    setDetailLoading(true)
    setDetailError('')
    setShowBoxes(true)
    setDetailItem(item)

    try {
      const response = await fetch(`${apiBase}/jiandu/gallery/items/${encodeURIComponent(itemId)}`)
      if (!response.ok) {
        throw new Error(`详情加载失败：${response.status}`)
      }
      const data = await response.json()
      setDetailItem(data.item || item)
    } catch (loadError) {
      setDetailError(loadError.message || '详情加载失败，请稍后再试。')
    } finally {
      setDetailLoading(false)
    }
  }

  const closeDetail = () => {
    setDetailOpen(false)
    setDetailLoading(false)
    setDetailError('')
    setDetailItem(null)
  }

  const detailWidth = Number(detailItem?.image_width || 1)
  const detailHeight = Number(detailItem?.image_height || 1)
  const detailBoxes = Array.isArray(detailItem?.boxes) ? detailItem.boxes : []
  const detailTitle = detailItem?.title || (detailItem?.id ? `简牍 ${detailItem.id}` : '简牍详情')
  const manifestNote = resolveManifestNote(manifest?.note)

  return (
    <>
      <section className="panel">
        <div className="gallery-toolbar">
          <div>
            <h2>简牍藏馆</h2>
            <p>
              基于 `deepjiandu_fullpage_eval_test200` 评测集构建，当前收录 {total || 0} 条秦简样例，
              支持瀑布流浏览与详情复核。
            </p>
          </div>
          <form className="gallery-search" onSubmit={handleSearchSubmit}>
            <input
              className="gallery-search-input"
              type="text"
              value={queryInput}
              onChange={(event) => setQueryInput(event.target.value)}
              placeholder="按编号或释文关键词筛选"
            />
            <button className="button primary" type="submit">
              检索
            </button>
            <button className="button ghost" type="button" onClick={handleReset}>
              重置
            </button>
          </form>
        </div>

        <div className="gallery-summary-strip">
          <span>{formatCount(total, '条样例')}</span>
          <span>{formatCount(manifest?.copied_images, '张图像')}</span>
          <span>{formatCount(manifest?.empty_pages, '页空白')}</span>
          <span>{manifest?.sample_mode ? `采样方式：${manifest.sample_mode}` : '采样方式：随机抽样'}</span>
        </div>

        {manifestNote && <div className="gallery-note">{manifestNote}</div>}

        {error && <div className="gallery-empty-state gallery-error">{error}</div>}

        {!error && items.length === 0 && !loading && (
          <div className="gallery-empty-state">当前筛选条件下没有可展示的简牍样例。</div>
        )}

        {items.length > 0 && (
          <div className="gallery-masonry">
            {items.map((item) => (
              <article className="gallery-card" key={item.id}>
                <div className="gallery-card-media">
                  <button
                    className="gallery-card-media-button"
                    type="button"
                    onClick={() => openDetail(item)}
                    aria-label={`查看${item.title || `简牍 ${item.id}`}详情`}
                  >
                    <img
                      src={`${item.thumbnail_url}?width=420`}
                      alt={item.title || `简牍 ${item.id}`}
                      loading="lazy"
                    />
                  </button>
                  <div className="gallery-card-badge">{formatCount(item.box_count, '个检测框')}</div>
                </div>

                <div className="gallery-card-body">
                  <div className="gallery-card-header">
                    <div>
                      <h3>{item.title || `简牍 ${item.id}`}</h3>
                      <div className="gallery-card-sub">编号 {item.id}</div>
                    </div>
                    <div className="gallery-card-metrics">
                      <span>{formatCount(item.num_chars, '字')}</span>
                      <span>
                        {item.image_width} × {item.image_height}
                      </span>
                    </div>
                  </div>

                  <div className="gallery-card-preview">{normalizePreviewText(item.preview_text)}</div>

                  <button className="button primary gallery-card-action" type="button" onClick={() => openDetail(item)}>
                    查看详情
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}

        <div className="gallery-load-more">
          {loading && <span className="gallery-loading-text">正在加载藏馆样例...</span>}
          {!loading && hasNext && (
            <button className="button ghost" type="button" onClick={() => setPage((prev) => prev + 1)}>
              加载更多
            </button>
          )}
          {!loading && items.length > 0 && !hasNext && <span className="gallery-loading-text">已加载全部样例。</span>}
        </div>
      </section>

      <Modal
        open={detailOpen}
        onClose={closeDetail}
        title={detailTitle}
        className="gallery-detail-modal"
        footer={
          <div className="modal-actions">
            <button className="button ghost" type="button" onClick={() => setShowBoxes((prev) => !prev)}>
              {showBoxes ? '隐藏检测框' : '显示检测框'}
            </button>
            {detailItem?.image_url && (
              <a
                className="button ghost gallery-link-button"
                href={detailItem.image_url}
                target="_blank"
                rel="noreferrer"
              >
                查看原图
              </a>
            )}
            <button className="button primary" type="button" onClick={closeDetail}>
              关闭
            </button>
          </div>
        }
      >
        {detailLoading && <div className="gallery-empty-state">正在加载详情...</div>}
        {!detailLoading && detailError && <div className="gallery-empty-state gallery-error">{detailError}</div>}

        {!detailLoading && !detailError && detailItem && (
          <div className="gallery-detail-layout">
            <div className="gallery-detail-visual">
              <div
                className="gallery-detail-image-frame"
                style={{ aspectRatio: `${detailWidth} / ${detailHeight}` }}
              >
                <img src={detailItem.image_url} alt={detailTitle} />
                {showBoxes &&
                  detailBoxes.map((box) => (
                    <div
                      key={`${detailItem.id}-${box.index}-${box.char}`}
                      className="gallery-detail-box"
                      style={{
                        left: `${(Number(box.x1 || 0) / detailWidth) * 100}%`,
                        top: `${(Number(box.y1 || 0) / detailHeight) * 100}%`,
                        width: `${(Math.max(0, Number(box.x2 || 0) - Number(box.x1 || 0)) / detailWidth) * 100}%`,
                        height: `${(Math.max(0, Number(box.y2 || 0) - Number(box.y1 || 0)) / detailHeight) * 100}%`,
                      }}
                    >
                      <span>{box.char || box.index}</span>
                    </div>
                  ))}
              </div>
            </div>

            <div className="gallery-detail-info">
              <div className="gallery-detail-meta">
                <span>编号 {detailItem.id}</span>
                <span>{formatCount(detailItem.box_count, '个检测框')}</span>
                <span>{formatCount(detailItem.num_chars, '字')}</span>
                <span>{detailItem.split || 'test'} 集</span>
              </div>

              <section className="gallery-detail-section">
                <div className="gallery-detail-label">参考释文</div>
                <div className="gallery-detail-text">{detailItem.reference_text || '暂无释文数据'}</div>
              </section>

              <section className="gallery-detail-section">
                <div className="gallery-detail-label">译文说明</div>
                <div className="gallery-detail-text muted">
                  {detailItem.translation_text ||
                    detailItem.translation_note ||
                    '当前数据集中未提供现代汉语译文。'}
                </div>
              </section>

              <section className="gallery-detail-section">
                <div className="gallery-detail-label">检测框字符序列</div>
                <div className="gallery-char-grid">
                  {detailBoxes.map((box) => (
                    <div className="gallery-char-chip" key={`chip-${detailItem.id}-${box.index}`}>
                      <span className="gallery-char-index">{box.index}</span>
                      <span className="gallery-char-value">{box.char || '□'}</span>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          </div>
        )}
      </Modal>
    </>
  )
}

export default GalleryPage
