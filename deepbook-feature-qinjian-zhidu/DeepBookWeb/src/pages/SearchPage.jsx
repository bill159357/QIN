import { useState } from 'react'
import Modal from '../components/Modal'
import ResultsPanel from '../components/ResultsPanel'

const escapeHtml = (value) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')

const PAGE_SIZE_OPTIONS = [10, 20, 50]

const formatDate = (value) => {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function SearchPage({ apiBase }) {
  const [searchQuery, setSearchQuery] = useState('')
  const [jobFilter, setJobFilter] = useState('')
  const [fileFilter, setFileFilter] = useState('')
  const [createdFrom, setCreatedFrom] = useState('')
  const [createdTo, setCreatedTo] = useState('')
  const [pageSize, setPageSize] = useState(20)
  const [searchResults, setSearchResults] = useState([])
  const [searchTotal, setSearchTotal] = useState(0)
  const [searchPage, setSearchPage] = useState(1)
  const [searchTotalPages, setSearchTotalPages] = useState(0)
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [searchRan, setSearchRan] = useState(false)
  const [selectedDoc, setSelectedDoc] = useState(null)
  const [docLoading, setDocLoading] = useState(false)
  const [docError, setDocError] = useState('')
  const [detailOpen, setDetailOpen] = useState(false)

  const handleItemUpdated = (_prevItem, updated) => {
    if (updated) {
      setSelectedDoc(updated)
    }
  }

  const renderSnippet = (snippet) => {
    if (!snippet) return ''
    const safe = escapeHtml(snippet)
    return safe.replace(/\[/g, '<mark>').replace(/\]/g, '</mark>')
  }

  const formatScore = (value) => {
    const num = Number(value)
    return Number.isFinite(num) ? num.toFixed(3) : '-'
  }

  const runSearch = async ({ targetPage = 1, targetPageSize = pageSize } = {}) => {
    const query = searchQuery.trim()
    const normalizedJob = jobFilter.trim()
    const normalizedFile = fileFilter.trim()
    setSearchError('')
    setSearchRan(true)
    setSearchLoading(true)
    try {
      const params = new URLSearchParams()
      if (query) params.set('q', query)
      if (normalizedJob) params.set('job_id', normalizedJob)
      if (normalizedFile) params.set('file_name', normalizedFile)
      if (createdFrom) params.set('created_from', createdFrom)
      if (createdTo) params.set('created_to', createdTo)
      params.set('page', String(targetPage))
      params.set('page_size', String(targetPageSize))

      const response = await fetch(
        `${apiBase}/search?${params.toString()}`,
      )
      if (!response.ok) {
        throw new Error('检索失败，请确认后端已启动。')
      }
      const data = await response.json()
      setSearchResults(data.results || [])
      setSearchTotal(Number(data.total) || 0)
      setSearchPage(Number(data.page) || targetPage)
      const totalPages =
        Number(data.total_pages) ||
        Math.ceil((Number(data.total) || 0) / Math.max(1, targetPageSize))
      setSearchTotalPages(totalPages)
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : '检索失败')
    } finally {
      setSearchLoading(false)
    }
  }

  const handleSearchSubmit = async (event) => {
    event.preventDefault()
    await runSearch({ targetPage: 1 })
  }

  const resetFilters = () => {
    setSearchQuery('')
    setJobFilter('')
    setFileFilter('')
    setCreatedFrom('')
    setCreatedTo('')
    setPageSize(20)
    setSearchResults([])
    setSearchTotal(0)
    setSearchPage(1)
    setSearchTotalPages(0)
    setSearchError('')
    setSearchRan(false)
  }

  const changePage = (targetPage) => {
    if (searchLoading) return
    runSearch({ targetPage })
  }

  const openDocument = async (docId) => {
    if (!docId) return
    setDocError('')
    setDocLoading(true)
    setDetailOpen(true)
    try {
      const response = await fetch(`${apiBase}/documents/${docId}`)
      if (!response.ok) {
        throw new Error('加载文档失败')
      }
      const data = await response.json()
      setSelectedDoc(data.document)
    } catch (err) {
      setDocError(err instanceof Error ? err.message : '加载文档失败')
    } finally {
      setDocLoading(false)
    }
  }

  const closeDetail = () => {
    setDetailOpen(false)
    setSelectedDoc(null)
  }

  return (
    <>
      <section className="panel search-panel">
        <div className="panel-head-row">
          <h2>全文检索</h2>
          <div className="panel-head-tip">条件筛选 · 分页浏览</div>
        </div>
        <form className="search-filter-form" onSubmit={handleSearchSubmit}>
          <div className="search-bar">
            <input
              className="search-input"
              type="text"
              placeholder="关键词（支持中文和英文）"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
            />
            <button className="button primary" type="submit" disabled={searchLoading}>
              {searchLoading ? '检索中...' : '搜索'}
            </button>
            <button className="button ghost" type="button" onClick={resetFilters}>
              重置条件
            </button>
          </div>
          <div className="filter-grid search-filter-grid">
            <label className="field">
              <span>任务 ID</span>
              <input
                type="text"
                placeholder="按任务号筛选"
                value={jobFilter}
                onChange={(event) => setJobFilter(event.target.value)}
              />
            </label>
            <label className="field">
              <span>文件名</span>
              <input
                type="text"
                placeholder="按文件名筛选"
                value={fileFilter}
                onChange={(event) => setFileFilter(event.target.value)}
              />
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
                  if (searchRan) {
                    runSearch({ targetPage: 1, targetPageSize: nextSize })
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
        {searchError && <div className="error">{searchError}</div>}
        {searchRan && !searchLoading && (
          <div className="result-summary">
            共 {searchTotal} 条结果，第 {searchPage} / {Math.max(searchTotalPages, 1)} 页
          </div>
        )}
        {searchResults.length > 0 ? (
          <div className="search-results">
            {searchResults.map((item) => (
              <div key={item.doc_id} className="search-item">
                <div className="search-meta">
                  <div className="search-title-group">
                    <div className="search-title">{item.title || '未命名文件'}</div>
                    <div className="search-sub">
                      任务 {item.job_id?.slice(0, 8) || '-'} · {formatDate(item.created_at)}
                    </div>
                  </div>
                  <div className="search-score">相关度 {formatScore(item.score)}</div>
                </div>
                {item.snippet && (
                  <div
                    className="search-snippet"
                    dangerouslySetInnerHTML={{ __html: renderSnippet(item.snippet) }}
                  />
                )}
                <div className="search-actions">
                  <button
                    className="button ghost small"
                    onClick={() => openDocument(item.doc_id)}
                    type="button"
                  >
                    查看详情
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          searchRan &&
          !searchLoading && (
            <div className="search-empty">当前条件下暂无检索结果。</div>
          )
        )}
        {searchRan && searchTotal > 0 && (
          <div className="pagination">
            <button
              className="button ghost small"
              type="button"
              onClick={() => changePage(searchPage - 1)}
              disabled={searchLoading || searchPage <= 1}
            >
              上一页
            </button>
            <div className="pagination-info">
              第 {searchPage} / {Math.max(searchTotalPages, 1)} 页
            </div>
            <button
              className="button ghost small"
              type="button"
              onClick={() => changePage(searchPage + 1)}
              disabled={searchLoading || searchPage >= searchTotalPages}
            >
              下一页
            </button>
          </div>
        )}
      </section>

      <Modal open={detailOpen} onClose={closeDetail} title="检索详情">
        {docError && <div className="error">{docError}</div>}
        {docLoading && <div className="search-loading">文档加载中...</div>}
        {!docLoading && selectedDoc && (
          <ResultsPanel
            title=""
            items={[selectedDoc]}
            apiBase={apiBase}
            fallbackJobId={selectedDoc.job_id}
            emptyText=""
            onItemUpdated={handleItemUpdated}
          />
        )}
      </Modal>
    </>
  )
}

export default SearchPage
