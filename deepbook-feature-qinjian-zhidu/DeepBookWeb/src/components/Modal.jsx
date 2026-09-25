import { useEffect } from 'react'

let scrollLockCount = 0
let savedBodyOverflow = ''
let savedBodyPaddingRight = ''

function Modal({ open, title, onClose, children, footer, className = '' }) {
  useEffect(() => {
    if (!open) return undefined
    const handler = (event) => {
      if (event.key === 'Escape') {
        onClose?.()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  useEffect(() => {
    if (!open) return undefined
    const { body, documentElement } = document
    if (scrollLockCount === 0) {
      savedBodyOverflow = body.style.overflow
      savedBodyPaddingRight = body.style.paddingRight
      const scrollbarWidth = window.innerWidth - documentElement.clientWidth

      body.style.overflow = 'hidden'
      if (scrollbarWidth > 0) {
        body.style.paddingRight = `${scrollbarWidth}px`
      }
    }
    scrollLockCount += 1

    return () => {
      scrollLockCount = Math.max(0, scrollLockCount - 1)
      if (scrollLockCount === 0) {
        body.style.overflow = savedBodyOverflow
        body.style.paddingRight = savedBodyPaddingRight
      }
    }
  }, [open])

  if (!open) return null
  const cardClassName = ['modal-card', className].filter(Boolean).join(' ')

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className={cardClassName} onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">{title}</div>
          <button className="modal-close" onClick={onClose} type="button">
            ×
          </button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-footer">{footer}</div>}
      </div>
    </div>
  )
}

export default Modal
