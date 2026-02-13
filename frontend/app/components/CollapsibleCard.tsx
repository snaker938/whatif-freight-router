'use client';

import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';

type Props = {
  title: ReactNode;
  hint?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  dataTutorialId?: string;
  defaultCollapsed?: boolean;
};

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 20 20"
      width="16"
      height="16"
      className={`collapsibleCard__chevron ${open ? 'isOpen' : ''}`}
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M5 8l5 5 5-5" />
    </svg>
  );
}

export default function CollapsibleCard({
  title,
  hint,
  children,
  className = '',
  bodyClassName = '',
  dataTutorialId,
  defaultCollapsed = true,
}: Props) {
  const [open, setOpen] = useState(!defaultCollapsed);
  const contentInnerRef = useRef<HTMLDivElement | null>(null);
  const [contentHeight, setContentHeight] = useState(0);
  const contentId = useId();
  const headerId = useId();

  const measureHeight = useCallback(() => {
    const node = contentInnerRef.current;
    if (!node) return;
    setContentHeight(Math.ceil(node.scrollHeight));
  }, []);

  useLayoutEffect(() => {
    measureHeight();
  }, [measureHeight]);

  useEffect(() => {
    setOpen(!defaultCollapsed);
  }, [defaultCollapsed]);

  useEffect(() => {
    if (open) measureHeight();
  }, [open, measureHeight]);

  useEffect(() => {
    const node = contentInnerRef.current;
    if (!node) return;

    measureHeight();

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(() => {
        measureHeight();
      });
      observer.observe(node);
      return () => observer.disconnect();
    }

    const onResize = () => measureHeight();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [measureHeight]);

  useEffect(() => {
    if (typeof document === 'undefined' || !('fonts' in document)) return;
    let cancelled = false;
    // Re-measure after font faces load to avoid clipped text in expanded cards.
    void (document as Document & { fonts?: FontFaceSet }).fonts?.ready.then(() => {
      if (cancelled) return;
      measureHeight();
    });
    return () => {
      cancelled = true;
    };
  }, [measureHeight]);

  return (
    <section
      className={`card collapsibleCard ${open ? 'isOpen' : ''} ${className}`.trim()}
      {...(dataTutorialId ? { 'data-tutorial-id': dataTutorialId } : {})}
    >
      <button
        type="button"
        id={headerId}
        className="collapsibleCard__header"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        aria-controls={contentId}
      >
        <span className="sectionTitle">{title}</span>
        <ChevronIcon open={open} />
      </button>

      <div
        id={contentId}
        role="region"
        aria-labelledby={headerId}
        className="collapsibleCard__content"
        aria-hidden={!open}
        style={{
          maxHeight: open ? `${Math.max(contentHeight + 32, 32)}px` : '0px',
          opacity: open ? 1 : 0,
          pointerEvents: open ? 'auto' : 'none',
          visibility: open ? 'visible' : 'hidden',
        }}
      >
        <div ref={contentInnerRef} className={`collapsibleCard__contentInner ${bodyClassName}`.trim()}>
          {hint ? <div className="sectionHint">{hint}</div> : null}
          {children}
        </div>
      </div>
    </section>
  );
}

