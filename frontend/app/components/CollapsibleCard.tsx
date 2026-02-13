'use client';

import { useState, type ReactNode } from 'react';

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

  return (
    <section
      className={`card collapsibleCard ${open ? 'isOpen' : ''} ${className}`.trim()}
      {...(dataTutorialId ? { 'data-tutorial-id': dataTutorialId } : {})}
    >
      <button
        type="button"
        className="collapsibleCard__header"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
      >
        <span className="sectionTitle">{title}</span>
        <ChevronIcon open={open} />
      </button>

      <div className="collapsibleCard__content" aria-hidden={!open}>
        <div className={`collapsibleCard__contentInner ${bodyClassName}`.trim()}>
          {hint ? <div className="sectionHint">{hint}</div> : null}
          {children}
        </div>
      </div>
    </section>
  );
}

