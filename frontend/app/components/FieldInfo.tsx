'use client';

import { useEffect, useId, useRef, useState } from 'react';

type Props = {
  text: string;
  id?: string;
};

export default function FieldInfo({ text, id }: Props) {
  const autoId = useId();
  const tooltipId = id ?? `field-info-${autoId.replace(/[:]/g, '-')}`;
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      const el = rootRef.current;
      if (!el) return;
      if (event.target instanceof Node && el.contains(event.target)) return;
      setOpen(false);
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [open]);

  return (
    <span ref={rootRef} className="fieldInfoWrap" data-open={open ? 'true' : 'false'}>
      <button
        type="button"
        className="fieldInfo"
        aria-label={`Field Information: ${text}`}
        aria-describedby={tooltipId}
        aria-expanded={open}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onClick={() => setOpen((prev) => !prev)}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            setOpen(false);
          }
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            setOpen((prev) => !prev);
          }
        }}
      >
        <span aria-hidden="true">i</span>
      </button>
      <span id={tooltipId} className="fieldInfo__tooltip" role="note" aria-hidden={!open}>
        {text}
      </span>
    </span>
  );
}
