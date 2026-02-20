'use client';

import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';

type Props = {
  text: string;
  id?: string;
};

export default function FieldInfo({ text, id }: Props) {
  const autoId = useId();
  const tooltipId = id ?? `field-info-${autoId.replace(/[:]/g, '-')}`;
  const [open, setOpen] = useState(false);
  const [tooltipSide, setTooltipSide] = useState<'right' | 'left'>('right');
  const [tooltipMaxWidth, setTooltipMaxWidth] = useState<number | null>(null);
  const rootRef = useRef<HTMLSpanElement | null>(null);
  const tooltipRef = useRef<HTMLSpanElement | null>(null);

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

  useLayoutEffect(() => {
    if (!open) return;
    const root = rootRef.current;
    const tooltip = tooltipRef.current;
    if (!root || !tooltip) return;

    const updatePlacement = () => {
      const margin = 12;
      const gap = 22;
      const rootRect = root.getBoundingClientRect();
      const tooltipRect = tooltip.getBoundingClientRect();
      const viewportWidth = window.innerWidth;

      const availableRight = Math.max(120, viewportWidth - (rootRect.right + gap + margin));
      const availableLeft = Math.max(120, rootRect.left - gap - margin);
      const preferredMax = 260;

      const fitsRight = rootRect.right + gap + tooltipRect.width <= viewportWidth - margin;
      const fitsLeft = rootRect.left - gap - tooltipRect.width >= margin;

      if (!fitsRight && (fitsLeft || availableLeft > availableRight)) {
        setTooltipSide('left');
        setTooltipMaxWidth(Math.min(preferredMax, Math.round(availableLeft)));
      } else {
        setTooltipSide('right');
        setTooltipMaxWidth(Math.min(preferredMax, Math.round(availableRight)));
      }
    };

    updatePlacement();
    window.addEventListener('resize', updatePlacement);
    window.addEventListener('scroll', updatePlacement, true);
    return () => {
      window.removeEventListener('resize', updatePlacement);
      window.removeEventListener('scroll', updatePlacement, true);
    };
  }, [open]);

  return (
    <span
      ref={rootRef}
      className="fieldInfoWrap"
      data-open={open ? 'true' : 'false'}
      data-side={tooltipSide}
    >
      <button
        type="button"
        className="fieldInfo"
        aria-label={`Field Information: ${text}`}
        aria-controls={tooltipId}
        aria-describedby={open ? tooltipId : undefined}
        aria-haspopup="true"
        aria-expanded={open}
        onFocus={() => setOpen(true)}
        onBlur={(event) => {
          if (event.relatedTarget instanceof Node && rootRef.current?.contains(event.relatedTarget)) {
            return;
          }
          setOpen(false);
        }}
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
      <span
        ref={tooltipRef}
        id={tooltipId}
        className="fieldInfo__tooltip"
        role="note"
        aria-hidden={!open}
        style={tooltipMaxWidth ? { maxWidth: `${tooltipMaxWidth}px` } : undefined}
      >
        {text}
      </span>
    </span>
  );
}
