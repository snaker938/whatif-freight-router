'use client';
// frontend/app/components/Select.tsx

import type React from 'react';
import { useEffect, useId, useMemo, useRef, useState } from 'react';

export type SelectOption<T extends string> = {
  value: T;
  label: string;
  description?: string;
};

type Props<T extends string> = {
  id?: string;
  value: T;
  options: SelectOption<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
  placeholder?: string;
  ariaLabel: string;
  showSelectionHint?: boolean;
  tutorialId?: string;
  tutorialAction?: string;
  tutorialActionPrefix?: string;
};

function ChevronDownIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.25"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

export default function Select<T extends string>({
  id: providedId,
  value,
  options,
  onChange,
  disabled,
  placeholder,
  ariaLabel,
  showSelectionHint = false,
  tutorialId,
  tutorialAction,
  tutorialActionPrefix,
}: Props<T>) {
  const generatedId = useId();
  const id = (providedId ?? generatedId).replace(/[:]/g, '-');
  const rootRef = useRef<HTMLDivElement | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const wasOpenRef = useRef(false);

  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number>(() =>
    Math.max(
      0,
      options.findIndex((o) => o.value === value),
    ),
  );
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const isDisabled = Boolean(disabled) || options.length === 0;

  const selected = useMemo(() => options.find((o) => o.value === value) ?? null, [options, value]);

  useEffect(() => {
    const selectedIndex = options.findIndex((o) => o.value === value);
    if (selectedIndex >= 0) {
      setActiveIndex(selectedIndex);
      return;
    }
    setActiveIndex((prev) => Math.max(0, Math.min(prev, Math.max(options.length - 1, 0))));
  }, [options, value]);

  useEffect(() => {
    optionRefs.current = optionRefs.current.slice(0, options.length);
  }, [options.length]);

  useEffect(() => {
    function onDocDown(e: MouseEvent | PointerEvent | TouchEvent) {
      const el = rootRef.current;
      if (!el) return;
      if (e.target instanceof Node && el.contains(e.target)) return;
      setOpen(false);
    }

    document.addEventListener('pointerdown', onDocDown);
    return () => document.removeEventListener('pointerdown', onDocDown);
  }, []);

  useEffect(() => {
    if (!open) {
      if (wasOpenRef.current) {
        buttonRef.current?.focus?.();
      }
      wasOpenRef.current = false;
      return;
    }
    wasOpenRef.current = true;
    const t = window.setTimeout(() => menuRef.current?.focus?.(), 0);
    return () => window.clearTimeout(t);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const node = optionRefs.current[activeIndex];
    node?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex, open]);

  function commit(idx: number) {
    const opt = options[idx];
    if (!opt) return;
    onChange(opt.value);
    setOpen(false);
  }

  function step(delta: number) {
    setActiveIndex((i) => {
      const next = i + delta;
      return Math.max(0, Math.min(options.length - 1, next));
    });
  }

  function onButtonKeyDown(e: React.KeyboardEvent<HTMLButtonElement>) {
    if (isDisabled) return;
    const selectedIndex = options.findIndex((opt) => opt.value === value);

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setOpen(true);
      if (open) step(1);
      else setActiveIndex(Math.max(selectedIndex, 0));
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setOpen(true);
      if (open) step(-1);
      else setActiveIndex(Math.max(selectedIndex, 0));
      return;
    }
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (open) commit(activeIndex);
      else setOpen(true);
      return;
    }
    if (e.key === 'Home') {
      e.preventDefault();
      setOpen(true);
      setActiveIndex(0);
      return;
    }
    if (e.key === 'End') {
      e.preventDefault();
      setOpen(true);
      setActiveIndex(Math.max(options.length - 1, 0));
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      setOpen(false);
    }
  }

  function onMenuKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      step(1);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      step(-1);
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      commit(activeIndex);
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      setOpen(false);
      return;
    }
    if (e.key === 'Tab') {
      setOpen(false);
      return;
    }
    if (e.key === 'Home') {
      e.preventDefault();
      setActiveIndex(0);
      return;
    }
    if (e.key === 'End') {
      e.preventDefault();
      setActiveIndex(Math.max(options.length - 1, 0));
      return;
    }
  }

  return (
    <div
      ref={rootRef}
      className={`select ${isDisabled ? 'isDisabled' : ''}`}
      aria-disabled={isDisabled}
      {...(tutorialId ? { 'data-tutorial-id': tutorialId } : {})}
    >
      <button
        ref={buttonRef}
        id={id}
        type="button"
        className="select__button"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={`${id}-menu`}
        disabled={isDisabled}
        onClick={() => !isDisabled && setOpen((o) => !o)}
        onKeyDown={onButtonKeyDown}
        {...(tutorialActionPrefix
          ? { 'data-tutorial-action': `${tutorialActionPrefix}:open` }
          : tutorialAction
            ? { 'data-tutorial-action': `${tutorialAction}:open` }
            : {})}
      >
        <span className="select__value">
          {selected?.label ?? placeholder ?? (options.length ? 'Select…' : 'No Options')}
        </span>
        <span className={`select__chev ${open ? 'isOpen' : ''}`}>
          <ChevronDownIcon />
        </span>
      </button>

      {open && (
        <div
          ref={menuRef}
          id={`${id}-menu`}
          className="select__menu"
          role="listbox"
          aria-labelledby={id}
          aria-activedescendant={`${id}-option-${activeIndex}`}
          tabIndex={-1}
          onKeyDown={onMenuKeyDown}
        >
          {options.map((opt, idx) => {
            const isSelected = opt.value === value;
            const isActive = idx === activeIndex;
            const optionTutorialAction = tutorialActionPrefix
              ? `${tutorialActionPrefix}:${String(opt.value)}`
              : tutorialAction;
            return (
              <button
                key={opt.value}
                id={`${id}-option-${idx}`}
                ref={(node) => {
                  optionRefs.current[idx] = node;
                }}
                type="button"
                role="option"
                aria-selected={isSelected}
                className={`select__option ${isSelected ? 'isSelected' : ''} ${
                  isActive ? 'isActive' : ''
                }`}
                onMouseEnter={() => setActiveIndex(idx)}
                onClick={() => commit(idx)}
                {...(optionTutorialAction
                  ? { 'data-tutorial-action': optionTutorialAction }
                  : {})}
              >
                <span className="select__optionText">
                  <span className="select__optionLabel">{opt.label}</span>
                  {opt.description && <span className="select__optionDesc">{opt.description}</span>}
                </span>
                <span className="select__optionIcon">{isSelected ? <CheckIcon /> : null}</span>
              </button>
            );
          })}
        </div>
      )}
      {showSelectionHint && selected?.description ? (
        <div className="select__selectionHint">{selected.description}</div>
      ) : null}
    </div>
  );
}
