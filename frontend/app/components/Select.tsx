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
  value: T;
  options: SelectOption<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
  placeholder?: string;
  ariaLabel: string;
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
  value,
  options,
  onChange,
  disabled,
  placeholder,
  ariaLabel,
}: Props<T>) {
  const id = useId();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number>(() =>
    Math.max(
      0,
      options.findIndex((o) => o.value === value),
    ),
  );

  const selected = useMemo(() => options.find((o) => o.value === value) ?? null, [options, value]);

  useEffect(() => {
    setActiveIndex(
      Math.max(
        0,
        options.findIndex((o) => o.value === value),
      ),
    );
  }, [options, value]);

  useEffect(() => {
    function onDocDown(e: MouseEvent) {
      const el = rootRef.current;
      if (!el) return;
      if (e.target instanceof Node && el.contains(e.target)) return;
      setOpen(false);
    }

    document.addEventListener('mousedown', onDocDown);
    return () => document.removeEventListener('mousedown', onDocDown);
  }, []);

  useEffect(() => {
    if (!open) {
      buttonRef.current?.focus?.();
      return;
    }
    const t = window.setTimeout(() => menuRef.current?.focus?.(), 0);
    return () => window.clearTimeout(t);
  }, [open]);

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
    if (disabled) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setOpen(true);
      step(1);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setOpen(true);
      step(-1);
      return;
    }
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (open) commit(activeIndex);
      else setOpen(true);
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
  }

  return (
    <div ref={rootRef} className={`select ${disabled ? 'isDisabled' : ''}`}>
      <button
        ref={buttonRef}
        id={id}
        type="button"
        className="select__button"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={`${id}-menu`}
        disabled={disabled}
        onClick={() => !disabled && setOpen((o) => !o)}
        onKeyDown={onButtonKeyDown}
      >
        <span className="select__value">{selected?.label ?? placeholder ?? 'Select…'}</span>
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
          tabIndex={-1}
          onKeyDown={onMenuKeyDown}
        >
          {options.map((opt, idx) => {
            const isSelected = opt.value === value;
            const isActive = idx === activeIndex;
            return (
              <button
                key={opt.value}
                type="button"
                role="option"
                aria-selected={isSelected}
                className={`select__option ${isSelected ? 'isSelected' : ''} ${
                  isActive ? 'isActive' : ''
                }`}
                onMouseEnter={() => setActiveIndex(idx)}
                onClick={() => commit(idx)}
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
    </div>
  );
}
