import type { Locale } from './i18n';

function fallbackNumber(value: number, digits: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  return value.toFixed(digits);
}

export function formatNumber(
  value: number,
  locale: Locale,
  opts: Intl.NumberFormatOptions = {},
): string {
  try {
    return new Intl.NumberFormat(locale, opts).format(value);
  } catch {
    const digits = opts.maximumFractionDigits ?? opts.minimumFractionDigits ?? 2;
    return fallbackNumber(value, digits);
  }
}

export function formatDateTime(
  value: string | Date | null | undefined,
  locale: Locale,
  opts: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  },
): string {
  if (!value) return 'n/a';
  const dt = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(dt.getTime())) return 'n/a';
  try {
    return new Intl.DateTimeFormat(locale, opts).format(dt);
  } catch {
    return dt.toISOString();
  }
}

