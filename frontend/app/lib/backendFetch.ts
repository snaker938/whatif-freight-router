const DEFAULT_ATTEMPT_TIMEOUT_MS = 1_200_000;
const DEFAULT_ROUTE_FALLBACK_TIMEOUT_MS = 900_000;

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return parsed;
}

export function backendBaseUrl(): string {
  return (
    process.env.BACKEND_INTERNAL_URL ??
    process.env.NEXT_PUBLIC_BACKEND_URL ??
    'http://localhost:8000'
  );
}

export function computeAttemptTimeoutMs(): number {
  return parsePositiveInt(
    process.env.COMPUTE_ATTEMPT_TIMEOUT_MS ?? process.env.NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS,
    DEFAULT_ATTEMPT_TIMEOUT_MS,
  );
}

export function computeRouteFallbackTimeoutMs(): number {
  return parsePositiveInt(
    process.env.COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS ??
      process.env.NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS,
    DEFAULT_ROUTE_FALLBACK_TIMEOUT_MS,
  );
}

function causeString(value: unknown): string {
  if (!value) return '';
  if (typeof value === 'string') return value.trim();
  if (value instanceof Error) return value.message.trim();
  try {
    return String(value).trim();
  } catch {
    return '';
  }
}

type TransportReasonCode =
  | 'backend_headers_timeout'
  | 'backend_body_timeout'
  | 'backend_connection_reset'
  | 'backend_unreachable';

export type BackendTransportError = {
  reasonCode: TransportReasonCode;
  cause: string;
  detail: string;
};

export function classifyBackendTransportError(error: unknown): BackendTransportError {
  const baseDetail = error instanceof Error ? error.message : 'Unknown backend transport error';
  const cause =
    error && typeof error === 'object' && 'cause' in error
      ? (error as { cause?: unknown }).cause
      : undefined;
  const causeCode =
    cause && typeof cause === 'object' && 'code' in cause
      ? String((cause as { code?: unknown }).code ?? '')
      : '';
  const causeMessage =
    cause && typeof cause === 'object' && 'message' in cause
      ? String((cause as { message?: unknown }).message ?? '')
      : '';
  const detail = `${baseDetail}${causeCode || causeMessage ? ` (${causeCode} ${causeMessage})` : ''}`.trim();
  const lowered = `${baseDetail} ${causeCode} ${causeMessage}`.toLowerCase();
  if (lowered.includes('backend_fetch_timeout')) {
    return {
      reasonCode: 'backend_headers_timeout',
      cause: causeString(cause) || causeMessage || causeCode || detail,
      detail,
    };
  }
  if (lowered.includes('und_err_headers_timeout') || lowered.includes('headers timeout')) {
    return {
      reasonCode: 'backend_headers_timeout',
      cause: causeString(cause) || causeMessage || causeCode || detail,
      detail,
    };
  }
  if (lowered.includes('und_err_body_timeout') || lowered.includes('body timeout')) {
    return {
      reasonCode: 'backend_body_timeout',
      cause: causeString(cause) || causeMessage || causeCode || detail,
      detail,
    };
  }
  if (lowered.includes('econnreset') || lowered.includes('connection reset')) {
    return {
      reasonCode: 'backend_connection_reset',
      cause: causeString(cause) || causeMessage || causeCode || detail,
      detail,
    };
  }
  return {
    reasonCode: 'backend_unreachable',
    cause: causeString(cause) || causeMessage || causeCode || detail,
    detail,
  };
}

type FetchBackendOptions = {
  timeoutMs?: number;
  requestSignal?: AbortSignal;
  method?: string;
  headers?: HeadersInit;
  body?: BodyInit | null;
};

export async function fetchBackend(path: string, options: FetchBackendOptions): Promise<Response> {
  const timeoutMs = parsePositiveInt(String(options.timeoutMs ?? ''), DEFAULT_ATTEMPT_TIMEOUT_MS);
  const ctrl = new AbortController();
  const externalSignal = options.requestSignal;
  const onExternalAbort = () => {
    try {
      ctrl.abort(externalSignal?.reason ?? new Error('frontend_request_aborted'));
    } catch {
      ctrl.abort();
    }
  };
  if (externalSignal) {
    if (externalSignal.aborted) {
      onExternalAbort();
    } else {
      externalSignal.addEventListener('abort', onExternalAbort, { once: true });
    }
  }
  const timeoutId = setTimeout(() => {
    try {
      ctrl.abort(new Error(`backend_fetch_timeout:${timeoutMs}`));
    } catch {
      ctrl.abort();
    }
  }, timeoutMs);
  try {
    return await fetch(`${backendBaseUrl()}${path}`, {
      method: options.method ?? 'GET',
      headers: options.headers,
      body: options.body,
      cache: 'no-store',
      signal: ctrl.signal,
    } as RequestInit);
  } finally {
    clearTimeout(timeoutId);
    if (externalSignal) {
      externalSignal.removeEventListener('abort', onExternalAbort);
    }
  }
}
