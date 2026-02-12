function extractErrorMessage(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return 'Request failed';

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (typeof parsed === 'string' && parsed.trim()) return parsed.trim();
    if (parsed && typeof parsed === 'object') {
      const detail = (parsed as Record<string, unknown>).detail;
      if (typeof detail === 'string' && detail.trim()) return detail.trim();
      if (Array.isArray(detail) && detail.length) {
        const joined = detail
          .map((entry) => {
            if (typeof entry === 'string') return entry;
            if (entry && typeof entry === 'object') {
              const msg = (entry as Record<string, unknown>).msg;
              return typeof msg === 'string' ? msg : '';
            }
            return '';
          })
          .filter(Boolean)
          .join('; ')
          .trim();
        if (joined) return joined;
      }
      const message = (parsed as Record<string, unknown>).message;
      if (typeof message === 'string' && message.trim()) return message.trim();
    }
  } catch {
    // Response was not JSON.
  }

  return trimmed;
}

async function readJSON<T>(res: Response): Promise<T> {
  const text = await res.text();
  if (!res.ok) {
    throw new Error(extractErrorMessage(text));
  }
  if (!text.trim()) {
    throw new Error('Server returned an empty response');
  }
  return JSON.parse(text) as T;
}

export async function postJSON<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
  headers?: HeadersInit,
): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(headers ?? {}) },
    body: JSON.stringify(body),
    cache: 'no-store',
    signal,
  });

  return readJSON<T>(res);
}

type PostNDJSONOptions<TEvent> = {
  signal?: AbortSignal;
  headers?: HeadersInit;
  onEvent: (event: TEvent) => void;
};

export async function postNDJSON<TEvent extends object>(
  path: string,
  body: unknown,
  options: PostNDJSONOptions<TEvent>,
): Promise<void> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(options.headers ?? {}) },
    body: JSON.stringify(body),
    cache: 'no-store',
    signal: options.signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(extractErrorMessage(text));
  }

  if (!res.body) {
    throw new Error('Streaming response was empty');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      options.onEvent(JSON.parse(trimmed) as TEvent);
    }
  }

  buffer += decoder.decode();
  const tail = buffer.trim();
  if (tail) {
    options.onEvent(JSON.parse(tail) as TEvent);
  }
}
