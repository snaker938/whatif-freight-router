export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    cache: 'no-store'
  });

  const text = await res.text();
  if (!res.ok) throw new Error(text);

  return JSON.parse(text) as T;
}
