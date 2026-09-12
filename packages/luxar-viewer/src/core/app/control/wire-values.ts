/**
 * Turning viewer values into things that survive a JSON socket.
 *
 * Two of the embedder API's return shapes are not JSON: `screenshot()` resolves
 * a `Blob`, and several events carry a live `Error`. A blind `JSON.stringify`
 * gets both wrong in the worst way — an `Error` serialises to `{}`, so a
 * controller receives an event that says a failure happened and nothing about
 * what it was, and a `Blob` serialises to `{}` too.
 *
 * The sanitiser below is deliberately **generic** rather than a per-event
 * table. A table has to be updated every time an event payload gains a field,
 * and the failure mode of forgetting is silent data loss on the wire. A rule
 * that applies to every value cannot be forgotten.
 */

/** An `Error` flattened to something a controller can read. */
export interface WireError {
  name: string;
  message: string;
}

/** A `Blob` encoded for transport. */
export interface WireBlob {
  mime: string;
  /** Base64, without a `data:` prefix — the receiver decides what to build. */
  base64: string;
}

/**
 * How deep the sanitiser walks before giving up.
 *
 * Every embedder payload is shallow (a pose, a dimension summary, a layer
 * list), so this is a cycle guard rather than a real limit: an object graph
 * that deep is a bug, and recursing it would hang the socket.
 */
export const MAX_WIRE_DEPTH = 8;

function isError(value: unknown): value is Error {
  return value instanceof Error;
}

/**
 * Copy `value` into JSON-safe form.
 *
 * - `Error` becomes `{name, message}` (no stack: it names viewer internals and
 *   a controller cannot act on it).
 * - Functions and `undefined` are dropped, as `JSON.stringify` would.
 * - `NaN` / `Infinity` become `null`, because that is what they decode as
 *   anyway and a silent `null` is better than a frame that fails to parse.
 * - Anything past {@link MAX_WIRE_DEPTH} becomes `null`.
 */
export function sanitizeForWire(value: unknown, depth = 0): unknown {
  if (depth > MAX_WIRE_DEPTH) return null;
  if (value === null) return null;
  if (isError(value)) {
    const wire: WireError = { name: value.name, message: value.message };
    return wire;
  }
  const scalar = sanitizeScalar(value);
  if (scalar.handled) return scalar.value;
  return sanitizeStructured(value, depth);
}

/**
 * Convert anything that is not a container.
 *
 * Returns `handled: false` — rather than a sentinel value — so `undefined` can
 * be a legitimate *result* (a dropped function) without being confused for
 * "this function did not apply".
 */
function sanitizeScalar(value: unknown): { handled: boolean; value?: unknown } {
  const kind = typeof value;
  if (kind === 'number') {
    return { handled: true, value: Number.isFinite(value as number) ? value : null };
  }
  if (kind === 'string' || kind === 'boolean') return { handled: true, value };
  if (kind === 'bigint') return { handled: true, value: (value as bigint).toString() };
  if (kind === 'function' || kind === 'undefined' || kind === 'symbol') return { handled: true };
  return { handled: false };
}

/** Arrays, typed arrays and plain objects. */
function sanitizeStructured(value: unknown, depth: number): unknown {
  if (Array.isArray(value)) {
    // A dropped array element becomes null rather than shifting its neighbours
    // down, which would silently change the meaning of a positional payload.
    return value.map((entry) => sanitizeForWire(entry, depth + 1) ?? null);
  }
  if (ArrayBuffer.isView(value)) {
    // Typed arrays are how poses and matrices travel inside the viewer.
    return Array.from(value as unknown as ArrayLike<number>);
  }
  if (typeof value === 'object') return sanitizeObject(value as Record<string, unknown>, depth);
  return null;
}

function sanitizeObject(value: Record<string, unknown>, depth: number): unknown {
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(value)) {
    const sanitized = sanitizeForWire(value[key], depth + 1);
    if (sanitized !== undefined) out[key] = sanitized;
  }
  return out;
}

/** Whether a resolved value needs blob encoding before it can be sent. */
export function isBlobLike(value: unknown): value is Blob {
  return typeof Blob !== 'undefined' && value instanceof Blob;
}

/**
 * Encode a `Blob` as `{mime, base64}`.
 *
 * Chunked rather than `String.fromCharCode(...bytes)`: spreading a
 * multi-megabyte screenshot into an argument list overflows the call stack,
 * and a screenshot of a 4K kiosk display is exactly that size.
 */
export async function encodeBlobForWire(blob: Blob): Promise<WireBlob> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const CHUNK = 0x8000;
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + CHUNK));
  }
  return { mime: blob.type || 'application/octet-stream', base64: btoa(binary) };
}
