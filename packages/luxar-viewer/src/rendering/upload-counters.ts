/**
 * GPU upload counters — wrap the backend's upload entry points ONCE so every
 * CPU→GPU byte the renderer pushes is tallied in {@link perfCounters}.
 *
 * Counters (all cumulative):
 * - `gpu.uploadCalls` — every wrapped upload call, whatever it carried.
 * - `gpu.uploadBytes` — bytes those calls uploaded (buffer + texture).
 * - `gpu.uploadBytes.buffer` — `bufferData` / `bufferSubData` /
 *   `queue.writeBuffer`.
 * - `gpu.uploadBytes.texture` — `texImage2D/3D` / `texSubImage2D/3D` /
 *   `queue.writeTexture` / `queue.copyExternalImageToTexture`.
 *
 * Byte rules: a typed-array source counts the bytes the call actually reads
 * (honouring the WebGL2 / WebGPU element-unit `srcOffset` / `length` args); a
 * numeric `bufferData` size counts that allocation; an image-like source
 * (image, canvas, ImageBitmap, video frame) counts `width * height * depth * 4`
 * when its size is determinable and 0 otherwise; a PBO offset or `null` counts
 * 0. The call is counted in every case.
 *
 * Installation is idempotent (the context / queue is marked with a symbol).
 * The wrappers are own properties of the context / queue object, which three
 * looks up on every call, and a WebGL context restore keeps the SAME context
 * object — the viewer rebuilds resources against it rather than creating a
 * new one — so the wrappers survive a restore without being reapplied.
 *
 * Instrumentation only: every wrapper forwards its exact arguments (WebGL
 * overload resolution depends on argument COUNT) and returns the original's
 * result. The byte accounting cannot throw into the caller.
 *
 * @module rendering/upload-counters
 */

import { perfCounters } from '../profiling/perf-counters';

const S_BYTES = perfCounters.slot('gpu.uploadBytes');
const S_CALLS = perfCounters.slot('gpu.uploadCalls');
const S_BUFFER_BYTES = perfCounters.slot('gpu.uploadBytes.buffer');
const S_TEXTURE_BYTES = perfCounters.slot('gpu.uploadBytes.texture');

/** Marks a context / queue whose upload methods are already wrapped. */
const WRAPPED = Symbol.for('luxar.uploadCounters');

type AnyFn = (...args: unknown[]) => unknown;
type Kind = 'buffer' | 'texture';
/** Bytes an upload call carries, from its argument list. */
type ByteCounter = (args: readonly unknown[]) => number;

function countUpload(kind: Kind, bytes: number): void {
  perfCounters.add(S_CALLS);
  if (!(bytes > 0)) return;
  perfCounters.add(S_BYTES, bytes);
  perfCounters.add(kind === 'buffer' ? S_BUFFER_BYTES : S_TEXTURE_BYTES, bytes);
}

/** Element size of a view (1 for DataView / non-typed views). */
function elementSize(view: ArrayBufferView): number {
  const bpe = (view as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT;
  return typeof bpe === 'number' ? bpe : 1;
}

/**
 * Bytes read from a buffer source, with optional element-unit `srcOffset` and
 * `length` (0 or absent = to the end), as WebGL2 `bufferData` /
 * `bufferSubData` and WebGPU `writeBuffer` define them.
 */
function sourceBytes(data: unknown, srcOffset: unknown, length: unknown): number {
  if (data instanceof ArrayBuffer) {
    const offset = typeof srcOffset === 'number' ? srcOffset : 0;
    const len = typeof length === 'number' && length > 0 ? length : data.byteLength - offset;
    return Math.max(0, len);
  }
  if (!ArrayBuffer.isView(data)) return 0;
  const size = elementSize(data);
  const elements = data.byteLength / size;
  const offset = typeof srcOffset === 'number' ? srcOffset : 0;
  const len = typeof length === 'number' && length > 0 ? length : elements - offset;
  return Math.max(0, len * size);
}

const WIDTH_KEYS = ['displayWidth', 'videoWidth', 'naturalWidth', 'width'] as const;
const HEIGHT_KEYS = ['displayHeight', 'videoHeight', 'naturalHeight', 'height'] as const;

/** First numeric property of `s` among `keys` (0 when none). */
function firstNumber(s: Record<string, unknown>, keys: readonly string[]): number {
  for (const key of keys) {
    const v = s[key];
    if (typeof v === 'number') return v;
  }
  return 0;
}

/**
 * Pixel size of an image-like texture source (0 when undeterminable). Video
 * frames and videos report their size under their own names, checked first.
 */
function imagePixels(source: unknown): number {
  if (source === null || typeof source !== 'object') return 0;
  const s = source as Record<string, unknown>;
  return firstNumber(s, WIDTH_KEYS) * firstNumber(s, HEIGHT_KEYS);
}

/**
 * Bytes of a texture upload's pixel argument. `dims` is the explicit
 * width*height*depth when the overload carries one (0 when it does not).
 */
function texelBytes(source: unknown, srcOffset: unknown, dims: number): number {
  if (source === null || source === undefined || typeof source === 'number') return 0;
  if (ArrayBuffer.isView(source)) return sourceBytes(source, srcOffset, undefined);
  return (dims > 0 ? dims : imagePixels(source)) * 4;
}

const num = (v: unknown): number => (typeof v === 'number' ? v : 0);

// --- WebGL(2) byte counters (argument positions per the WebGL2 IDL) ---------

const bufferDataBytes: ByteCounter = (a) =>
  typeof a[1] === 'number' ? a[1] : sourceBytes(a[1], a[3], a[4]);
const bufferSubDataBytes: ByteCounter = (a) => sourceBytes(a[2], a[3], a[4]);
// texImage2D: 6-arg (…, format, type, source) or 9/10-arg sized form.
const texImage2DBytes: ByteCounter = (a) =>
  a.length === 6 ? texelBytes(a[5], undefined, 0) : texelBytes(a[8], a[9], num(a[3]) * num(a[4]));
const texImage3DBytes: ByteCounter = (a) =>
  texelBytes(a[9], a[10], num(a[3]) * num(a[4]) * num(a[5]));
// texSubImage2D: 7-arg (…, format, type, source) or 9/10-arg sized form.
const texSubImage2DBytes: ByteCounter = (a) =>
  a.length === 7 ? texelBytes(a[6], undefined, 0) : texelBytes(a[8], a[9], num(a[4]) * num(a[5]));
const texSubImage3DBytes: ByteCounter = (a) =>
  texelBytes(a[10], a[11], num(a[5]) * num(a[6]) * num(a[7]));

// --- WebGPU byte counters ----------------------------------------------------

// writeBuffer(buffer, bufferOffset, data, dataOffset?, size?) — element units
// for a typed array, bytes for an ArrayBuffer (sourceBytes handles both).
const writeBufferBytes: ByteCounter = (a) => sourceBytes(a[2], a[3], a[4]);
// writeTexture(destination, data, dataLayout, size)
const writeTextureBytes: ByteCounter = (a) => {
  const offset = num((a[2] as { offset?: unknown } | undefined)?.offset);
  const data = a[1];
  const total = data instanceof ArrayBuffer || ArrayBuffer.isView(data) ? data.byteLength : 0;
  return Math.max(0, total - offset);
};
/** Texel count of a GPUExtent3D (array or dictionary form). */
function extentTexels(size: unknown): number {
  if (Array.isArray(size)) return num(size[0]) * (num(size[1]) || 1) * (num(size[2]) || 1);
  if (size === null || typeof size !== 'object') return 0;
  const e = size as { width?: unknown; height?: unknown; depthOrArrayLayers?: unknown };
  return num(e.width) * (num(e.height) || 1) * (num(e.depthOrArrayLayers) || 1);
}
// copyExternalImageToTexture(source, destination, copySize)
const copyExternalImageBytes: ByteCounter = (a) => extentTexels(a[2]) * 4;

function wrap(target: Record<string, unknown>, name: string, kind: Kind, bytes: ByteCounter): void {
  const original = target[name];
  if (typeof original !== 'function') return;
  const fn = original as AnyFn;
  target[name] = function countedUpload(this: unknown, ...args: unknown[]): unknown {
    let n = 0;
    try {
      n = bytes(args);
    } catch {
      // Exotic argument shapes count the call with 0 bytes.
    }
    countUpload(kind, n);
    return fn.apply(this, args);
  };
}

function markOnce(target: object): boolean {
  const t = target as Record<symbol, unknown>;
  if (t[WRAPPED]) return false;
  Object.defineProperty(t, WRAPPED, { value: true });
  return true;
}

/**
 * Wrap a WebGL(2) context's upload methods. Idempotent.
 *
 * @returns true when this call installed the wrappers
 */
export function wrapWebGLUploads(gl: object): boolean {
  if (!markOnce(gl)) return false;
  const t = gl as Record<string, unknown>;
  wrap(t, 'bufferData', 'buffer', bufferDataBytes);
  wrap(t, 'bufferSubData', 'buffer', bufferSubDataBytes);
  wrap(t, 'texImage2D', 'texture', texImage2DBytes);
  wrap(t, 'texImage3D', 'texture', texImage3DBytes);
  wrap(t, 'texSubImage2D', 'texture', texSubImage2DBytes);
  wrap(t, 'texSubImage3D', 'texture', texSubImage3DBytes);
  return true;
}

/**
 * Wrap a WebGPU queue's upload methods. Idempotent.
 *
 * @returns true when this call installed the wrappers
 */
export function wrapWebGPUUploads(queue: object): boolean {
  if (!markOnce(queue)) return false;
  const t = queue as Record<string, unknown>;
  wrap(t, 'writeBuffer', 'buffer', writeBufferBytes);
  wrap(t, 'writeTexture', 'texture', writeTextureBytes);
  wrap(t, 'copyExternalImageToTexture', 'texture', copyExternalImageBytes);
  return true;
}

/** The renderer shapes {@link installUploadCounters} knows how to reach into. */
interface RendererLike {
  getContext?: () => unknown;
  backend?: { device?: { queue?: unknown } | null; gl?: unknown } | null;
}

/**
 * Wrap the upload entry points of an initialised renderer: a classic
 * `WebGLRenderer`'s context, a `WebGPURenderer`'s device queue, or the WebGL2
 * context of a `WebGPURenderer` on its WebGL fallback backend. Call after
 * `renderer.init()` for WebGPU. Idempotent; never throws.
 *
 * @returns which backend was wrapped, or null when none was reachable
 */
export function installUploadCounters(renderer: unknown): 'webgl' | 'webgpu' | null {
  try {
    const r = renderer as RendererLike;
    const queue = r.backend?.device?.queue;
    if (isObject(queue)) {
      wrapWebGPUUploads(queue);
      return 'webgpu';
    }
    // A WebGPURenderer on its WebGL2 fallback exposes `backend.gl`; a classic
    // WebGLRenderer has no backend and hands out its context.
    const gl = r.backend ? r.backend.gl : r.getContext?.();
    if (isObject(gl)) {
      wrapWebGLUploads(gl);
      return 'webgl';
    }
  } catch {
    // Instrumentation must never break renderer setup.
  }
  return null;
}

function isObject(v: unknown): v is object {
  return typeof v === 'object' && v !== null;
}
