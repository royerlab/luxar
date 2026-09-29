/**
 * Rows-only element-texture uploads on three's WebGPU backends (#2944).
 *
 * Element data lives in RGBA32F `DataTexture`s whose writers register the
 * dirty span as per-row `updateRanges` (`./element-storage`). The classic
 * `WebGLRenderer` honours them; three r185's `WebGPURenderer` backends do
 * not — both the native WebGPU backend (`WebGPUTextureUtils.updateTexture`
 * → one `queue.writeTexture` of the whole image) and the WebGL2 fallback
 * (`WebGLTextureUtils.updateTexture` → one whole-image `texSubImage2D`)
 * re-upload the full CAPACITY on every `needsUpdate`, so a playback commit
 * moved ~3x the bytes of the same commit on WebGL.
 *
 * Interception point: {@link installElementTextureRowUploads} replaces the
 * backend instance's `updateTexture(texture, options)` ONCE, at renderer
 * creation. That is the single call three's common `Textures.updateTexture`
 * makes to move texel bytes, and it runs AFTER three has done all its own
 * bookkeeping for the update — `createTexture` on first use, the
 * `version` / `generation` stamps, and (after we return) `onUpdate`. The
 * wrapper therefore changes only WHICH bytes reach the GPU resource three
 * already allocated, never three's view of the texture's state: nothing
 * three tracks can go stale, and no later full re-upload is triggered.
 *
 * Why the GPU contents equal a full upload: by induction over a GPU
 * resource's uploads. Every CPU write to an element texture registers its
 * span as a range, and ranges are consumed only by an upload, so the
 * texels OUTSIDE the pending ranges already hold their current CPU value
 * on the GPU whenever the resource is "in sync". Uploading the whole rows
 * the ranges touch, from the same `image.data`, then leaves every texel
 * equal to what a full upload would have written, and the ranges are
 * CONSUMED (cleared), as the classic renderer does, so the next commit
 * uploads only its own rows. A resource is in sync when either
 * - this wrapper last brought exactly this resource object fully up to
 *   date (a full upload or a ranged one), or
 * - the texture has NEVER been uploaded by any renderer (its `onUpdate`
 *   has never fired): its backing store was zero-filled at attach and
 *   every write since is still in the pending ranges, while three has just
 *   allocated the resource and both APIs zero-initialise new textures.
 *   This is the ranged first upload the classic renderer already does, and
 *   it is what keeps growth-headroom rows off a progressive load.
 *
 * Three's full upload is used instead (the original method, verbatim)
 * whenever any of these holds:
 * - the texture is not an element texture (every other texture);
 * - no ranges are pending, or a full upload is pending
 *   (`markElementTextureFullDirty`, the ≥75%-dirty fold, context restore);
 * - the resource is not in sync — a reallocation (three hands back a new
 *   `GPUTexture` / `WebGLTexture`), a device / context loss that recreates
 *   resources, or a texture another renderer or backend already uploaded;
 * - the texture or its GPU resource is not the exact shape this writer
 *   knows (RGBA32F 2D, no mips, no flipY, `image.data` a full-size
 *   Float32Array, descriptor size equal to the image);
 * - a range is out of bounds;
 * - the row write throws.
 *
 * Pool re-acquire is ordinary here: the geometry keeps its texture and GPU
 * resource, and the new tenant's writes register their spans like any
 * other commit.
 *
 * The rows are written through the backend's own `device.queue` /
 * `gl`, whose upload methods `./upload-counters` wraps, and are passed as
 * an exact `subarray` so the counters tally the bytes actually written.
 *
 * The classic `WebGLRenderer` has no backend and is left alone.
 *
 * @module rendering/element-texture-row-upload
 */

import * as THREE from 'three';
import {
  hasElementTextureBeenUploaded,
  hasPendingElementTextureFullUpload,
  isElementTexture,
} from './element-storage';

/** Marks a backend whose `updateTexture` is already wrapped. */
const WRAPPED = Symbol.for('luxar.elementTextureRowUploads');

/** RGBA32F: 4 floats per texel. */
const FLOATS_PER_TEXEL = 4;
const BYTES_PER_TEXEL = FLOATS_PER_TEXEL * Float32Array.BYTES_PER_ELEMENT;

/** A contiguous run of whole texture rows. */
export interface RowSpan {
  firstRow: number;
  rowCount: number;
}

/** The slice of three's `Textures.updateTexture` options this module reads. */
interface UpdateOptions {
  image?: unknown;
  width?: number;
  height?: number;
}

type UpdateTextureFn = (texture: THREE.Texture, options: UpdateOptions) => void;

/** Structural view of three's backends (native WebGPU + WebGL2 fallback). */
interface BackendLike {
  isWebGPUBackend?: boolean;
  isWebGLBackend?: boolean;
  updateTexture?: UpdateTextureFn;
  get(object: object): Record<string, unknown>;
  device?: { queue?: { writeTexture?: (...args: unknown[]) => unknown } } | null;
  gl?: Record<string, unknown> | null;
  state?: { bindTexture?: (target: unknown, texture: unknown) => void } | null;
  textureUtils?: { setTextureParameters?: (target: unknown, texture: THREE.Texture) => void };
}

/** Image shape of an element `DataTexture`. */
interface ElementImage {
  data: Float32Array;
  width: number;
  height: number;
}

/**
 * One backend's way of writing rows into a texture's existing GPU resource.
 * `resource` identifies that resource (undefined = none allocated);
 * `writeRows` returns false when the resource is not a shape it can write,
 * so the caller falls back to three's full upload.
 */
export interface RowWriter {
  resource(texture: THREE.Texture): unknown;
  writeRows(texture: THREE.Texture, image: ElementImage, span: RowSpan): boolean;
}

/**
 * GPU resource each element texture was last brought fully in sync with,
 * through a wrapped backend (by a full upload or a successful row upload). Keyed by the texture; the resource object is
 * backend-specific, so a different backend or a reallocated resource never
 * matches.
 */
const syncedResource = new WeakMap<THREE.Texture, unknown>();

/** True for a full-size RGBA Float32 image of at least one texel. */
function isFullSizeFloatImage(image: Partial<ElementImage>): image is ElementImage {
  const { data, width, height } = image;
  if (!(data instanceof Float32Array)) return false;
  if (typeof width !== 'number' || typeof height !== 'number') return false;
  return width >= 1 && height >= 1 && data.length === width * height * FLOATS_PER_TEXEL;
}

/** True for a mip-less, unflipped RGBA32F `DataTexture`. */
function isPlainRgba32fDataTexture(texture: THREE.Texture): boolean {
  return [
    (texture as THREE.DataTexture).isDataTexture === true,
    texture.format === THREE.RGBAFormat,
    texture.type === THREE.FloatType,
    texture.flipY === false,
    texture.mipmaps.length === 0,
    !texture.generateMipmaps,
  ].every(Boolean);
}

/** The element image, when the texture is exactly the shape rows can be written for. */
export function elementImageForRowUpload(
  texture: THREE.Texture,
  options: UpdateOptions
): ElementImage | null {
  const image = texture.image as Partial<ElementImage> | null | undefined;
  if (!image || options.image !== image || !isFullSizeFloatImage(image)) return null;
  if (options.width !== image.width || options.height !== image.height) return null;
  return isPlainRgba32fDataTexture(texture) ? image : null;
}

/**
 * The whole-row span covering every pending update range (float units of
 * `image.data`, RGBA), or null when none is pending or any is out of
 * bounds. A covering span is a superset of the ranges — always correct.
 */
export function planDirtyRows(
  ranges: readonly { start: number; count: number }[],
  image: ElementImage
): RowSpan | null {
  if (ranges.length === 0) return null;
  const rowFloats = image.width * FLOATS_PER_TEXEL;
  let start = Infinity;
  let end = -Infinity;
  for (const range of ranges) {
    const rangeEnd = range.start + range.count;
    const valid =
      Number.isInteger(range.start) &&
      Number.isInteger(range.count) &&
      range.start >= 0 &&
      range.count > 0 &&
      rangeEnd <= image.data.length;
    if (!valid) return null;
    start = Math.min(start, range.start);
    end = Math.max(end, rangeEnd);
  }
  const firstRow = Math.floor(start / rowFloats);
  const lastRow = Math.floor((end - 1) / rowFloats);
  return { firstRow, rowCount: lastRow - firstRow + 1 };
}

/** The `image.data` floats of `span`, as an exact view (so counters see its bytes). */
function rowData(image: ElementImage, span: RowSpan): Float32Array {
  const rowFloats = image.width * FLOATS_PER_TEXEL;
  const begin = span.firstRow * rowFloats;
  return image.data.subarray(begin, begin + span.rowCount * rowFloats);
}

/** The slice of the native backend's per-texture record this module reads. */
interface WebGPUTextureRecord {
  texture?: unknown;
  textureDescriptorGPU?: {
    format?: string;
    size?: { width?: number; height?: number; depthOrArrayLayers?: number };
    mipLevelCount?: number;
    sampleCount?: number;
  };
}

/** True when the allocated GPU texture is a single-level RGBA32F 2D texture of `image`'s size. */
function isRowWritableGPUTexture(record: WebGPUTextureRecord, image: ElementImage): boolean {
  const desc = record.textureDescriptorGPU;
  if (record.texture == null || desc === undefined) return false;
  const size = desc.size ?? {};
  return [
    desc.format === 'rgba32float',
    size.width === image.width,
    size.height === image.height,
    (size.depthOrArrayLayers ?? 1) === 1,
    (desc.mipLevelCount ?? 1) === 1,
    (desc.sampleCount ?? 1) === 1,
  ].every(Boolean);
}

/** Row writer for the native WebGPU backend (`queue.writeTexture`). */
export function webgpuRowWriter(backend: BackendLike): RowWriter {
  return {
    resource: (texture) => backend.get(texture).texture,
    writeRows(texture, image, span) {
      const record = backend.get(texture) as WebGPUTextureRecord;
      const queue = backend.device?.queue;
      if (typeof queue?.writeTexture !== 'function') return false;
      if (!isRowWritableGPUTexture(record, image)) return false;
      queue.writeTexture(
        { texture: record.texture, mipLevel: 0, origin: { x: 0, y: span.firstRow, z: 0 } },
        rowData(image, span),
        { offset: 0, bytesPerRow: image.width * BYTES_PER_TEXEL, rowsPerImage: span.rowCount },
        { width: image.width, height: span.rowCount, depthOrArrayLayers: 1 }
      );
      return true;
    },
  };
}

/** The slice of the fallback backend's per-texture record this module reads. */
interface WebGLTextureRecord {
  textureGPU?: unknown;
  glTextureType?: unknown;
  glFormat?: unknown;
  glType?: unknown;
}

/** True when the allocated WebGL texture is an RGBA / FLOAT `TEXTURE_2D`. */
function isRowWritableGLTexture(record: WebGLTextureRecord, gl: Record<string, unknown>): boolean {
  return [
    record.textureGPU != null,
    record.glTextureType === gl.TEXTURE_2D,
    record.glFormat === gl.RGBA,
    record.glType === gl.FLOAT,
    typeof gl.texSubImage2D === 'function',
  ].every(Boolean);
}

/** Row writer for `WebGPURenderer`'s WebGL2 fallback backend (`texSubImage2D`). */
export function webglFallbackRowWriter(backend: BackendLike): RowWriter {
  return {
    resource: (texture) => backend.get(texture).textureGPU,
    writeRows(texture, image, span) {
      const record = backend.get(texture) as WebGLTextureRecord;
      const { gl, state, textureUtils } = backend;
      if (gl == null || !isRowWritableGLTexture(record, gl)) return false;
      if (typeof state?.bindTexture !== 'function') return false;
      if (typeof textureUtils?.setTextureParameters !== 'function') return false;
      // Exactly what three's own full upload does first: bind, then set the
      // unpack state (flipY / premultiply / alignment / colour conversion).
      state.bindTexture(record.glTextureType, record.textureGPU);
      textureUtils.setTextureParameters(record.glTextureType, texture);
      (gl.texSubImage2D as (...args: unknown[]) => void).call(
        gl,
        record.glTextureType,
        0,
        0,
        span.firstRow,
        image.width,
        span.rowCount,
        record.glFormat,
        record.glType,
        rowData(image, span)
      );
      return true;
    },
  };
}

/**
 * Upload an element texture's dirty rows through `writer`, or return false
 * so the caller runs three's full upload. On success the ranges are
 * consumed.
 */
function tryRowUpload(
  texture: THREE.DataTexture,
  options: UpdateOptions,
  writer: RowWriter
): boolean {
  if (hasPendingElementTextureFullUpload(texture)) return false;
  const resource = writer.resource(texture);
  if (resource == null) return false;
  const inSync =
    syncedResource.get(texture) === resource || !hasElementTextureBeenUploaded(texture);
  if (!inSync) return false;
  const image = elementImageForRowUpload(texture, options);
  if (image === null) return false;
  const span = planDirtyRows(texture.updateRanges, image);
  if (span === null) return false;
  try {
    if (!writer.writeRows(texture, image, span)) return false;
  } catch {
    return false;
  }
  // A ranged first upload leaves the resource as in sync as a full one (its
  // unwritten rows are the zeroed storage the CPU image also holds there);
  // without the record the next commit would re-upload the full capacity.
  syncedResource.set(texture, resource);
  texture.clearUpdateRanges();
  return true;
}

/**
 * The replacement `updateTexture`: rows for an in-sync element texture,
 * three's original full upload for everything else. After a full upload of
 * an element texture its resource is recorded as in sync and the ranges it
 * covered are consumed.
 */
export function createRowUploadUpdateTexture(
  original: UpdateTextureFn,
  writer: RowWriter
): UpdateTextureFn {
  return function updateTextureRows(this: unknown, texture, options) {
    if (!isElementTexture(texture)) {
      original.call(this, texture, options);
      return;
    }
    if (tryRowUpload(texture, options, writer)) return;
    original.call(this, texture, options);
    const resource = writer.resource(texture);
    if (resource == null || elementImageForRowUpload(texture, options) === null) {
      syncedResource.delete(texture);
      return;
    }
    syncedResource.set(texture, resource);
    texture.clearUpdateRanges();
  };
}

/**
 * Wrap a `WebGPURenderer` backend's `updateTexture` so element textures
 * upload only their dirty rows. Call after `renderer.init()`. Idempotent;
 * never throws; a classic `WebGLRenderer` (no backend) is left untouched.
 *
 * @returns which backend was wrapped, or null when none was
 */
export function installElementTextureRowUploads(
  renderer: unknown
): 'webgpu' | 'webgl-fallback' | null {
  try {
    const backend = (renderer as { backend?: BackendLike | null } | null)?.backend;
    if (!backend || typeof backend.updateTexture !== 'function') return null;
    const kind = backend.isWebGPUBackend
      ? 'webgpu'
      : backend.isWebGLBackend
        ? 'webgl-fallback'
        : null;
    if (kind === null) return null;
    const marked = backend as unknown as Record<symbol, unknown>;
    if (marked[WRAPPED]) return kind;
    const writer = kind === 'webgpu' ? webgpuRowWriter(backend) : webglFallbackRowWriter(backend);
    backend.updateTexture = createRowUploadUpdateTexture(backend.updateTexture, writer);
    Object.defineProperty(marked, WRAPPED, { value: true });
    return kind;
  } catch {
    // An unexpected backend shape keeps three's own uploads.
    return null;
  }
}
