/**
 * Native-WebGPU partial texture uploads — depth-sorting spec §7 Stage 3.
 *
 * three r184's WebGPU backend ignores `Texture.updateRanges`: its
 * `WebGPUTextureUtils._copyBufferToTexture` issues one whole-image
 * `queue.writeTexture` (origin {0,0}, full width×height) on every
 * `texture.version` bump, so every element-texture commit re-uploaded
 * the entire capacity-sized RGBA32F backing store. Probed on Dawn/Metal
 * (M4 Max, headless Chrome): `writeTexture` runs at only ~2.2 GB/s, so
 * a 5M-splat texture (305 MB) cost ~129 ms per commit while a ladder
 * append needs ~7 ms — and per-ROW ranged calls beat even the single
 * contiguous block (Dawn's staging path favors smaller copies). This
 * wrapper closes the gap by consuming the SAME per-row `updateRanges`
 * the classic WebGL path uses (registered by
 * `element-storage.ts::registerElementTexelDirtyRange`) into per-range
 * `queue.writeTexture` calls.
 *
 * Seam: `WebGPUBackend.updateTexture(texture, options)` is a one-line
 * delegate to its own `textureUtils` instance — we wrap it ON THE
 * CONSTRUCTED BACKEND INSTANCE (no three fork, no prototype patching).
 * The wrapper self-neutralizes if a future three consumes texture
 * ranges upstream (ranges arrive already cleared → delegate untouched),
 * and falls back to the stock full-upload behavior (status quo,
 * correct) if the seam's shape ever drifts (see
 * {@link installWebGPUPartialTextureUploads}'s shape check).
 *
 * CONSUME-AND-CLEAR is load-bearing: nothing else clears texture ranges
 * on this backend, and `registerElementTexelDirtyRange` folds pending
 * ranges into every new span — without the clear, the union grows
 * monotonically and "partial" degrades back to full asymptotically.
 * Clearing here mirrors the classic renderer (`WebGLTextures.js`
 * clears after consuming) and preserves the accumulate-while-hidden →
 * single-flush contract.
 *
 * Eligibility is deliberately narrow (the element-texture shape):
 * plain 2D `DataTexture`, non-empty `updateRanges`, no mipmaps, no
 * flipY, typed-array image data. Everything else — including the
 * pendingFullUpload full-dirty encoding (needsUpdate + EMPTY ranges)
 * and fresh textures — delegates to the stock path. A ranged FIRST
 * upload of a fresh texture is also safe (WebGPU zero-initializes
 * texture storage; shaders only read `[0, count)`), matching the
 * classic-WebGL argument in element-storage.ts.
 *
 * The `forceWebGL` fallback backend is NOT wrapped (renderer-setup
 * gates on `backend.isWebGLBackend !== true`): it is a TSL parity
 * diagnostic surface and keeps the stock full-upload behavior.
 *
 * @module rendering/webgpu-partial-texture-upload
 */

import type * as THREE from 'three';
import { log, Modules } from '../utils/log';

/** Bytes per texel for the RGBA32F element textures (4 × float32). */
const BYTES_PER_TEXEL = 16;

/** Duck-typed slice of the three WebGPU backend the wrapper relies on. */
interface WebGPUBackendSeam {
  isWebGLBackend?: boolean;
  device?: {
    queue?: {
      writeTexture: (
        destination: { texture: unknown; mipLevel?: number; origin: { x: number; y: number; z?: number } },
        data: ArrayBufferView,
        dataLayout: { offset: number; bytesPerRow: number },
        size: { width: number; height: number; depthOrArrayLayers?: number }
      ) => void;
    };
  };
  get?: (texture: THREE.Texture) => { texture?: unknown } | undefined;
  updateTexture?: (texture: THREE.Texture, options?: unknown) => void;
}

interface RangedTexture extends THREE.DataTexture {
  updateRanges: Array<{ start: number; count: number }>;
}

/**
 * True when `texture` matches the narrow element-texture shape the
 * ranged path handles. Exported for tests.
 */
export function isRangedUploadEligible(texture: THREE.Texture): texture is RangedTexture {
  const t = texture as RangedTexture;
  return (
    t.isDataTexture === true &&
    Array.isArray(t.updateRanges) &&
    t.updateRanges.length > 0 &&
    (t.mipmaps === undefined || t.mipmaps.length === 0) &&
    t.flipY === false &&
    ArrayBuffer.isView(t.image?.data)
  );
}

/**
 * Install the ranged-upload wrapper on a constructed WebGPURenderer.
 * Returns `true` when installed; `false` (with a warning log) when the
 * backend seam's shape has drifted — the renderer then keeps the stock
 * full-upload behavior, which is the correct status quo.
 *
 * Call AFTER `renderer.init()` (the backend instance must exist) and
 * only for the native backend (`backend.isWebGLBackend !== true`) —
 * renderer-setup owns that gate.
 */
export function installWebGPUPartialTextureUploads(renderer: {
  backend?: WebGPUBackendSeam;
}): boolean {
  const backend = renderer.backend;
  if (
    !backend ||
    typeof backend.updateTexture !== 'function' ||
    typeof backend.get !== 'function' ||
    typeof backend.device?.queue?.writeTexture !== 'function'
  ) {
    log.warning(
      Modules.RENDERER,
      'WebGPU partial-texture-upload seam shape drifted (backend.updateTexture/get/device.queue.writeTexture) — keeping stock full uploads.'
    );
    return false;
  }

  const original = backend.updateTexture.bind(backend);

  backend.updateTexture = (texture: THREE.Texture, options?: unknown): void => {
    if (!isRangedUploadEligible(texture)) {
      original(texture, options);
      return;
    }
    const entry = backend.get!(texture);
    const gpuTexture = entry?.texture;
    if (!gpuTexture) {
      // First sight (createTexture ran but our lookup came back empty) or
      // an unexpected registry shape — stock path is always correct.
      original(texture, options);
      return;
    }

    const image = texture.image as { data: ArrayBufferView; width: number; height: number };
    const width = image.width;
    const bytesPerRow = width * BYTES_PER_TEXEL;
    const data = image.data;

    // Each range is float-indexed and, by registerElementTexelDirtyRange's
    // no-row-straddle invariant, confined to one row. Per-range (= per-row)
    // writeTexture is ALSO the fast shape on Dawn/Metal (probe: per-row
    // beats the monolithic block). A malformed straddling range from a
    // foreign producer falls back to the stock full upload rather than
    // uploading wrong bytes.
    for (const range of (texture as RangedTexture).updateRanges) {
      const startTexel = range.start / 4;
      const row = Math.floor(startTexel / width);
      const x = startTexel - row * width;
      const texelCount = range.count / 4;
      if (
        !Number.isInteger(startTexel) ||
        !Number.isInteger(texelCount) ||
        texelCount <= 0 ||
        x + texelCount > width ||
        row >= image.height
      ) {
        original(texture, options);
        return;
      }
      backend.device!.queue!.writeTexture(
        { texture: gpuTexture, mipLevel: 0, origin: { x, y: row } },
        data,
        { offset: range.start * 4 /* float index → bytes */, bytesPerRow },
        { width: texelCount, height: 1 }
      );
    }

    // Consume-and-clear — the mirror of the classic renderer's
    // clear-after-upload; see the module header for why this is
    // load-bearing on WebGPU.
    texture.clearUpdateRanges();

    // Mirror the tail of the stock path that matters for luxar:
    // common/Textures.js invokes texture.onUpdate AFTER
    // backend.updateTexture returns (r184), which clears the
    // pendingFullUpload guard — nothing to replicate here.
  };

  log.info(Modules.RENDERER, 'WebGPU partial texture uploads installed (spec §7 Stage 3).');
  return true;
}
