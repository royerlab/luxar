/**
 * Splat-texture layout — the single authority for how gsplat data is
 * laid out in the RGBA32F splat texture (Phase 1 of the depth-sorting
 * plan, `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §4).
 *
 * Each splat occupies **4 consecutive texels** (16 floats):
 *
 * | texel | rgba                                        |
 * |-------|---------------------------------------------|
 * | 0     | center.xyz, amplitude                       |
 * | 1     | cholesky01.xy, cholesky23.xy                |
 * | 2     | cholesky45.xy, color.rg                     |
 * | 3     | color.b, 0, 0, 0                            |
 *
 * The texture width is a **session constant**: `min(4096,
 * maxTextureSize)`, configured once at renderer init from
 * `RendererCapabilities.maxTextureSize` (mirroring the
 * `gpu-byte-budget.ts` live-authority pattern — the GPU buffer pool is
 * constructed far from the renderer, so capabilities can't be threaded
 * through its constructor). Texture height varies per allocation:
 * `ceil(capacity × 4 / width)`. Shaders don't bake the width at all —
 * they read it per-draw via `textureSize(uSplatTex, 0)`, so addressing
 * is correct for any texture bound to the node (no define lifecycle);
 * only the texture binding itself changes per node.
 *
 * Unconfigured (unit tests, headless), the width defaults to 4096 and
 * the capacity bound assumes a 4096² texture — the conservative
 * WebGL2-era floor.
 */
import * as THREE from 'three';
import { log, Modules } from '../utils/log';

/** Texels consumed per splat (16 floats = 64 B in RGBA32F). */
export const SPLAT_TEXELS_PER_SPLAT = 4;

/** Floats per splat row in the texture's backing store. */
export const SPLAT_FLOATS_PER_SPLAT = SPLAT_TEXELS_PER_SPLAT * 4;

/** Preferred (and maximum) splat-texture width in texels. */
export const SPLAT_TEXTURE_MAX_WIDTH = 4096;

let configuredMaxTextureSize: number | null = null;
let warnedCapacityClamp = false;

/**
 * Configure the layout from the live renderer capabilities. Called
 * once per renderer init (`renderer-setup.ts`); safe to call again on
 * renderer swap — allocations pick up the new bound lazily.
 */
export function configureSplatTextureLayout(maxTextureSize: number): void {
  if (!Number.isFinite(maxTextureSize) || maxTextureSize <= 0) return;
  configuredMaxTextureSize = Math.floor(maxTextureSize);
}

/** Reset to the unconfigured default (tests only). */
export function resetSplatTextureLayoutForTests(): void {
  configuredMaxTextureSize = null;
  warnedCapacityClamp = false;
}

/** Effective max texture dimension (defaults to the 4096 floor). */
function effectiveMaxTextureSize(): number {
  return configuredMaxTextureSize ?? SPLAT_TEXTURE_MAX_WIDTH;
}

/**
 * Splat-texture width in texels — a session constant, forced to a
 * multiple of 4 so a splat's 4 texels always share one row (the
 * shader fetches `base + (1|2|3)` on x only; a row straddle would
 * read garbage).
 */
export function getSplatTextureWidth(): number {
  return Math.min(SPLAT_TEXTURE_MAX_WIDTH, effectiveMaxTextureSize()) & ~3;
}

/**
 * Hard per-node splat capacity: `width × maxTextureSize / 4` texels.
 * 4.19M splats on a 4096-class device, 8.38M at 8192 — comfortably
 * above the 250K/part tiles idiom.
 */
export function getMaxSplatCapacityPerNode(): number {
  return Math.floor((getSplatTextureWidth() * effectiveMaxTextureSize()) / SPLAT_TEXELS_PER_SPLAT);
}

/**
 * Clamp a requested splat capacity to the per-node texture bound,
 * warning once per session on the first clamp (data loss — the tail
 * of the node's splats will never render; the tiles idiom is the fix).
 */
export function clampSplatCapacity(requested: number): number {
  const max = getMaxSplatCapacityPerNode();
  if (requested <= max) return requested;
  if (!warnedCapacityClamp) {
    warnedCapacityClamp = true;
    log.warning(
      Modules.GPU_BUFFER_POOL,
      `Splat capacity ${requested.toLocaleString()} exceeds the per-node texture bound ` +
        `${max.toLocaleString()} (width ${getSplatTextureWidth()} × maxTextureSize ` +
        `${effectiveMaxTextureSize()}); clamping. Repartition the dataset ` +
        '(e.g. `luxar gsplat lod --recipe tiles`) to render every splat.'
    );
  }
  return max;
}

/** Texture height (rows) needed for `capacity` splats at the session width. */
export function splatTextureHeightForCapacity(capacity: number): number {
  return Math.max(1, Math.ceil((capacity * SPLAT_TEXELS_PER_SPLAT) / getSplatTextureWidth()));
}

let placeholderSplatTexture: THREE.DataTexture | null = null;

/**
 * Shared 4×1 RGBA32F placeholder bound to gsplat materials before
 * their first commit rebinds the real pool texture. One instance for
 * the whole session — materials never own or dispose it.
 */
export function getPlaceholderSplatTexture(): THREE.DataTexture {
  if (!placeholderSplatTexture) {
    placeholderSplatTexture = new THREE.DataTexture(
      new Float32Array(16),
      4,
      1,
      THREE.RGBAFormat,
      THREE.FloatType
    );
    placeholderSplatTexture.magFilter = THREE.NearestFilter;
    placeholderSplatTexture.minFilter = THREE.NearestFilter;
    placeholderSplatTexture.generateMipmaps = false;
    placeholderSplatTexture.flipY = false;
    placeholderSplatTexture.needsUpdate = true;
  }
  return placeholderSplatTexture;
}
