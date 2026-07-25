/**
 * Element-texture layout — the single authority for how per-element
 * geometry data is laid out in an RGBA32F data texture (depth-sorting
 * plan, `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §4/§8).
 *
 * The core is geometry-agnostic and parameterized by an
 * {@link ElementTextureLayout} descriptor (texels per element + log
 * wording); the gsplat- and point-bound bindings below pin the two
 * concrete layouts:
 *
 * - **GSplats** ({@link SPLAT_TEXTURE_LAYOUT}): 4 texels/splat
 *   (16 floats):
 *
 *   | texel | rgba                                        |
 *   |-------|---------------------------------------------|
 *   | 0     | center.xyz, amplitude                       |
 *   | 1     | L00, L10, L11, L20                          |
 *   | 2     | L21, L22, color.rg                          |
 *   | 3     | color.b, alpha, 0, 0                        |
 *
 *   texel3.y is the per-splat opacity alpha (RGBA colors; written 1.0 =
 *   opaque when the dataset is RGB — pool textures are reused, so the
 *   writer never leaves it unspecified).
 *
 * - **Points** ({@link POINT_TEXTURE_LAYOUT}): 3 texels/point. The
 *   per-texel layout is documented in `point-geometry.ts` (the texel
 *   writer lives there too).
 *
 * - **Lines** ({@link LINE_TEXTURE_LAYOUT}): 6 texels/segment. The
 *   per-texel layout is documented in `line-geometry.ts` (the texel
 *   writer lives there too).
 *
 * The texture width is a **session constant**: `min(4096,
 * maxTextureSize)` rounded down to a multiple of the layout's
 * texels-per-element, configured once at renderer init from
 * `RendererCapabilities.maxTextureSize` (mirroring the
 * `gpu-byte-budget.ts` live-authority pattern — the GPU buffer pool is
 * constructed far from the renderer, so capabilities can't be threaded
 * through its constructor). Texture height varies per allocation:
 * `ceil(capacity × texelsPerElement / width)`. Shaders don't bake the
 * width at all — they read it per-draw via `textureSize(uSplatTex, 0)`,
 * so addressing is correct for any texture bound to the node (no
 * define lifecycle); only the texture binding itself changes per node.
 *
 * Unconfigured (unit tests, headless), the width defaults to 4096 and
 * the capacity bound assumes a 4096² texture — the conservative
 * WebGL2-era floor.
 */
import * as THREE from 'three';
import { log, Modules } from '../utils/log';

/** Descriptor parameterizing the generic layout math per geometry type. */
export interface ElementTextureLayout {
  /** Texels consumed per element (all four floats of each texel). */
  readonly texelsPerElement: number;
  /** texelsPerElement * 4. */
  readonly floatsPerElement: number;
  /** Element noun for log messages ('splat' | 'point' | 'segment'). */
  readonly label: string;
  /** Remediation hint appended to the capacity-clamp warning. */
  readonly clampHint: string;
}

/** Preferred (and maximum) element-texture width in texels. */
export const ELEMENT_TEXTURE_MAX_WIDTH = 4096;

let configuredMaxTextureSize: number | null = null;
const warnedCapacityClamp = new Set<ElementTextureLayout>();

/**
 * Configure the layout from the live renderer capabilities. Called
 * once per renderer init (`renderer-setup.ts`); safe to call again on
 * renderer swap — allocations pick up the new bound lazily. One shared
 * session value for every layout.
 */
export function configureElementTextureLayout(maxTextureSize: number): void {
  if (!Number.isFinite(maxTextureSize) || maxTextureSize <= 0) return;
  configuredMaxTextureSize = Math.floor(maxTextureSize);
}

/** Reset to the unconfigured default (tests only). */
export function resetElementTextureLayoutForTests(): void {
  configuredMaxTextureSize = null;
  warnedCapacityClamp.clear();
}

/** Effective max texture dimension (defaults to the 4096 floor). */
function effectiveMaxTextureSize(): number {
  return configuredMaxTextureSize ?? ELEMENT_TEXTURE_MAX_WIDTH;
}

/**
 * Element-texture width in texels — a session constant, forced to a
 * multiple of the layout's texels-per-element so an element's texels
 * always share one row (the shader fetches `base + k` on x only; a
 * row straddle would read garbage).
 */
export function getElementTextureWidth(layout: ElementTextureLayout): number {
  // Floor of one element: a sub-element maxTextureSize (impossible on
  // real devices — the WebGL2 spec floor is 2048) would otherwise round
  // to width 0 and divide-by-zero the height math.
  const capped = Math.min(ELEMENT_TEXTURE_MAX_WIDTH, effectiveMaxTextureSize());
  return Math.max(
    layout.texelsPerElement,
    Math.floor(capped / layout.texelsPerElement) * layout.texelsPerElement
  );
}

/**
 * Hard per-node element capacity: `width × maxTextureSize /
 * texelsPerElement` texels. For gsplats: 4.19M splats on a 4096-class
 * device, 8.38M at 8192 — comfortably above the 250K/part tiles idiom.
 */
export function getMaxElementCapacityPerNode(layout: ElementTextureLayout): number {
  return Math.floor(
    (getElementTextureWidth(layout) * effectiveMaxTextureSize()) / layout.texelsPerElement
  );
}

/**
 * Clamp a requested element capacity to the per-node texture bound,
 * warning once per session and layout on the first clamp (data loss —
 * the tail of the node's elements will never render; the layout's
 * `clampHint` names the fix).
 */
export function clampElementCapacity(requested: number, layout: ElementTextureLayout): number {
  const max = getMaxElementCapacityPerNode(layout);
  if (requested <= max) return requested;
  if (!warnedCapacityClamp.has(layout)) {
    warnedCapacityClamp.add(layout);
    const noun = layout.label.charAt(0).toUpperCase() + layout.label.slice(1);
    log.warning(
      Modules.GPU_BUFFER_POOL,
      `${noun} capacity ${requested.toLocaleString()} exceeds the per-node texture bound ` +
        `${max.toLocaleString()} (width ${getElementTextureWidth(layout)} × maxTextureSize ` +
        `${effectiveMaxTextureSize()}); clamping. ${layout.clampHint}`
    );
  }
  return max;
}

/** Texture height (rows) needed for `capacity` elements at the session width. */
export function elementTextureHeightForCapacity(
  capacity: number,
  layout: ElementTextureLayout
): number {
  return Math.max(
    1,
    Math.ceil((capacity * layout.texelsPerElement) / getElementTextureWidth(layout))
  );
}

let placeholderElementTexture: THREE.DataTexture | null = null;

/**
 * Shared 12×1 RGBA32F placeholder bound to element-texture materials
 * before their first commit rebinds the real pool texture. One
 * instance for the whole session — materials never own or dispose it.
 */
export function getPlaceholderElementTexture(): THREE.DataTexture {
  if (!placeholderElementTexture) {
    // 12×1: the LCM of the layouts' texels-per-element (4, 3 and 6), so the
    // "an element's texels never straddle a row" invariant the shader
    // prologues state holds for the placeholder too (an OOB texelFetch is
    // defined-safe in WebGL2, but keeping the invariant true costs nothing).
    placeholderElementTexture = new THREE.DataTexture(
      new Float32Array(12 * 4),
      12,
      1,
      THREE.RGBAFormat,
      THREE.FloatType
    );
    placeholderElementTexture.magFilter = THREE.NearestFilter;
    placeholderElementTexture.minFilter = THREE.NearestFilter;
    placeholderElementTexture.generateMipmaps = false;
    placeholderElementTexture.flipY = false;
    placeholderElementTexture.needsUpdate = true;
  }
  return placeholderElementTexture;
}

// ---------------------------------------------------------------------------
// GSplat-bound bindings (4 texels/splat)
// ---------------------------------------------------------------------------

/** The gsplat layout: 4 texels/splat (16 floats = 64 B in RGBA32F). */
export const SPLAT_TEXTURE_LAYOUT: ElementTextureLayout = {
  texelsPerElement: 4,
  floatsPerElement: 16,
  label: 'splat',
  clampHint:
    'Repartition the dataset (e.g. `luxar gsplat lod --recipe tiles`) to render every splat.',
};

/** Texels consumed per splat (16 floats = 64 B in RGBA32F). */
export const SPLAT_TEXELS_PER_SPLAT = SPLAT_TEXTURE_LAYOUT.texelsPerElement;

/** Floats per splat row in the texture's backing store. */
export const SPLAT_FLOATS_PER_SPLAT = SPLAT_TEXTURE_LAYOUT.floatsPerElement;

/** Splat-texture width in texels (multiple of 4 — see the generic core). */
export function getSplatTextureWidth(): number {
  return getElementTextureWidth(SPLAT_TEXTURE_LAYOUT);
}

/** Hard per-node splat capacity (see {@link getMaxElementCapacityPerNode}). */
export function getMaxSplatCapacityPerNode(): number {
  return getMaxElementCapacityPerNode(SPLAT_TEXTURE_LAYOUT);
}

/** Clamp a requested splat capacity (see {@link clampElementCapacity}). */
export function clampSplatCapacity(requested: number): number {
  return clampElementCapacity(requested, SPLAT_TEXTURE_LAYOUT);
}

/** Texture height (rows) needed for `capacity` splats at the session width. */
export function splatTextureHeightForCapacity(capacity: number): number {
  return elementTextureHeightForCapacity(capacity, SPLAT_TEXTURE_LAYOUT);
}

// ---------------------------------------------------------------------------
// Point-bound bindings (3 texels/point — per-texel layout and the texel
// writer live in point-geometry.ts)
// ---------------------------------------------------------------------------

/** The point layout: 3 texels/point (12 floats = 48 B in RGBA32F). */
export const POINT_TEXTURE_LAYOUT: ElementTextureLayout = {
  texelsPerElement: 3,
  floatsPerElement: 12,
  label: 'point',
  clampHint: 'Split the dataset into multiple nodes to render every point.',
};

/** Texels consumed per point (12 floats = 48 B in RGBA32F). */
export const POINT_TEXELS_PER_POINT = POINT_TEXTURE_LAYOUT.texelsPerElement;

/** Floats per point row in the texture's backing store. */
export const POINT_FLOATS_PER_POINT = POINT_TEXTURE_LAYOUT.floatsPerElement;

/** Point-texture width in texels (multiple of 3 — see the generic core). */
export function getPointTextureWidth(): number {
  return getElementTextureWidth(POINT_TEXTURE_LAYOUT);
}

/** Hard per-node point capacity (see {@link getMaxElementCapacityPerNode}). */
export function getMaxPointCapacityPerNode(): number {
  return getMaxElementCapacityPerNode(POINT_TEXTURE_LAYOUT);
}

/** Clamp a requested point capacity (see {@link clampElementCapacity}). */
export function clampPointCapacity(requested: number): number {
  return clampElementCapacity(requested, POINT_TEXTURE_LAYOUT);
}

/** Texture height (rows) needed for `capacity` points at the session width. */
export function pointTextureHeightForCapacity(capacity: number): number {
  return elementTextureHeightForCapacity(capacity, POINT_TEXTURE_LAYOUT);
}

// ---------------------------------------------------------------------------
// Line-bound bindings (6 texels/segment — per-texel layout and the texel
// writer live in line-geometry.ts)
// ---------------------------------------------------------------------------

/** The line layout: 6 texels/segment (24 floats = 96 B in RGBA32F). */
export const LINE_TEXTURE_LAYOUT: ElementTextureLayout = {
  texelsPerElement: 6,
  floatsPerElement: 24,
  label: 'segment',
  clampHint: 'Split the dataset into multiple nodes to render every segment.',
};

/** Texels consumed per segment (24 floats = 96 B in RGBA32F). */
export const LINE_TEXELS_PER_SEGMENT = LINE_TEXTURE_LAYOUT.texelsPerElement;

/** Floats per segment row in the texture's backing store. */
export const LINE_FLOATS_PER_SEGMENT = LINE_TEXTURE_LAYOUT.floatsPerElement;

/** Line-texture width in texels (multiple of 6 — see the generic core). */
export function getLineTextureWidth(): number {
  return getElementTextureWidth(LINE_TEXTURE_LAYOUT);
}

/** Hard per-node segment capacity (see {@link getMaxElementCapacityPerNode}). */
export function getMaxLineCapacityPerNode(): number {
  return getMaxElementCapacityPerNode(LINE_TEXTURE_LAYOUT);
}

/** Clamp a requested segment capacity (see {@link clampElementCapacity}). */
export function clampLineCapacity(requested: number): number {
  return clampElementCapacity(requested, LINE_TEXTURE_LAYOUT);
}

/** Texture height (rows) needed for `capacity` segments at the session width. */
export function lineTextureHeightForCapacity(capacity: number): number {
  return elementTextureHeightForCapacity(capacity, LINE_TEXTURE_LAYOUT);
}
