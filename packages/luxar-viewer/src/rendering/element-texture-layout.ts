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
 *   | 3     | color.b, alpha, label_index, 0              |
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
 * `ceil(capacity × texelsPerElement / width)`. Shaders bake the width
 * as a COMPILE-TIME constant (the layout's `widthDefine` on the
 * GLSL materials; a literal int node in the TSL graphs), read from
 * {@link getElementTextureWidth} at material construction. A constant
 * (unlike the per-draw `textureSize(uSplatTex, 0)` query this replaced)
 * lets the shader compiler strength-reduce the per-vertex `%`/`/`
 * addressing math — measured −7% on the quad line primitive's whole
 * GPU pass at 4 M segments (RTX 3070, WebGPU timestamps); a uniform
 * captured almost none of that (−1%), so a define it is. The baked
 * value is correct because the ADDRESSING AUTHORITY is the bound
 * texture's own width: materials pre-stamp the session width at
 * construction and re-stamp from the texture at every bind
 * ({@link applyElementTextureWidthDefine}); the TSL graphs re-bake on
 * texture rebind (their texture node is factory-time bound, so a
 * rebind rebuilds the graph). Pool textures allocate at the session
 * width (capped at {@link ELEMENT_TEXTURE_MAX_WIDTH}), so re-stamps
 * are no-ops in the common path and nothing recompiles.
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
  /**
   * Name of the GLSL define carrying this layout's texture width
   * (see {@link elementTextureWidthDefines}).
   */
  readonly widthDefine: string;
  /** Remediation hint appended to the capacity-clamp warning. */
  readonly clampHint: string;
}

/** Preferred (and maximum) element-texture width in texels. */
export const ELEMENT_TEXTURE_MAX_WIDTH = 4096;

/**
 * The GLSL `#define`s carrying each layout's element-texture width.
 * Pre-stamped at material construction (visual + picking, all three
 * geometry types) with `String(getElementTextureWidth(layout))` via
 * the layout's own `widthDefine` name, then re-stamped from the bound
 * texture on every texture update ({@link
 * applyElementTextureWidthDefine}) — pool textures allocate at the
 * session width, so the re-stamp is a no-op in the common path. The
 * TSL twins bake the same value as a literal int node at graph build
 * instead of a define ({@link resolveElementTextureWidth}). The names are PER GEOMETRY
 * (widths differ: texels-per-element 6/3/4 → 4092/4095/4096) so a
 * harness compiling raw shader sources can inject all of them
 * unconditionally via {@link elementTextureWidthDefines} — an unused
 * define is inert.
 */
export function elementTextureWidthDefines(): Record<string, string> {
  const defines: Record<string, string> = {};
  for (const layout of [LINE_TEXTURE_LAYOUT, POINT_TEXTURE_LAYOUT, SPLAT_TEXTURE_LAYOUT]) {
    defines[layout.widthDefine] = String(getElementTextureWidth(layout));
  }
  return defines;
}

/**
 * Width to bake for a material about to draw `texture`: the texture's
 * own width when a real one is bound (bind-time authority — see
 * {@link applyElementTextureWidthDefine}), else the session width
 * (constructor pre-stamp / placeholder phase, whose draws are empty).
 * The TSL factories call this at graph build — the texture node is
 * factory-time bound and a rebind rebuilds the graph, so the literal
 * always matches the texture the graph will sample.
 */
export function resolveElementTextureWidth(
  layout: ElementTextureLayout,
  texture: { image?: { width?: number } } | null | undefined
): number {
  if (texture && texture !== (placeholderElementTexture as unknown)) {
    const width = texture.image?.width;
    if (typeof width === 'number' && Number.isFinite(width) && width > 0) return width;
  }
  return getElementTextureWidth(layout);
}

/**
 * Re-stamp a material's width define from the texture actually being
 * bound. The addressing authority is the BOUND texture's own width,
 * never the global session value (same rationale as the row math in
 * `element-storage.ts`: a renderer swap can reconfigure the session
 * width while an existing texture keeps its allocated width). The
 * constructor pre-stamps the session width so the common first bind —
 * a pool texture allocated at that same width — changes nothing and
 * never recompiles; this only triggers a program rebuild (cached by
 * define set) when a genuinely different-width texture binds
 * (harness fixtures, post-swap stragglers).
 *
 * The placeholder binding is exempt: its draws are empty
 * (`instanceCount` 0 until the first commit), so restamping for its
 * 12-texel width would only force a pointless recompile round-trip.
 */
export function applyElementTextureWidthDefine(
  material: { defines?: Record<string, unknown>; needsUpdate: boolean },
  layout: ElementTextureLayout,
  texture: THREE.Texture | null
): void {
  if (!texture || texture === placeholderElementTexture) return;
  const width = (texture as THREE.DataTexture).image?.width;
  if (!Number.isFinite(width) || width <= 0) return;
  const next = String(width);
  if (!material.defines) material.defines = {};
  if (material.defines[layout.widthDefine] !== next) {
    material.defines[layout.widthDefine] = next;
    material.needsUpdate = true;
  }
}

let configuredMaxTextureSize: number | null = null;
/**
 * Keyed by `${layout.label}:${requested}`, NOT by the layout object. A clamp is
 * silent DATA LOSS, and a per-layout flag reports only the FIRST offender: a
 * scene with two oversized Lines nodes announced one of them and dropped the
 * other's tail without a word. Keying on the requested count reports each
 * distinct clamped size once, while still collapsing per-frame repeats. Nodes
 * with the same label and requested size intentionally fold into one line.
 */
const warnedCapacityClamp = new Set<string>();

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
 * reporting each distinct clamped size once (data loss — the tail of the
 * node's elements will never render; the layout's `clampHint` names the
 * fix). Identically sized offenders of one geometry type share that report.
 *
 * Reported at ERROR level, not warning. A clamp means part of the scene the
 * author asked for is not on screen and never will be, and because elements
 * are written in spatial (Hilbert / BSP) order the dropped tail is one
 * COMPACT REGION rather than a thin scatter — it reads as a hole in the data,
 * not as degraded quality. That is indistinguishable from a broken dataset
 * unless the message stands out from the couple of hundred ordinary log lines
 * a scene load emits (#1957, where an ocean-currents node lost the North
 * Atlantic and the one `console.warn` saying so went unnoticed).
 *
 * The Python writers warn about the same overflow at AUTHORING time
 * (`io/_compiler/node_common.py::warn_if_over_element_cap`, #1957), but only
 * against the 4096-class floor and only for scenes Luxar authored. This is the
 * load-time backstop: it knows the REAL device bound and fires for any store,
 * however it was produced.
 */
export function clampElementCapacity(
  requested: number,
  layout: ElementTextureLayout,
  reportLoss = true
): number {
  const max = getMaxElementCapacityPerNode(layout);
  if (requested <= max) return requested;
  const key = `${layout.label}:${requested}`;
  if (reportLoss && !warnedCapacityClamp.has(key)) {
    warnedCapacityClamp.add(key);
    const noun = layout.label.charAt(0).toUpperCase() + layout.label.slice(1);
    log.error(
      Modules.GPU_BUFFER_POOL,
      `${noun} capacity ${requested.toLocaleString()} exceeds the per-node texture bound ` +
        `${max.toLocaleString()} (width ${getElementTextureWidth(layout)} × maxTextureSize ` +
        `${effectiveMaxTextureSize()}); clamping — the last ` +
        `${(requested - max).toLocaleString()} ${layout.label}s of this node will NEVER ` +
        'render, and being spatially ordered they are one contiguous region of the scene. ' +
        `${layout.clampHint}`
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
  widthDefine: 'LUXAR_SPLAT_TEX_W',
  clampHint:
    'Repartition the dataset (e.g. `luxar gsplat lod --recipe tiles`) to render every splat.',
};

/** Floats per splat row in the texture's backing store. */
export const SPLAT_FLOATS_PER_SPLAT = SPLAT_TEXTURE_LAYOUT.floatsPerElement;

/** Clamp a requested splat capacity (see {@link clampElementCapacity}). */
export function clampSplatCapacity(requested: number, reportLoss = true): number {
  return clampElementCapacity(requested, SPLAT_TEXTURE_LAYOUT, reportLoss);
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
  widthDefine: 'LUXAR_POINT_TEX_W',
  clampHint: 'Split the dataset into multiple nodes to render every point.',
};

/** Floats per point row in the texture's backing store. */
export const POINT_FLOATS_PER_POINT = POINT_TEXTURE_LAYOUT.floatsPerElement;

/** Clamp a requested point capacity (see {@link clampElementCapacity}). */
export function clampPointCapacity(requested: number, reportLoss = true): number {
  return clampElementCapacity(requested, POINT_TEXTURE_LAYOUT, reportLoss);
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
  widthDefine: 'LUXAR_LINE_TEX_W',
  clampHint: 'Split the dataset into multiple nodes to render every segment.',
};

/** Floats per segment row in the texture's backing store. */
export const LINE_FLOATS_PER_SEGMENT = LINE_TEXTURE_LAYOUT.floatsPerElement;

/** Clamp a requested segment capacity (see {@link clampElementCapacity}). */
export function clampLineCapacity(requested: number, reportLoss = true): number {
  return clampElementCapacity(requested, LINE_TEXTURE_LAYOUT, reportLoss);
}
