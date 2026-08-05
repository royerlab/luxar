/**
 * Material factory tables + backend resolution.
 *
 * Keeps the `MaterialManager` class focused on lifecycle
 * orchestration.
 *
 * Everything here is **stateless** — pure constructor lookups. The
 * single piece of state (renderer capabilities) is threaded through
 * `resolveMaterialBackend` by the caller; we don't import the
 * singleton.
 *
 * @module rendering/material-manager/factories
 */

import type { BlendingMode } from '../../types/blending';
import { PointMaterial } from '../materials/point/material-glsl';
import { LineMaterial } from '../materials/line/material-glsl';
import { GSplatMaterial } from '../materials/gsplat/material-glsl';
import { MeshMaterial } from '../materials/mesh/material-glsl';
import { PointTSLMaterial } from '../materials/point/material-tsl';
import { LineTSLMaterial } from '../materials/line/material-tsl';
import { GSplatTSLMaterial } from '../materials/gsplat/material-tsl';
import { MeshTSLMaterial } from '../materials/mesh/material-tsl';
import { PointPickingMaterial } from '../picking/point/material';
import { LinePickingMaterial } from '../picking/line/material';
import { GSplatPickingMaterial } from '../picking/gsplat/material';
import { PointPickingTSLMaterial } from '../picking/point/material-tsl';
import { LinePickingTSLMaterial } from '../picking/line/material-tsl';
import { GSplatPickingTSLMaterial } from '../picking/gsplat/material-tsl';
import { MegaShaderMaterial } from '../post-processing/mega/material';
import { MegaShaderTSLMaterial } from '../post-processing/mega/material-tsl';
import type { RendererCapabilities } from '../renderer-capabilities';

/**
 * Supported blending modes for materials.
 *
 * - 'normal': Standard alpha blending (semi-transparent). For
 *   **Points** and **Lines** the shader emits a straight per-fragment
 *   alpha (opacity × edge softness) over SrcAlpha/OneMinusSrcAlpha.
 *   For **GSplats** the shader emits PREMULTIPLIED coverage alpha
 *   (`clamp(intensity·opacity, 0, 1)`, `LUXAR_NORMAL_PREMULT` define)
 *   over `One / OneMinusSrcAlpha` — see
 *   `getGSplatNormalBlendingState()` in blending-state.ts. The
 *   framebuffer behind a splat IS revealed in proportion to coverage;
 *   dim splats occlude proportionally little (emitter-with-occlusion,
 *   deliberate for HDR scientific data). GSplat normal mode never
 *   writes depth, so it does not occlude additive layers behind it.
 *   Normal-mode gsplat compositing is order-dependent; the viewer
 *   depth-sorts them per frame (GSPLAT_DEPTH_SORTING_SPEC.md, shipped
 *   Phases 0-3).
 * - 'additive': Classic additive blending, ignores depth (renders on top of everything)
 * - 'max': Maximum of source and destination (brightest wins)
 * - 'opaque': Solid rendering with depth write (closest object wins)
 * - 'luminous': Same as additive visually, but respects depth occlusion (occluded by closer objects)
 * - 'volumetric': Emission–absorption (VOLUMETRIC_BLENDING_SPEC.md). All
 *   three geometry types: the shader emits premultiplied self-screened
 *   emission with the physical absorption alpha `1 − e^(−τ)`,
 *   τ = κ·opacity·rayMass (`LUXAR_VOLUMETRIC` define, `uAbsorption`
 *   uniform; for every family rayMass is the SAME quantity its additive
 *   branch emits — see the ray-mass unification banner in
 *   VOLUMETRIC_BLENDING_SPEC.md), over the same `One / OneMinusSrcAlpha`
 *   state as gsplat normal; order-dependent and depth-sorted
 *   (`needsDepthSort`), never depth-writes; κ = 0 renders exactly like
 *   'additive'.
 *
 * Re-exported from `types/blending.ts` — the single source of truth for
 * the mode set (adding a mode there updates this union, the per-node attr
 * types, `normalizeBlendingMode`, and the panel dropdown together).
 */
export type { BlendingMode };

/** Point material properties driving cache key + constructor config. */
export interface PointMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  /** Absorption coefficient κ (volumetric mode; identity/default 1.0). */
  absorption?: number;
  gamma: number;
  intensity: number;
  offset: number;
  /** Scale factor for radius normalization (e.g., 1/255 for uint8) */
  radiusScale?: number;
}

/** Line material properties driving cache key + constructor config. */
export interface LineMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  /** Absorption coefficient κ (volumetric mode; identity/default 1.0). */
  absorption?: number;
  gamma: number;
  intensity: number;
  offset: number;
}

/** GSplat material properties driving cache key + constructor config. */
export interface GSplatMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  /** Absorption coefficient κ (volumetric mode; identity/default 1.0). */
  absorption?: number;
  gamma: number;
  intensity: number;
  offset: number;
  /** Default 3.0 */
  truncationRadius?: number;
}

/**
 * Mesh material properties driving the constructor config.
 *
 * The compositing half matches its three siblings field for field. What differs is
 * named rather than quietly omitted, because each omission is a §9 exclusion or a
 * §2.2 structural difference rather than an oversight:
 *
 * - no `absorption` — that uniform exists only for `volumetric`, and a
 *   zero-thickness surface has no path length for it to attenuate over (§6.3). The
 *   `hasElementAlpha` flag goes with it: it gates nothing but the volumetric
 *   `w(a) = −ln(1−a)` optical-depth map, and mesh's per-vertex alpha is a plain
 *   coverage term in every mode it supports, so there is nothing to gate;
 * - no `radiusScale` / `truncationRadius` — both normalize a per-element extent,
 *   and a triangle's extent is its own vertices;
 * - no camera surface at all (see `LuxarMeshMaterial`): a mesh's size IS its
 *   geometry, so there is no screen-space extent to recompute per camera change;
 * - `blendingMode` defaults to `'opaque'`, not `'additive'` — the only mode
 *   unconditionally correct without per-triangle depth sorting (§6.3);
 * - `flatNormal` is new: mesh is the first shaded type, and the stored-normal vs
 *   derivative-normal choice is a compile-time shader variant the caller resolves
 *   once per node.
 */
export interface MeshMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  gamma: number;
  intensity: number;
  offset: number;
  /** Shade from screen-space derivatives instead of the stored `normal` attribute. */
  flatNormal?: boolean;
  /**
   * Headlight shade floor, clamped to `[0, 1]` (`1.0` = flat/emissive). Optional
   * because the writer never stamps it — it reaches here only when an author passed
   * it through `add_mesh(**attrs)`.
   */
  ambient?: number;
  /** Headlight wrap exponent, clamped positive. */
  shadeExponent?: number;
  /** `opaque`-mode cutout threshold, clamped to `[0, 1]`. */
  alphaCutoff?: number;
}

/**
 * The material backend tag used in cache keys and as the index into
 * the factory tables. `'tsl'` selects the `NodeMaterial`-derived
 * implementation built for WebGPURenderer; `'glsl'` selects the
 * `ShaderMaterial`-derived implementation for `THREE.WebGLRenderer`.
 */
export type MaterialBackend = 'glsl' | 'tsl';

/**
 * Resolve the active material backend from a `RendererCapabilities`
 * snapshot. Returns `'glsl'` when caps are unset so unit tests that
 * touch material creation without configuring caps get the WebGL2
 * dispatch — same default as before this helper existed.
 */
export function resolveMaterialBackend(caps: RendererCapabilities | null): MaterialBackend {
  return caps?.apiSurface === 'webgpu' ? 'tsl' : 'glsl';
}

/**
 * Constructor table for the visual material pair of each geometry
 * type. `MaterialManager.get{Point,Line,GSplat}Material` looks up
 * `VISUAL_FACTORIES[kind][backend]` to pick the class to instantiate.
 */
export const VISUAL_FACTORIES = {
  point: { glsl: PointMaterial, tsl: PointTSLMaterial },
  line: { glsl: LineMaterial, tsl: LineTSLMaterial },
  gsplat: { glsl: GSplatMaterial, tsl: GSplatTSLMaterial },
  mesh: { glsl: MeshMaterial, tsl: MeshTSLMaterial },
} as const;

/**
 * Constructor table for the picking material pair of each geometry
 * type. Mirror of {@link VISUAL_FACTORIES} for the picking pipeline;
 * the `create*PickingMaterial` methods look up
 * `PICKING_FACTORIES[kind][backend]` and instantiate it directly
 * (picking materials are not cached).
 */
export const PICKING_FACTORIES = {
  point: { glsl: PointPickingMaterial, tsl: PointPickingTSLMaterial },
  line: { glsl: LinePickingMaterial, tsl: LinePickingTSLMaterial },
  gsplat: { glsl: GSplatPickingMaterial, tsl: GSplatPickingTSLMaterial },
} as const;

/**
 * Constructor pair for the post-processing mega-shader. Looked up by
 * `createMegaShaderMaterial`; one entry per backend, no per-geometry
 * indirection (there is only one mega-shader).
 */
export const MEGA_SHADER_FACTORIES = {
  glsl: MegaShaderMaterial,
  tsl: MegaShaderTSLMaterial,
} as const;

// (The historical `lineCacheKey` + bucketing helpers are gone: line
// materials went PER NODE with the texture-backed storage migration —
// each carries the node's own `uLineTex` — so no material kind is
// cached or keyed anymore.)
