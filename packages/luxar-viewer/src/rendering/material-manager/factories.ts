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

import type * as THREE from 'three';
import type { BlendingMode } from '../../types/blending';
import type { LineJoinStyle } from '../../types/line-join';
import type { LinePrimitive } from '../../types/line-primitive';
import { PointMaterial } from '../materials/point/material-glsl';
import { LineMaterial } from '../materials/line/material-glsl';
import type { MeshShadingMode } from '../materials/mesh/appearance';
import { GSplatMaterial } from '../materials/gsplat/material-glsl';
import { MeshMaterial } from '../materials/mesh/material-glsl';
import { PhysicalMeshMaterial } from '../materials/mesh-physical/material-glsl';
import { PointPickingMaterial } from '../picking/point/material';
import { LinePickingMaterial } from '../picking/line/material';
import { GSplatPickingMaterial } from '../picking/gsplat/material';
import { MeshPickingMaterial } from '../picking/mesh/material';
import { MegaShaderMaterial } from '../post-processing/mega/material';
import type { RendererCapabilities } from '../renderer-capabilities';
import { requireTslMaterials } from '../tsl/slot';

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

/** Point material properties driving the constructor config. */
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

/** Line material properties driving the constructor config. */
export interface LineMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  /** Absorption coefficient κ (volumetric mode; identity/default 1.0). */
  absorption?: number;
  gamma: number;
  intensity: number;
  offset: number;
  /**
   * Join style at degree-2 polyline joints (#790). Omitted ⇒ the session
   * default. No cache-key concern: line materials are PER NODE (each owns its
   * `uLineTex`), so unlike the point materials there is no LRU entry two nodes
   * with different styles could collide on — which matters here because the TSL
   * backend BAKES this into the graph.
   */
  join?: LineJoinStyle;
  /**
   * Line rendering primitive (#1352). Omitted ⇒ the session-wide
   * resolution (override > forced policy > default). `createLinesNode`
   * passes the per-node `auto`-policy result here so the built shader
   * matches the node's size class; same per-node/no-cache-key story as
   * `join` above.
   */
  primitive?: LinePrimitive;
}

/** GSplat material properties driving the constructor config. */
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
 * - only HALF a camera surface (see `LuxarMeshMaterial`): a mesh's size IS its
 *   geometry, so there is no screen-space extent to recompute per camera change and
 *   `updateCameraParams` ignores resolution/isOrtho — but it does consume
 *   `nearCull`, because the shared near fade applies to a surface too (#1431);
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
  /**
   * How normals are obtained: stored, screen-space derivatives, or none (unlit).
   *
   * One enum rather than the earlier `flatNormal` boolean, because a second
   * boolean for the unlit arm would admit a meaningless `flatNormal && noShading`
   * combination.
   */
  shading?: MeshShadingMode;
  /** Base-colour texture, sampled per fragment. Excludes the other two sources. */
  baseColorTexture?: THREE.Texture;
  /** Whether {@link baseColorTexture} is single-channel (red replicated to RGB). */
  baseColorTextureLuminance?: boolean;
  /**
   * Wrapped-diffuse shade floor, clamped to `[0, 1]` (`1.0` = flat diffuse). Optional
   * because the writer never stamps it — it reaches here only when an author passed
   * it through `add_mesh(**attrs)`.
   */
  ambient?: number;
  /** Wrapped-diffuse exponent, clamped positive. */
  shadeExponent?: number;
  /** Additive specular strength, clamped to `[0, 1]`. */
  specular?: number;
  /** Specular exponent, clamped positive. */
  shininess?: number;
  /** `opaque`-mode cutout threshold, clamped to `[0, 1]`. */
  alphaCutoff?: number;
}

/**
 * The material backend tag used as the index into the factory
 * tables. `'tsl'` selects the `NodeMaterial`-derived
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
 * type. `MaterialManager.get{Point,Line,GSplat,Mesh}Material` looks up
 * `VISUAL_FACTORIES[kind][backend]()` to pick the class to instantiate.
 *
 * Every cell is a **thunk** rather than the class itself. The `glsl` ones do
 * not need to be, but keeping both backends the same shape is what lets the
 * call sites stay a uniform `[backend]()` lookup — and the `tsl` ones must be,
 * because their classes live behind the lazy `three/webgpu` boundary
 * (`rendering/tsl/load.ts`) and do not exist until it has been awaited. A thunk
 * that resolves late is also why the laziness is visible at the point of use
 * instead of hiding in a getter that throws when a debugger inspects it.
 */
export const VISUAL_FACTORIES = {
  point: { glsl: () => PointMaterial, tsl: () => requireTslMaterials().materials.point },
  line: { glsl: () => LineMaterial, tsl: () => requireTslMaterials().materials.line },
  gsplat: { glsl: () => GSplatMaterial, tsl: () => requireTslMaterials().materials.gsplat },
  mesh: { glsl: () => MeshMaterial, tsl: () => requireTslMaterials().materials.mesh },
  /**
   * The mesh's second material FAMILY, not a fifth geometry type: three's own
   * physically based material behind the Luxar leaf surface
   * (`MESH_PHYSICAL_MATERIALS_SPEC.md` §3.2). A separate key rather than a variant of
   * `mesh` because none of the house contracts apply to it — no codegen snapshot, no
   * per-epoch shading define, no blend-mode state — and, deliberately, NO matching
   * `PICKING_FACTORIES` entry: picking renders geometry, not appearance, so a physical
   * mesh picks through the house `mesh` pick material.
   */
  meshPhysical: {
    glsl: () => PhysicalMeshMaterial,
    tsl: () => requireTslMaterials().materials.meshPhysical,
  },
} as const;

/**
 * The geometry KINDS — the keys that must have both a visual and a picking pair.
 * `meshPhysical` is excluded on purpose (see its entry above).
 */
export const GEOMETRY_KINDS = ['point', 'line', 'gsplat', 'mesh'] as const;

/**
 * Constructor table for the picking material pair of each geometry
 * type. Mirror of {@link VISUAL_FACTORIES} for the picking pipeline;
 * the `create*PickingMaterial` methods look up
 * `PICKING_FACTORIES[kind][backend]()` and instantiate it directly
 * (picking materials are not cached).
 */
export const PICKING_FACTORIES = {
  point: { glsl: () => PointPickingMaterial, tsl: () => requireTslMaterials().picking.point },
  line: { glsl: () => LinePickingMaterial, tsl: () => requireTslMaterials().picking.line },
  gsplat: { glsl: () => GSplatPickingMaterial, tsl: () => requireTslMaterials().picking.gsplat },
  mesh: { glsl: () => MeshPickingMaterial, tsl: () => requireTslMaterials().picking.mesh },
} as const;

/**
 * Constructor pair for the post-processing mega-shader. Looked up by
 * `createMegaShaderMaterial`; one entry per backend, no per-geometry
 * indirection (there is only one mega-shader).
 */
export const MEGA_SHADER_FACTORIES = {
  glsl: () => MegaShaderMaterial,
  tsl: () => requireTslMaterials().mega,
} as const;

// (The historical `lineCacheKey` + bucketing helpers are gone: line
// materials went PER NODE with the texture-backed storage migration —
// each carries the node's own `uLineTex` — so no material kind is
// cached or keyed anymore.)
