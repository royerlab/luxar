/**
 * Material Manager for Luxar
 *
 * This module manages all materials in the scene, providing caching,
 * global uniform updates, and support for multiple material types.
 * Supports point, line, and GSplat materials.
 */

import * as THREE from 'three';
import { PointMaterial } from './materials/point/material-glsl';
import { LineMaterial } from './materials/line/material-glsl';
import { GSplatMaterial } from './materials/gsplat/material-glsl';
import { MeshMaterial } from './materials/mesh/material-glsl';
import { PointTSLMaterial } from './materials/point/material-tsl';
import { LineTSLMaterial } from './materials/line/material-tsl';
import { GSplatTSLMaterial } from './materials/gsplat/material-tsl';
import { MeshTSLMaterial } from './materials/mesh/material-tsl';
import type { PointPickingMaterial } from './picking/point/material';
import type { LinePickingMaterial } from './picking/line/material';
import type { GSplatPickingMaterial } from './picking/gsplat/material';
import type { PointPickingTSLMaterial } from './picking/point/material-tsl';
import type { LinePickingTSLMaterial } from './picking/line/material-tsl';
import type { GSplatPickingTSLMaterial } from './picking/gsplat/material-tsl';
import type { MeshPickingMaterial } from './picking/mesh/material';
import type { MeshPickingTSLMaterial } from './picking/mesh/material-tsl';
import type { MegaShaderMaterial } from './post-processing/mega/material';
import type { MegaShaderTSLMaterial } from './post-processing/mega/material-tsl';
import type { PointPickingMaterialConfig } from './picking/point/material';
import type { LinePickingMaterialConfig } from './picking/line/material';
import type { GSplatPickingMaterialConfig } from './picking/gsplat/material';
import type { MeshPickingMaterialConfig } from './picking/mesh/material';
import type { MegaShaderConfig } from './post-processing/mega/material';
import {
  isCameraAwareMaterial,
  type CameraAwareMaterial,
} from './materials/_shared/camera-aware-material';
import type { RendererCapabilities } from './renderer-capabilities';
import { log, Modules } from '../utils/log';
import {
  VISUAL_FACTORIES,
  PICKING_FACTORIES,
  MEGA_SHADER_FACTORIES,
  resolveMaterialBackend,
  type BlendingMode,
  type PointMaterialProperties,
  type LineMaterialProperties,
  type GSplatMaterialProperties,
  type MeshMaterialProperties,
  type MaterialBackend,
} from './material-manager/factories';
import {
  SOFT_DISPOSE_FLAG,
  subscribeToDispose,
  removeFromRegistries,
  type LifecycleCtx,
} from './material-manager/lifecycle';
import { getCacheStats as buildCacheStats } from './material-manager/stats';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../config/constants';

// Re-export the sentinel for callers that import from material-manager.
export { SOFT_DISPOSE_FLAG };

// Re-export factory types so external callers don't need to know
// about the helper subfolder — the material-manager module remains
// the single public surface for material configuration.
export {
  resolveMaterialBackend,
  type BlendingMode,
  type PointMaterialProperties,
  type LineMaterialProperties,
  type GSplatMaterialProperties,
  type MeshMaterialProperties,
  type MaterialBackend,
};

/**
 * Per-geometry-type material returned by
 * `MaterialManager.get{Point,Line,GSplat}Material`. The concrete
 * class is either the GLSL `*Material` (WebGL2 path) or the TSL
 * `*TSLMaterial` (WebGPU / fallback-via-WebGPURenderer path). Both
 * classes expose the same surface — `updateOpacity`,
 * `updateCameraParams`, `applyBlendingMode`, `clone()`, etc. — so
 * call sites treat the return type as a single union.
 */
export type LuxarPointMaterial = PointMaterial | PointTSLMaterial;
export type LuxarLineMaterial = LineMaterial | LineTSLMaterial;
export type LuxarGSplatMaterial = GSplatMaterial | GSplatTSLMaterial;
/**
 * Same union shape as its three siblings, with one member of the shared surface
 * absent on purpose: no `updateCameraParams`. A mesh has no screen-space size to
 * recompute, so it is not camera-aware and does not join the camera broadcast — see
 * `LifecycleCtx.staticMaterials`.
 */
export type LuxarMeshMaterial = MeshMaterial | MeshTSLMaterial;

/**
 * Per-geometry-type picking material returned by
 * `MaterialManager.create{Point,Line,GSplat,Mesh}PickingMaterial`.
 * Picking materials have a per-mesh lifetime (not cached).
 */
export type LuxarPointPickingMaterial = PointPickingMaterial | PointPickingTSLMaterial;
export type LuxarLinePickingMaterial = LinePickingMaterial | LinePickingTSLMaterial;
export type LuxarGSplatPickingMaterial = GSplatPickingMaterial | GSplatPickingTSLMaterial;
/**
 * Like its three siblings, minus `updateCameraParams` — the mesh pick pass has no
 * screen-space footprint to size, so it is not camera-aware and joins
 * `staticMaterials` rather than the camera broadcast, exactly as the visual mesh
 * material does.
 */
export type LuxarMeshPickingMaterial = MeshPickingMaterial | MeshPickingTSLMaterial;

/**
 * The single post-processing mega-shader material managed by
 * `PostProcessingManager`.
 */
export type LuxarMegaShaderMaterial = MegaShaderMaterial | MegaShaderTSLMaterial;

/**
 * Manages all materials in the scene with lifecycle tracking and
 * global camera-uniform updates. Supports points, lines, and gsplats.
 *
 * ALL visual materials are PER NODE (each carries the node's own
 * element texture — `uPointTex` / `uLineTex` / `uSplatTex`) and are never
 * cached: sharing one would rebind a node's texture onto another node's mesh at
 * every commit. The historical line-material LRU was the last cached kind and
 * died with the lines texture-storage migration.
 */
export class MaterialManager {
  private registeredMaterials = new Set<THREE.Material & CameraAwareMaterial>();
  /**
   * Renderer capabilities — drives the GLSL vs. TSL dispatch in the
   * three `getXMaterial` factories. `SceneManager.setupRenderer` calls
   * {@link setCaps} once the renderer is alive; before that hook
   * fires, the manager defaults to the WebGL2 path so unit tests
   * that touch material creation don't need to know about caps.
   */
  private caps: RendererCapabilities | null = null;
  /**
   * Materials whose `dispose` event we have already wired a listener for.
   * Separate from `registeredMaterials` because `register()` /
   * `getXMaterial()` can be called repeatedly with the same instance
   * (cache hits, clones re-registered explicitly), and a second
   * `addEventListener('dispose', ...)` would silently stack listeners on
   * THREE's EventDispatcher.
   */
  private subscribedMaterials = new WeakSet<THREE.Material & CameraAwareMaterial>();
  /**
   * Diagnostic: cumulative wall-clock time spent constructing
   * materials (Point/Line/GSplat). Useful as a proxy for "how much
   * time does the user spend waiting for material-creation work" —
   * first-use stutter shows up as a single large delta on the
   * affected animation frame. Exposed in `getCacheStats()`.
   */
  private totalCreateMs = 0;
  private createCount = 0;
  /**
   * Materials that entered through `register()` rather than a manager
   * factory (per-node point/line/gsplat/mesh materials live in
   * `registeredMaterials` / `staticMaterials` only).
   *
   * Examples: GPU-picking materials and colormap clones. These need
   * manager-level disposal — and global camera uniforms when they are
   * camera-aware — but are tracked separately for leak diagnostics.
   *
   * Typed as plain `THREE.Material` because `register()` accepts one: a
   * non-camera-aware entry (a mesh colormap clone) belongs in the leak
   * diagnostic exactly as much as a camera-aware one, so this set must
   * span both rather than silently omitting half of them.
   */
  private ownedMaterials = new Set<THREE.Material>();
  /**
   * Per-node materials that are tracked for disposal but take NO camera broadcast.
   *
   * Mesh materials only, and structurally so: a mesh draws real geometry, so it has
   * no screen-space extent to recompute from fov/resolution and therefore no
   * `updateCameraParams`. Adding an empty one just to fit `registeredMaterials`
   * would be a lie that also costs a per-frame call per node — see
   * {@link LifecycleCtx.staticMaterials}.
   */
  private staticMaterials = new Set<THREE.Material>();
  private currentFov = (60 * Math.PI) / 180; // Current FOV in radians (or frustumHeight for ortho)
  private currentResolution = new THREE.Vector2(1920, 1080); // Use reasonable default
  private currentIsOrtho = false;
  private currentNearCull: number | undefined = undefined;

  /**
   * Build the lifecycle context handed to `subscribeToDispose` and
   * `removeFromRegistries`. Lazy getter — captures `this` references
   * once, reused across calls.
   */
  private get lifecycleCtx(): LifecycleCtx {
    return {
      registeredMaterials: this.registeredMaterials,
      ownedMaterials: this.ownedMaterials,
      staticMaterials: this.staticMaterials,
      subscribedMaterials: this.subscribedMaterials,
    };
  }

  /**
   * Set the renderer capabilities. Called once by SceneManager after
   * the renderer is alive. Determines which backend the dispatch in
   * `getPointMaterial` (and Line/GSplat equivalents) picks.
   *
   * Materials already handed out keep their original class; switching caps only
   * affects which class subsequent calls construct.
   */
  setCaps(caps: RendererCapabilities): void {
    this.caps = caps;
  }

  /**
   * Create a point material — PER NODE, no LRU cache.
   *
   * Point data lives in a per-node texture (`uPointTex`), so two nodes
   * can never share a point material: sharing would rebind one node's
   * texture onto another's mesh at every commit. Every call creates a
   * fresh material that the node owns for its lifetime (the node
   * factory stamps `_layerMaterialCloned: true`, so LayersPanel /
   * LOD-cross-fade mutate it directly instead of clone-on-first-use).
   * `pointMaterialCache` stays permanently empty — it remains in the
   * lifecycle/stats context shapes shared with lines, where an empty
   * map is a truthful no-op. Mirrors {@link getGSplatMaterial}.
   *
   * Dispatches to `PointTSLMaterial` (NodeMaterial / TSL) when the
   * active renderer reports `caps.apiSurface === 'webgpu'`, otherwise to the
   * GLSL `PointMaterial`. Both classes expose the same update surface
   * (see {@link LuxarPointMaterial}), so callers in node-factory and
   * the layers panel don't need to branch.
   */
  getPointMaterial(props: PointMaterialProperties): LuxarPointMaterial {
    const backend = resolveMaterialBackend(this.caps);

    const createStart = performance.now();
    const material = new VISUAL_FACTORIES.point[backend]({
      opacity: props.opacity,
      absorption: props.absorption,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
      radiusScale: props.radiusScale,
    });
    this.totalCreateMs += performance.now() - createStart;
    this.createCount++;

    this.registeredMaterials.add(material);
    subscribeToDispose(material, this.lifecycleCtx);
    material.updateCameraParams(
      this.currentFov,
      this.currentResolution,
      this.currentIsOrtho,
      this.currentNearCull
    );

    log.info(Modules.RENDERER, `Created per-node point material (${backend})`);
    return material;
  }

  /**
   * Create a line material — PER NODE, no LRU cache.
   *
   * Segment data lives in a per-node texture (`uLineTex`, since the
   * texture-backed storage migration), so two nodes can never share a
   * line material: sharing would rebind one node's texture onto
   * another's mesh at every commit. Every call creates a fresh material
   * that the node owns for its lifetime (the node factory stamps
   * `_layerMaterialCloned: true`, so LayersPanel / LOD-cross-fade
   * mutate it directly instead of clone-on-first-use).
   * `lineMaterialCache` stays permanently empty — it remains in the
   * lifecycle/stats context shapes, where an empty map is a truthful
   * no-op. Mirrors {@link getPointMaterial} / {@link getGSplatMaterial}.
   *
   * Dispatches to `LineTSLMaterial` (NodeMaterial / TSL) when the
   * active renderer reports `caps.apiSurface === 'webgpu'`, otherwise to the
   * GLSL `LineMaterial`. Both classes expose the same update surface
   * via {@link LuxarLineMaterial}.
   */
  getLineMaterial(props: LineMaterialProperties): LuxarLineMaterial {
    const backend = resolveMaterialBackend(this.caps);

    const createStart = performance.now();
    const material = new VISUAL_FACTORIES.line[backend]({
      opacity: props.opacity,
      absorption: props.absorption,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
    });
    this.totalCreateMs += performance.now() - createStart;
    this.createCount++;

    this.registeredMaterials.add(material);
    subscribeToDispose(material, this.lifecycleCtx);
    material.updateCameraParams(
      this.currentFov,
      this.currentResolution,
      this.currentIsOrtho,
      this.currentNearCull
    );

    log.info(Modules.RENDERER, `Created per-node line material (${backend})`);
    return material;
  }

  /**
   * Create a gsplat material — PER NODE, no LRU cache.
   *
   * Splat data lives in a per-node texture (`uSplatTex`), so two nodes
   * can never share a gsplat material: sharing would rebind one node's
   * texture onto another's mesh at every commit. Every call creates a
   * fresh material that the node owns for its lifetime (the node
   * factory stamps `_layerMaterialCloned: true`, so LayersPanel /
   * LOD-cross-fade mutate it directly instead of clone-on-first-use).
   * `gsplatMaterialCache` stays permanently empty — it remains in the
   * lifecycle/stats context shapes shared with points/lines, where an
   * empty map is a truthful no-op.
   *
   * Dispatches to `GSplatTSLMaterial` (NodeMaterial / TSL) when the
   * active renderer reports `caps.apiSurface === 'webgpu'`, otherwise to the
   * GLSL `GSplatMaterial`. Both classes expose the same update
   * surface via {@link LuxarGSplatMaterial}.
   */
  getGSplatMaterial(props: GSplatMaterialProperties): LuxarGSplatMaterial {
    const backend = resolveMaterialBackend(this.caps);

    const createStart = performance.now();
    const material = new VISUAL_FACTORIES.gsplat[backend]({
      opacity: props.opacity,
      absorption: props.absorption ?? 1.0,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
      truncationRadius: props.truncationRadius ?? GSPLAT_DEFAULT_TRUNCATION_RADIUS,
    });
    this.totalCreateMs += performance.now() - createStart;
    this.createCount++;

    this.registeredMaterials.add(material);
    subscribeToDispose(material, this.lifecycleCtx);
    material.updateCameraParams(
      this.currentFov,
      this.currentResolution,
      this.currentIsOrtho,
      this.currentNearCull
    );

    log.info(Modules.RENDERER, `Created per-node gsplat material (${backend})`);
    return material;
  }

  /**
   * Create a mesh material — PER NODE, no LRU cache.
   *
   * Per node for a different reason than its three siblings: they carry the node's
   * own element texture, so sharing would rebind one node's data onto another's
   * mesh. A mesh material holds no per-node texture at all — but it does hold two
   * pieces of per-node state that make sharing wrong anyway: the `flatNormal`
   * compile-time variant (a function of that node's `shading` and its normals'
   * validity for the active view) and `side` (re-applied per epoch by
   * `applyMeshSide`). Sharing would let one node's shading model and face-sidedness
   * follow another's.
   *
   * Deliberately does NOT enter `registeredMaterials`: a mesh has no screen-space
   * size, so it has no `updateCameraParams` to broadcast to. It is tracked in
   * `staticMaterials` instead, which keeps disposal and the stats counters honest
   * without a per-frame no-op call per node. There is no fourth empty
   * `meshMaterialCache` either — the three vestigial maps exist only to keep
   * `getCacheStats()`'s historical shape, and adding to them would be inventing a
   * cache that never existed.
   *
   * Dispatches to `MeshTSLMaterial` when the active renderer reports
   * `caps.apiSurface === 'webgpu'`, otherwise the GLSL `MeshMaterial`.
   */
  getMeshMaterial(props: MeshMaterialProperties): LuxarMeshMaterial {
    const backend = resolveMaterialBackend(this.caps);

    const createStart = performance.now();
    const material = new VISUAL_FACTORIES.mesh[backend]({
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
      flatNormal: props.flatNormal,
      ambient: props.ambient,
      shadeExponent: props.shadeExponent,
      alphaCutoff: props.alphaCutoff,
    });
    this.totalCreateMs += performance.now() - createStart;
    this.createCount++;

    this.staticMaterials.add(material);
    subscribeToDispose(material, this.lifecycleCtx);

    log.info(Modules.RENDERER, `Created per-node mesh material (${backend})`);
    return material;
  }

  /**
   * Create a per-mesh point picking material, dispatching on
   * `caps.apiSurface`. The returned material is NOT cached — picking
   * materials have per-mesh lifetimes; the NodeFactory disposes
   * them when the mesh disposes. Camera updates flow through
   * `register()` like any other camera-aware material.
   */
  createPointPickingMaterial(config: PointPickingMaterialConfig): LuxarPointPickingMaterial {
    return new PICKING_FACTORIES.point[resolveMaterialBackend(this.caps)](config);
  }

  /** Same shape as `createPointPickingMaterial`, for lines. */
  createLinePickingMaterial(config: LinePickingMaterialConfig): LuxarLinePickingMaterial {
    return new PICKING_FACTORIES.line[resolveMaterialBackend(this.caps)](config);
  }

  /** Same shape as `createPointPickingMaterial`, for gsplats. */
  createGSplatPickingMaterial(config: GSplatPickingMaterialConfig): LuxarGSplatPickingMaterial {
    return new PICKING_FACTORIES.gsplat[resolveMaterialBackend(this.caps)](config);
  }

  /** Same shape as `createPointPickingMaterial`, for meshes. */
  createMeshPickingMaterial(config: MeshPickingMaterialConfig): LuxarMeshPickingMaterial {
    return new PICKING_FACTORIES.mesh[resolveMaterialBackend(this.caps)](config);
  }

  /**
   * Create the post-processing mega-shader material, dispatching on
   * `caps.apiSurface`. PostProcessingManager owns the single instance; this
   * is the seam between `THREE.ShaderMaterial`-derived
   * `MegaShaderMaterial` and the TSL `MegaShaderTSLMaterial`.
   */
  createMegaShaderMaterial(cfg: MegaShaderConfig): LuxarMegaShaderMaterial {
    return new MEGA_SHADER_FACTORIES[resolveMaterialBackend(this.caps)](cfg);
  }

  /** Update camera parameters for all registered materials. */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.currentFov = fov;
    this.currentResolution.copy(resolution);
    this.currentIsOrtho = isOrtho;
    this.currentNearCull = nearCull;

    for (const material of this.registeredMaterials) {
      material.updateCameraParams(fov, resolution, isOrtho, nearCull);
    }
  }

  /**
   * Register a material the manager did not construct, for lifecycle tracking and —
   * where applicable — global camera-parameter updates.
   *
   * Use this for non-cached materials such as per-node colormap clones and picking
   * materials. Materials from the four `getXMaterial` factories register themselves.
   *
   * Takes a plain `THREE.Material` and DISPATCHES on the capability rather than
   * demanding it, which is the same `isCameraAwareMaterial` pattern the picking
   * system already uses. A camera-aware material joins the broadcast registry and
   * receives the current camera state immediately; one without a screen-space extent
   * (a mesh material, whose size IS its geometry) is tracked for disposal only.
   *
   * Dispatching here rather than at the call site is deliberate: it leaves ONE public
   * entry point that cannot be called wrongly. Requiring `& CameraAwareMaterial`
   * instead pushed the problem outward — the layers panel's `LuxarMaterial` had to
   * claim a method it never calls just to satisfy this signature, which made a
   * perfectly valid non-camera-aware leaf material unrepresentable in the panel.
   */
  register(material: THREE.Material): void {
    subscribeToDispose(material, this.lifecycleCtx);
    if (!isCameraAwareMaterial(material)) {
      // No screen-space size to recompute — see `staticMaterials`.
      this.staticMaterials.add(material);
      this.ownedMaterials.add(material);
      return;
    }
    this.registeredMaterials.add(material);
    this.ownedMaterials.add(material);
    material.updateCameraParams(
      this.currentFov,
      this.currentResolution,
      this.currentIsOrtho,
      this.currentNearCull
    );
  }

  /**
   * Drop a material from every registry without disposing the underlying GPU object.
   * The picking-material classes use this in their custom `dispose()` paths so the
   * manager stops broadcasting camera updates before the caller disposes it directly.
   *
   * Widened to `THREE.Material` alongside {@link register}, so anything that can be
   * registered can be unregistered — the pair must accept the same set.
   */
  unregister(material: THREE.Material): void {
    removeFromRegistries(material, this.lifecycleCtx);
  }

  /** Dispose all managed materials. */
  dispose(): void {
    // Union of both registries so an ownedMaterials-only entry (e.g. a
    // register()-entered material) can't leak its GPU program at
    // teardown.
    const materials = new Set<THREE.Material>([
      ...this.registeredMaterials,
      ...this.ownedMaterials,
      // Mesh materials live only here (no camera broadcast), so omitting this set
      // would leak their GPU programs at teardown.
      ...this.staticMaterials,
    ]);
    this.registeredMaterials.clear();
    this.ownedMaterials.clear();
    this.staticMaterials.clear();
    for (const material of materials) {
      material.dispose();
    }
    // Release the page-level singleton reference: after a full dispose
    // the instance is a husk (empty registries, disposed programs), and
    // the lazy Proxy would keep re-serving it to a dispose-then-reinit
    // embedder. Nulling here makes the next access construct a fresh
    // manager instead. (Tests use __resetMaterialManagerForTests, which
    // does the same without disposing.)
    if (_materialManagerInstance === this) {
      _materialManagerInstance = undefined;
    }
  }

  /**
   * Context-restore hook, called by `webgl-context-recovery` between the
   * renderer rebuild and `NodeFactory.rebuildAfterContextRestore`.
   *
   * There is nothing to rebuild here: materials are per node, so the node
   * factory reconstructs them along with their meshes and textures. This used
   * to drop the per-type allocation caches, which no longer exist.
   *
   * It must stay a no-op rather than clearing the registries —
   * `registeredMaterials` / `ownedMaterials` track materials attached to
   * visible meshes, which must keep receiving `updateCameraParams()` across a
   * restore. Kept as an explicit member so the recovery sequence stays
   * readable and its ordering test keeps a real subject.
   */
  rebuildAfterContextRestore(): void {
    // Intentionally empty — see above.
  }

  /** Get cache statistics (delegates to stats.ts). */
  getCacheStats() {
    return buildCacheStats({
      ownedMaterials: this.ownedMaterials,
      registeredMaterials: this.registeredMaterials,
      staticMaterials: this.staticMaterials,
      totalCreateMs: this.totalCreateMs,
      createCount: this.createCount,
    });
  }
}

/**
 * Page-level singleton instance of the material manager.
 *
 * Construction is **deferred until first access** via a Proxy. Tests can
 * call {@link __resetMaterialManagerForTests} to start fresh between
 * cases. Call-site syntax is unchanged from a directly-exported instance.
 */
let _materialManagerInstance: MaterialManager | undefined;

export const materialManager: MaterialManager = new Proxy({} as MaterialManager, {
  get(_target, prop, _receiver) {
    _materialManagerInstance ??= new MaterialManager();
    const value = Reflect.get(_materialManagerInstance, prop, _materialManagerInstance);
    return typeof value === 'function' ? value.bind(_materialManagerInstance) : value;
  },
  set(_target, prop, value, _receiver) {
    _materialManagerInstance ??= new MaterialManager();
    return Reflect.set(_materialManagerInstance, prop, value, _materialManagerInstance);
  },
  has(_target, prop) {
    _materialManagerInstance ??= new MaterialManager();
    return prop in _materialManagerInstance;
  },
});

/**
 * Discard the current singleton so the next access constructs a fresh
 * instance. Intended for tests; safe to leave un-called in production.
 */
export const __resetMaterialManagerForTests = (): void => {
  _materialManagerInstance = undefined;
};
