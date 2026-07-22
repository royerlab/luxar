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
import { PointTSLMaterial } from './materials/point/material-tsl';
import { LineTSLMaterial } from './materials/line/material-tsl';
import { GSplatTSLMaterial } from './materials/gsplat/material-tsl';
import type { PointPickingMaterial } from './picking/point/material';
import type { LinePickingMaterial } from './picking/line/material';
import type { GSplatPickingMaterial } from './picking/gsplat/material';
import type { PointPickingTSLMaterial } from './picking/point/material-tsl';
import type { LinePickingTSLMaterial } from './picking/line/material-tsl';
import type { GSplatPickingTSLMaterial } from './picking/gsplat/material-tsl';
import type { MegaShaderMaterial } from './post-processing/mega/material';
import type { MegaShaderTSLMaterial } from './post-processing/mega/material-tsl';
import type { PointPickingMaterialConfig } from './picking/point/material';
import type { LinePickingMaterialConfig } from './picking/line/material';
import type { GSplatPickingMaterialConfig } from './picking/gsplat/material';
import type { MegaShaderConfig } from './post-processing/mega/material';
import type { CameraAwareMaterial } from './materials/_shared/camera-aware-material';
import type { RendererCapabilities } from './renderer-capabilities';
import { log, Modules } from '../utils/log';
import { config } from '../config';
import {
  VISUAL_FACTORIES,
  PICKING_FACTORIES,
  MEGA_SHADER_FACTORIES,
  resolveMaterialBackend,
  lineCacheKey,
  type BlendingMode,
  type PointMaterialProperties,
  type LineMaterialProperties,
  type GSplatMaterialProperties,
  type MaterialBackend,
} from './material-manager/factories';
import { lruGet, lruSet } from './material-manager/lru-cache';
import {
  SOFT_DISPOSE_FLAG,
  subscribeToDispose,
  removeFromRegistries,
  type LifecycleCtx,
} from './material-manager/lifecycle';
import { getCacheStats as buildCacheStats } from './material-manager/stats';

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
 * Per-geometry-type picking material returned by
 * `MaterialManager.create{Point,Line,GSplat}PickingMaterial`.
 * Picking materials have a per-mesh lifetime (not cached).
 */
export type LuxarPointPickingMaterial = PointPickingMaterial | PointPickingTSLMaterial;
export type LuxarLinePickingMaterial = LinePickingMaterial | LinePickingTSLMaterial;
export type LuxarGSplatPickingMaterial = GSplatPickingMaterial | GSplatPickingTSLMaterial;

/**
 * The single post-processing mega-shader material managed by
 * `PostProcessingManager`.
 */
export type LuxarMegaShaderMaterial = MegaShaderMaterial | MegaShaderTSLMaterial;

/**
 * Manages all materials in the scene with caching and global updates.
 * Supports points, lines, and future material types.
 *
 * The LINE cache is a bounded LRU map: each `getLineMaterial()` call
 * promotes the entry to most-recently-used by re-inserting it; on
 * insert past `cacheMaxSize`, the least-recently-used entry is
 * disposed and dropped. Without this bound the cache would grow
 * unbounded as users animate attribute sliders, leaking GPU shader
 * programs. Point and gsplat materials are PER NODE (each carries the
 * node's own element texture) and are never cached — their maps stay
 * permanently empty (kept for the lifecycle/stats context shapes).
 *
 * The bound is configured from
 * `config.dataLoading.performance.materialCacheMaxSize` (default 200).
 * `0` disables eviction.
 */
export class MaterialManager {
  private pointMaterialCache = new Map<string, LuxarPointMaterial>();
  private lineMaterialCache = new Map<string, LuxarLineMaterial>();
  private gsplatMaterialCache = new Map<string, LuxarGSplatMaterial>();
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
  /** Total LRU evictions (only the line cache can evict — point/gsplat materials are per node). */
  private evictionCount = 0;

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
   * Materials that entered through `register()` or LRU eviction rather than
   * a manager factory (per-node point/gsplat materials live in
   * `registeredMaterials` only).
   *
   * Examples: GPU-picking materials and evicted line materials awaiting
   * defer-dispose. These
   * still need global camera uniforms and manager-level disposal, but they must
   * be tracked separately from cached shared materials for leak diagnostics.
   */
  private ownedMaterials = new Set<THREE.Material & CameraAwareMaterial>();
  private currentFov = (60 * Math.PI) / 180; // Current FOV in radians (or frustumHeight for ortho)
  private currentResolution = new THREE.Vector2(1920, 1080); // Use reasonable default
  private currentIsOrtho = false;
  private currentNearCull: number | undefined = undefined;

  /**
   * Callback used by `lruSet` on eviction. DEFER-DISPOSE: cached
   * materials are shared and attached directly to live meshes (only the
   * colormap / layers-panel paths clone), so eviction must not dispose —
   * a disposed-but-still-rendered material is auto-recompiled by Three
   * but its dispose listener has unregistered it, so it silently stops
   * receiving `updateCameraParams` and renders with stale
   * resolution/FOV/nearCull after the next resize. Instead the entry
   * merely leaves the cache (the bound is enforced by `lruSet`, which
   * deleted the map entry before calling this); the material stays in
   * `registeredMaterials` (camera updates keep flowing) and moves to
   * `ownedMaterials` (leak diagnostics + teardown), so `dispose()` still
   * cleans it up at end of life. Trade-off: an evicted-and-truly-unused
   * material is retained until manager disposal — one CPU-side uniforms
   * object per distinct key ever created (its GPU program is shared and
   * refcounted by shader key in Three), strictly cheaper than the
   * use-after-evict bug. Bound as an arrow field so each `lruSet` call
   * site can pass it without rebinding `this`.
   */
  private readonly handleEviction = (
    _key: string,
    material: THREE.Material & CameraAwareMaterial
  ): void => {
    this.ownedMaterials.add(material);
    this.evictionCount++;
  };

  /**
   * Build the lifecycle context handed to `subscribeToDispose` and
   * `removeFromRegistries`. Lazy getter — captures `this` references
   * once, reused across calls.
   */
  private get lifecycleCtx(): LifecycleCtx {
    return {
      registeredMaterials: this.registeredMaterials,
      ownedMaterials: this.ownedMaterials,
      subscribedMaterials: this.subscribedMaterials,
      pointMaterialCache: this.pointMaterialCache,
      lineMaterialCache: this.lineMaterialCache,
      gsplatMaterialCache: this.gsplatMaterialCache,
    };
  }

  /**
   * Set the renderer capabilities. Called once by SceneManager after
   * the renderer is alive. Determines which backend the dispatch in
   * `getPointMaterial` (and Line/GSplat equivalents) picks.
   *
   * Switching caps after materials have been cached invalidates the
   * cache because cached entries are class-specific (PointMaterial vs
   * PointTSLMaterial). We clear all three caches defensively.
   */
  setCaps(caps: RendererCapabilities): void {
    if (this.caps && this.caps.apiSurface !== caps.apiSurface) {
      this.pointMaterialCache.clear();
      this.lineMaterialCache.clear();
      this.gsplatMaterialCache.clear();
    }
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
   * Get or create a line material with caching.
   *
   * Dispatches to `LineTSLMaterial` (NodeMaterial / TSL) when the
   * active renderer reports `caps.apiSurface === 'webgpu'`, otherwise to the
   * GLSL `LineMaterial`. Both classes expose the same update surface
   * via {@link LuxarLineMaterial}.
   */
  getLineMaterial(props: LineMaterialProperties): LuxarLineMaterial {
    const backend = resolveMaterialBackend(this.caps);
    const key = lineCacheKey(props, backend);

    let material = lruGet(this.lineMaterialCache, key);
    if (material) return material;

    const createStart = performance.now();
    material = new VISUAL_FACTORIES.line[backend]({
      opacity: props.opacity,
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
    lruSet(
      this.lineMaterialCache,
      key,
      material,
      config.dataLoading.performance.materialCacheMaxSize,
      this.handleEviction
    );

    log.info(Modules.RENDERER, `Created line material: ${key}`);
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
      truncationRadius: props.truncationRadius ?? 3.0,
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
   * Register a material for global camera parameter updates.
   *
   * Use this for non-cached materials such as per-node colormap clones and
   * picking materials. Cached materials are registered internally by
   * getPointMaterial()/getLineMaterial()/getGSplatMaterial().
   */
  register(material: THREE.Material & CameraAwareMaterial): void {
    this.registeredMaterials.add(material);
    this.ownedMaterials.add(material);
    subscribeToDispose(material, this.lifecycleCtx);
    material.updateCameraParams(
      this.currentFov,
      this.currentResolution,
      this.currentIsOrtho,
      this.currentNearCull
    );
  }

  /**
   * Drop a material from the global registry without disposing the
   * underlying GPU object. The picking-material classes use this in
   * their custom `dispose()` paths so the manager stops broadcasting
   * camera updates to the material before the caller disposes it
   * directly.
   */
  unregister(material: THREE.Material & CameraAwareMaterial): void {
    removeFromRegistries(material, this.lifecycleCtx);
  }

  /** Dispose all cached materials. */
  dispose(): void {
    // Union of both registries: an LRU-evicted line material is parked
    // in ownedMaterials only (handleEviction) — registeredMaterials
    // alone would miss it and leak its GPU program at teardown.
    const materials = new Set([...this.registeredMaterials, ...this.ownedMaterials]);
    this.registeredMaterials.clear();
    this.ownedMaterials.clear();
    this.pointMaterialCache.clear();
    this.lineMaterialCache.clear();
    this.gsplatMaterialCache.clear();
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
   * Rebuild GPU-bound material state after a WebGL context-restore
   * event. Drops the per-type allocation caches; the renderer's next
   * request via `getXMaterial` will compile fresh shaders against the
   * new context.
   *
   * Do NOT clear `registeredMaterials` / `ownedMaterials` — those track
   * materials currently attached to visible scene meshes and must keep
   * receiving `updateCameraParams()` after restore.
   *
   * Idempotent. Unlike `dispose()`, does NOT call `material.dispose()` on
   * cached entries — those programs are already detached from a dead
   * WebGL context, and disposing them tends to throw on some drivers.
   */
  rebuildAfterContextRestore(): void {
    this.pointMaterialCache.clear();
    this.lineMaterialCache.clear();
    this.gsplatMaterialCache.clear();
    // registeredMaterials / ownedMaterials deliberately preserved.
  }

  /** Get cache statistics (delegates to stats.ts). */
  getCacheStats() {
    return buildCacheStats({
      pointMaterialCache: this.pointMaterialCache,
      lineMaterialCache: this.lineMaterialCache,
      gsplatMaterialCache: this.gsplatMaterialCache,
      ownedMaterials: this.ownedMaterials,
      registeredMaterials: this.registeredMaterials,
      evictionCount: this.evictionCount,
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
