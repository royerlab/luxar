/**
 * Material Manager for Luxar
 *
 * This module manages all materials in the scene, providing caching,
 * global uniform updates, and support for multiple material types.
 * Supports point, line, and GSplat materials.
 */

import * as THREE from 'three';
import { PointMaterial } from './point-material';
import { LineMaterial } from './line-material';
import { GSplatMaterial } from './gsplat-material';
import { PointTSLMaterial } from './point-material-tsl';
import type { CameraAwareMaterial } from './camera-aware-material';
import type { RendererCapabilities } from './renderer-capabilities';
import { log, Modules } from '../utils/log';
import { clamp } from '../utils/clamp';
import { config } from '../config';

/**
 * Supported blending modes for materials.
 *
 * - 'normal': Standard alpha blending (semi-transparent). For
 *   **Points** and **Lines** this works as expected — the shader
 *   emits a per-fragment alpha derived from opacity and edge
 *   softness. For **GSplats** the shader emits `alpha = 1.0` and
 *   modulates RGB by uOpacity instead, so 'normal' on a GSplat layer
 *   behaves like "opaque dimmed by opacity": the framebuffer behind
 *   the splat is not revealed. Proper alpha-on-GSplats requires
 *   premultiplied-alpha output + a `ONE` / `ONE_MINUS_SRC_ALPHA`
 *   blend func, which is a deeper shader change deferred until
 *   needed. Users wanting semi-transparent splats today should use
 *   'luminous' or 'additive'.
 * - 'additive': Classic additive blending, ignores depth (renders on top of everything)
 * - 'max': Maximum of source and destination (brightest wins)
 * - 'opaque': Solid rendering with depth write (closest object wins)
 * - 'luminous': Same as additive visually, but respects depth occlusion (occluded by closer objects)
 *
 * Depth behavior:
 * - 'additive': depthTest=false, depthWrite=false (ignores depth entirely)
 * - 'luminous': depthTest=true, depthWrite=false (respects occlusion, doesn't occlude others)
 */
export type BlendingMode = 'normal' | 'additive' | 'max' | 'opaque' | 'luminous';

// Point material properties
export interface PointMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  gamma: number;
  intensity: number;
  offset: number;
  radiusScale?: number; // Scale factor for radius normalization (e.g., 1/255 for uint8)
  sharpnessScale?: number; // Scale factor for sharpness normalization (e.g., 1/255 for uint8)
}

// Line material properties
export interface LineMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  gamma: number;
  intensity: number;
  offset: number;
}

// GSplat material properties
export interface GSplatMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  gamma: number;
  intensity: number;
  offset: number;
  truncationRadius?: number; // Default 3.0
}

/**
 * Compute the integer-bucketed cache-key components shared by all three
 * material caches (Points / Lines / GSplats). All four properties have
 * the same valid ranges and bucketing rules across material types, so
 * having one helper avoids drift the next time the rules change.
 *
 * Returned ranges:
 *   - opacity: 0–100
 *   - gamma:   0–1000
 *   - intensity: 0–10000
 *   - offset:  0–200 (input -10..10 shifted up to 0..20 then ×10)
 */
function getCommonMaterialBuckets(props: {
  opacity: number;
  gamma: number;
  intensity: number;
  offset: number;
}): { opacityBucket: number; gammaBucket: number; intensityBucket: number; offsetBucket: number } {
  return {
    opacityBucket: Math.round(clamp(props.opacity, 0, 1) * 100),
    gammaBucket: Math.round(clamp(props.gamma, 0, 10) * 100),
    intensityBucket: Math.round(clamp(props.intensity, 0, 100) * 100),
    offsetBucket: Math.round((clamp(props.offset, -10, 10) + 10) * 10),
  };
}

/**
 * Manages all materials in the scene with caching and global updates.
 * Supports points, lines, and future material types.
 *
 * The three caches are bounded LRU maps: each `getXMaterial()` call
 * promotes the entry to most-recently-used by re-inserting it; on
 * insert past `cacheMaxSize`, the least-recently-used entry is
 * disposed and dropped. Without this bound the cache would grow
 * unbounded as users animate attribute sliders, leaking GPU shader
 * programs.
 *
 * The bound is configured from
 * `config.dataLoading.performance.materialCacheMaxSize` (default 200).
 * `0` disables eviction.
 */
/**
 * Point material returned by `MaterialManager.getPointMaterial`. The
 * concrete class is either the GLSL `PointMaterial` (WebGL2 path) or
 * the TSL `PointTSLMaterial` (WebGPU / fallback-via-WebGPURenderer
 * path). Both classes expose the same surface — `updateOpacity`,
 * `updateCameraParams`, `applyBlendingMode`, `clone()`, etc. — so
 * call sites treat the return type as a single LuxarPointMaterial.
 */
export type LuxarPointMaterial = PointMaterial | PointTSLMaterial;

export class MaterialManager {
  private pointMaterialCache = new Map<string, LuxarPointMaterial>();
  private lineMaterialCache = new Map<string, LineMaterial>();
  private gsplatMaterialCache = new Map<string, GSplatMaterial>();
  private registeredMaterials = new Set<THREE.Material & CameraAwareMaterial>();
  /**
   * Renderer capabilities — drives the GLSL vs. TSL dispatch in
   * `getPointMaterial` and (future) `getLineMaterial` /
   * `getGSplatMaterial`. `SceneManager.setupRenderer` calls
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
  /** Per-cache eviction count (read by getCacheStats; no behavior). */
  private evictionCount = 0;

  /**
   * Diagnostic: cumulative wall-clock time spent constructing
   * materials (Point/Line/GSplat). Each `getXMaterial()` cache miss
   * runs `new XMaterial(...)` which builds the shader source string
   * and allocates uniforms; the WebGL program compile cost itself
   * happens later during the first render. The number here is a
   * useful proxy for "how much time does the user spend waiting for
   * material-creation work" — first-use stutter shows up as a single
   * large delta in this counter on the affected animation frame.
   *
   * Exposed in `getCacheStats()` as `{ totalCreateMs, createCount }`.
   */
  private totalCreateMs = 0;
  private createCount = 0;
  /**
   * Materials registered for camera updates but not owned by a cache entry.
   *
   * Examples: per-node colormap material clones and GPU-picking materials. These
   * still need global camera uniforms and manager-level disposal, but they must
   * be tracked separately from cached shared materials for leak diagnostics.
   */
  private ownedMaterials = new Set<THREE.Material & CameraAwareMaterial>();
  private currentFov = (60 * Math.PI) / 180; // Current FOV in radians (or frustumHeight for ortho)
  private currentResolution = new THREE.Vector2(1920, 1080); // Use reasonable default
  private currentIsOrtho = false;
  private currentNearCull: number | undefined = undefined;
  // Note: Global HDR multiplier has been replaced by exposure/offset/gamma in post-processing

  /**
   * LRU-aware cache lookup. On hit, promote the entry to
   * most-recently-used by re-inserting it (Map preserves insertion
   * order, so the first key is the LRU). Returns the cached value or
   * undefined.
   */
  private lruGet<T>(cache: Map<string, T>, key: string): T | undefined {
    const value = cache.get(key);
    if (value !== undefined) {
      // Re-insert to bump to MRU.
      cache.delete(key);
      cache.set(key, value);
    }
    return value;
  }

  /**
   * LRU-aware cache insert. If the cache is at its bound, evict the
   * LRU entry (first key in insertion order), dispose the material,
   * and remove it from the registered-materials set so global camera
   * updates stop targeting it. `materialCacheMaxSize: 0` disables
   * eviction and allows unbounded cache growth.
   */
  private lruSet<T extends THREE.Material & CameraAwareMaterial>(
    cache: Map<string, T>,
    key: string,
    value: T
  ): void {
    const maxSize = config.dataLoading.performance.materialCacheMaxSize;
    if (maxSize > 0) {
      while (cache.size >= maxSize) {
        const lruKey = cache.keys().next().value;
        if (lruKey === undefined) break;
        const lruMat = cache.get(lruKey);
        cache.delete(lruKey);
        if (lruMat) {
          this.registeredMaterials.delete(lruMat);
          try {
            lruMat.dispose();
          } catch (err) {
            log.warning(Modules.RENDERER, `Error disposing evicted material '${lruKey}': ${err}`);
          }
          this.evictionCount++;
        }
      }
    }
    cache.set(key, value);
  }

  /**
   * Subscribe to a material's `dispose` event so the manager can clean
   * up its registry / cache entries automatically. THREE.Material's
   * EventDispatcher fires `dispose` synchronously inside `dispose()`,
   * so by the time super.dispose() returns, the manager has already
   * forgotten about this material.
   *
   * Wiring cleanup this way (manager → material) instead of having
   * materials call `materialManager.unregister(this)` (material →
   * manager) avoids an import cycle between material-manager.ts and
   * {point,line,gsplat}-material.ts.
   */
  private subscribeToDispose(material: THREE.Material & CameraAwareMaterial): void {
    if (this.subscribedMaterials.has(material)) return;
    const onDispose = (): void => {
      this.subscribedMaterials.delete(material);
      this.removeFromRegistries(material);
      material.removeEventListener('dispose', onDispose);
    };
    material.addEventListener('dispose', onDispose);
    this.subscribedMaterials.add(material);
  }

  /**
   * Internal cleanup: remove `material` from every registry and cache.
   * Called from the dispose listener and (for backwards compatibility)
   * from the public `unregister` method. Idempotent.
   */
  private removeFromRegistries(material: THREE.Material & CameraAwareMaterial): void {
    this.registeredMaterials.delete(material);
    this.ownedMaterials.delete(material);

    if (material instanceof PointMaterial || material instanceof PointTSLMaterial) {
      for (const [key, cachedMaterial] of this.pointMaterialCache.entries()) {
        if (cachedMaterial === material) {
          this.pointMaterialCache.delete(key);
          break;
        }
      }
    } else if (material instanceof LineMaterial) {
      for (const [key, cachedMaterial] of this.lineMaterialCache.entries()) {
        if (cachedMaterial === material) {
          this.lineMaterialCache.delete(key);
          break;
        }
      }
    } else if (material instanceof GSplatMaterial) {
      for (const [key, cachedMaterial] of this.gsplatMaterialCache.entries()) {
        if (cachedMaterial === material) {
          this.gsplatMaterialCache.delete(key);
          break;
        }
      }
    }
  }

  /**
   * Set the renderer capabilities. Called once by SceneManager after
   * the renderer is alive. Determines which backend the dispatch in
   * `getPointMaterial` (and future Line/GSplat equivalents) picks.
   *
   * Switching caps after materials have been cached invalidates the
   * cache because cached entries are class-specific (PointMaterial vs
   * PointTSLMaterial). We clear all three caches defensively — the
   * existing materials remain in `registeredMaterials` so global
   * camera updates still reach them until their owning mesh disposes
   * them.
   */
  setCaps(caps: RendererCapabilities): void {
    if (this.caps && this.caps.api !== caps.api) {
      // Backend changed mid-session — drop the allocation caches so
      // the next request reaches the new dispatch branch. Same
      // pattern as `rebuildAfterContextRestore`.
      this.pointMaterialCache.clear();
      this.lineMaterialCache.clear();
      this.gsplatMaterialCache.clear();
    }
    this.caps = caps;
  }

  /**
   * Get or create a point material with caching.
   *
   * Dispatches to `PointTSLMaterial` (NodeMaterial / TSL) when the
   * active renderer reports `caps.api === 'webgpu'`, otherwise to the
   * GLSL `PointMaterial`. Both classes expose the same update surface
   * (see {@link LuxarPointMaterial}), so callers in node-factory and
   * the layers panel don't need to branch.
   */
  getPointMaterial(props: PointMaterialProperties): LuxarPointMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    // This prevents floating-point precision issues while still grouping similar values
    // Clamp values to valid ranges to handle edge cases gracefully
    const { opacityBucket, gammaBucket, intensityBucket, offsetBucket } =
      getCommonMaterialBuckets(props);
    const radiusBucket = props.radiusScale
      ? Math.round(Math.max(0, props.radiusScale) * 1000)
      : 1000;
    const sharpnessBucket = props.sharpnessScale
      ? Math.round(Math.max(0, props.sharpnessScale) * 1000)
      : 1000;

    const isOpaque = this.isOpaqueMode(props.blendingMode);
    const transparent = !isOpaque;
    const useTSL = this.caps?.api === 'webgpu';
    const key = `point_${useTSL ? 'tsl' : 'glsl'}_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_r${radiusBucket}_s${sharpnessBucket}_t${transparent ? 1 : 0}`;

    // Check cache first (LRU-promoting)
    let material = this.lruGet(this.pointMaterialCache, key);
    if (material) {
      return material;
    }

    // Create new material instance with neutral blending defaults;
    // the canonical state for `props.blendingMode` is then applied via
    // `applyBlendingMode()` so creation-time and runtime transitions
    // share one code path. Without this, the LayersPanel runtime path
    // would diverge from creation (notably max mode, which needs
    // OneFactor/OneFactor blend factors AND the
    // LUXAR_MAX_RGB_CONTRIBUTION shader define).
    const createStart = performance.now();
    const constructorConfig = {
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      // Pass `transparent` and a neutral default; applyBlendingMode
      // will overwrite blending/depth fields immediately below.
      transparent: !isOpaque,
      radiusScale: props.radiusScale,
      sharpnessScale: props.sharpnessScale,
    };
    material = useTSL
      ? new PointTSLMaterial(constructorConfig)
      : new PointMaterial(constructorConfig);
    material.applyBlendingMode(props.blendingMode);
    this.totalCreateMs += performance.now() - createStart;
    this.createCount++;

    // Register for global updates
    this.registeredMaterials.add(material);
    this.subscribeToDispose(material);

    // Update with current camera params
    material.updateCameraParams(this.currentFov, this.currentResolution, this.currentIsOrtho);

    // Debug log the camera params being set
    log.info(
      Modules.RENDERER,
      `Material camera params: FOV=${((this.currentFov * 180) / Math.PI).toFixed(1)}°, ` +
        `Resolution=${this.currentResolution.x}x${this.currentResolution.y}`
    );

    // Cache it (LRU-bounded)
    this.lruSet(this.pointMaterialCache, key, material);

    log.info(Modules.RENDERER, `Created point material: ${key}`);
    return material;
  }

  /**
   * Get or create a line material with caching
   */
  getLineMaterial(props: LineMaterialProperties): LineMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    const { opacityBucket, gammaBucket, intensityBucket, offsetBucket } =
      getCommonMaterialBuckets(props);

    const lineTransparent = props.blendingMode !== 'opaque';
    const key = `line_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_t${lineTransparent ? 1 : 0}`;

    // Check cache first (LRU-promoting)
    let material = this.lruGet(this.lineMaterialCache, key);
    if (material) {
      return material;
    }

    // Create new LineMaterial instance
    const createStart = performance.now();
    material = new LineMaterial({
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
    });
    this.totalCreateMs += performance.now() - createStart;
    this.createCount++;

    // Register for global updates
    this.registeredMaterials.add(material);
    this.subscribeToDispose(material);

    // Update with current camera params
    material.updateCameraParams(this.currentFov, this.currentResolution, this.currentIsOrtho);

    // Cache it (LRU-bounded)
    this.lruSet(this.lineMaterialCache, key, material);

    log.info(Modules.RENDERER, `Created line material: ${key}`);
    return material;
  }

  /**
   * Get or create a gsplat material with caching
   */
  getGSplatMaterial(props: GSplatMaterialProperties): GSplatMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    const { opacityBucket, gammaBucket, intensityBucket, offsetBucket } =
      getCommonMaterialBuckets(props);
    const truncBucket = Math.round((props.truncationRadius ?? 3.0) * 10);

    const gsplatTransparent = props.blendingMode !== 'opaque';
    const key = `gsplat_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_tr${truncBucket}_t${gsplatTransparent ? 1 : 0}`;

    // Check cache first (LRU-promoting)
    let material = this.lruGet(this.gsplatMaterialCache, key);
    if (material) {
      return material;
    }

    // Create new GSplatMaterial instance
    const createStart = performance.now();
    material = new GSplatMaterial({
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
      truncationRadius: props.truncationRadius ?? 3.0,
    });
    this.totalCreateMs += performance.now() - createStart;
    this.createCount++;

    // Register for global updates
    this.registeredMaterials.add(material);
    this.subscribeToDispose(material);

    // Update with current camera params
    material.updateCameraParams(this.currentFov, this.currentResolution, this.currentIsOrtho);

    // Cache it (LRU-bounded)
    this.lruSet(this.gsplatMaterialCache, key, material);

    log.info(Modules.RENDERER, `Created gsplat material: ${key}`);
    return material;
  }

  /**
   * Check if a blending mode is opaque (solid rendering)
   */
  private isOpaqueMode(mode: BlendingMode): boolean {
    return mode === 'opaque';
  }

  /**
   * Update camera parameters for all registered materials
   */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    // Store current values for future material creation
    this.currentFov = fov;
    this.currentResolution.copy(resolution);
    this.currentIsOrtho = isOrtho;
    this.currentNearCull = nearCull;

    // Update all registered materials
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
    this.subscribeToDispose(material);
    // Immediately update with current camera params so the material is in sync
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
   * directly. (When the caller does eventually call `material.dispose()`,
   * the dispose-event listener from `subscribeToDispose` is a no-op
   * because the material is already gone from the registries.)
   */
  unregister(material: THREE.Material & CameraAwareMaterial): void {
    this.removeFromRegistries(material);
  }

  /**
   * Detach a pooled material from the global camera-update set without
   * evicting it from the LRU material cache. Used at clone sites in
   * `NodeFactory`: the per-node clone takes over the role of "this
   * geometry's material" while the pooled original stays in the cache
   * for the next caller. Without this, cloning + registering the clone
   * leaves the pooled material in `registeredMaterials`, so `disposeAll`
   * disposes the pooled material and the next cache hit returns a
   * disposed material.
   */
  detachFromGlobalUpdates(material: THREE.Material & CameraAwareMaterial): void {
    this.registeredMaterials.delete(material);
    this.ownedMaterials.delete(material);
  }

  /**
   * Dispose all cached materials
   */
  dispose(): void {
    // Snapshot and clear first so unregister() calls during dispose are no-ops
    const materials = [...this.registeredMaterials];
    this.registeredMaterials.clear();
    this.ownedMaterials.clear();
    this.pointMaterialCache.clear();
    this.lineMaterialCache.clear();
    this.gsplatMaterialCache.clear();
    for (const material of materials) {
      material.dispose();
    }
  }

  /**
   * Rebuild GPU-bound material state after a WebGL context-restore
   * event. The previous shader programs are now invalid (the WebGL
   * context they were compiled against is gone), so we drop the per-
   * type allocation caches. The renderer's next request via
   * `getPointMaterial` / `getLineMaterial` / `getGSplatMaterial` will
   * compile fresh shaders against the new context.
   *
   * Do NOT clear `registeredMaterials` / `ownedMaterials`. Those are
   * the camera-uniform update tracker, not allocation caches: they
   * hold materials currently attached to visible scene meshes and need
   * to keep receiving `updateCameraParams()` after restore. Clearing
   * them strands existing visible materials — subsequent resize/ortho/
   * DPR changes wouldn't reach their uniforms, and layer-cloned
   * materials (registered as "owned" at clone time via
   * `materialManager.register()` from layers-panel) would silently
   * miss camera updates.
   *
   * Implementation choice: lazy rebuild for ALLOCATION. Eager
   * re-creation would require us to remember every (props, key) pair
   * that was ever cached and to re-create all of them, but most are
   * transient (hidden layers, off-screen panels) and the renderer
   * will re-request only the materials it actually needs.
   *
   * The method is idempotent and safe to call repeatedly. Unlike
   * `dispose()`, it does NOT call `material.dispose()` on the cached
   * entries — those programs are already detached from a dead WebGL
   * context, and calling `dispose` on them tends to throw on some
   * drivers.
   */
  rebuildAfterContextRestore(): void {
    // Drop allocation caches only — fresh shaders will be compiled on
    // demand against the new context.
    this.pointMaterialCache.clear();
    this.lineMaterialCache.clear();
    this.gsplatMaterialCache.clear();
    // `registeredMaterials` / `ownedMaterials` deliberately preserved
    // — see comment above.
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    // Returns cache and ownership statistics for diagnostics/tests.
    return {
      pointMaterials: this.pointMaterialCache.size,
      lineMaterials: this.lineMaterialCache.size,
      gsplatMaterials: this.gsplatMaterialCache.size,
      ownedMaterials: this.ownedMaterials.size,
      cachedMaterials:
        this.pointMaterialCache.size + this.lineMaterialCache.size + this.gsplatMaterialCache.size,
      totalRegistered: this.registeredMaterials.size,
      /** Cumulative LRU evictions across all three caches since creation. */
      evictions: this.evictionCount,
      /** Configured cache bound (`0` = disabled). */
      maxSize: config.dataLoading.performance.materialCacheMaxSize,
      /**
       * Cumulative wall-clock ms spent inside `new XMaterial(...)`
       * calls (cache-miss path). Excludes WebGL program compilation,
       * which happens lazily on first render. Useful to spot
       * first-use stutter — divide by `createCount` for an average.
       */
      totalCreateMs: this.totalCreateMs,
      /** Number of `new XMaterial(...)` calls (cache misses). */
      createCount: this.createCount,
      keys: [
        ...Array.from(this.pointMaterialCache.keys()),
        ...Array.from(this.lineMaterialCache.keys()),
        ...Array.from(this.gsplatMaterialCache.keys()),
      ],
    };
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
