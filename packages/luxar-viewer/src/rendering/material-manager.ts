/**
 * Material Manager for Luxar
 *
 * This module manages all materials in the scene, providing caching,
 * global uniform updates, and support for multiple material types.
 * Supports point materials and line materials.
 */

import * as THREE from 'three';
import { PointMaterial } from './point-material';
import { LineMaterial } from './line-material';
import { GSplatMaterial } from './gsplat-material';
import type { CameraAwareMaterial } from './camera-aware-material';
import { log, Modules } from '../utils/log';
import { config } from '../config';

/**
 * Supported blending modes for materials.
 *
 * - 'normal': Standard alpha blending (semi-transparent)
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
export class MaterialManager {
  private pointMaterialCache = new Map<string, PointMaterial>();
  private lineMaterialCache = new Map<string, LineMaterial>();
  private gsplatMaterialCache = new Map<string, GSplatMaterial>();
  private registeredMaterials = new Set<THREE.Material & CameraAwareMaterial>();
  /** Per-cache eviction count (read by getCacheStats; no behavior). */
  private evictionCount = 0;
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
   * eviction (legacy unbounded behavior).
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
   * manager) breaks the import cycle that previously existed between
   * material-manager.ts and {point,line,gsplat}-material.ts.
   */
  private subscribeToDispose(material: THREE.Material & CameraAwareMaterial): void {
    const onDispose = (): void => {
      this.removeFromRegistries(material);
      material.removeEventListener('dispose', onDispose);
    };
    material.addEventListener('dispose', onDispose);
  }

  /**
   * Internal cleanup: remove `material` from every registry and cache.
   * Called from the dispose listener and (for backwards compatibility)
   * from the public `unregister` method. Idempotent.
   */
  private removeFromRegistries(material: THREE.Material & CameraAwareMaterial): void {
    this.registeredMaterials.delete(material);
    this.ownedMaterials.delete(material);

    if (material instanceof PointMaterial) {
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
   * Get or create a point material with caching
   */
  getPointMaterial(props: PointMaterialProperties): PointMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    // This prevents floating-point precision issues while still grouping similar values
    // Clamp values to valid ranges to handle edge cases gracefully
    const opacityBucket = Math.round(Math.max(0, Math.min(1, props.opacity)) * 100); // 0-100 range
    const gammaBucket = Math.round(Math.max(0, Math.min(10, props.gamma)) * 100); // 0-1000 range
    const intensityBucket = Math.round(Math.max(0, Math.min(100, props.intensity)) * 100); // 0-10000 range
    const offsetBucket = Math.round((Math.max(-10, Math.min(10, props.offset)) + 10) * 10); // 0-200 range
    const radiusBucket = props.radiusScale
      ? Math.round(Math.max(0, props.radiusScale) * 1000)
      : 1000;
    const sharpnessBucket = props.sharpnessScale
      ? Math.round(Math.max(0, props.sharpnessScale) * 1000)
      : 1000;

    const isOpaque = this.isOpaqueMode(props.blendingMode);
    const transparent = !isOpaque;
    const key = `point_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_r${radiusBucket}_s${sharpnessBucket}_t${transparent ? 1 : 0}`;

    // Check cache first (LRU-promoting)
    let material = this.lruGet(this.pointMaterialCache, key);
    if (material) {
      return material;
    }

    // Determine material properties based on mode
    const isAdditive = props.blendingMode === 'additive';

    // Create new PointMaterial instance
    material = new PointMaterial({
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blending: this.getThreeBlending(props.blendingMode),
      // Opaque: write depth. All others: no depth write
      depthWrite: isOpaque || (props.blendingMode === 'normal' && props.opacity >= 0.99),
      // Additive ignores depth entirely (renders on top of everything)
      depthTest: !isAdditive,
      transparent: !isOpaque,
      radiusScale: props.radiusScale,
      sharpnessScale: props.sharpnessScale,
    });

    // Configure custom blending for max mode
    if (props.blendingMode === 'max') {
      material.blendEquation = THREE.MaxEquation; // Max(source, destination)
      material.blendSrc = THREE.OneFactor;
      material.blendDst = THREE.OneFactor;
    }

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
    const opacityBucket = Math.round(Math.max(0, Math.min(1, props.opacity)) * 100);
    const gammaBucket = Math.round(Math.max(0, Math.min(10, props.gamma)) * 100); // 0-1000 range
    const intensityBucket = Math.round(Math.max(0, Math.min(100, props.intensity)) * 100);
    const offsetBucket = Math.round((Math.max(-10, Math.min(10, props.offset)) + 10) * 10);

    const lineTransparent = props.blendingMode !== 'opaque';
    const key = `line_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_t${lineTransparent ? 1 : 0}`;

    // Check cache first (LRU-promoting)
    let material = this.lruGet(this.lineMaterialCache, key);
    if (material) {
      return material;
    }

    // Create new LineMaterial instance
    material = new LineMaterial({
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
    });

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
    const opacityBucket = Math.round(Math.max(0, Math.min(1, props.opacity)) * 100);
    const gammaBucket = Math.round(Math.max(0, Math.min(10, props.gamma)) * 100); // 0-1000 range
    const intensityBucket = Math.round(Math.max(0, Math.min(100, props.intensity)) * 100);
    const offsetBucket = Math.round((Math.max(-10, Math.min(10, props.offset)) + 10) * 10);
    const truncBucket = Math.round((props.truncationRadius ?? 3.0) * 10);

    const gsplatTransparent = props.blendingMode !== 'opaque';
    const key = `gsplat_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_tr${truncBucket}_t${gsplatTransparent ? 1 : 0}`;

    // Check cache first (LRU-promoting)
    let material = this.lruGet(this.gsplatMaterialCache, key);
    if (material) {
      return material;
    }

    // Create new GSplatMaterial instance
    material = new GSplatMaterial({
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
      truncationRadius: props.truncationRadius ?? 3.0,
    });

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
   * Convert our blending mode to Three.js blending constant
   */
  private getThreeBlending(mode: BlendingMode): THREE.Blending {
    switch (mode) {
      case 'opaque':
        return THREE.NormalBlending; // Solid rendering
      case 'normal':
        return THREE.NormalBlending; // Semi-transparent alpha blending
      case 'additive':
      case 'luminous':
        // Both use AdditiveBlending (SrcAlpha, One) - same visual output
        // Difference is only in depthTest (additive=false, luminous=true)
        return THREE.AdditiveBlending;
      case 'max':
        return THREE.CustomBlending; // Max blending uses CustomBlending with MaxEquation
      default:
        log.warning(Modules.RENDERER, `Unknown blending mode: ${mode}, using normal`);
        return THREE.NormalBlending;
    }
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
   * Unregister a material from global updates.
   *
   * @deprecated Cleanup is now automatic via the material's `dispose`
   *   event listener registered by `subscribeToDispose`. This method
   *   stays for the rare external caller that needs to drop a material
   *   from the registry without disposing it.
   */
  unregister(material: THREE.Material & CameraAwareMaterial): void {
    this.removeFromRegistries(material);
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
   * context they were compiled against is gone), so we drop the cache
   * and the registry. The renderer's next request via
   * `getPointMaterial` / `getLineMaterial` / `getGSplatMaterial` will
   * compile fresh shaders against the new context.
   *
   * Implementation choice: lazy rebuild. Eager re-creation would
   * require us to remember every (props, key) pair that was ever
   * cached and to re-create all of them, but most are transient
   * (hidden layers, off-screen panels) and the renderer will
   * re-request only the materials it actually needs in the next frame.
   *
   * Mirror of the durable-state pattern in
   * `rendering/post-processing/context-recovery.ts`: this method is
   * idempotent and safe to call repeatedly. Unlike `dispose()`,
   * however, it does NOT call `material.dispose()` on the entries —
   * those programs are already detached from a dead WebGL context, and
   * calling `dispose` on them tends to throw on some drivers.
   */
  rebuildAfterContextRestore(): void {
    this.registeredMaterials.clear();
    this.ownedMaterials.clear();
    this.pointMaterialCache.clear();
    this.lineMaterialCache.clear();
    this.gsplatMaterialCache.clear();
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
