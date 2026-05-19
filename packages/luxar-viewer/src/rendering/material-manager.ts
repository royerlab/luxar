/**
 * Material Manager for Luxar
 *
 * This module manages all materials in the scene, providing caching,
 * global uniform updates, and support for multiple material types.
 * Supports point, line, and GSplat materials.
 */

import * as THREE from 'three';
import { PointMaterial } from './materials/point/material-glsl';
import { LineMaterial } from './line-material';
import { GSplatMaterial } from './gsplat-material';
import { PointTSLMaterial } from './materials/point/material-tsl';
import { LineTSLMaterial } from './line-material-tsl';
import { GSplatTSLMaterial } from './gsplat-material-tsl';
import { PointPickingMaterial } from './picking/point-picking-material';
import { LinePickingMaterial } from './picking/line-picking-material';
import { GSplatPickingMaterial } from './picking/gsplat-picking-material';
import { PointPickingTSLMaterial } from './picking/point-picking-material-tsl';
import { LinePickingTSLMaterial } from './picking/line-picking-material-tsl';
import { GSplatPickingTSLMaterial } from './picking/gsplat-picking-material-tsl';
import { MegaShaderMaterial } from './post-processing/mega-shader-material';
import { MegaShaderTSLMaterial } from './post-processing/mega-shader-material-tsl';
import type { PointPickingMaterialConfig } from './picking/point-picking-material';
import type { LinePickingMaterialConfig } from './picking/line-picking-material';
import type { GSplatPickingMaterialConfig } from './picking/gsplat-picking-material';
import type { MegaShaderConfig } from './post-processing/mega-shader-material';
import type { CameraAwareMaterial } from './camera-aware-material';
import type { RendererCapabilities } from './renderer-capabilities';
import { log, Modules } from '../utils/log';
import { clamp } from '../utils/clamp';
import { config } from '../config';

/**
 * Sentinel symbol that callers set transiently on a material when
 * they dispatch a `'dispose'` event purely to evict Three's cached
 * `RenderObject` — NOT because the material is actually being torn
 * down. `subscribeToDispose`'s listener checks for this flag and
 * skips registry / cache cleanup when it is present, so the material
 * continues to receive global camera updates and stays cached.
 *
 * Used by `data/scene-loader/invalidate-render-object.ts`, which is
 * fired when the GPU buffer pool rebuilds a geometry's underlying
 * `InstancedInterleavedBuffer` and the mesh's cached `RenderObject`
 * needs to drop its stale `vertexBuffers` set. Exported so the
 * dispatcher and the listener stay name-coupled.
 *
 * Symbol-keyed so the flag can't collide with Three's internal
 * properties or with userspace `userData` keys, and so it's invisible
 * to enumeration / serialization.
 */
export const SOFT_DISPOSE_FLAG = Symbol.for('luxar.invalidateRenderObject.softDispose');

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
 * The material backend tag used in cache keys and as the index into
 * the factory tables below. `'tsl'` selects the `NodeMaterial`-derived
 * implementation built for WebGPURenderer; `'glsl'` selects the
 * `ShaderMaterial`-derived implementation for `THREE.WebGLRenderer`.
 */
export type MaterialBackend = 'glsl' | 'tsl';

/**
 * Resolve the active material backend from a `RendererCapabilities`
 * snapshot. Returns `'glsl'` when caps are unset so unit tests that
 * touch material creation without configuring caps get the WebGL2
 * dispatch — same default as before this helper existed.
 *
 * Exported so the rendering layer has a single source of truth for
 * the GLSL/TSL decision: every `MaterialManager` factory routes its
 * dispatch through here, and tests can assert the mapping directly
 * without instantiating the manager.
 */
export function resolveMaterialBackend(caps: RendererCapabilities | null): MaterialBackend {
  return caps?.apiSurface === 'webgpu' ? 'tsl' : 'glsl';
}

/**
 * Constructor table for the visual material pair of each geometry
 * type. `MaterialManager.get{Point,Line,GSplat}Material` looks up
 * `VISUAL_FACTORIES[kind][backend]` to pick the class to instantiate,
 * replacing what used to be inline `useTSL ? new XTSL(...) : new
 * X(...)` ternaries. Each pair's two constructors share their
 * `*MaterialConfig` shape (the type union behind `LuxarXMaterial`),
 * so the indexed access is type-safe at the call site.
 */
const VISUAL_FACTORIES = {
  point: { glsl: PointMaterial, tsl: PointTSLMaterial },
  line: { glsl: LineMaterial, tsl: LineTSLMaterial },
  gsplat: { glsl: GSplatMaterial, tsl: GSplatTSLMaterial },
} as const;

/**
 * Per-geometry-type picking material returned by
 * `MaterialManager.create{Point,Line,GSplat}PickingMaterial`. The
 * concrete class is either the GLSL `*PickingMaterial` (WebGL2 path)
 * or the TSL `*PickingTSLMaterial` (WebGPU path). Picking materials
 * have a per-mesh lifetime (not cached); the dispatch routes through
 * {@link resolveMaterialBackend} like the visual-material counterparts.
 */
export type LuxarPointPickingMaterial = PointPickingMaterial | PointPickingTSLMaterial;
export type LuxarLinePickingMaterial = LinePickingMaterial | LinePickingTSLMaterial;
export type LuxarGSplatPickingMaterial = GSplatPickingMaterial | GSplatPickingTSLMaterial;

/**
 * Constructor table for the picking material pair of each geometry
 * type. Mirror of {@link VISUAL_FACTORIES} for the picking pipeline;
 * the `create*PickingMaterial` methods look up
 * `PICKING_FACTORIES[kind][backend]` and instantiate it directly
 * (picking materials are not cached).
 */
const PICKING_FACTORIES = {
  point: { glsl: PointPickingMaterial, tsl: PointPickingTSLMaterial },
  line: { glsl: LinePickingMaterial, tsl: LinePickingTSLMaterial },
  gsplat: { glsl: GSplatPickingMaterial, tsl: GSplatPickingTSLMaterial },
} as const;

/**
 * The single post-processing mega-shader material managed by
 * `PostProcessingManager`. The concrete class is either the GLSL
 * `MegaShaderMaterial` or the TSL `MegaShaderTSLMaterial`; both
 * expose the same setter / toggle / getter surface.
 */
export type LuxarMegaShaderMaterial = MegaShaderMaterial | MegaShaderTSLMaterial;

/**
 * Constructor pair for the post-processing mega-shader. Looked up by
 * `createMegaShaderMaterial`; one entry per backend, no per-geometry
 * indirection (there is only one mega-shader).
 */
const MEGA_SHADER_FACTORIES = {
  glsl: MegaShaderMaterial,
  tsl: MegaShaderTSLMaterial,
} as const;

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
  private pointMaterialCache = new Map<string, LuxarPointMaterial>();
  private lineMaterialCache = new Map<string, LuxarLineMaterial>();
  private gsplatMaterialCache = new Map<string, LuxarGSplatMaterial>();
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
      // Soft-dispose: `data/scene-loader/invalidate-render-object.ts`
      // dispatches a `'dispose'` event purely to evict Three's cached
      // `RenderObject` when the GPU buffer pool rebuilds a geometry's
      // underlying buffer. The material is NOT being torn down in
      // that case — skip registry/cache cleanup so it keeps receiving
      // global camera updates and stays in its allocation cache.
      // See `SOFT_DISPOSE_FLAG`.
      const tagged = material as unknown as Record<symbol, boolean | undefined>;
      if (tagged[SOFT_DISPOSE_FLAG]) return;
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
    } else if (material instanceof LineMaterial || material instanceof LineTSLMaterial) {
      for (const [key, cachedMaterial] of this.lineMaterialCache.entries()) {
        if (cachedMaterial === material) {
          this.lineMaterialCache.delete(key);
          break;
        }
      }
    } else if (material instanceof GSplatMaterial || material instanceof GSplatTSLMaterial) {
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
    if (this.caps && this.caps.apiSurface !== caps.apiSurface) {
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
   * active renderer reports `caps.apiSurface === 'webgpu'`, otherwise to the
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
    const backend = resolveMaterialBackend(this.caps);
    const key = `point_${backend}_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_r${radiusBucket}_s${sharpnessBucket}_t${transparent ? 1 : 0}`;

    // Check cache first (LRU-promoting)
    let material = this.lruGet(this.pointMaterialCache, key);
    if (material) {
      return material;
    }

    // Pass `blendingMode` through the constructor — the wrapper's
    // own constructor calls `applyBlendingMode` internally so
    // creation-time and runtime transitions share one code path.
    // Matches the Line + GSplat factories exactly (three-geometry
    // symmetry).
    const createStart = performance.now();
    const constructorConfig = {
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
      radiusScale: props.radiusScale,
      sharpnessScale: props.sharpnessScale,
    };
    material = new VISUAL_FACTORIES.point[backend](constructorConfig);
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
   * Get or create a line material with caching.
   *
   * Dispatches to `LineTSLMaterial` (NodeMaterial / TSL) when the
   * active renderer reports `caps.apiSurface === 'webgpu'`, otherwise to the
   * GLSL `LineMaterial`. Both classes expose the same update surface
   * via {@link LuxarLineMaterial}.
   */
  getLineMaterial(props: LineMaterialProperties): LuxarLineMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    const { opacityBucket, gammaBucket, intensityBucket, offsetBucket } =
      getCommonMaterialBuckets(props);

    const lineTransparent = props.blendingMode !== 'opaque';
    const backend = resolveMaterialBackend(this.caps);
    const key = `line_${backend}_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_t${lineTransparent ? 1 : 0}`;

    // Check cache first (LRU-promoting)
    let material = this.lruGet(this.lineMaterialCache, key);
    if (material) {
      return material;
    }

    // Create new material instance
    const createStart = performance.now();
    const constructorConfig = {
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
    };
    material = new VISUAL_FACTORIES.line[backend](constructorConfig);
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
   * Get or create a gsplat material with caching.
   *
   * Dispatches to `GSplatTSLMaterial` (NodeMaterial / TSL) when the
   * active renderer reports `caps.apiSurface === 'webgpu'`, otherwise to the
   * GLSL `GSplatMaterial`. Both classes expose the same update
   * surface via {@link LuxarGSplatMaterial}.
   */
  getGSplatMaterial(props: GSplatMaterialProperties): LuxarGSplatMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    const { opacityBucket, gammaBucket, intensityBucket, offsetBucket } =
      getCommonMaterialBuckets(props);
    const truncBucket = Math.round((props.truncationRadius ?? 3.0) * 10);

    const gsplatTransparent = props.blendingMode !== 'opaque';
    const backend = resolveMaterialBackend(this.caps);
    const key = `gsplat_${backend}_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_tr${truncBucket}_t${gsplatTransparent ? 1 : 0}`;

    // Check cache first (LRU-promoting)
    let material = this.lruGet(this.gsplatMaterialCache, key);
    if (material) {
      return material;
    }

    // Create new material instance
    const createStart = performance.now();
    const constructorConfig = {
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
      truncationRadius: props.truncationRadius ?? 3.0,
    };
    material = new VISUAL_FACTORIES.gsplat[backend](constructorConfig);
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
   * Create a per-mesh point picking material, dispatching on
   * `caps.apiSurface`. The returned material is NOT cached — picking
   * materials have per-mesh lifetimes; the NodeFactory disposes
   * them when the mesh disposes. Camera updates flow through
   * `register()` like any other camera-aware material.
   *
   * Same dispatch rationale as `getPointMaterial`: under
   * `WebGPURenderer`, a `THREE.ShaderMaterial`-derived picking
   * material would crash the NodeBuilder.
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
