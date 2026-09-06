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
import type { PointTSLMaterial } from './materials/point/material-tsl';
import type { LineTSLMaterial } from './materials/line/material-tsl';
import type { GSplatTSLMaterial } from './materials/gsplat/material-tsl';
import type { MeshTSLMaterial } from './materials/mesh/material-tsl';
import type { PhysicalMeshMaterial } from './materials/mesh-physical/material-glsl';
import type { PhysicalMeshTSLMaterial } from './materials/mesh-physical/material-tsl';
import type { PhysicalMeshMaterialConfig } from './materials/mesh-physical/config';
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
  type PhysicalMeshMaterialConfig,
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
 * Same union shape as its three siblings, including `updateCameraParams` — though a
 * mesh consumes only half of it. There is no screen-space size to recompute from
 * fov/resolution, but the projection mode and near-cull distance drive the shared
 * near fade, which applies to a surface exactly as it does to a sprite. So mesh
 * joins the camera broadcast like everything else.
 */
export type LuxarMeshMaterial = MeshMaterial | MeshTSLMaterial;
/**
 * The mesh's PHYSICAL family — three's own physically based material behind the Luxar
 * leaf surface (`MESH_PHYSICAL_MATERIALS_SPEC.md` §3.2). Not part of
 * {@link LuxarMeshMaterial}: it has no camera surface (the near fade is a house-shader
 * feature) and none of the house update methods, so the two unions stay separate and a
 * call site that needs `updateShading` cannot be handed this by accident.
 */
export type LuxarPhysicalMeshMaterial = PhysicalMeshMaterial | PhysicalMeshTSLMaterial;

/**
 * Per-geometry-type picking material returned by
 * `MaterialManager.create{Point,Line,GSplat,Mesh}PickingMaterial`.
 * Picking materials have a per-mesh lifetime (not cached).
 */
export type LuxarPointPickingMaterial = PointPickingMaterial | PointPickingTSLMaterial;
export type LuxarLinePickingMaterial = LinePickingMaterial | LinePickingTSLMaterial;
export type LuxarGSplatPickingMaterial = GSplatPickingMaterial | GSplatPickingTSLMaterial;
/**
 * Like its three siblings, including `updateCameraParams` — the mesh pick pass has
 * no screen-space footprint to size, but it does have to reproduce the visual near
 * fade, so it takes the same two camera inputs and joins the same broadcast,
 * exactly as the visual mesh material does.
 */
export type LuxarMeshPickingMaterial = MeshPickingMaterial | MeshPickingTSLMaterial;

/**
 * The single post-processing mega-shader material managed by
 * `PostProcessingManager`.
 */
export type LuxarMegaShaderMaterial = MegaShaderMaterial | MegaShaderTSLMaterial;

/**
 * Manages all materials in the scene with lifecycle tracking and
 * global camera-uniform updates. Supports points, lines, gsplats and mesh.
 *
 * ALL visual materials are PER NODE and are never cached. Point, line and
 * gsplat materials each carry the node's own element texture (`uPointTex` /
 * `uLineTex` / `uSplatTex`), so sharing one would rebind a node's texture onto
 * another node's mesh at every commit; a mesh material carries no element
 * texture but holds per-node shading state instead (see
 * {@link MaterialManager.getMeshMaterial}). The historical line-material LRU
 * was the last cached kind and died with the lines texture-storage migration.
 */
export class MaterialManager {
  private registeredMaterials = new Set<THREE.Material & CameraAwareMaterial>();
  /**
   * Renderer capabilities — drives the GLSL vs. TSL dispatch in the
   * four `getXMaterial` factories. `SceneManager.setupRenderer` calls
   * {@link setCaps} once the renderer is alive; before that hook
   * fires, the manager defaults to the WebGL2 path so unit tests
   * that touch material creation don't need to know about caps.
   */
  private caps: RendererCapabilities | null = null;
  /**
   * Materials whose `dispose` event we have already wired a listener for.
   * Separate from `registeredMaterials` because `register()` /
   * `getXMaterial()` can be called repeatedly with the same instance
   * (a re-`register()`, or a clone re-registered explicitly), and a second
   * `addEventListener('dispose', ...)` would silently stack listeners on
   * THREE's EventDispatcher.
   */
  private subscribedMaterials = new WeakSet<THREE.Material & CameraAwareMaterial>();
  /**
   * Diagnostic: cumulative wall-clock time spent constructing
   * materials (Point/Line/GSplat/Mesh). Useful as a proxy for "how much
   * time does the user spend waiting for material-creation work" —
   * first-use stutter shows up as a single large delta on the
   * affected animation frame. Exposed in `getCacheStats()`.
   */
  private totalCreateMs = 0;
  private createCount = 0;
  /**
   * Listeners for {@link onPhysicalMaterialCreated}. The scene environment that
   * lights physical meshes is built lazily, and this manager is the one place a
   * physical material is born — so this hook is how the `SceneManager` learns it
   * is time to build it, without the node factory knowing about renderers.
   */
  private physicalMaterialListeners = new Set<() => void>();
  /**
   * Materials that entered through `register()` rather than a manager
   * factory (per-node point/line/gsplat/mesh materials live in
   * `registeredMaterials` only).
   *
   * Examples: GPU-picking materials and colormap clones. These need
   * manager-level disposal — and global camera uniforms when they are
   * camera-aware — but are tracked separately for leak diagnostics.
   *
   * Typed as plain `THREE.Material` because `register()` accepts one: a
   * non-camera-aware entry belongs in the leak diagnostic exactly as much
   * as a camera-aware one, so this set must span both rather than silently
   * omitting half of them.
   */
  private ownedMaterials = new Set<THREE.Material>();
  /**
   * Materials that are tracked for disposal but take NO camera broadcast.
   *
   * The generic fallback for anything without `updateCameraParams`: {@link register}
   * dispatches on `isCameraAwareMaterial` and lands the rest here. All four geometry
   * types (mesh included, since #1431 gave it the near fade) are camera-aware today,
   * so nothing from the `getXMaterial` factories reaches this set.
   *
   * It is NOT what keeps such a material from leaking: `register()` adds to
   * `ownedMaterials` on the same path, and {@link dispose} unions that in. Two jobs
   * are left, and both are real. It is the destination that is *not* the camera
   * broadcast — `updateCameraParams` iterates `registeredMaterials`, so a material
   * with no `updateCameraParams` has to land somewhere else or the broadcast would
   * throw on it. And it is a term in the stats snapshot: `totalRegistered` is
   * `registeredMaterials.size + staticMaterials.size`, so a non-camera-aware entry
   * shows up in the leak diagnostic instead of vanishing from it.
   * See {@link LifecycleCtx.staticMaterials}.
   */
  private staticMaterials = new Set<THREE.Material>();
  private currentFov = (60 * Math.PI) / 180; // Current FOV in radians (or frustumHeight for ortho)
  private currentResolution = new THREE.Vector2(1920, 1080); // Use reasonable default
  private currentIsOrtho = false;
  private currentNearCull: number | undefined = undefined;
  private currentPixelRatio = 1;

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
   * The two capabilities a mesh texture upload depends on.
   *
   * Exposed as a narrow accessor rather than the whole `RendererCapabilities`
   * because the texture upload is the only consumer outside this class, and it
   * needs exactly these two. Returns the conservative answer when caps have not
   * been set (unit tests, pre-renderer): `filterableFloatTextures: false` selects a
   * HalfFloat upload, which filters correctly on every backend — degrading HDR
   * precision is recoverable, whereas a float32 texture the device cannot filter
   * silently samples blocky.
   */
  getTextureCapabilities(): { filterableFloatTextures: boolean; maxAnisotropy: number } {
    return {
      filterableFloatTextures: this.caps?.hdr.filterableFloatTextures ?? false,
      maxAnisotropy: 8,
    };
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
   * There is no material cache: every material is per-node, so
   * `getCacheStats()` reports only registry size and create-time, never a
   * cache size. Mirrors {@link getGSplatMaterial}.
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
    const material = new (VISUAL_FACTORIES.point[backend]())({
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
      this.currentNearCull,
      this.currentPixelRatio
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
   * There is no material cache: every material is per-node. Mirrors
   * {@link getPointMaterial} / {@link getGSplatMaterial}.
   *
   * Dispatches to `LineTSLMaterial` (NodeMaterial / TSL) when the
   * active renderer reports `caps.apiSurface === 'webgpu'`, otherwise to the
   * GLSL `LineMaterial`. Both classes expose the same update surface
   * via {@link LuxarLineMaterial}.
   */
  getLineMaterial(props: LineMaterialProperties): LuxarLineMaterial {
    const backend = resolveMaterialBackend(this.caps);

    const createStart = performance.now();
    const material = new (VISUAL_FACTORIES.line[backend]())({
      opacity: props.opacity,
      absorption: props.absorption,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
      join: props.join,
      primitive: props.primitive,
    });
    this.totalCreateMs += performance.now() - createStart;
    this.createCount++;

    this.registeredMaterials.add(material);
    subscribeToDispose(material, this.lifecycleCtx);
    material.updateCameraParams(
      this.currentFov,
      this.currentResolution,
      this.currentIsOrtho,
      this.currentNearCull,
      this.currentPixelRatio
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
   * There is no material cache: every material is per-node.
   *
   * Dispatches to `GSplatTSLMaterial` (NodeMaterial / TSL) when the
   * active renderer reports `caps.apiSurface === 'webgpu'`, otherwise to the
   * GLSL `GSplatMaterial`. Both classes expose the same update
   * surface via {@link LuxarGSplatMaterial}.
   */
  getGSplatMaterial(props: GSplatMaterialProperties): LuxarGSplatMaterial {
    const backend = resolveMaterialBackend(this.caps);

    const createStart = performance.now();
    const material = new (VISUAL_FACTORIES.gsplat[backend]())({
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
      this.currentNearCull,
      this.currentPixelRatio
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
   * pieces of per-node state that make sharing wrong anyway: the `shading`
   * compile-time variant (a function of that node's `shading` and its normals'
   * validity for the active view) and `side` (re-applied per epoch by
   * `applyMeshSide`). Sharing would let one node's shading model and face-sidedness
   * follow another's.
   *
   * Enters `registeredMaterials` and takes the camera broadcast like its three
   * siblings. It consumes only half of it — there is no screen-space size to
   * recompute from fov/resolution — but the projection mode and near-cull distance
   * drive the shared near fade (#1431), and a mesh left out of the broadcast would
   * fade against the constructor's 0.1 default instead of the scene's. There is no
   * `meshMaterialCache`: no type has a material cache — every material is per-node,
   * so `getCacheStats()` reports only registry size and create-time, never a cache
   * size.
   *
   * Dispatches to `MeshTSLMaterial` when the active renderer reports
   * `caps.apiSurface === 'webgpu'`, otherwise the GLSL `MeshMaterial`.
   */
  getMeshMaterial(props: MeshMaterialProperties): LuxarMeshMaterial {
    const backend = resolveMaterialBackend(this.caps);

    const createStart = performance.now();
    const material = new (VISUAL_FACTORIES.mesh[backend]())({
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
      shading: props.shading,
      baseColorTexture: props.baseColorTexture,
      baseColorTextureLuminance: props.baseColorTextureLuminance,
      ambient: props.ambient,
      shadeExponent: props.shadeExponent,
      specular: props.specular,
      shininess: props.shininess,
      alphaCutoff: props.alphaCutoff,
    });
    this.totalCreateMs += performance.now() - createStart;
    this.createCount++;

    this.registeredMaterials.add(material);
    subscribeToDispose(material, this.lifecycleCtx);
    material.updateCameraParams(
      this.currentFov,
      this.currentResolution,
      this.currentIsOrtho,
      this.currentNearCull,
      this.currentPixelRatio
    );

    log.info(Modules.RENDERER, `Created per-node mesh material (${backend})`);
    return material;
  }

  /**
   * Create a per-node PHYSICAL mesh material — three's `MeshPhysicalMaterial` (GLSL)
   * or `MeshPhysicalNodeMaterial` (TSL) behind the Luxar leaf surface
   * (`MESH_PHYSICAL_MATERIALS_SPEC.md` §3.2).
   *
   * Deliberately NOT `getMeshMaterial` with a flag: the two families share nothing
   * but the geometry. This one takes no camera broadcast (it has no near fade — it
   * enters through {@link register}, which files it as static), stamps no blending
   * mode unless opaque, and is lit by the scene environment rather than a shader
   * constant — which is why every creation notifies
   * {@link onPhysicalMaterialCreated}: the environment is built lazily, on the
   * first of these, and never for a scene that has none.
   */
  getMeshPhysicalMaterial(config: PhysicalMeshMaterialConfig): LuxarPhysicalMeshMaterial {
    const backend = resolveMaterialBackend(this.caps);
    const createStart = performance.now();
    const material = new (VISUAL_FACTORIES.meshPhysical[backend]())(config);
    this.totalCreateMs += performance.now() - createStart;
    this.createCount++;
    this.register(material);
    for (const listener of this.physicalMaterialListeners) listener();
    log.info(Modules.RENDERER, `Created per-node physical mesh material (${backend})`);
    return material;
  }

  /**
   * Subscribe to physical-material creation. Fired synchronously inside
   * {@link getMeshPhysicalMaterial}, after the material exists, on EVERY creation —
   * the subscriber is expected to be idempotent (`SceneEnvironment.ensure` is).
   *
   * @returns An unsubscribe function.
   */
  onPhysicalMaterialCreated(listener: () => void): () => void {
    this.physicalMaterialListeners.add(listener);
    return () => {
      this.physicalMaterialListeners.delete(listener);
    };
  }

  /**
   * Create a per-mesh point picking material, dispatching on
   * `caps.apiSurface`. The returned material is NOT cached — picking
   * materials have per-mesh lifetimes; the NodeFactory disposes
   * them when the mesh disposes. Camera updates flow through
   * `register()` like any other camera-aware material.
   */
  createPointPickingMaterial(config: PointPickingMaterialConfig): LuxarPointPickingMaterial {
    return new (PICKING_FACTORIES.point[resolveMaterialBackend(this.caps)]())(config);
  }

  /** Same shape as `createPointPickingMaterial`, for lines. */
  createLinePickingMaterial(config: LinePickingMaterialConfig): LuxarLinePickingMaterial {
    return new (PICKING_FACTORIES.line[resolveMaterialBackend(this.caps)]())(config);
  }

  /** Same shape as `createPointPickingMaterial`, for gsplats. */
  createGSplatPickingMaterial(config: GSplatPickingMaterialConfig): LuxarGSplatPickingMaterial {
    return new (PICKING_FACTORIES.gsplat[resolveMaterialBackend(this.caps)]())(config);
  }

  /** Same shape as `createPointPickingMaterial`, for meshes. */
  createMeshPickingMaterial(config: MeshPickingMaterialConfig): LuxarMeshPickingMaterial {
    return new (PICKING_FACTORIES.mesh[resolveMaterialBackend(this.caps)]())(config);
  }

  /**
   * Create the post-processing mega-shader material, dispatching on
   * `caps.apiSurface`. PostProcessingManager owns the single instance; this
   * is the seam between `THREE.ShaderMaterial`-derived
   * `MegaShaderMaterial` and the TSL `MegaShaderTSLMaterial`.
   */
  createMegaShaderMaterial(cfg: MegaShaderConfig): LuxarMegaShaderMaterial {
    return new (MEGA_SHADER_FACTORIES[resolveMaterialBackend(this.caps)]())(cfg);
  }

  /** Update camera parameters for all registered materials. */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean,
    nearCull: number | undefined,
    pixelRatio: number
  ): void {
    this.currentFov = fov;
    this.currentResolution.copy(resolution);
    this.currentIsOrtho = isOrtho;
    this.currentNearCull = nearCull;
    this.currentPixelRatio = pixelRatio;

    for (const material of this.registeredMaterials) {
      material.updateCameraParams(fov, resolution, isOrtho, nearCull, pixelRatio);
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
   * receives the current camera state immediately; one that reads no camera uniform
   * at all is tracked for disposal only.
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
      // Reads no camera uniform at all — tracked for disposal only. See
      // `staticMaterials`.
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
      this.currentNearCull,
      this.currentPixelRatio
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
    // Two entry paths, and `registeredMaterials` only covers one of them: the four
    // `getXMaterial` factories add there and nowhere else, so it is load-bearing.
    // Everything that arrives through `register()` instead lands in
    // `ownedMaterials` — and, when it is not camera-aware, ALSO in
    // `staticMaterials`. So those last two are two overlapping views of the same
    // path and either one alone would already complete the cover; both are spread
    // because the redundancy is free and neither set exists for teardown's sake in
    // the first place (see their field docs). What must never be dropped is a whole
    // PATH: `registeredMaterials`, or both of the other two.
    const materials = new Set<THREE.Material>([
      ...this.registeredMaterials,
      ...this.ownedMaterials,
      ...this.staticMaterials,
    ]);
    this.registeredMaterials.clear();
    this.ownedMaterials.clear();
    this.staticMaterials.clear();
    this.physicalMaterialListeners.clear();
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
