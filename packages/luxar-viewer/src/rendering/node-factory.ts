/**
 * NodeFactory — Creates THREE.js scene nodes from loaded data.
 *
 * Extracted from SceneLoader to separate node creation (geometry, material,
 * userData, transforms) from data loading and orchestration.
 *
 * This separation enables the picking system to hook into node creation
 * and create parallel pick-scene shadow nodes.
 *
 * Implicit dependency: uses the `materialManager` singleton from
 * ../rendering/material-manager (same pattern as SceneLoader).
 */

import * as THREE from 'three';
import {
  materialManager,
  type BlendingMode,
  type LuxarPointMaterial,
  type LuxarLineMaterial,
  type LuxarGSplatMaterial,
} from './material-manager';
import { getColormapTexture } from './colormap-textures';
import {
  createInstancedLinesMesh,
  isAllSharpnessTwo,
  type InstancedLinesMeshConfig,
} from './line-geometry';
import { createInstancedGSplatsMesh, type InstancedGSplatsMeshConfig } from './gsplat-geometry';
import type { LoadedPointsData, DataLoader } from '../data/data-loader-types';
import type { PointsMetadata, PointsUserData } from '../types/points';
import type { LinesMetadata, LinesUserData, LinesDataLoader } from '../types/lines';
import type { GSplatsMetadata, GSplatsUserData, GSplatsDataLoader } from '../types/gsplats';
import { log, Modules } from '../utils/log';
import type { PickingSystem } from './picking/picking-system';
import {
  validateLoadedPointsData as validateLoadedPointsDataImpl,
  validateColorMode as validateColorModeImpl,
  validateTransformFormat as validateTransformFormatImpl,
} from './node-factory/validation';
import { applyTransform as applyTransformImpl } from './node-factory/transforms';
import {
  createPointsGeometry as createPointsGeometryImpl,
  createPointsMaterial as createPointsMaterialImpl,
} from './node-factory/create-points-node';
// Picking materials are constructed via `materialManager.create*PickingMaterial`
// helpers so the GLSL vs. TSL dispatch on `caps.apiSurface` lives in one place. The
// concrete types are still imported elsewhere (e.g. material-sync-helpers).

export class NodeFactory {
  private pickingSystem: PickingSystem | null = null;

  /** Wire up the picking system. When set, all subsequent node creations
   *  will also create shadow pick-scene nodes. */
  setPickingSystem(ps: PickingSystem | null): void {
    this.pickingSystem = ps;
  }

  /**
   * Invalidate the cached pick buffer (call after geometry updates).
   * Also invalidates the picking system's world-AABB cache, since
   * geometry changes can move the bounding box. (Camera-only motion
   * does NOT need to invalidate boxes and reaches `markDirty()` via
   * the controls 'change' event, not this method.)
   */
  markPickingDirty(): void {
    this.pickingSystem?.markDirty();
    this.pickingSystem?.invalidateBoxes();
  }

  /**
   * Retroactively register already-loaded scene nodes with the picking system.
   * Called after initPicking() since the scene is loaded before picking is wired up.
   */
  registerExistingSceneNodes(root: THREE.Object3D): void {
    if (!this.pickingSystem) return;

    root.traverse((obj) => {
      const nodeType = obj.userData?.nodeType as string | undefined;
      if (!nodeType || obj.userData.pickId != null) return; // skip non-data or already registered

      if (nodeType === 'points' && obj instanceof THREE.Mesh) {
        const pickId = this.pickingSystem!.allocatePickId();
        obj.userData.pickId = pickId;
        const radiusScale = obj.geometry?.userData?.radiusScale ?? 1.0;
        const sharpnessScale = obj.geometry?.userData?.sharpnessScale ?? 1.0;
        const pickMaterial = materialManager.createPointPickingMaterial({
          nodeId: pickId,
          radiusScale,
          sharpnessScale,
        });
        materialManager.register(pickMaterial);
        const pickNode = new THREE.Mesh(obj.geometry, pickMaterial);
        pickNode.frustumCulled = false;
        pickNode.matrixWorld.copy(obj.matrixWorld);
        this.pickingSystem!.registerNode(obj, pickNode, pickId);
      } else if (nodeType === 'lines' && obj instanceof THREE.Mesh) {
        const pickId = this.pickingSystem!.allocatePickId();
        obj.userData.pickId = pickId;
        const pickMaterial = materialManager.createLinePickingMaterial({ nodeId: pickId });
        materialManager.register(pickMaterial);
        const pickNode = new THREE.Mesh(obj.geometry, pickMaterial);
        pickNode.matrixWorld.copy(obj.matrixWorld);
        this.pickingSystem!.registerNode(obj, pickNode, pickId);
      } else if (nodeType === 'gsplats' && obj instanceof THREE.Mesh) {
        const pickId = this.pickingSystem!.allocatePickId();
        obj.userData.pickId = pickId;
        const pickMaterial = materialManager.createGSplatPickingMaterial({ nodeId: pickId });
        materialManager.register(pickMaterial);
        const pickNode = new THREE.Mesh(obj.geometry, pickMaterial);
        pickNode.matrixWorld.copy(obj.matrixWorld);
        this.pickingSystem!.registerNode(obj, pickNode, pickId);
      }
    });

    log.info(
      Modules.SCENE_LOADER,
      `Registered ${this.pickingSystem!.registeredNodeCount} existing nodes for picking`
    );
  }

  /**
   * Rebuild picking-system registrations after a WebGL context-restore
   * event. The pick materials in `pickingSystem.nodeMap` were compiled
   * against the now-dead WebGL context, so we drop the registrations
   * (without disposing — see `PickingSystem.clearRegistrationsForRebuild`)
   * and re-create them via {@link registerExistingSceneNodes}, which
   * produces fresh pick materials against the new context.
   *
   * Mirror of `MaterialManager.rebuildAfterContextRestore` — both are
   * called from `SceneManager.contextRestoredHandler` in the order
   * post-processing → materials → nodes.
   */
  rebuildAfterContextRestore(root: THREE.Object3D): void {
    if (!this.pickingSystem) return;
    this.pickingSystem.clearRegistrationsForRebuild();
    // Reset every scene node's pickId so registerExistingSceneNodes
    // re-allocates a fresh one on the rebuilt picking system.
    root.traverse((obj) => {
      if (obj.userData?.pickId != null) {
        obj.userData.pickId = undefined;
      }
    });
    this.registerExistingSceneNodes(root);
  }

  // ============================================================================
  // Points Node Creation
  // ============================================================================

  /**
   * Create a THREE.Mesh object from loaded point data.
   *
   * Each point is rendered as an instanced quad sprite, matching the
   * line + gsplat geometry pattern. The mesh's geometry is built by
   * `createPointsGeometry`, which attaches per-instance attributes
   * (aCenter, aRadius, aSharpness, aColor, optional aScalar) to a
   * shared unit-quad base.
   *
   * Handles geometry creation, material selection, userData, and
   * transforms.
   */
  createPointsNode(
    path: string,
    attrs: PointsMetadata,
    data: LoadedPointsData,
    loader: DataLoader
  ): THREE.Mesh {
    const maxRadius = attrs.max_radius ?? 1.0;
    const maxSharpness = attrs.max_sharpness ?? 31.0;
    const geometry = this.createPointsGeometry(data, maxRadius, maxSharpness);

    const radiusScale = geometry.userData.radiusScale ?? 1.0;
    const sharpnessScale = geometry.userData.sharpnessScale ?? 1.0;
    const material = this.createPointsMaterial(attrs, radiusScale, sharpnessScale, geometry, path);

    const points = new THREE.Mesh(geometry, material);
    points.name = path;
    // Three's per-mesh frustum culling tests the bounding sphere of
    // the base quad geometry, not the spread of instances. Disable so
    // we don't lose all points because the unit-sized base quad sits
    // outside the camera frustum.
    points.frustumCulled = false;

    points.userData = {
      nodeType: 'points',
      loader,
      attrs,
      maxRadius: attrs.max_radius ?? 1.0,
      visiblePointCount: data.pointCount,
    } as PointsUserData;

    if (attrs.transform) {
      this.applyTransform(points, attrs.transform);
    }

    // Create picking shadow node if picking system is active
    if (this.pickingSystem) {
      const pickId = this.pickingSystem.allocatePickId();
      points.userData.pickId = pickId;
      const pickMaterial = materialManager.createPointPickingMaterial({
        nodeId: pickId,
        radiusScale,
        sharpnessScale,
      });
      materialManager.register(pickMaterial);
      const pickNode = new THREE.Mesh(geometry, pickMaterial);
      pickNode.frustumCulled = false;
      pickNode.matrixWorld.copy(points.matrixWorld);
      this.pickingSystem.registerNode(points, pickNode, pickId);
    }

    return points;
  }

  // ============================================================================
  // Lines Node Creation
  // ============================================================================

  /**
   * Create a THREE.Mesh (instanced lines) from processed line data.
   */
  createLinesNode(
    path: string,
    nodeAttrs: Record<string, unknown>,
    attrs: LinesMetadata,
    processed: InstancedLinesMeshConfig,
    loader: LinesDataLoader
  ): THREE.Mesh {
    let material: LuxarLineMaterial = materialManager.getLineMaterial({
      opacity: (attrs.opacity as number | undefined) ?? 1.0,
      gamma: (attrs.gamma as number | undefined) ?? 1.0,
      intensity: (attrs.intensity as number | undefined) ?? 1.0,
      offset: (attrs.offset as number | undefined) ?? 0.0,
      blendingMode: (attrs.blending_mode as string | undefined as BlendingMode) ?? 'additive',
    });

    // Apply colormap if specified and scalar data exists. When the
    // line's metadata declares `colormap='custom'`, the scene loader
    // has already attached the LUT bytes as `nodeAttrs.customLutBytes`.
    const lnColormapName = nodeAttrs.colormap as string | undefined;
    const lnHasScalars = !!nodeAttrs.has_scalars;
    const linesScalarsReady = 'startScalars' in processed && 'endScalars' in processed;
    if (lnColormapName && lnHasScalars) {
      if (!linesScalarsReady) {
        log.warning(
          Modules.SCENE_LOADER,
          `[${path}] Line scalar colormap requested but scalar attributes are not bound. Colormap suppressed.`
        );
      } else {
        const lnLutBytes = nodeAttrs.customLutBytes as Uint8Array | undefined;
        const lnColormapTex = getColormapTexture(lnColormapName, lnLutBytes);
        if (lnColormapTex) {
          // B.1: detach the pooled original from global updates BEFORE
          // cloning, so disposeAll doesn't dispose the cache entry that
          // still serves other callers. The clone takes the global-
          // update slot; the pooled material stays in lineMaterialCache.
          materialManager.detachFromGlobalUpdates(material);
          material = material.clone() as typeof material;
          materialManager.register(material);
          material.updateColormapTexture(lnColormapTex);
          const lnScalarRange = (nodeAttrs.scalar_data_range as [number, number]) ?? [0, 1];
          material.updateScalarRange(lnScalarRange[0], lnScalarRange[1]);
        }
      }
    }

    // Auto-detect the sharpness fast path: when every per-vertex
    // sharpness in this geometry is 2.0 (the default), the wrapper
    // toggles `LUXAR_SHARPNESS_TWO` so the fragment shader replaces
    // its `pow(x, vSharpness)` with `x*x`. O(N) over segments, runs
    // once at upload.
    const sharpnessFastPath = isAllSharpnessTwo(processed);
    material.setSharpnessAllTwo(sharpnessFastPath);

    const mesh = createInstancedLinesMesh(processed, material);
    mesh.name = path;

    mesh.userData = {
      nodeType: 'lines',
      loader,
      attrs,
      maxWidth: attrs.max_width ?? 1.0,
      visibleSegmentCount: processed.segmentCount,
    } as LinesUserData;

    if (attrs.transform) {
      this.applyTransform(mesh, attrs.transform);
    }

    // Create picking shadow node if picking system is active
    if (this.pickingSystem) {
      const pickId = this.pickingSystem.allocatePickId();
      mesh.userData.pickId = pickId;
      const pickMaterial = materialManager.createLinePickingMaterial({ nodeId: pickId });
      // Mirror the visual material's sharpness fast path on the
      // picking material so the pick shader skips its `pow(...)` too.
      pickMaterial.setSharpnessAllTwo(sharpnessFastPath);
      materialManager.register(pickMaterial);
      // Share the same InstancedBufferGeometry — only material differs
      const pickNode = new THREE.Mesh(mesh.geometry, pickMaterial);
      pickNode.matrixWorld.copy(mesh.matrixWorld);
      this.pickingSystem.registerNode(mesh, pickNode, pickId);
    }

    return mesh;
  }

  // ============================================================================
  // GSplats Node Creation
  // ============================================================================

  /**
   * Create a THREE.Mesh (instanced gsplats) from processed splat data.
   * Works for both standard and progressive loading paths — initial creation is identical.
   */
  createGSplatsNode(
    path: string,
    nodeAttrs: Record<string, unknown>,
    attrs: GSplatsMetadata,
    meshConfig: InstancedGSplatsMeshConfig,
    loader: GSplatsDataLoader
  ): THREE.Mesh {
    let material: LuxarGSplatMaterial = materialManager.getGSplatMaterial({
      opacity: (attrs.opacity as number | undefined) ?? 1.0,
      gamma: (attrs.gamma as number | undefined) ?? 1.0,
      intensity: (attrs.intensity as number | undefined) ?? 1.0,
      offset: (attrs.offset as number | undefined) ?? 0.0,
      blendingMode: (attrs.blending_mode as string | undefined as BlendingMode) ?? 'additive',
      truncationRadius: (attrs.truncation_radius as number | undefined) ?? 3.0,
    });

    // Apply colormap if specified.
    //
    // when colormap='custom', use the bytes stashed by the scene
    // loader (`nodeAttrs.customLutBytes`). `getColormapTexture` falls
    // back to the viridis built-in if the bytes are missing/invalid,
    // so the GSplat-only "not yet implemented" warning is gone.
    const gsColormapName = nodeAttrs.colormap as string | undefined;
    let gsplatMaterialCloned = false;
    if (gsColormapName) {
      const gsLutBytes = nodeAttrs.customLutBytes as Uint8Array | undefined;
      const gsColormapTex = getColormapTexture(gsColormapName, gsLutBytes);
      if (gsColormapTex) {
        // B.1: detach pooled material from global updates before cloning.
        // See lines/points clone sites for the rationale.
        materialManager.detachFromGlobalUpdates(material);
        // Both clones (GSplatMaterial.clone() → GSplatMaterial,
        // GSplatTSLMaterial.clone() → GSplatTSLMaterial) satisfy the
        // LuxarGSplatMaterial union — `as typeof material` keeps the
        // backend-agnostic type and avoids narrowing to the WebGL2
        // class.
        material = material.clone() as typeof material;
        materialManager.register(material);
        gsplatMaterialCloned = true;
        material.updateColormapTexture(gsColormapTex);
        const ampRange = nodeAttrs.amplitude_data_range as [number, number] | undefined;
        const gsScalarRange = ampRange ?? [0, 1];
        material.updateScalarRange(gsScalarRange[0], gsScalarRange[1]);
      }
    }

    const mesh = createInstancedGSplatsMesh(meshConfig, material);
    mesh.name = path;

    mesh.userData = {
      nodeType: 'gsplats',
      loader,
      attrs,
      visibleSplatCount: meshConfig.splatCount,
      _layerMaterialCloned: gsplatMaterialCloned,
    } as GSplatsUserData;

    if (attrs.transform) {
      this.applyTransform(mesh, attrs.transform);
    }

    // Create picking shadow node if picking system is active
    if (this.pickingSystem) {
      const pickId = this.pickingSystem.allocatePickId();
      mesh.userData.pickId = pickId;
      const pickMaterial = materialManager.createGSplatPickingMaterial({ nodeId: pickId });
      materialManager.register(pickMaterial);
      // Share the same InstancedBufferGeometry — only material differs
      const pickNode = new THREE.Mesh(mesh.geometry, pickMaterial);
      pickNode.matrixWorld.copy(mesh.matrixWorld);
      this.pickingSystem.registerNode(mesh, pickNode, pickId);
    }

    return mesh;
  }

  // ============================================================================
  // Empty placeholder factories
  // ============================================================================
  //
  // These build a fully-formed THREE node with empty geometry/instance
  // buffers, ready to be attached to the scene before any data fetch
  // happens. The scene-loader uses them so an initial-load failure leaves
  // a placeholder in place: the commit helpers can find it by name and
  // populate it once data finally arrives, and `retryFailedLoader()` can
  // read its `userData.attrs` to derive the retry view state.
  //
  // Empty data flows through the same `createXNode` factories used by
  // the success path, so userData, transforms, picking shadow nodes,
  // and material clone bookkeeping are all set up identically. The
  // commit helpers naturally take the "different size" branch (0 → N)
  // when the real data arrives.

  /**
   * Create a `THREE.Points` placeholder with an empty geometry.
   *
   * Constructs a minimal {@link LoadedPointsData} inline rather than
   * routing through `createEmptyPointsData()` (which needs a full
   * `ProjectionContext` with `chunkIndex`); the values that distinguish
   * the two paths (`ndim`, `dtypes`) are overwritten on the first
   * successful commit.
   */
  createEmptyPointsNode(path: string, attrs: PointsMetadata, loader: DataLoader): THREE.Mesh {
    const emptyData: LoadedPointsData = {
      positions: new Float32Array(0) as LoadedPointsData['positions'],
      pointCount: 0,
      ndim: 3,
      metadata: {
        totalPoints: attrs.n_points ?? 0,
        loadedPoints: 0,
        bounds: new THREE.Box3(),
        usedSpatialIndex: true,
        dtypes: {},
      },
    };
    return this.createPointsNode(path, attrs, emptyData, loader);
  }

  /**
   * Create a `THREE.Mesh` (instanced lines) placeholder with empty
   * instance buffers.
   */
  createEmptyLinesNode(
    path: string,
    nodeAttrs: Record<string, unknown>,
    attrs: LinesMetadata,
    loader: LinesDataLoader
  ): THREE.Mesh {
    const emptyConfig: InstancedLinesMeshConfig = {
      startPositions: new Float32Array(0),
      endPositions: new Float32Array(0),
      startColors: new Float32Array(0),
      endColors: new Float32Array(0),
      startWidths: new Float32Array(0),
      endWidths: new Float32Array(0),
      startSharpness: new Float32Array(0),
      endSharpness: new Float32Array(0),
      segmentLengths: new Float32Array(0),
      startClipped: new Uint8Array(0),
      endClipped: new Uint8Array(0),
      segmentCount: 0,
    };
    return this.createLinesNode(path, nodeAttrs, attrs, emptyConfig, loader);
  }

  /**
   * Create a `THREE.Mesh` (instanced gsplats) placeholder with empty
   * instance buffers.
   */
  createEmptyGSplatsNode(
    path: string,
    nodeAttrs: Record<string, unknown>,
    attrs: GSplatsMetadata,
    loader: GSplatsDataLoader
  ): THREE.Mesh {
    const emptyConfig: InstancedGSplatsMeshConfig = {
      centers: new Float32Array(0),
      cholesky01: new Float32Array(0),
      cholesky23: new Float32Array(0),
      cholesky45: new Float32Array(0),
      amplitudes: new Float32Array(0),
      colors: new Float32Array(0),
      splatCount: 0,
    };
    return this.createGSplatsNode(path, nodeAttrs, attrs, emptyConfig, loader);
  }

  // ============================================================================
  // Validation Helpers (public for testing)
  // ============================================================================

  /**
   * Validate points data for edge cases and malformed data.
   * Logs detailed diagnostics to browser console for debugging.
   */
  validateLoadedPointsData(data: LoadedPointsData): void {
    validateLoadedPointsDataImpl(data);
  }

  validateColorMode(
    colors: Uint8Array | Uint16Array | Float32Array,
    nodeMetadata: Record<string, unknown> | null | undefined
  ): void {
    validateColorModeImpl(colors, nodeMetadata);
  }

  validateTransformFormat(transform: readonly number[]): void {
    validateTransformFormatImpl(transform);
  }

  applyTransform(object: THREE.Object3D, transform: readonly number[]): void {
    applyTransformImpl(object, transform);
  }

  // ============================================================================
  // Private Helpers (delegated to node-factory/create-points-node.ts)
  // ============================================================================

  createPointsGeometry(
    data: LoadedPointsData,
    maxRadius: number = 1.0,
    maxSharpness: number = 31.0
  ): THREE.BufferGeometry {
    return createPointsGeometryImpl(data, maxRadius, maxSharpness);
  }

  createPointsMaterial(
    attrs: Partial<PointsMetadata>,
    radiusScale: number = 1.0,
    sharpnessScale: number = 1.0,
    geometry?: THREE.BufferGeometry,
    path?: string
  ): LuxarPointMaterial {
    return createPointsMaterialImpl(attrs, radiusScale, sharpnessScale, geometry, path);
  }
}
