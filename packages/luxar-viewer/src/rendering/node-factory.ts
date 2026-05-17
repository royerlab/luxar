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
import { supportsScalarColormap } from './material-colormap-helpers';
import { createInstancedLinesMesh, type InstancedLinesMeshConfig } from './line-geometry';
import { createInstancedGSplatsMesh, type InstancedGSplatsMeshConfig } from './gsplat-geometry';
import { createPointQuadGeometry } from './point-geometry';
import type { LoadedPointsData, DataLoader } from '../data/data-loader-types';
import type { PointsMetadata, PointsUserData } from '../types/points';
import type { LinesMetadata, LinesUserData, LinesDataLoader } from '../types/lines';
import type { GSplatsMetadata, GSplatsUserData, GSplatsDataLoader } from '../types/gsplats';
import { log, Modules } from '../utils/log';
import type { PickingSystem } from './picking/picking-system';
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

  /** Invalidate the cached pick buffer (call after geometry updates). */
  markPickingDirty(): void {
    this.pickingSystem?.markDirty();
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
    const pointCount = data.positions.length / 3;

    log.info(Modules.SCENE_LOADER, 'Points Data Validation:', {
      pointCount,
      positionsLength: data.positions.length,
      positionsType: data.positions.constructor.name,
      hasColors: !!data.colors,
      colorsType: data.colors?.constructor.name,
      colorsLength: data.colors?.length,
      hasRadii: !!data.radii,
      radiiType: data.radii?.constructor.name,
      radiiLength: data.radii?.length,
      hasSharpness: !!data.sharpness,
      sharpnessType: data.sharpness?.constructor.name,
      sharpnessLength: data.sharpness?.length,
    });

    if (pointCount === 0) {
      log.warning(Modules.SCENE_LOADER, 'Empty dataset detected - no points to render');
      return;
    }

    if (data.positions.length % 3 !== 0) {
      const error = `Malformed positions array: length ${data.positions.length} is not divisible by 3`;
      log.error(Modules.SCENE_LOADER, error);
      throw new Error(error);
    }

    if (data.colors && data.colors.length !== data.positions.length) {
      const expected = data.positions.length;
      const actual = data.colors.length;
      log.warning(
        Modules.SCENE_LOADER,
        `Colors length mismatch: expected ${expected}, got ${actual}`,
        { expected, actual }
      );
    }

    if (data.radii && data.radii.length !== pointCount) {
      const expected = pointCount;
      const actual = data.radii.length;
      log.warning(
        Modules.SCENE_LOADER,
        `Radii length mismatch: expected ${expected}, got ${actual}`,
        { expected, actual }
      );
    }

    if (data.sharpness && data.sharpness.length !== pointCount) {
      const expected = pointCount;
      const actual = data.sharpness.length;
      log.warning(
        Modules.SCENE_LOADER,
        `Sharpness length mismatch: expected ${expected}, got ${actual}`,
        { expected, actual }
      );
    }

    log.success(Modules.SCENE_LOADER, `Points data validated: ${pointCount} points`);
  }

  /**
   * Validate color mode consistency.
   * Ensures color array type matches expected encoding.
   *
   * `nodeMetadata` is typed loosely as `Record<string, unknown>` because
   * it can come from either the typed `LoadedPointsData.metadata`
   * (no `color_mode` today) or from a zarr attrs dict in tests. We
   * only read `color_mode` from it.
   */
  validateColorMode(
    colors: Uint8Array | Uint16Array | Float32Array,
    nodeMetadata: Record<string, unknown> | null | undefined
  ): void {
    const isHDR = colors instanceof Float32Array;
    const isSDR = colors instanceof Uint8Array || colors instanceof Uint16Array;

    if (isSDR && nodeMetadata?.color_mode === 'hdr') {
      log.warning(
        Modules.SCENE_LOADER,
        `Node metadata indicates HDR colors but array is ${colors.constructor.name}. ` +
          'HDR colors should use Float32Array. This may indicate incorrect encoding.'
      );
    }

    if (isHDR) {
      const hasHDRValues = Array.from(colors).some((v) => v > 1.0);
      if (!hasHDRValues && nodeMetadata?.color_mode === 'hdr') {
        log.info(
          Modules.SCENE_LOADER,
          'HDR color mode specified but all values in [0, 1] range. Consider using SDR mode for better compression.'
        );
      }
    }

    const colorType = colors.constructor.name;
    const colorMode = isHDR ? 'HDR (float32)' : 'SDR (normalized integer)';
    log.info(Modules.SCENE_LOADER, `Colors: ${colorType} - ${colorMode}`);
  }

  /**
   * Validate transform matrix format. Throws when the matrix appears to be
   * stored row-major (NumPy) rather than column-major (THREE.js / OpenGL),
   * which is almost always a producer bug — column-major translation lives
   * at indices [12,13,14], row-major at [3,7,11].
   */
  validateTransformFormat(transform: readonly number[]): void {
    const colMajorTranslation = [transform[12], transform[13], transform[14]];
    const rowMajorTranslation = [transform[3], transform[7], transform[11]];

    const colMajorNonZero = colMajorTranslation.some((v) => Math.abs(v) > 0.001);
    const rowMajorNonZero = rowMajorTranslation.some((v) => Math.abs(v) > 0.001);

    if (rowMajorNonZero && !colMajorNonZero) {
      throw new Error(
        'Transform matrix appears to be stored in row-major (NumPy) format ' +
          'instead of column-major (THREE.js). Translation detected at ' +
          'indices [3,7,11] instead of [12,13,14]. Python should transpose ' +
          'before storing: matrix.T.ravel().tolist()'
      );
    }
  }

  /**
   * Apply transformation matrix to a THREE.js object. Throws when the
   * transform is malformed or stored row-major.
   */
  applyTransform(object: THREE.Object3D, transform: readonly number[]): void {
    if (transform.length !== 16) {
      throw new Error(`Invalid transform length: ${transform.length} (expected 16)`);
    }

    this.validateTransformFormat(transform);

    // THREE.Matrix4.fromArray takes ArrayLike<number>; the readonly tuple is fine.
    const matrix = new THREE.Matrix4().fromArray(transform as number[]);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();

    matrix.decompose(position, quaternion, scale);

    object.position.copy(position);
    object.quaternion.copy(quaternion);
    object.scale.copy(scale);
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  /**
   * Create THREE.js geometry from points data.
   * Public because SceneLoader's update path also needs to create geometry.
   */
  createPointsGeometry(
    data: LoadedPointsData,
    maxRadius: number = 1.0,
    maxSharpness: number = 31.0
  ): THREE.BufferGeometry {
    // Start from the shared unit-quad base. Each Points node gets
    // its own BufferGeometry instance with cloned base + per-point
    // InstancedBufferAttributes.
    const geometry = createPointQuadGeometry();

    this.validateLoadedPointsData(data);

    const pointCount = data.positions.length / 3;

    // Per-instance centre positions. Float16 datasets are widened to
    // Float32 because InstancedBufferAttribute doesn't accept Float16
    // typed arrays directly.
    let centersTyped: Float32Array;
    if (
      typeof globalThis.Float16Array !== 'undefined' &&
      data.positions instanceof globalThis.Float16Array
    ) {
      centersTyped = new Float32Array(data.positions);
    } else {
      centersTyped = data.positions as Float32Array;
    }
    geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(centersTyped, 3));

    // Per-instance colours. Always present in the new path — if the
    // loader didn't supply colours, fill with white. The material
    // toggles between aColor and aScalar via the USE_COLORMAP define;
    // both attributes can coexist.
    if (data.colors) {
      this.validateColorMode(data.colors, data.metadata);
      const needsNormalization =
        data.colors instanceof Uint8Array || data.colors instanceof Uint16Array;
      geometry.setAttribute(
        'aColor',
        new THREE.InstancedBufferAttribute(data.colors, 3, needsNormalization)
      );
    } else {
      const defaultColors = new Float32Array(pointCount * 3).fill(1.0);
      geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(defaultColors, 3));
    }

    // Per-instance radii. Same dtype-handling rules as before; the
    // resulting `radiusScale` is consumed by the material uniform.
    let radiusScale = 1.0;
    if (data.radii) {
      if (
        typeof globalThis.Float16Array !== 'undefined' &&
        data.radii instanceof globalThis.Float16Array
      ) {
        geometry.setAttribute(
          'aRadius',
          new THREE.InstancedBufferAttribute(new Float32Array(data.radii), 1)
        );
        radiusScale = 1.0;
      } else if (data.radii instanceof Uint8Array) {
        geometry.setAttribute('aRadius', new THREE.InstancedBufferAttribute(data.radii, 1, true));
        radiusScale = maxRadius;
      } else {
        geometry.setAttribute(
          'aRadius',
          new THREE.InstancedBufferAttribute(data.radii as Float32Array, 1, false)
        );
        radiusScale = 1.0;
      }
    } else {
      const defaultRadii = new Float32Array(pointCount).fill(0.5);
      geometry.setAttribute('aRadius', new THREE.InstancedBufferAttribute(defaultRadii, 1));
    }

    // Per-instance sharpness.
    let sharpnessScale = 1.0;
    if (data.sharpness) {
      if (
        typeof globalThis.Float16Array !== 'undefined' &&
        data.sharpness instanceof globalThis.Float16Array
      ) {
        geometry.setAttribute(
          'aSharpness',
          new THREE.InstancedBufferAttribute(new Float32Array(data.sharpness), 1)
        );
        sharpnessScale = 1.0;
      } else if (data.sharpness instanceof Uint8Array) {
        geometry.setAttribute(
          'aSharpness',
          new THREE.InstancedBufferAttribute(data.sharpness, 1, true)
        );
        sharpnessScale = maxSharpness;
      } else {
        geometry.setAttribute(
          'aSharpness',
          new THREE.InstancedBufferAttribute(data.sharpness as Float32Array, 1, false)
        );
        sharpnessScale = 1.0;
      }
    } else {
      const defaultSharpness = new Float32Array(pointCount).fill(2.0);
      geometry.setAttribute('aSharpness', new THREE.InstancedBufferAttribute(defaultSharpness, 1));
    }

    // Per-instance scalar (USE_COLORMAP only). Attached as `aScalar`
    // so the shader can read it via `in float aScalar` under the
    // USE_COLORMAP define.
    if (data.scalars) {
      const scalarsTyped = data.scalars;
      if (
        typeof globalThis.Float16Array !== 'undefined' &&
        scalarsTyped instanceof globalThis.Float16Array
      ) {
        geometry.setAttribute(
          'aScalar',
          new THREE.InstancedBufferAttribute(new Float32Array(scalarsTyped), 1)
        );
      } else if (scalarsTyped instanceof Uint8Array) {
        geometry.setAttribute('aScalar', new THREE.InstancedBufferAttribute(scalarsTyped, 1, true));
      } else {
        geometry.setAttribute(
          'aScalar',
          new THREE.InstancedBufferAttribute(scalarsTyped as Float32Array, 1, false)
        );
      }
    }

    // WebGLRenderer only issues an instanced draw for InstancedBufferGeometry
    // when instanceCount is set. The base quad has 6 indices; instanceCount
    // is the number of point sprites to draw.
    geometry.instanceCount = pointCount;
    geometry.setDrawRange(0, 6);

    // Bounding box/sphere of the per-instance positions (used by spatial
    // queries / debug/camera paths). The base-quad bounds are irrelevant;
    // frustum culling is disabled on the mesh because the sprites expand
    // in screen space.
    geometry.boundingBox = data.metadata.bounds.clone();
    geometry.boundingSphere = new THREE.Sphere();
    geometry.boundingBox.getBoundingSphere(geometry.boundingSphere);

    // Store radius and sharpness scales as user data for material creation
    if (!geometry.userData) {
      geometry.userData = {};
    }
    geometry.userData.radiusScale = radiusScale;
    geometry.userData.sharpnessScale = sharpnessScale;
    geometry.userData.pointCount = pointCount;

    return geometry;
  }

  /**
   * Create material for points rendering.
   * Public because SceneLoader tests and update paths access it.
   *
   * Accepts a `Partial<PointsMetadata>` because callers (and tests)
   * frequently pass narrowed attribute subsets.
   */
  createPointsMaterial(
    attrs: Partial<PointsMetadata>,
    radiusScale: number = 1.0,
    sharpnessScale: number = 1.0,
    geometry?: THREE.BufferGeometry,
    path?: string
  ): LuxarPointMaterial {
    let material = materialManager.getPointMaterial({
      opacity: attrs.opacity ?? 1.0,
      gamma: attrs.gamma ?? 1.0,
      intensity: attrs.intensity ?? 1.0,
      offset: attrs.offset ?? 0.0,
      blendingMode: (attrs.blending_mode as BlendingMode) ?? 'additive',
      radiusScale: radiusScale,
      sharpnessScale: sharpnessScale,
    });

    const ptColormapName = attrs.colormap;
    const ptHasScalars = !!attrs.has_scalars;
    if (ptColormapName && ptHasScalars) {
      // Point shader's USE_COLORMAP path requires a `scalar` attribute.
      // When `geometry` is provided (placeholder/init path), check the
      // actual binding; when it's absent (e.g. tests calling
      // createPointsMaterial directly), skip the guard and trust the
      // caller.
      const guardOK = !geometry || supportsScalarColormap('points', geometry);
      if (!guardOK) {
        log.warning(
          Modules.SCENE_LOADER,
          `[${path ?? '<points>'}] Scalar colormap requested but 'scalar' attribute is not bound on geometry. Colormap suppressed; rendering with vertex colors.`
        );
      } else {
        // pass customLutBytes when colormap='custom'.
        const ptLutBytes = (attrs as { customLutBytes?: Uint8Array }).customLutBytes;
        const ptColormapTex = getColormapTexture(ptColormapName, ptLutBytes);
        if (ptColormapTex) {
          // B.1: detach pooled material from global updates before cloning.
          // See lines clone site for the full rationale.
          materialManager.detachFromGlobalUpdates(material);
          material = material.clone() as typeof material;
          materialManager.register(material);
          material.updateColormapTexture(ptColormapTex);
          const ptScalarRange = attrs.scalar_data_range ?? [0, 1];
          material.updateScalarRange(ptScalarRange[0], ptScalarRange[1]);
        }
      }
    }

    return material;
  }
}
