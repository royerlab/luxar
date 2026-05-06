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
import { materialManager, type BlendingMode } from '../rendering/material-manager';
import { getColormapTexture } from '../rendering/colormap-textures';
import {
  createInstancedLinesMesh,
  type InstancedLinesMeshConfig,
} from '../rendering/line-geometry';
import {
  createInstancedGSplatsMesh,
  type InstancedGSplatsMeshConfig,
} from '../rendering/gsplat-geometry';
import { GSplatMaterial } from '../rendering/gsplat-material';
import type { LoadedPointsData, DataLoader } from './data-loader-types';
import type { PointsMetadata, PointsUserData } from '../types/points';
import type { LinesMetadata, LinesUserData, LinesDataLoader } from '../types/lines';
import type { GSplatsMetadata, GSplatsUserData, GSplatsDataLoader } from '../types/gsplats';
import { log, Modules } from '../utils/log';
import type { PickingSystem } from '../rendering/picking/picking-system';
import { PointPickingMaterial } from '../rendering/picking/point-picking-material';
import { LinePickingMaterial } from '../rendering/picking/line-picking-material';
import { GSplatPickingMaterial } from '../rendering/picking/gsplat-picking-material';

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

      if (nodeType === 'points' && obj instanceof THREE.Points) {
        const pickId = this.pickingSystem!.allocatePickId();
        obj.userData.pickId = pickId;
        const radiusScale = obj.geometry?.userData?.radiusScale ?? 1.0;
        const sharpnessScale = obj.geometry?.userData?.sharpnessScale ?? 1.0;
        const pickMaterial = new PointPickingMaterial({
          nodeId: pickId,
          radiusScale,
          sharpnessScale,
        });
        materialManager.register(pickMaterial);
        const pickNode = new THREE.Points(obj.geometry, pickMaterial);
        pickNode.matrixWorld.copy(obj.matrixWorld);
        this.pickingSystem!.registerNode(obj, pickNode, pickId);
      } else if (nodeType === 'lines' && obj instanceof THREE.Mesh) {
        const pickId = this.pickingSystem!.allocatePickId();
        obj.userData.pickId = pickId;
        const pickMaterial = new LinePickingMaterial({ nodeId: pickId });
        materialManager.register(pickMaterial);
        const pickNode = new THREE.Mesh(obj.geometry, pickMaterial);
        pickNode.matrixWorld.copy(obj.matrixWorld);
        this.pickingSystem!.registerNode(obj, pickNode, pickId);
      } else if (nodeType === 'gsplats' && obj instanceof THREE.Mesh) {
        const pickId = this.pickingSystem!.allocatePickId();
        obj.userData.pickId = pickId;
        const pickMaterial = new GSplatPickingMaterial({ nodeId: pickId });
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
   * Create a THREE.Points object from loaded point data.
   * Handles geometry creation, material selection, userData, and transforms.
   */
  createPointsNode(
    path: string,
    attrs: PointsMetadata,
    data: LoadedPointsData,
    loader: DataLoader
  ): THREE.Points {
    const maxRadius = (attrs.max_radius as number | undefined) ?? 1.0;
    const maxSharpness = ((attrs as any).max_sharpness as number | undefined) ?? 31.0;
    const geometry = this.createPointsGeometry(data, maxRadius, maxSharpness);

    const radiusScale = geometry.userData.radiusScale ?? 1.0;
    const sharpnessScale = geometry.userData.sharpnessScale ?? 1.0;
    const material = this.createPointsMaterial(attrs, radiusScale, sharpnessScale);

    const points = new THREE.Points(geometry, material);
    points.name = path;

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
      const pickMaterial = new PointPickingMaterial({
        nodeId: pickId,
        radiusScale,
        sharpnessScale,
      });
      materialManager.register(pickMaterial);
      const pickNode = new THREE.Points(geometry, pickMaterial);
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
    let material = materialManager.getLineMaterial({
      opacity: (attrs.opacity as number | undefined) ?? 1.0,
      gamma: (attrs.gamma as number | undefined) ?? 1.0,
      intensity: (attrs.intensity as number | undefined) ?? 1.0,
      offset: (attrs.offset as number | undefined) ?? 0.0,
      blendingMode: (attrs.blending_mode as string | undefined as BlendingMode) ?? 'additive',
    });

    // Apply colormap if specified AND scalar data exists
    const lnColormapName = nodeAttrs.colormap as string | undefined;
    const lnHasScalars = !!nodeAttrs.has_scalars;
    if (lnColormapName && lnHasScalars) {
      const lnColormapTex = getColormapTexture(lnColormapName);
      if (lnColormapTex) {
        material = material.clone() as typeof material;
        materialManager.register(material);
        material.updateColormapTexture(lnColormapTex);
        const lnScalarRange = (nodeAttrs.scalar_data_range as [number, number]) ?? [0, 1];
        material.updateScalarRange(lnScalarRange[0], lnScalarRange[1]);
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
      const pickMaterial = new LinePickingMaterial({ nodeId: pickId });
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
    let material: GSplatMaterial = materialManager.getGSplatMaterial({
      opacity: (attrs.opacity as number | undefined) ?? 1.0,
      gamma: (attrs.gamma as number | undefined) ?? 1.0,
      intensity: (attrs.intensity as number | undefined) ?? 1.0,
      offset: (attrs.offset as number | undefined) ?? 0.0,
      blendingMode: (attrs.blending_mode as string | undefined as BlendingMode) ?? 'additive',
      truncationRadius: (attrs.truncation_radius as number | undefined) ?? 3.0,
    });

    // Apply colormap if specified
    const gsColormapName = nodeAttrs.colormap as string | undefined;
    let gsplatMaterialCloned = false;
    if (gsColormapName) {
      const gsColormapTex = getColormapTexture(gsColormapName);
      if (gsColormapTex) {
        material = material.clone() as GSplatMaterial;
        materialManager.register(material);
        gsplatMaterialCloned = true;
        material.updateColormapTexture(gsColormapTex);
        const ampRange = nodeAttrs.amplitude_data_range as [number, number] | undefined;
        const gsScalarRange = ampRange ?? [0, 1];
        material.updateScalarRange(gsScalarRange[0], gsScalarRange[1]);
      } else if (gsColormapName === 'custom') {
        log.warning(
          Modules.SCENE_LOADER,
          `Custom colormap LUT loading not yet implemented for ${path}`
        );
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
      const pickMaterial = new GSplatPickingMaterial({ nodeId: pickId });
      materialManager.register(pickMaterial);
      // Share the same InstancedBufferGeometry — only material differs
      const pickNode = new THREE.Mesh(mesh.geometry, pickMaterial);
      pickNode.matrixWorld.copy(mesh.matrixWorld);
      this.pickingSystem.registerNode(mesh, pickNode, pickId);
    }

    return mesh;
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
   */
  validateColorMode(colors: Uint8Array | Uint16Array | Float32Array, nodeMetadata: any): void {
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
      throw new Error(
        `Invalid transform length: ${transform.length} (expected 16)`
      );
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
    const geometry = new THREE.BufferGeometry();

    this.validateLoadedPointsData(data);

    // Set positions (handle Float16Array conversion if needed)
    if (
      typeof (globalThis as any).Float16Array !== 'undefined' &&
      data.positions instanceof (globalThis as any).Float16Array
    ) {
      const float32Positions = new Float32Array(data.positions);
      geometry.setAttribute('position', new THREE.BufferAttribute(float32Positions, 3));
    } else {
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(data.positions as Float32Array, 3)
      );
    }

    // Set colors if available
    if (data.colors) {
      this.validateColorMode(data.colors, data.metadata as any);
      const needsNormalization =
        data.colors instanceof Uint8Array || data.colors instanceof Uint16Array;
      geometry.setAttribute('color', new THREE.BufferAttribute(data.colors, 3, needsNormalization));
    }

    // Set radii if available, or use default
    let radiusScale = 1.0;

    if (data.radii) {
      if (
        typeof (globalThis as any).Float16Array !== 'undefined' &&
        data.radii instanceof (globalThis as any).Float16Array
      ) {
        const float32Radii = new Float32Array(data.radii);
        geometry.setAttribute('radius', new THREE.BufferAttribute(float32Radii, 1));
        radiusScale = 1.0;
      } else if (data.radii instanceof Uint8Array) {
        geometry.setAttribute('radius', new THREE.BufferAttribute(data.radii, 1, true));
        radiusScale = maxRadius;
      } else {
        geometry.setAttribute(
          'radius',
          new THREE.BufferAttribute(data.radii as Float32Array, 1, false)
        );
        radiusScale = 1.0;
      }
    } else {
      const numPoints = data.positions.length / 3;
      const defaultRadii = new Float32Array(numPoints).fill(0.5);
      geometry.setAttribute('radius', new THREE.BufferAttribute(defaultRadii, 1));
      radiusScale = 1.0;
    }

    // Set sharpness if available, or use default
    let sharpnessScale = 1.0;

    if (data.sharpness) {
      if (
        typeof (globalThis as any).Float16Array !== 'undefined' &&
        data.sharpness instanceof (globalThis as any).Float16Array
      ) {
        const float32Sharpness = new Float32Array(data.sharpness);
        geometry.setAttribute('sharpness', new THREE.BufferAttribute(float32Sharpness, 1));
        sharpnessScale = 1.0;
      } else if (data.sharpness instanceof Uint8Array) {
        geometry.setAttribute('sharpness', new THREE.BufferAttribute(data.sharpness, 1, true));
        sharpnessScale = maxSharpness;
      } else {
        geometry.setAttribute(
          'sharpness',
          new THREE.BufferAttribute(data.sharpness as Float32Array, 1, false)
        );
        sharpnessScale = 1.0;
      }
    } else {
      const numPoints = data.positions.length / 3;
      const defaultSharpness = new Float32Array(numPoints).fill(2.0);
      geometry.setAttribute('sharpness', new THREE.BufferAttribute(defaultSharpness, 1));
      sharpnessScale = 1.0;
    }

    // Compute bounding box
    geometry.boundingBox = data.metadata.bounds.clone();

    // Store radius and sharpness scales as user data for material creation
    if (!geometry.userData) {
      geometry.userData = {};
    }
    geometry.userData.radiusScale = radiusScale;
    geometry.userData.sharpnessScale = sharpnessScale;

    return geometry;
  }

  /**
   * Create material for points rendering.
   * Public because SceneLoader tests and update paths access it.
   */
  createPointsMaterial(
    attrs: any,
    radiusScale: number = 1.0,
    sharpnessScale: number = 1.0
  ): THREE.ShaderMaterial {
    let material = materialManager.getPointMaterial({
      opacity: attrs.opacity ?? 1.0,
      gamma: attrs.gamma ?? 1.0,
      intensity: attrs.intensity ?? 1.0,
      offset: attrs.offset ?? 0.0,
      blendingMode: (attrs.blending_mode as BlendingMode) ?? 'additive',
      radiusScale: radiusScale,
      sharpnessScale: sharpnessScale,
    });

    const ptColormapName = attrs.colormap as string | undefined;
    const ptHasScalars = !!attrs.has_scalars;
    if (ptColormapName && ptHasScalars) {
      const ptColormapTex = getColormapTexture(ptColormapName);
      if (ptColormapTex) {
        material = material.clone() as typeof material;
        materialManager.register(material);
        material.updateColormapTexture(ptColormapTex);
        const ptScalarRange = (attrs.scalar_data_range as [number, number]) ?? [0, 1];
        material.updateScalarRange(ptScalarRange[0], ptScalarRange[1]);
      }
    }

    return material;
  }
}
