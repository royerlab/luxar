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
import { materialManager, type LuxarPointMaterial } from './material-manager';
import { type InstancedLinesMeshConfig } from './line-geometry';
import { type InstancedGSplatsMeshConfig } from './gsplat-geometry';
import { getElementTexture, markElementTextureFullDirty } from './element-storage';
import type { LoadedPointsData, DataLoader } from '../data/data-loader-types';
import type { PointsMetadata } from '../types/points';
import type { LinesMetadata, LinesDataLoader } from '../types/lines';
import type { GSplatsMetadata, GSplatsDataLoader } from '../types/gsplats';
import { isPooledGeometry } from '../types/geometry-capabilities';
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
  createPointsNode as createPointsNodeImpl,
  createEmptyPointsNode as createEmptyPointsNodeImpl,
} from './node-factory/create-points-node';
import {
  createLinesNode as createLinesNodeImpl,
  createEmptyLinesNode as createEmptyLinesNodeImpl,
} from './node-factory/create-lines-node';
import {
  createGSplatsNode as createGSplatsNodeImpl,
  createEmptyGSplatsNode as createEmptyGSplatsNodeImpl,
} from './node-factory/create-gsplats-node';
import { createEmptyMeshNode as createEmptyMeshNodeImpl } from './node-factory/create-mesh-node';
import type { MeshDataLoader, MeshMetadata } from '../types/mesh';
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
        const pickMaterial = materialManager.createPointPickingMaterial({
          nodeId: pickId,
          radiusScale,
        });
        materialManager.register(pickMaterial);
        // Shares the visual geometry (instance-spanning, footprint-expanded
        // bounds), so the pick node culls safely — same as lines/gsplats.
        const pickNode = new THREE.Mesh(obj.geometry, pickMaterial);
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
   *
   * Also re-uploads geometry GPU buffers for every POOLED geometry type. A
   * context loss zeroes the GPU-side storage — the element textures +
   * `aSortedIndex` (gsplats, points, and lines alike since the lines
   * texture-storage migration) — while the CPU mirror survives, so we
   * mark everything full-dirty (empty ranges → three's full upload) and
   * clear the append-fast-path flag `gpuPrefixIntact` (depth-sorting
   * Phase 4 Stage 2). The flag is load-bearing: without it the next
   * commit could take the append path and DOWNGRADE the pending full
   * upload to a suffix-only partial, leaving the prefix stale. This runs
   * unconditionally (picking may be disabled).
   */
  rebuildAfterContextRestore(root: THREE.Object3D): void {
    root.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      if (isPooledGeometry(obj.userData?.nodeType)) {
        // Texture-backed storage (pool AND non-pool geometries alike):
        // mark the element texture + aSortedIndex full-dirty. Restricted to
        // pooled types — a geometry type rendered from a plain
        // `BufferGeometry` has no element texture to re-upload (see
        // `types/geometry-capabilities`).
        const geom = obj.geometry as THREE.InstancedBufferGeometry;
        const tex = getElementTexture(geom);
        if (tex) {
          // Registers the pending-full state too, so a pre-flush ranged
          // write can't downgrade the restore's full re-upload.
          markElementTextureFullDirty(tex);
        }
        // BOTH ordering buffers: a context loss zeroes the GPU side while
        // the CPU mirrors survive, and either one may be the active slot
        // (or become it when an in-flight stream completes). Marking only
        // the front buffer would leave a freshly-flipped back buffer
        // reading as zeros. attachElementStorage allocates them as two
        // DISTINCT buffers, so both need the full re-upload. The active
        // SLOT is deliberately left alone: the buffer it points at still
        // holds a whole permutation, so resetting it would swap in the
        // other, staler one.
        for (const name of ['aSortedIndex', 'aSortedIndexB']) {
          const idx = geom.getAttribute(name) as THREE.InstancedBufferAttribute | undefined;
          if (idx) {
            idx.clearUpdateRanges();
            idx.needsUpdate = true;
          }
        }
        obj.userData.gpuPrefixIntact = false;
      }
    });

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

  createPointsNode(
    path: string,
    attrs: PointsMetadata,
    data: LoadedPointsData,
    loader: DataLoader,
    isPlaceholder: boolean = false,
    leafAttrs?: Partial<PointsMetadata>
  ): THREE.Mesh {
    return createPointsNodeImpl(
      path,
      attrs,
      data,
      loader,
      this.pickingSystem,
      isPlaceholder,
      leafAttrs
    );
  }

  createLinesNode(
    path: string,
    nodeAttrs: Record<string, unknown>,
    attrs: LinesMetadata,
    processed: InstancedLinesMeshConfig,
    loader: LinesDataLoader
  ): THREE.Mesh {
    return createLinesNodeImpl(path, nodeAttrs, attrs, processed, loader, this.pickingSystem);
  }

  // ============================================================================
  // GSplats Node Creation
  // ============================================================================

  createGSplatsNode(
    path: string,
    nodeAttrs: Record<string, unknown>,
    attrs: GSplatsMetadata,
    meshConfig: InstancedGSplatsMeshConfig,
    loader: GSplatsDataLoader
  ): THREE.Mesh {
    return createGSplatsNodeImpl(path, nodeAttrs, attrs, meshConfig, loader, this.pickingSystem);
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
   * Create a `THREE.Mesh` (instanced points) placeholder with an empty
   * geometry.
   */
  createEmptyPointsNode(
    path: string,
    attrs: PointsMetadata,
    loader: DataLoader,
    leafAttrs?: Partial<PointsMetadata>
  ): THREE.Mesh {
    return createEmptyPointsNodeImpl(path, attrs, loader, this.pickingSystem, leafAttrs);
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
    return createEmptyLinesNodeImpl(path, nodeAttrs, attrs, loader, this.pickingSystem);
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
    return createEmptyGSplatsNodeImpl(path, nodeAttrs, attrs, loader, this.pickingSystem);
  }

  /**
   * Create an empty placeholder for a mesh node — a plain `THREE.Mesh` with an
   * indexed `BufferGeometry`, not an instanced quad.
   *
   * Takes no `nodeAttrs` and no `pickingSystem`, unlike its three siblings. No
   * `nodeAttrs` because that second bag exists to tell a leaf-authored colormap
   * window from an inherited ancestor gain, and mesh has no colormap path in this
   * phase. No `pickingSystem` because mesh picking uses `gl_VertexID` rather than an
   * element-texture texel and needs its own pick material pair, which arrives with
   * the shading phase — threading the system in now would look like picking works.
   */
  createEmptyMeshNode(path: string, attrs: MeshMetadata, loader: MeshDataLoader): THREE.Mesh {
    return createEmptyMeshNodeImpl(path, attrs, loader);
  }

  // ============================================================================
  // Validation Helpers (public for testing)
  // ============================================================================

  /**
   * Validate points data for edge cases and malformed data.
   * Logs detailed diagnostics to browser console for debugging.
   */
  validateLoadedPointsData(data: LoadedPointsData, isPlaceholder = false): void {
    validateLoadedPointsDataImpl(data, isPlaceholder);
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
    isPlaceholder: boolean = false
  ): THREE.BufferGeometry {
    return createPointsGeometryImpl(data, maxRadius, isPlaceholder);
  }

  createPointsMaterial(
    attrs: Partial<PointsMetadata>,
    radiusScale: number = 1.0,
    geometry?: THREE.BufferGeometry,
    path?: string,
    leafAttrs?: Partial<PointsMetadata>
  ): LuxarPointMaterial {
    return createPointsMaterialImpl(attrs, radiusScale, geometry, path, leafAttrs);
  }
}
