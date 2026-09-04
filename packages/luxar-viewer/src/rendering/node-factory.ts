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
  type LuxarMeshPickingMaterial,
  type LuxarPointMaterial,
} from './material-manager';
import { type InstancedLinesMeshConfig } from './line-geometry';
import { type InstancedGSplatsMeshConfig } from './gsplat-geometry';
import { getElementTexture, markElementTextureFullDirty } from './element-storage';
import {
  syncGSplatMaterialWithGeometry,
  syncLineMaterialWithGeometry,
  syncPointMaterialWithGeometry,
} from './material-sync-helpers';
import type { LoadedPointsData, DataLoader } from '../data/data-loader-types';
import type { PointsMetadata } from '../types/points';
import type { LinesMetadata, LinesDataLoader } from '../types/lines';
import type { GSplatsMetadata, GSplatsDataLoader } from '../types/gsplats';
import { isGeometryType, isPooledGeometry } from '../types/geometry-capabilities';
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
import {
  createEmptyMeshNode as createEmptyMeshNodeImpl,
  resolveRequestedMeshMode,
} from './node-factory/create-mesh-node';
import type { MeshDataLoader, MeshMetadata } from '../types/mesh';
import { isMeshPickAwareMaterial } from './picking/mesh/pick-mode';
import type { GeometryTypeName } from '../types/format-contract';
import { lineJoinStyleFromUniform, type LineJoinStyle } from '../types/line-join';
import type { LinePrimitive } from '../types/line-primitive';
// Picking materials are constructed via `materialManager.create*PickingMaterial`
// helpers so the GLSL vs. TSL dispatch on `caps.apiSurface` lives in one place. The
// concrete types are still imported elsewhere (e.g. material-sync-helpers).

/**
 * How to give one geometry type a pick material, for the retro-registration pass.
 *
 * `build` constructs it; the optional `afterRegister` runs once the picking system has
 * stamped `userData.pickNode`, for a type whose pick material has to copy state off its
 * visual twin.
 */
interface PickMaterialRecipe {
  build(obj: THREE.Mesh, pickId: number): THREE.Material;
  /** Optional post-registration step. Runs AFTER `registerNode`. */
  afterRegister?(obj: THREE.Mesh): void;
}

/**
 * Per-geometry-type pick-material recipe, for the retro-registration pass.
 *
 * A `Record<GeometryTypeName, …>` rather than an `else if` chain because this pass is
 * the one production actually runs (see {@link NodeFactory.registerExistingSceneNodes}),
 * so a type missing from it is unpickable everywhere — a silent omission with no
 * compile error and, until this was table-driven, no test that would have failed.
 * As a table, a new geometry type is a `TS2739` here.
 *
 * `afterRegister` is part of the table for the same reason `build` is. It began life as
 * an `if (nodeType === 'mesh')` line inside the loop — which is precisely the branch the
 * table exists to abolish, so the loop's compile-time guarantee stopped one step short
 * of the claim made for it. As a table field, "does this type need a post-registration
 * step?" is answered where the type is declared rather than in the traversal.
 *
 * Each recipe reads whatever its type's pick material needs off the visual node: points
 * need the geometry's `radiusScale` (the 80%-radius pick footprint derives from it),
 * lines need the join style AND the primitive (the pick pass rasterizes the same
 * stencil the visual material draws, and under the `auto` policy that is a per-node
 * choice — see {@link linePrimitiveFromVisual}), mesh
i * needs the node opacity and cutout threshold (they are its coverage term), while
 * gsplats restore the live class filter after a context rebuild.
 *
 * The three pooled types ALSO run their commit-time material sync here
 * (`sync*MaterialWithGeometry`): a pick material is born on the shared placeholder
 * element texture and only a COMMIT rebinds the geometry-owned one through
 * `userData.pickNode` — but this pass runs after `loadScene` has already committed every
 * node, so without the sync a first-load pick pass samples the placeholder and nothing
 * is ever hit. That stayed invisible for as long as the post-load slice update
 * re-committed every node a second time; once that redundant pass was skipped
 * (view state unchanged), first-load picking silently died on the retro-registered
 * points/lines nodes. The sync is idempotent (no-op on an unchanged identity).
 */
const PICK_MATERIAL_RECIPES: Record<GeometryTypeName, PickMaterialRecipe> = {
  points: {
    build: (obj, pickId) =>
      materialManager.createPointPickingMaterial({
        nodeId: pickId,
        radiusScale: (obj.geometry?.userData?.radiusScale as number | undefined) ?? 1.0,
      }),
    afterRegister: syncPointMaterialWithGeometry,
  },
  lines: {
    build: (obj, pickId) =>
      materialManager.createLinePickingMaterial({
        nodeId: pickId,
        join: lineJoinStyleFromVisual(obj),
        primitive: linePrimitiveFromVisual(obj),
      }),
    afterRegister: syncLineMaterialWithGeometry,
  },
  gsplats: {
    build: (_obj, pickId) => materialManager.createGSplatPickingMaterial({ nodeId: pickId }),
    afterRegister: (obj) => {
      syncGSplatMaterialWithGeometry(obj);
      syncGSplatPickMaterialToVisual(obj);
    },
  },
  mesh: {
    build: (obj, pickId) => {
      const attrs = (obj.userData?.attrs ?? {}) as MeshMetadata;
      const visual = Array.isArray(obj.material) ? obj.material[0] : obj.material;
      const baseColorTexture = (visual as THREE.ShaderMaterial | undefined)?.uniforms?.uBaseColorTex
        ?.value as THREE.Texture | undefined;
      return materialManager.createMeshPickingMaterial({
        nodeId: pickId,
        opacity: attrs.opacity ?? 1.0,
        alphaCutoff: attrs.alpha_cutoff,
        baseColorTexture,
      });
    },
    afterRegister: syncMeshPickMaterialToVisual,
  },
};

/**
 * Recover a lines node's join style (#790) from its LIVE VISUAL MATERIAL.
 *
 * The pick pass builds the same screen-space quad as the visual one, so a divergence
 * leaves the outer wedge of a mitred corner pickable while nothing renders there —
 * and, because this pass is the one production takes on a first load while
 * `createLinesNode` handles the second, the same scene would pick differently on a
 * re-load.
 *
 * Read from the material rather than from `obj.userData.attrs`, which holds the RAW
 * leaf attrs and by construction lacks a wrapper-composed `join` (it rides on the
 * `kind=partition` / `kind=lod` wrapper — see COMPOSITING_ATTRS). The material is the
 * only place the composed value survives.
 *
 * The two backends store it differently, so both are checked: TSL bakes the style into
 * the graph and keeps the UNRESOLVED style on `userData.lineJoin`, GLSL keeps the
 * resolved code in the `uLineJoin` uniform. The unresolved form is preferred where it
 * exists; either way re-resolving is idempotent, since `resolveLineJoin` applies the
 * same `?lineJoin=` session override that produced the value being read.
 *
 * `undefined` (no lines material, or one with neither marker) means "unauthored" and
 * resolves through the normal precedence — NOT `'none'`.
 */
function lineJoinStyleFromVisual(obj: THREE.Mesh): LineJoinStyle | undefined {
  const visual = obj.material as THREE.Material | THREE.Material[] | undefined;
  const single = Array.isArray(visual) ? visual[0] : visual;
  if (!single) return undefined;
  const stored = single.userData?.lineJoin as LineJoinStyle | undefined;
  if (stored !== undefined) return stored;
  const uniforms = (single as THREE.Material & { uniforms?: Record<string, { value?: unknown }> })
    .uniforms;
  const code = uniforms?.uLineJoin?.value;
  return lineJoinStyleFromUniform(typeof code === 'number' ? code : undefined);
}

/**
 * Recover a lines node's PRIMITIVE (#1352) from its live visual material —
 * the `join` mirror above, for the same reason: this retro pass is the one
 * production takes on a first load, and under the `auto` policy the
 * primitive is per-node (sized at the visual material's build), so
 * re-resolving here without the node's size would give large nodes a
 * capsule pick footprint over a quad render. Every wrapper stamps the
 * RESOLVED primitive on `userData.linePrimitive` at construction, so this
 * read is exact; `undefined` (not a lines material) falls through to the
 * session-wide resolution, which is correct for everything except a
 * missing stamp — and the stamp is unconditional.
 */
function linePrimitiveFromVisual(obj: THREE.Mesh): LinePrimitive | undefined {
  const visual = obj.material as THREE.Material | THREE.Material[] | undefined;
  const single = Array.isArray(visual) ? visual[0] : visual;
  return single?.userData?.linePrimitive as LinePrimitive | undefined;
}

/** Copy the live visual class filter onto a freshly-created gsplat pick material. */
function syncGSplatPickMaterialToVisual(obj: THREE.Mesh): void {
  const pickMaterial = (obj.userData?.pickNode as THREE.Mesh | undefined)?.material;
  if (!pickMaterial || Array.isArray(pickMaterial) || !('updateLabelFilter' in pickMaterial)) {
    return;
  }
  const visual = obj.material as THREE.Material | THREE.Material[] | undefined;
  const single = Array.isArray(visual) ? visual[0] : visual;
  const uniforms = (
    single as THREE.Material & {
      uniforms?: Record<string, { value?: unknown }>;
    }
  )?.uniforms;
  const liveFilter = uniforms?.uLabelFilterIndex?.value;
  if (typeof liveFilter !== 'number') return;
  (
    pickMaterial as THREE.Material & { updateLabelFilter(filterIndex: number): void }
  ).updateLabelFilter(liveFilter);
}

/**
 * Copy the visual material's per-epoch state onto a mesh's freshly-created pick
 * material.
 *
 * Mesh is the one type whose pick pass mirrors the visual material's face culling
 * and blending-derived behaviour (spec §6.5). The picking
 * system re-pushes both on every pick render, so this governs only the window before
 * the first one — but that window contains the first hover, which is exactly when a
 * user would notice picking a face that isn't drawn.
 *
 * Every value is read from the LIVE VISUAL MATERIAL rather than from the node attrs,
 * and that distinction is the point on the context-restore path:
 *
 * - `side` — by the time this runs the commit may already have forced `DoubleSide` for
 *   an undecidable `displayDims` frame, so the authored `double_sided` is stale.
 * - `blendingMode` — `userData.blendingMode` is the mode the material RESOLVED, which
 *   for a `volumetric`-by-inheritance mesh is already `opaque`. Re-deriving it from
 *   attrs would hand the pick pass a mode the shader is not in.
 * - `uOpacity` / `uAlphaCutoff` — these can have been dragged in the layers panel
 *   since load. `rebuildAfterContextRestore` re-runs this pass with FRESH pick
 *   materials (the old ones were compiled against the dead context) while the visual
 *   material survives with the user's edits, so seeding from attrs would silently
 *   revert the pick coverage to the authored values. Rare — it needs a context loss
 *   *and* a prior slider edit — and it self-heals on the next panel edit, which is
 *   exactly the kind of thing that never gets found later.
 */
function syncMeshPickMaterialToVisual(obj: THREE.Mesh): void {
  const pickMaterial = (obj.userData?.pickNode as THREE.Mesh | undefined)?.material;
  if (!pickMaterial || Array.isArray(pickMaterial)) return;
  if (!isMeshPickAwareMaterial(pickMaterial)) return;
  const visual = obj.material as THREE.Material | THREE.Material[] | undefined;
  const single = Array.isArray(visual) ? visual[0] : visual;
  if (!single) return;
  pickMaterial.setPickSide(single.side);
  // The RESOLVED mode the material stamped, not the authored one.
  pickMaterial.setPickMode(
    (single.userData?.blendingMode as BlendingMode | undefined) ??
      resolveRequestedMeshMode((obj.userData?.attrs ?? {}) as MeshMetadata)
  );
  const uniforms = (single as THREE.Material & { uniforms?: Record<string, { value?: unknown }> })
    .uniforms;
  const pick = pickMaterial as LuxarMeshPickingMaterial;
  const liveOpacity = uniforms?.uOpacity?.value;
  if (typeof liveOpacity === 'number') pick.updateOpacityUniform(liveOpacity);
  const liveCutoff = uniforms?.uAlphaCutoff?.value;
  if (typeof liveCutoff === 'number') pick.updateAlphaCutoff(liveCutoff);
}

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
   *
   * **This — not the per-node factory — is the path production actually takes.** The
   * scene loads before picking is wired up (`initPicking` traverses the finished
   * scene to decide whether any node declares labels, and only then constructs the
   * `PickingSystem` and calls `setPickingSystem`), so at node-creation time
   * `this.pickingSystem` is still null on a first load. A geometry type wired into
   * `createEmptyXNode` but missing HERE is unpickable in every real scene, and appears
   * to work only on a SECOND dataset load. Hence the table above rather than an
   * `else if` chain: adding a geometry type is a compile error at one place instead of
   * a branch someone forgets.
   */
  registerExistingSceneNodes(root: THREE.Object3D): void {
    if (!this.pickingSystem) return;

    root.traverse((obj) => {
      const nodeType = obj.userData?.nodeType;
      // Skip non-data nodes and already-registered ones. The `instanceof` gate is
      // load-bearing rather than defensive: the pick node SHARES `obj.geometry`, so an
      // object without one would register a pick node with nothing to draw.
      if (!isGeometryType(nodeType) || obj.userData.pickId != null) return;
      if (!(obj instanceof THREE.Mesh)) return;

      const pickId = this.pickingSystem!.allocatePickId();
      obj.userData.pickId = pickId;
      const recipe = PICK_MATERIAL_RECIPES[nodeType];
      const pickMaterial = recipe.build(obj, pickId);
      materialManager.register(pickMaterial);
      // Shares the visual geometry — for the three instanced types that is the
      // instance-spanning, footprint-expanded bounds, so the pick node culls safely;
      // for mesh it is the indexed BufferGeometry, whose drawRange the slice
      // compaction rewrites and which the picking system re-syncs per pick render.
      const pickNode = new THREE.Mesh(obj.geometry, pickMaterial);
      pickNode.matrixWorld.copy(obj.matrixWorld);
      this.pickingSystem!.registerNode(obj, pickNode, pickId);
      // Post-registration step, declared per type in the table above rather than
      // branched on here. Must run AFTER registerNode, which is what stamps
      // `userData.pickNode` that the step reads.
      recipe.afterRegister?.(obj);
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
   * `attrs` is the COMPOSED effective attrs and `leafAttrs` the node's RAW ones —
   * the same two-bag arrangement the three siblings use, needed for exactly one
   * decision: telling a leaf-authored colormap window from an inherited ancestor
   * gain (`resolveColormapWindow`). The order is reversed from the siblings' because
   * the raw bag is optional here, and it defaults to `attrs`, which is right
   * whenever no ancestor authored a gain.
   *
   * Registers a pick node like the three siblings do, with its own pick material
   * pair: mesh picking keys on `gl_VertexID` rather than an element-texture texel
   * (spec §6.5), so it could not reuse theirs.
   */
  createEmptyMeshNode(
    path: string,
    attrs: MeshMetadata,
    loader: MeshDataLoader,
    leafAttrs?: Partial<MeshMetadata>
  ): THREE.Mesh {
    return createEmptyMeshNodeImpl(path, attrs, loader, this.pickingSystem, leafAttrs);
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
