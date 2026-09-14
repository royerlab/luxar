/**
 * Debug-state computer for `window.__luxarDebug.getState()`.
 *
 * Pure scene-walking helper extracted from `core/app.ts` so the
 * point/gsplat counting + camera/dim reporting can be unit-tested
 * directly against real `THREE.Points` and `THREE.Mesh` fixtures
 * (THREE objects work fine in jsdom — only the WebGL renderer
 * doesn't).
 *
 * The function takes its dependencies as parameters rather than
 * reaching for the LuxarApp instance, so consumers can swap in stubs
 * (or call it from a notebook embed where `getInstance()` shorthand
 * isn't available).
 *
 * @module core/app/debug/debug-state
 */

import * as THREE from 'three';
import type { SimpleDims } from '../../../types/dims';
import { LOADER_TYPES, type LoaderTypeName } from '../../../types/format-contract';
import { readVisibleElementCount } from '../../../data/scene-loader/monitor/visible-counts';
import type { RefinementResidencyStop } from '../../../data/scene-loader/progressive/residency-budget';

/** Additive reveal-ladder state shared by every drawable leaf type. */
export interface AdditiveLadderDebugInfo {
  committedLadderComplete?: boolean;
  committedEnergyFraction?: number;
  loadedLODCount?: number;
  totalLODCount?: number;
  lastAllResident?: boolean;
}

/** Per-mesh point-cloud info reported by getState(). */
export interface PointCloudInfo extends AdditiveLadderDebugInfo {
  name: string;
  pointCount: number;
  visible: boolean;
  hasColors: boolean;
  hasRadii: boolean;
  hasSharpness: boolean;
  requestedElementCount: number;
  grantedElementCount: number;
  droppedElementCount: number;
}

/** Per-mesh gsplat info reported by getState(). */
export interface GSplatMeshInfo extends AdditiveLadderDebugInfo {
  name: string;
  splatCount: number;
  visible: boolean;
  requestedElementCount: number;
  grantedElementCount: number;
  droppedElementCount: number;
}

/** Per-mesh line-instance info reported by getState(). */
export interface LineMeshInfo extends AdditiveLadderDebugInfo {
  name: string;
  segmentCount: number;
  visible: boolean;
  hasColormap: boolean;
  requestedElementCount: number;
  grantedElementCount: number;
  droppedElementCount: number;
}

/**
 * Per-node mesh (triangle-surface) info reported by getState().
 *
 * `triangleCount` counts the triangles ACTUALLY DRAWN this epoch, which for a mesh is
 * `drawRange.count / 3` rather than an instance count: the nD slice compaction rewrites
 * the index buffer and narrows `drawRange` (spec §5.4), leaving the vertex arrays
 * untouched. Reading `index.count` instead would report the whole surface no matter
 * where the slice sits — the one number a debug driver most needs to be honest about.
 *
 * `flatNormal` and `alphaCutout` are the two shader VARIANTS a mesh can be built in,
 * surfaced because neither is visible from the geometry or the node attrs alone: the
 * flat/smooth choice folds in the live `displayDims` (§3.4) and the cutout follows the
 * composed blending mode. They are what an E2E test asserting "this mesh is shading
 * from stored normals right now" has to read.
 */
export interface MeshNodeInfo extends AdditiveLadderDebugInfo {
  name: string;
  /** Triangles in the current draw range — what is on screen, not what was loaded. */
  triangleCount: number;
  /**
   * Vertices committed to the node. Invariant across slices; the pick-id domain.
   *
   * Read from `userData.committedVertexCount` when stamped, NOT from
   * `position.count` — that attribute is capacity-sized for a reveal ladder
   * (#1521), so it reports the ladder's LIFETIME total from level 0 on rather
   * than what has actually committed.
   */
  vertexCount: number;
  visible: boolean;
  /** `true` when shading from screen-space derivatives instead of stored normals. */
  flatNormal: boolean;
  /** `true` in `opaque` mode: the fragment stage applies a hard alpha cutout. */
  alphaCutout: boolean;
  hasColormap: boolean;
}

/**
 * Minimal byte-stats shape surfaced from the GPU buffer pool. Mirrors
 * a subset of `PoolStats` so the debug UI doesn't need to import the
 * full type from the rendering module.
 */
export interface GPUPoolDebugStats {
  activeBuffers: number;
  pooledBuffers: number;
  activeBytes: number;
  pooledBytes: number;
  totalBytes: number;
  largestPooledBytes: number;
  evictions: number;
  /**
   * The subset of `evictions` forced by the byte budget (`PoolStats
   * .byteBudgetEvictions`). Kept distinct because only this one means the pool
   * went over its VRAM budget and shed pooled geometry — including levels the
   * LOD registry had demoted from active — to get back under it, which makes
   * what stayed resident a property of the machine; ordinary LRU recycling of
   * released buffers says nothing at all. See the `PoolStats` field doc for
   * what it does NOT prove.
   */
  byteBudgetEvictions: number;
}

/**
 * Substitutive `kind=lod` group summary. `activeLevel` is the index of
 * the currently-visible child (the level the registry selected); `-1`
 * when none is visible.
 */
export interface LODGroupDebugInfo {
  name: string;
  levelCount: number;
  activeLevel: number;
  selector: 'screen-area' | 'coverage';
  footprintStamped: boolean;
}

/** `kind=partition` BSP group summary. */
export interface PartitionDebugInfo {
  name: string;
  partCount: number;
  visibleParts: number;
}

/** Result shape returned by `computeDebugState()`. */
export interface DebugState {
  totalPoints: number;
  totalGSplats: number;
  /**
   * Line segments summed over ALL line meshes, hidden ones included — the
   * per-node `lineMeshes` entries carry `visible` for filtering.
   */
  totalLines: number;
  /**
   * Current-draw-range TRIANGLES summed over ALL mesh nodes. Like every
   * aggregate above, a hidden node still counts (its draw range is intact) —
   * the per-node `meshNodes` entries carry `visible` for filtering.
   */
  totalTriangles: number;
  totalElements: number;
  /**
   * Elements omitted by per-node element-texture capacity clamps across ALL nodes,
   * including hidden nodes and every substitutive-LOD level. Mesh is not texture-backed.
   */
  totalDroppedElements: number;
  pointClouds: PointCloudInfo[];
  gsplatMeshes: GSplatMeshInfo[];
  /** Per-mesh line summary. */
  lineMeshes: LineMeshInfo[];
  /** Per-node mesh (triangle-surface) summary. */
  meshNodes: MeshNodeInfo[];
  /** Substitutive LOD groups (kind=lod) with their active level. */
  lodGroups: LODGroupDebugInfo[];
  /** Partition groups (kind=partition) with part / visible-part counts. */
  partitions: PartitionDebugInfo[];
  /** GPU buffer pool byte stats (undefined when the pool is disabled). */
  gpuPool?: GPUPoolDebugStats;
  /**
   * The progressive-refinement BYTE-ceiling stop, when refinement declined at
   * least one rung while this scene was loaded; ABSENT when it never did.
   * Loader-scoped and cumulative within that life — see
   * {@link RefinementResidencyStop} for the contract.
   *
   * Absence is "no stop", NOT "no information": an older viewer build simply
   * does not carry the field, and a consumer must not read that as a scene in
   * trouble. Presence is the whole signal — the counts in this snapshot then
   * describe a PARTIAL scene whose composition is nondeterministic, which is
   * why `capture-readiness.ts` refuses on it (#2508).
   */
  refinementResidency?: RefinementResidencyStop;
  dimensions: { ndim: number; displayed: number[]; currentStep: number[] } | null;
  camera: {
    position: { x: number; y: number; z: number };
    fov: number;
  };
  /** Flat camera position kept for debug-state compatibility. */
  cameraPosition: { x: number; y: number; z: number };
  /** Flat camera field-of-view kept for debug-state compatibility. */
  cameraFov: number;
  isAnimating: boolean;
  initialized: boolean;
  /**
   * Whether a LOAD PASS is outstanding on at least one registered scene loader:
   * an `updateView` sweep (fetch / decode / upload) up to its geometry commit,
   * a failed-loader retry sweep, which takes the same lock, or a view-state
   * QUEUED behind either (the requested slice has not begun loading, so it is
   * still an unfinished pass — without it a nav that lands during a refinement
   * hold would read idle immediately).
   *
   * It does NOT cover, and must not be read as covering:
   *   - the INITIAL `loadScene` — that path only touches the loader's lock at
   *     its very end (to hand it to the post-load refinement kick). For the
   *     first load `initialized` is the flag to wait on; for an in-page DATASET
   *     SWITCH neither helps, because `initialized` stays true and the fresh
   *     loader is registered before its `loadScene` runs, so `isLoading` reads
   *     false throughout the switch's load.
   *   - lazy substitutive-LOD / deferred-partition `ensureLoaded` promotions,
   *     which run outside any `updateView` cycle and surface as content-change
   *     notifications instead.
   *   - the progressive-LOD refinement drain, which inherits the same lock
   *     after the current view has committed. Excluded deliberately: this is
   *     first-commit latency, not full-ladder latency — see
   *     `SceneLoader.isLoadPassInProgress`.
   *
   * The E2E "wait for data" helpers in `tests/e2e/helpers.ts` poll this field
   * to decide when a load has settled — `waitForDataLoaded`,
   * `waitForDimensionNavigation`, `waitForSpatialQuery`,
   * `waitForSpatialQueryOrThrow`, `waitForNavigationComplete`,
   * `waitForNavigationCompleteOrThrow`, and the state-based fallbacks in
   * `waitForRenderStable` / `waitForNextRender` (eight in all) — as do
   * `tests/e2e/real-dataset-loading.spec.ts` and the two capture specs under
   * `tests/screenshots/`. So it must always be a real boolean rather than
   * absent: `!undefined` is `true`, which would gate on nothing.
   */
  isLoading: boolean;
}

/**
 * One data mesh's cross-node draw-order record, reported by
 * `window.__luxarDebug.getDrawOrder()`. Mirrors what the depth-sort
 * coordinator's `renderOrder` pass and the material's blending state
 * decide, so a viewer bug (a backdrop drawn after the content in front of
 * it) can be diagnosed from the console without a renderer capture.
 */
export interface DrawOrderEntry {
  /** Scene-graph path / name of the mesh (node-factory stamps `mesh.name = path`). */
  path: string;
  /**
   * Blending bucket: `'transparent'` sorts, `'opaque'` is drawn depth-first.
   * THREE renders the whole opaque list before the transparent list, so the
   * bucket outranks `renderOrder` in the effective draw order.
   */
  bucket: 'opaque' | 'transparent';
  /** Whether this mesh writes depth (`material.depthWrite`). */
  depthWrite: boolean;
  /** Resolved `mesh.renderOrder` (ascending within a bucket → lowest drawn first). */
  renderOrder: number;
  /**
   * Drawn-primitive count for the node: points / segments / splats / TRIANGLES.
   *
   * Instanced types read `instanceCount`; mesh, which has no instances, reads the
   * committed `visibleTriangleCount` through the shared per-type reader.
   */
  elements: number;
}

/** Surface this helper needs from the parent LuxarApp's components. */
export interface DebugStateContext {
  scene: THREE.Object3D;
  camera: THREE.Camera;
  currentFov: number;
  isAnimating: boolean;
  initialized: boolean;
  /**
   * Whether a load pass is in flight on any registered scene loader — see
   * {@link DebugState.isLoading} for the exact scope (and what it excludes).
   * Passed in like `isAnimating` and `initialized` rather than read from a
   * singleton here, so this helper stays pure; the production caller supplies
   * `SceneLoaderManager.getInstance().isAnyLoadPassInProgress()` per snapshot.
   */
  isLoading: boolean;
  dims: SimpleDims | null;
  /** Optional pool-stats provider so the debug state can surface byte usage. */
  gpuPoolStats?: () => GPUPoolDebugStats | undefined;
  /**
   * Optional refinement-stop provider, supplied the same way and for the same
   * reason as `gpuPoolStats`: the production caller passes
   * `SceneLoaderManager.getInstance().refinementResidencyStop()` per snapshot so
   * this helper stays pure. Omitted (tests, embeds) means "unknown", which
   * surfaces as an absent {@link DebugState.refinementResidency} — never as a
   * synthesised "no stop".
   */
  refinementResidency?: () => RefinementResidencyStop | undefined;
}

function readAdditiveLadderDebugInfo(userData: Record<string, unknown>): AdditiveLadderDebugInfo {
  const loader = userData.loader as
    { loadedLODCount?: unknown; totalLODCount?: unknown; lastAllResident?: unknown } | undefined;
  if (typeof loader?.totalLODCount !== 'number' || loader.totalLODCount <= 1) return {};

  return {
    ...(typeof userData.committedLadderComplete === 'boolean'
      ? { committedLadderComplete: userData.committedLadderComplete }
      : {}),
    ...(typeof userData.committedEnergyFraction === 'number'
      ? { committedEnergyFraction: userData.committedEnergyFraction }
      : {}),
    ...(typeof loader.loadedLODCount === 'number' ? { loadedLODCount: loader.loadedLODCount } : {}),
    totalLODCount: loader.totalLODCount,
    ...(typeof loader.lastAllResident === 'boolean'
      ? { lastAllResident: loader.lastAllResident }
      : {}),
  };
}

/**
 * Walk the scene graph and report cumulative point/gsplat counts plus
 * per-mesh detail. The traversal:
 *   - Counts every points mesh (`userData.nodeType === 'points'`); uses
 *     `InstancedBufferGeometry.instanceCount` because pooled attributes are
 *     over-allocated and drawRange only covers the 6-index base quad.
 *   - Counts every `THREE.Mesh` with `userData.nodeType === 'gsplats'`
 *     and an `InstancedBufferGeometry`; uses `instanceCount` directly.
 *
 * Returns a JSON-serialisable structure suitable for hand-off to test
 * harnesses, AI debug drivers, or the recording panel.
 */
export function computeDebugState(ctx: DebugStateContext): DebugState {
  let totalPoints = 0;
  let totalGSplats = 0;
  let totalLines = 0;
  let totalTriangles = 0;
  let totalDroppedElements = 0;
  const pointClouds: PointCloudInfo[] = [];
  const gsplatMeshes: GSplatMeshInfo[] = [];
  const lineMeshes: LineMeshInfo[] = [];
  const meshNodes: MeshNodeInfo[] = [];
  const lodGroups: LODGroupDebugInfo[] = [];
  const partitions: PartitionDebugInfo[] = [];

  ctx.scene.traverse((object) => {
    // Specialized-group containers carry their kind in userData (set by
    // load-lod-group-node / load-partition-group-node). Surface their
    // structure + live active level so debug/E2E can assert on LOD
    // selection and partitioning without reaching into the registry.
    const kind = (object.userData as { kind?: string })?.kind;
    if (kind === 'lod') {
      const children = object.children;
      const metadata = object.userData as {
        lodSelector?: 'screen-area' | 'coverage';
        footprintStamped?: boolean;
      };
      lodGroups.push({
        name: object.name || 'unnamed',
        levelCount: children.length,
        activeLevel: children.findIndex((c) => c.visible),
        selector: metadata.lodSelector ?? 'coverage',
        footprintStamped: metadata.footprintStamped ?? false,
      });
    } else if (kind === 'partition') {
      const children = object.children;
      partitions.push({
        name: object.name || 'unnamed',
        partCount: children.length,
        visibleParts: children.reduce((n, c) => n + (c.visible ? 1 : 0), 0),
      });
    }

    // Points render as THREE.Mesh + InstancedBufferGeometry.
    // `instanceCount` is the source of truth for visible-point count;
    // attribute count can be pooled capacity.
    if (
      object instanceof THREE.Mesh &&
      (object.userData as { nodeType?: string })?.nodeType === 'points'
    ) {
      const geometry = object.geometry as THREE.InstancedBufferGeometry;
      const visiblePointCount = (object.userData as { visiblePointCount?: number })
        ?.visiblePointCount;
      const pointCount =
        geometry?.isInstancedBufferGeometry && Number.isFinite(geometry.instanceCount)
          ? geometry.instanceCount
          : (visiblePointCount ?? 0);
      // Production nodes are created empty and stamped at commit. The fallback
      // covers synthetic debug/test nodes created directly with geometry data.
      const requestedElementCount =
        (object.userData as { requestedElementCount?: number }).requestedElementCount ?? pointCount;
      const droppedElementCount =
        (object.userData as { droppedElementCount?: number }).droppedElementCount ?? 0;
      totalDroppedElements += droppedElementCount;
      totalPoints += pointCount;
      // Per-point data lives in the point texture (fixed 3-texel layout;
      // absent fields get identity fills), so field presence can no
      // longer be read off geometry attributes — the texel writers stamp
      // source presence on geometry.userData instead (the zarr node attrs
      // lack has_colors/has_radii/has_sharpness on pre-stamp datasets).
      const presence = geometry?.userData as
        { hasColors?: boolean; hasRadii?: boolean; hasSharpness?: boolean } | undefined;
      pointClouds.push({
        name: object.name || 'unnamed',
        pointCount,
        visible: object.visible,
        hasColors: !!presence?.hasColors,
        hasRadii: !!presence?.hasRadii,
        hasSharpness: !!presence?.hasSharpness,
        requestedElementCount,
        grantedElementCount: pointCount,
        droppedElementCount,
        ...readAdditiveLadderDebugInfo(object.userData),
      });
    }

    if (
      object instanceof THREE.Mesh &&
      (object.userData as { nodeType?: string })?.nodeType === 'gsplats' &&
      object.geometry instanceof THREE.InstancedBufferGeometry
    ) {
      const splatCount = (object.geometry as THREE.InstancedBufferGeometry).instanceCount;
      const requestedElementCount =
        (object.userData as { requestedElementCount?: number }).requestedElementCount ?? splatCount;
      const droppedElementCount =
        (object.userData as { droppedElementCount?: number }).droppedElementCount ?? 0;
      totalDroppedElements += droppedElementCount;
      totalGSplats += splatCount;
      gsplatMeshes.push({
        name: object.name || 'unnamed',
        splatCount,
        visible: object.visible,
        requestedElementCount,
        grantedElementCount: splatCount,
        droppedElementCount,
        ...readAdditiveLadderDebugInfo(object.userData),
      });
    }

    // Count Lines meshes (instanced quads with nodeType='lines').
    if (
      object instanceof THREE.Mesh &&
      (object.userData as { nodeType?: string })?.nodeType === 'lines' &&
      object.geometry instanceof THREE.InstancedBufferGeometry
    ) {
      const segmentCount = (object.geometry as THREE.InstancedBufferGeometry).instanceCount;
      const requestedElementCount =
        (object.userData as { requestedElementCount?: number }).requestedElementCount ??
        segmentCount;
      const droppedElementCount =
        (object.userData as { droppedElementCount?: number }).droppedElementCount ?? 0;
      totalDroppedElements += droppedElementCount;
      totalLines += segmentCount;
      // ShaderMaterial (GLSL) and NodeMaterial (TSL) both expose
      // `defines` — read structurally so this works on either backend.
      const mat = object.material as
        | (THREE.Material & { defines?: Record<string, unknown> })
        | (THREE.Material & { defines?: Record<string, unknown> })[]
        | undefined;
      const firstMat = Array.isArray(mat) ? mat[0] : mat;
      // PRESENCE, not truthiness — see the mesh arm below. Production sets
      // `defines.USE_COLORMAP = ''` (three emits a bare `#define`), and `!!''` is
      // false, so this had ALWAYS reported `hasColormap: false` for a colormapped line
      // node. Its unit test passed only because the fixture used `1` where production
      // uses `''` — a vacuous assertion, found when the same read was written for mesh
      // and checked against a real render.
      const hasColormap = !!firstMat?.defines && 'USE_COLORMAP' in firstMat.defines;
      lineMeshes.push({
        name: object.name || 'unnamed',
        segmentCount,
        visible: object.visible,
        hasColormap,
        requestedElementCount,
        grantedElementCount: segmentCount,
        droppedElementCount,
        ...readAdditiveLadderDebugInfo(object.userData),
      });
    }

    // Mesh: a plain indexed BufferGeometry, NOT an instanced one — so the
    // `instanceCount` every branch above reads does not exist here, and the
    // `InstancedBufferGeometry` guard they use would exclude it. The drawn count comes
    // from the DRAW RANGE, because that is what the nD slice compaction narrows (§5.4):
    // `index.count` would report the whole surface regardless of the slice position.
    if (
      object instanceof THREE.Mesh &&
      (object.userData as { nodeType?: string })?.nodeType === 'mesh' &&
      !(object.geometry instanceof THREE.InstancedBufferGeometry)
    ) {
      const geometry = object.geometry;
      const index = geometry?.index;
      const drawCount = geometry?.drawRange?.count;
      // `drawRange.count` defaults to Infinity ("draw everything"), so fall back to the
      // index length rather than reporting Infinity/3 as a triangle count.
      const indices =
        typeof drawCount === 'number' && Number.isFinite(drawCount)
          ? Math.min(drawCount, index?.count ?? 0)
          : (index?.count ?? 0);
      const triangleCount = Math.floor(indices / 3);
      totalTriangles += triangleCount;
      const mat = object.material as
        | (THREE.Material & { defines?: Record<string, unknown> })
        | (THREE.Material & { defines?: Record<string, unknown> })[]
        | undefined;
      const firstMat = Array.isArray(mat) ? mat[0] : mat;
      // Read from the material's DEFINES rather than from the node attrs, because the
      // variant is the resolved live state: the flat/smooth choice folds in the current
      // `displayDims` and the cutout follows the composed blending mode. On the TSL
      // backend the wrappers mirror the same flags into `defines` for exactly this
      // reason (there is no GLSL preprocessor there), so this reads the same on both.
      const defines = firstMat?.defines;
      // PRESENCE, not truthiness. A GLSL define's conventional value here is the empty
      // string (`defines[flag] = ''`, which three emits as a bare `#define FLAG`), and
      // `!!''` is false — so a truthiness test reports every mesh variant as OFF while
      // the shader is compiled WITH it. Caught by the first end-to-end render: the
      // `flat_patch` node carried `LUXAR_MESH_FLAT_NORMAL` in `defines` and still
      // reported `flatNormal: false`.
      const hasDefine = (flag: string): boolean => !!defines && flag in defines;
      // Prefer the commit's own stamp over `position.count`, which is
      // capacity-sized for a reveal ladder (#1521) and would otherwise report the
      // ladder's lifetime total from level 0 on. Falls back to `position.count`
      // for a node that never committed — the placeholder, or a unit-test
      // geometry built without going through `commitMeshGeometry`.
      const committedVertexCount = (object.userData as { committedVertexCount?: number })
        .committedVertexCount;
      meshNodes.push({
        name: object.name || 'unnamed',
        triangleCount,
        vertexCount: committedVertexCount ?? geometry?.getAttribute('position')?.count ?? 0,
        visible: object.visible,
        flatNormal: hasDefine('LUXAR_MESH_FLAT_NORMAL'),
        alphaCutout: hasDefine('LUXAR_MESH_ALPHA_CUTOUT'),
        hasColormap: hasDefine('USE_COLORMAP'),
        ...readAdditiveLadderDebugInfo(object.userData),
      });
    }
  });

  const dimensionsInfo = ctx.dims
    ? {
        ndim: ctx.dims.ndim,
        displayed: ctx.dims.displayed,
        currentStep: ctx.dims.currentStep,
      }
    : null;

  const cameraPosition = {
    x: ctx.camera.position.x,
    y: ctx.camera.position.y,
    z: ctx.camera.position.z,
  };

  // Surface GPU pool byte stats when a provider is wired in.
  const gpuPool = ctx.gpuPoolStats ? ctx.gpuPoolStats() : undefined;
  const refinementResidency = ctx.refinementResidency ? ctx.refinementResidency() : undefined;

  return {
    totalPoints,
    totalGSplats,
    totalLines,
    totalTriangles,
    // Every geometry type's DRAWN-primitive count, summed. Triangles join on the same
    // footing as segments and splats: it is the primitive the mesh actually draws, and
    // the noun the monitor and the visible-counts walk already use for it.
    totalElements: totalPoints + totalGSplats + totalLines + totalTriangles,
    totalDroppedElements,
    pointClouds,
    gsplatMeshes,
    lineMeshes,
    meshNodes,
    lodGroups,
    partitions,
    gpuPool,
    refinementResidency,
    dimensions: dimensionsInfo,
    camera: {
      position: cameraPosition,
      fov: ctx.currentFov,
    },
    cameraPosition,
    cameraFov: ctx.currentFov,
    isAnimating: ctx.isAnimating,
    initialized: ctx.initialized,
    isLoading: ctx.isLoading,
  };
}

/**
 * The viewer-drawable node types that carry a material + draw order — the
 * loader set (points / lines / gsplats), shared with
 * `data/scene-loader/monitor/draw-order-provider.ts` so drawability stays a
 * single capability and a future `mesh` loader is admitted in one place.
 */
const DATA_NODE_TYPES: ReadonlySet<string> = new Set<LoaderTypeName>(LOADER_TYPES);

/**
 * Walk the scene and report the effective cross-node draw order of every
 * VISIBLE data mesh: opaque meshes first — THREE renders its whole opaque
 * list before the transparent list, so the bucket outranks `renderOrder`,
 * which only orders meshes WITHIN a list — then `renderOrder` ascending.
 * Residual ties keep traversal (scene-graph) order as a deterministic report
 * order (THREE itself then compares material id / view-z, which this snapshot
 * doesn't reproduce). Powers `window.__luxarDebug.getDrawOrder()`.
 *
 * Reads live THREE state: the blending bucket + `depthWrite` from the mesh's
 * material and the resolved `renderOrder` the depth-sort coordinator assigned.
 * Hidden subtrees are pruned (like `data/scene-loader/monitor/visible-counts.ts`):
 * `renderOrder` is only assigned to visible sorted meshes and never reset, so a
 * toggled-off layer or inactive LOD level would otherwise report a stale order.
 * Element counts reuse the same `instanceCount` / `visible*Count` source of
 * truth as {@link computeDebugState}.
 */
export function computeDrawOrder(scene: THREE.Object3D): DrawOrderEntry[] {
  const entries: DrawOrderEntry[] = [];

  // Manual recursion rather than `traverse`, which visits `visible === false`
  // subtrees; those keep a stale `renderOrder` and must not be reported.
  const visit = (object: THREE.Object3D): void => {
    if (!object.visible) return;
    if (
      object instanceof THREE.Mesh &&
      DATA_NODE_TYPES.has((object.userData as { nodeType?: string })?.nodeType ?? '')
    ) {
      // Materials can be arrays; the render bucket + depthWrite are shared, so
      // the first material is representative.
      const material = Array.isArray(object.material) ? object.material[0] : object.material;

      const geometry = object.geometry as THREE.BufferGeometry;
      // The committed per-type count, read through the SHARED reader rather than a
      // local `visiblePointCount ?? visibleSplatCount ?? visibleSegmentCount` chain.
      // That chain was a partial copy of `VISIBLE_COUNT_READERS` and reported **0 for
      // every mesh**, because `visibleTriangleCount` was not in it and a missing field
      // reads as "no count" rather than as an error.
      const fallbackCount = readVisibleElementCount(object.userData) ?? 0;
      const elements =
        geometry instanceof THREE.InstancedBufferGeometry && Number.isFinite(geometry.instanceCount)
          ? geometry.instanceCount
          : fallbackCount;

      entries.push({
        path: object.name || 'unnamed',
        bucket: material?.transparent ? 'transparent' : 'opaque',
        depthWrite: !!material?.depthWrite,
        renderOrder: object.renderOrder,
        elements,
      });
    }
    for (const child of object.children) visit(child);
  };
  visit(scene);

  // Bucket first (THREE draws every opaque mesh before any transparent one,
  // regardless of renderOrder — and an opaque mesh can carry a stale positive
  // renderOrder from a live blending-mode switch, since it is never reset),
  // then renderOrder within the bucket. The sort is stable, so residual ties
  // keep traversal (scene-graph) order.
  return entries.sort((a, b) => {
    if (a.bucket !== b.bucket) return a.bucket === 'opaque' ? -1 : 1;
    return a.renderOrder - b.renderOrder;
  });
}
