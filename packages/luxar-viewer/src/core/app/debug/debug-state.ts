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

/** Per-mesh point-cloud info reported by getState(). */
export interface PointCloudInfo {
  name: string;
  pointCount: number;
  visible: boolean;
  hasColors: boolean;
  hasRadii: boolean;
  hasSharpness: boolean;
}

/** Per-mesh gsplat info reported by getState(). */
export interface GSplatMeshInfo {
  name: string;
  splatCount: number;
  visible: boolean;
}

/** Per-mesh line-instance info reported by getState(). */
export interface LineMeshInfo {
  name: string;
  segmentCount: number;
  visible: boolean;
  hasColormap: boolean;
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
export interface MeshNodeInfo {
  name: string;
  /** Triangles in the current draw range — what is on screen, not what was loaded. */
  triangleCount: number;
  /** Vertices bound on the geometry. Invariant across slices; the pick-id domain. */
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
  /** Total visible line segments across all line meshes. */
  totalLines: number;
  /** Total visible TRIANGLES across all mesh nodes. */
  totalTriangles: number;
  totalElements: number;
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
  dims: SimpleDims | null;
  /** Optional pool-stats provider so the debug state can surface byte usage. */
  gpuPoolStats?: () => GPUPoolDebugStats | undefined;
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
      lodGroups.push({
        name: object.name || 'unnamed',
        levelCount: children.length,
        activeLevel: children.findIndex((c) => c.visible),
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
      });
    }

    if (
      object instanceof THREE.Mesh &&
      (object.userData as { nodeType?: string })?.nodeType === 'gsplats' &&
      object.geometry instanceof THREE.InstancedBufferGeometry
    ) {
      const splatCount = (object.geometry as THREE.InstancedBufferGeometry).instanceCount;
      totalGSplats += splatCount;
      gsplatMeshes.push({
        name: object.name || 'unnamed',
        splatCount,
        visible: object.visible,
      });
    }

    // Count Lines meshes (instanced quads with nodeType='lines').
    if (
      object instanceof THREE.Mesh &&
      (object.userData as { nodeType?: string })?.nodeType === 'lines' &&
      object.geometry instanceof THREE.InstancedBufferGeometry
    ) {
      const segmentCount = (object.geometry as THREE.InstancedBufferGeometry).instanceCount;
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
      meshNodes.push({
        name: object.name || 'unnamed',
        triangleCount,
        vertexCount: geometry?.getAttribute('position')?.count ?? 0,
        visible: object.visible,
        flatNormal: hasDefine('LUXAR_MESH_FLAT_NORMAL'),
        alphaCutout: hasDefine('LUXAR_MESH_ALPHA_CUTOUT'),
        hasColormap: hasDefine('USE_COLORMAP'),
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

  return {
    totalPoints,
    totalGSplats,
    totalLines,
    totalTriangles,
    // Every geometry type's DRAWN-primitive count, summed. Triangles join on the same
    // footing as segments and splats: it is the primitive the mesh actually draws, and
    // the noun the monitor and the visible-counts walk already use for it.
    totalElements: totalPoints + totalGSplats + totalLines + totalTriangles,
    pointClouds,
    gsplatMeshes,
    lineMeshes,
    meshNodes,
    lodGroups,
    partitions,
    gpuPool,
    dimensions: dimensionsInfo,
    camera: {
      position: cameraPosition,
      fov: ctx.currentFov,
    },
    cameraPosition,
    cameraFov: ctx.currentFov,
    isAnimating: ctx.isAnimating,
    initialized: ctx.initialized,
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
