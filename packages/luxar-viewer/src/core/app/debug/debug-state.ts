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
  totalElements: number;
  pointClouds: PointCloudInfo[];
  gsplatMeshes: GSplatMeshInfo[];
  /** Per-mesh line summary. */
  lineMeshes: LineMeshInfo[];
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
  const pointClouds: PointCloudInfo[] = [];
  const gsplatMeshes: GSplatMeshInfo[] = [];
  const lineMeshes: LineMeshInfo[] = [];
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
      // carry no has_colors/has_radii/has_sharpness).
      const presence = geometry?.userData as
        | { hasColors?: boolean; hasRadii?: boolean; hasSharpness?: boolean }
        | undefined;
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
      const hasColormap = !!firstMat?.defines?.USE_COLORMAP;
      lineMeshes.push({
        name: object.name || 'unnamed',
        segmentCount,
        visible: object.visible,
        hasColormap,
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
    // Include lines in the cumulative element count.
    totalElements: totalPoints + totalGSplats + totalLines,
    pointClouds,
    gsplatMeshes,
    lineMeshes,
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
