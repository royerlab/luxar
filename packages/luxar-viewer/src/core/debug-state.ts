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
 * @module core/debug-state
 */

import * as THREE from 'three';
import type { SimpleDims } from '../types/dims';

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

/** Result shape returned by `computeDebugState()`. */
export interface DebugState {
  totalPoints: number;
  totalGSplats: number;
  totalElements: number;
  pointClouds: PointCloudInfo[];
  gsplatMeshes: GSplatMeshInfo[];
  dimensions: { ndim: number; displayed: number[]; currentStep: number[] } | null;
  camera: {
    position: { x: number; y: number; z: number };
    fov: number;
  };
  /** Legacy flat camera position kept for backward compatibility. */
  cameraPosition: { x: number; y: number; z: number };
  /** Legacy flat fov kept for backward compatibility. */
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
}

/**
 * Walk the scene graph and report cumulative point/gsplat counts plus
 * per-mesh detail. The traversal:
 *   - Counts every `THREE.Points` mesh; uses `geometry.drawRange.count`
 *     when set (the GPU buffer pool uses drawRange to limit rendering
 *     after relocation), falling back to the position-attribute count.
 *   - Counts every `THREE.Mesh` with `userData.nodeType === 'gsplats'`
 *     and an `InstancedBufferGeometry`; uses `instanceCount` directly.
 *
 * Returns a JSON-serialisable structure suitable for hand-off to test
 * harnesses, AI debug drivers, or the recording panel.
 */
export function computeDebugState(ctx: DebugStateContext): DebugState {
  let totalPoints = 0;
  let totalGSplats = 0;
  const pointClouds: PointCloudInfo[] = [];
  const gsplatMeshes: GSplatMeshInfo[] = [];

  ctx.scene.traverse((object) => {
    if (object instanceof THREE.Points) {
      const geometry = object.geometry;
      const drawRangeCount = geometry?.drawRange?.count;
      const bufferCount = geometry?.attributes?.position?.count || 0;
      // Infinity means "draw all" — fall back to the buffer count.
      const pointCount =
        drawRangeCount !== undefined && drawRangeCount !== Infinity
          ? Math.min(drawRangeCount, bufferCount)
          : bufferCount;
      totalPoints += pointCount;
      pointClouds.push({
        name: object.name || 'unnamed',
        pointCount,
        visible: object.visible,
        hasColors: !!geometry?.attributes?.color,
        hasRadii: !!geometry?.attributes?.radius,
        hasSharpness: !!geometry?.attributes?.sharpness,
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

  return {
    totalPoints,
    totalGSplats,
    totalElements: totalPoints + totalGSplats,
    pointClouds,
    gsplatMeshes,
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
