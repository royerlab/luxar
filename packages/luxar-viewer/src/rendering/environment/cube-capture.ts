/**
 * The exact scene capture behind `viewer_config.environment.source = "scene"`.
 *
 * Six renders of the REAL scene with the REAL shaders from a `CubeCamera` at the probe,
 * into a half-float cube render target. Nothing is approximated: what a chrome sphere
 * reflects is what the viewer draws, colormaps, intensity, opacity, LOD energy
 * compensation and all (spec §3.3 — the approximate emitter projection was rejected for
 * duplicating that pipeline). Assigning the target's texture to `scene.environment` is
 * enough on both backends: three re-prefilters (PMREM) in place whenever
 * `CubeCamera.update` bumps the texture's `pmremVersion`.
 *
 * Two details keep the capture correct:
 *
 * - **No self-reflection.** Physical meshes are hidden for the six draws (their
 *   previous `visible` is restored afterwards, so a user- or LOD-hidden mesh stays
 *   hidden). Visibility rather than a `layers` bit, because the pick camera and the main
 *   camera would both need the bit and nothing else in the viewer uses layers.
 * - **Sprite sizes.** Point and line footprints come from the camera params the material
 *   manager broadcasts (fov, drawing-buffer size, pixel ratio). The caller pushes the
 *   cube camera's params before the six renders and restores the main camera's after —
 *   see `SceneEnvironment.captureScene`.
 *
 * @module rendering/environment/cube-capture
 */

import * as THREE from 'three';
import { isPhysicalMeshMaterial } from '../materials/mesh-physical/config';

/** The slice of a cube render target this module needs — the same on both backends. */
export interface CubeTargetLike {
  texture: THREE.CubeTexture;
  width: number;
  height: number;
  dispose(): void;
}

/** What one capture needs, all injected so a unit test can hand in stubs. */
export interface CubeCaptureRequest {
  /** Either backend's renderer; `CubeCamera.update` dispatches on it itself. */
  renderer: unknown;
  scene: THREE.Scene;
  target: CubeTargetLike;
  /** World position the six faces look out from. */
  probe: THREE.Vector3;
  near: number;
  far: number;
  /** The subtree whose physical meshes are hidden during the draws (the scene root). */
  root: THREE.Object3D | null;
}

/** True for an object drawn with one of the two physical wrappers. */
export function isPhysicalMeshObject(obj: THREE.Object3D): boolean {
  const material = (obj as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
  return !!material && !Array.isArray(material) && isPhysicalMeshMaterial(material);
}

/**
 * Near/far planes for a capture from `probe` of a scene with the given bounding sphere:
 * far reaches past the farthest point of the bounds with margin, near is a small
 * fraction of that so a probe inside a cluster does not clip it away.
 */
export function captureClipPlanes(
  probe: THREE.Vector3,
  bounds: THREE.Sphere | null
): { near: number; far: number } {
  if (!bounds) return { near: 0.01, far: 1000 };
  const reach = probe.distanceTo(bounds.center) + bounds.radius;
  const far = Math.max(reach * 2, 1e-3);
  return { near: Math.max(far * 1e-4, 1e-6), far };
}

/**
 * Render the six faces. Synchronous; the target's texture carries the result and its
 * `needsPMREMUpdate` flag is set by `CubeCamera.update`. Returns how many physical
 * meshes were hidden for the draws (a diagnostic the tests pin).
 */
export function captureSceneCube(req: CubeCaptureRequest): number {
  const hidden: THREE.Object3D[] = [];
  req.root?.traverse((obj) => {
    if (obj.visible && isPhysicalMeshObject(obj)) {
      obj.visible = false;
      hidden.push(obj);
    }
  });
  // Also keep the previous capture (or the room) out of the picture: a stale copy
  // of the environment showing up in the new one is the failure mode. Physical
  // meshes are hidden, so nothing samples `scene.environment` during the draws.
  const previousEnvironment = req.scene.environment;
  req.scene.environment = null;
  try {
    const camera = new THREE.CubeCamera(
      req.near,
      req.far,
      req.target as unknown as THREE.WebGLCubeRenderTarget
    );
    camera.position.copy(req.probe);
    // The target's texture is prefiltered by three (PMREM) — its own mip chain would
    // only be work; and a fresh CubeCamera must not inherit a stale one.
    req.target.texture.generateMipmaps = false;
    camera.update(req.renderer as THREE.WebGLRenderer, req.scene);
  } finally {
    req.scene.environment = previousEnvironment;
    for (const obj of hidden) obj.visible = true;
  }
  return hidden.length;
}
