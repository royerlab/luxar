/**
 * Cross-family helpers for the TSL ↔ GLSL parity harness: the default
 * ortho / behind-camera perspective cameras shared by every registry
 * entry that doesn't override `buildCamera`, and the deterministic
 * colormap LUT used by the point/line/gsplat colormap-parity cases.
 *
 * @module tests/e2e/harnesses/tsl-harness/shared
 */

import * as THREE from 'three';

/**
 * Deterministic 256×1 RGBA colormap LUT for the colormap-parity cases.
 * A diagonal gradient (R ramps up, B ramps down, G a triangle) so the
 * sampled colour varies meaningfully with the lookup coordinate `t` —
 * making gamma-on-value warping observable. Matches the production
 * colormap texture layout/filtering (`colormap-textures.ts`) so both
 * backends sample it identically.
 */
export function buildColormapTexture(): THREE.DataTexture {
  const n = 256;
  const data = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    data[o] = i; // R: 0 → 255
    data[o + 1] = i < 128 ? i * 2 : (255 - i) * 2; // G: triangle peak at mid
    data[o + 2] = 255 - i; // B: 255 → 0
    data[o + 3] = 255;
  }
  const tex = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Default parity camera: `OrthographicCamera` at (0,0,1) looking at the
 * origin, so world (0,0,0) projects to NDC centre. Shared by every case that
 * doesn't override `buildCamera`.
 */
export function buildDefaultCamera(): THREE.Camera {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  return camera;
}

/**
 * Orthographic camera at (0,0,1) whose frustum height is `height` world
 * units. Point, line and splat shaders derive their size scale from the
 * projection matrix (|P11| = 2 / height), so a case that needs a specific
 * size factor states it through the camera it renders with: e.g. a point
 * size factor of `4 * resY / height`.
 */
export function buildOrthoCamera(height: number): THREE.Camera {
  const half = height / 2;
  const camera = new THREE.OrthographicCamera(-half, half, half, -half, 0.1, 10);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  return camera;
}

/**
 * The +X face camera of a `THREE.CubeCamera` at the origin, built the way
 * three builds it (WebGL coordinate system): fov −90 — a negative fov is
 * three's convention for a cube face, and it FLIPS the projection matrix —
 * up (0, −1, 0), looking down +X. The scene-captured environment renders the
 * data through six of these.
 */
export function buildCubeFaceCamera(): THREE.Camera {
  const camera = new THREE.PerspectiveCamera(-90, 1, 0.1, 10);
  camera.up.set(0, -1, 0);
  camera.lookAt(1, 0, 0);
  return camera;
}

/**
 * An ordinary +90° camera that sees exactly what {@link buildCubeFaceCamera}
 * sees: the −90 fov negates both NDC axes, which is a 180° roll about the
 * view axis, so the same view direction with up (0, +1, 0) produces the same
 * image. A geometry whose size or position ignores the projection's sign
 * renders differently through the two.
 */
export function buildCubeFaceEquivalentCamera(): THREE.Camera {
  const camera = new THREE.PerspectiveCamera(90, 1, 0.1, 10);
  camera.up.set(0, 1, 0);
  camera.lookAt(1, 0, 0);
  return camera;
}

/**
 * Perspective camera at (0,0,1) looking down −Z, for the behind-camera guard
 * cases. A point at world z=3 lands at view-space z=+2 (behind the camera),
 * so the perspective-only guard (view z >= 0 under a perspective projection) fires.
 */
export function buildBehindCamera(): THREE.Camera {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 10);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  return camera;
}
