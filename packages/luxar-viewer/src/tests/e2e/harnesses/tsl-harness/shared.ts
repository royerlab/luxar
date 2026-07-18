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
 * Perspective camera at (0,0,1) looking down −Z, for the behind-camera guard
 * cases. A point at world z=3 lands at view-space z=+2 (behind the camera),
 * so the perspective-only guard (`uIsOrtho == 0 && mvPosition.z >= 0`) fires.
 */
export function buildBehindCamera(): THREE.Camera {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 10);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  return camera;
}
