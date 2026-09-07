/**
 * A baked environment map (`environment/faces-<digest>` in the store) as a three
 * `CubeTexture`.
 *
 * The store carries the RAW capture — six faces of IEEE half-float bits, RGBA, in
 * three's `px, nx, py, ny, pz, nz` order — and three prefilters (PMREM) it at load, on
 * either backend, the moment it is assigned to `scene.environment`. Storing the raw
 * faces rather than a prefiltered map keeps the prefilter math three's and the file
 * small (a 128 px cube is ~0.8 MB of halves before zstd).
 *
 * Orientation: the faces are read back from the capture target in GL memory order
 * (bottom-up rows — `readPixelsCompactAsync(..., { flipY: true })`) and uploaded here
 * with `flipY = false`, so every texel lands where the render target had it. The
 * header's `coordinate_system` records which backend captured them; three's `CubeCamera`
 * compensates for the backends' differing framebuffer conventions with per-backend
 * face orientations precisely so the RESULTING cube map is the same conventional cube
 * map, which is what lets a map baked on one backend load on the other.
 *
 * @module rendering/environment/baked
 */

import * as THREE from 'three';
import type { BakedEnvironment } from '../../types/environment';
import { ENVIRONMENT_FACE_ORDER } from '../../types/environment';

/** Build the cube texture for a baked map. The caller owns its disposal. */
export function buildBakedCubeTexture(map: BakedEnvironment): THREE.CubeTexture {
  if (map.faces.length !== ENVIRONMENT_FACE_ORDER.length) {
    throw new Error(
      `A baked environment needs ${ENVIRONMENT_FACE_ORDER.length} faces, got ${map.faces.length}`
    );
  }
  const size = map.resolution;
  const images = map.faces.map((face, i) => {
    if (face.length !== size * size * 4) {
      throw new Error(
        `Baked environment face ${ENVIRONMENT_FACE_ORDER[i]} has ${face.length} samples; ` +
          `a ${size}px RGBA face needs ${size * size * 4}`
      );
    }
    const data = new THREE.DataTexture(face, size, size, THREE.RGBAFormat, THREE.HalfFloatType);
    data.colorSpace = THREE.LinearSRGBColorSpace;
    data.flipY = false;
    data.needsUpdate = true;
    return data;
  });
  const cube = new THREE.CubeTexture(images, THREE.CubeReflectionMapping);
  cube.type = THREE.HalfFloatType;
  cube.format = THREE.RGBAFormat;
  cube.colorSpace = THREE.LinearSRGBColorSpace;
  cube.generateMipmaps = false;
  cube.minFilter = THREE.LinearFilter;
  cube.magFilter = THREE.LinearFilter;
  cube.flipY = false;
  cube.needsUpdate = true;
  return cube;
}
