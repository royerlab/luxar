/**
 * Transform-matrix application for NodeFactory.
 *
 * Pure over its inputs; throws on bad length or row-major (NumPy)
 * layout — the producer-side test guard.
 *
 * @module rendering/node-factory/transforms
 */

import * as THREE from 'three';
import { validateTransformFormat } from './validation';

/**
 * Apply a 16-element column-major transform to a THREE.Object3D, preserving the
 * full affine matrix. Throws on malformed input via the validation guard.
 *
 * The matrix is installed DIRECTLY rather than decomposed. `Matrix4.decompose()`
 * factors into position / quaternion / scale, and TRS cannot represent SHEAR —
 * so a composition as ordinary as `rotate ∘ non-uniform-scale` (which
 * `luxar.transforms.compose` advertises, and which
 * `packages/luxar/examples/transform_example.py` authors) came back altered.
 * Measured before this change: the authored matrix
 * `[[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]` should send `(0,1,0)` to `(1,1,0)`;
 * decomposition yielded `(0.5, 1.319, 0)`. Silent geometry corruption, which for a
 * scientific viewer is the worst failure mode available.
 *
 * Nothing downstream needs the TRS fields: no production code reads
 * `.position`/`.quaternion`/`.scale` off a data node (every such access is a
 * camera), and bounds, picking, frustum culling and depth sorting all consume the
 * full `matrixWorld`. The shaders take `modelViewMatrix` whole — including the
 * gsplat covariance congruence Σ' = A Σ Aᵀ, which is already general-linear and so
 * is correct under shear the moment the matrix survives to the GPU.
 *
 * `updateMatrixWorld(true)` is load-bearing: with `matrixAutoUpdate` off, three.js
 * no longer recomposes `matrix` from TRS, and nothing else would mark
 * `matrixWorld` stale — so the world matrix would never refresh. This mirrors
 * `core/layer/luxar-layer.ts`, which already installs an arbitrary 4×4 this way.
 */
export function applyTransform(object: THREE.Object3D, transform: readonly number[]): void {
  if (transform.length !== 16) {
    throw new Error(`Invalid transform length: ${transform.length} (expected 16)`);
  }

  validateTransformFormat(transform);

  // THREE.Matrix4.fromArray takes ArrayLike<number>; readonly tuple is fine.
  object.matrixAutoUpdate = false;
  object.matrix.fromArray(transform as number[]);
  object.updateMatrixWorld(true);
}
