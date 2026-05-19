/**
 * Transform-matrix application for NodeFactory.
 *
 * Extracted in P6/step 6.2. Pure over its inputs; throws on bad
 * length or row-major (NumPy) layout — the producer-side test guard.
 *
 * @module rendering/node-factory/transforms
 */

import * as THREE from 'three';
import { validateTransformFormat } from './validation';

/**
 * Decompose a 16-element column-major transform onto a THREE.Object3D
 * (position / quaternion / scale). Throws on malformed input via the
 * validation guard.
 */
export function applyTransform(object: THREE.Object3D, transform: readonly number[]): void {
  if (transform.length !== 16) {
    throw new Error(`Invalid transform length: ${transform.length} (expected 16)`);
  }

  validateTransformFormat(transform);

  // THREE.Matrix4.fromArray takes ArrayLike<number>; readonly tuple is fine.
  const matrix = new THREE.Matrix4().fromArray(transform as number[]);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();

  matrix.decompose(position, quaternion, scale);

  object.position.copy(position);
  object.quaternion.copy(quaternion);
  object.scale.copy(scale);
}
