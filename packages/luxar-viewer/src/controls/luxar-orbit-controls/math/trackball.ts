/**
 * Virtual-trackball rotation math (vendored from ArcballControls / Shoemake).
 * Extracted from `luxar-orbit-controls.ts` so the orchestrator file
 * stays focused on its sequenced update step and DOM lifecycle.
 *
 * Pure functions: no orbit-control state, no allocations beyond the
 * caller-visible return values.
 */

import * as THREE from 'three';

/** Project NDC coordinates onto virtual trackball (sphere + hyperboloid). */
export function projectOnTrackball(ndcX: number, ndcY: number, radius: number): THREE.Vector3 {
  const r2 = radius * radius;
  const d2 = ndcX * ndcX + ndcY * ndcY;
  let z: number;
  if (d2 <= r2 * 0.5) {
    z = Math.sqrt(r2 - d2); // On the sphere
  } else {
    z = (r2 * 0.5) / Math.sqrt(d2); // On the hyperboloid (smooth falloff at edges)
  }
  return new THREE.Vector3(ndcX, ndcY, z).normalize();
}

/** Compute rotation quaternion from arcball drag (start → end in NDC). */
export function computeArcballRotation(
  startNDC: THREE.Vector2,
  endNDC: THREE.Vector2,
  trackballRadius: number,
  rotateSpeed: number
): THREE.Quaternion {
  const p1 = projectOnTrackball(startNDC.x, startNDC.y, trackballRadius);
  const p2 = projectOnTrackball(endNDC.x, endNDC.y, trackballRadius);

  const axis = new THREE.Vector3().crossVectors(p1, p2);
  if (axis.lengthSq() < 1e-10) return new THREE.Quaternion(); // No rotation

  axis.normalize();
  const angle = Math.acos(THREE.MathUtils.clamp(p1.dot(p2), -1, 1)) * rotateSpeed;

  // Negate angle: camera orbits opposite to the drag direction
  // (dragging right rotates the view rightward = camera moves left around target)
  return new THREE.Quaternion().setFromAxisAngle(axis, -angle);
}
