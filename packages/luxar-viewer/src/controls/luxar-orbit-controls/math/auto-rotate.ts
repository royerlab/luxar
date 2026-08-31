/**
 * Turntable (auto-rotation) axis math.
 *
 * One tiny pure function, split out of the update sequencer and the
 * orchestrator's `applyOrbitRotation` so BOTH read the axis from the same
 * place — the interactive turntable and the recorded turntable cannot pick
 * different axes and ship a video that rotates unlike its own preview.
 */

import * as THREE from 'three';
import type { AutoRotateAxis } from '../../types';

/**
 * Camera-space direction each {@link AutoRotateAxis} names, as the axis
 * POINTS: up, right, and the view direction (-Z — the camera looks down its
 * own negative Z). Rotation then follows the right-hand rule about that
 * direction, uniformly: sighting back down the axis from its far end, the
 * scene turns counter-clockwise. `view` matching -Z is also what makes a
 * positive auto-rotation roll the same way as a positive Shift+scroll roll
 * delta, which uses the same vector.
 */
const AXIS_IN_CAMERA_SPACE: Record<AutoRotateAxis, readonly [number, number, number]> = {
  vertical: [0, 1, 0],
  horizontal: [1, 0, 0],
  view: [0, 0, -1],
};

/**
 * Write the world-space rotation axis for `axis` into `out`.
 *
 * The camera-space direction is rotated by the live `orientation`, so the
 * result is the screen-relative axis expressed in world space — which is what
 * the orbit rotation (a world-frame premultiply) needs.
 *
 * Every axis is invariant under its own rotation, so re-deriving it each frame
 * is stable rather than drifting: `vertical` holds the up vector fixed and
 * sweeps the camera around a circle, `horizontal` holds the right vector fixed
 * and tumbles the camera over the top, and `view` holds the camera POSITION
 * fixed (the offset lies along the axis) and spins only the up vector — a pure
 * roll.
 *
 * @param axis - Which camera-frame axis to revolve around.
 * @param orientation - The live orbit orientation quaternion.
 * @param out - Destination vector, mutated and returned (no allocation on the
 *   per-frame path).
 */
export function autoRotateAxisVector(
  axis: AutoRotateAxis,
  orientation: THREE.Quaternion,
  out: THREE.Vector3
): THREE.Vector3 {
  // Fall back rather than throw: this runs inside the render loop, and an
  // unrecognized token (a hand-edited scene attr, a newer file) should degrade
  // to the historical turntable instead of killing every subsequent frame.
  const [x, y, z] = AXIS_IN_CAMERA_SPACE[axis] ?? AXIS_IN_CAMERA_SPACE.vertical;
  return out.set(x, y, z).applyQuaternion(orientation);
}
