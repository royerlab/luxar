/**
 * Turntable (auto-rotation) axis math.
 *
 * One tiny pure function, split out of the update sequencer and the
 * orchestrator's `applyOrbitRotation` so BOTH read the axis from the same
 * place — the interactive turntable and the recorded turntable cannot pick
 * different axes and ship a video that rotates unlike its own preview.
 */

import * as THREE from 'three';
import { DEFAULT_AUTO_ROTATE_AXIS, isAutoRotateAxis, type AutoRotateAxis } from '../../types';

/**
 * Which frame an axis vector is expressed in — `camera` vectors get rotated by
 * the live orientation, `world` vectors are already world-space constants.
 */
type AxisFrame = 'camera' | 'world';

interface AxisSpec {
  readonly frame: AxisFrame;
  readonly v: readonly [number, number, number];
}

/**
 * The direction each {@link AutoRotateAxis} names, as the axis POINTS, plus the
 * frame that direction is given in.
 *
 * Camera-frame entries are up, right, and the view direction (-Z — the camera
 * looks down its own negative Z); world-frame entries are the scene's own
 * basis. Rotation then follows the right-hand rule about that direction,
 * uniformly across both families: sighting back down the axis from its far
 * end, the scene turns counter-clockwise. Two consequences worth knowing:
 * `view` matching -Z is what makes a positive auto-rotation roll the same way
 * as a positive Shift+scroll roll delta (same vector), and `world-y` agrees
 * with `vertical` exactly while the camera is level — the two only diverge
 * once the camera has elevation, which is the whole reason to offer both.
 *
 * Typed as a total `Record` over the union so adding a token to
 * {@link AutoRotateAxis} without a direction here fails to compile.
 */
const AXIS_SPECS: Record<AutoRotateAxis, AxisSpec> = {
  vertical: { frame: 'camera', v: [0, 1, 0] },
  horizontal: { frame: 'camera', v: [1, 0, 0] },
  view: { frame: 'camera', v: [0, 0, -1] },
  'world-x': { frame: 'world', v: [1, 0, 0] },
  'world-y': { frame: 'world', v: [0, 1, 0] },
  'world-z': { frame: 'world', v: [0, 0, 1] },
};

/**
 * Write the world-space rotation axis for `axis` into `out`.
 *
 * A camera-frame direction is rotated by the live `orientation` so the result
 * is the screen-relative axis in world space; a world-frame direction is
 * already world-space and is returned as the constant it is. Either way the
 * caller gets what the orbit rotation (a world-frame premultiply) needs.
 *
 * Nothing here can destabilize the orbit. Each camera-frame axis is invariant
 * under its own rotation, so re-deriving it every frame is stable rather than
 * drifting: `vertical` holds the up vector fixed and sweeps the camera around a
 * circle, `horizontal` holds the right vector fixed and tumbles the camera over
 * the top, and `view` holds the camera POSITION fixed (the offset lies along
 * the axis) and spins only the up vector — a pure roll. A world axis has no
 * feedback at all: it is a constant, so the camera simply sweeps the cone of
 * whatever latitude it started at, and the subject spins about its own axis
 * instead of precessing. The degenerate world case — an axis parallel to the
 * view direction — is benign rather than singular: the camera offset lies along
 * the axis, so it stays put and the image rolls, just as `view` does.
 *
 * @param axis - Which axis to revolve around, camera-frame or world.
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
  const spec = AXIS_SPECS[isAutoRotateAxis(axis) ? axis : DEFAULT_AUTO_ROTATE_AXIS];
  const [x, y, z] = spec.v;
  out.set(x, y, z);
  return spec.frame === 'camera' ? out.applyQuaternion(orientation) : out;
}
