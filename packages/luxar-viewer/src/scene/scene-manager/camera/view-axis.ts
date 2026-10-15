/**
 * Axis-aligned opening views (`viewer_config.camera.view_axis`, `?viewAxis=`).
 *
 * A preset names the plane the camera faces, horizontal axis first: `xy` looks
 * at the plane of the dimensions named x and y (along z), `xz` along y, `yz`
 * along x. The letters are scene DIMENSION NAMES resolved against the displayed
 * dims, so a fitted z-stack whose columns stayed in `(z, y, x)` order (a grafted
 * partition) still faces its XY plane; when a name is missing the letter falls
 * back to world X/Y/Z.
 *
 * The scene's up vector chooses the convention, not the direction: its sign
 * along the preset's vertical axis says whether that axis points up (the
 * default, world +Y) or DOWN on screen — the image convention of Fiji/napari,
 * which `luxar gsplat convert` authors as `up = (0, -1, 0)`. When the authored
 * up is orthogonal to the vertical axis (a `?viewAxis=` override picking a face
 * the author did not), the sign of the up vector's dominant component decides.
 * The camera basis then follows from `right = forward × up` with the horizontal
 * axis pointing right, so every preset is a proper rotation — never a mirror.
 *
 * @module scene/scene-manager/camera/view-axis
 */

import * as THREE from 'three';
import type { ViewAxis } from '../../../types/view-axis';

/** Camera orientation a preset resolves to: look along `forward`, `up` on screen. */
export interface ViewAxisPose {
  /** Unit view direction (from the camera toward the target). */
  forward: THREE.Vector3;
  /** Unit screen-up vector, orthogonal to `forward`. */
  up: THREE.Vector3;
}

const WORLD_LETTERS = ['x', 'y', 'z'] as const;

/**
 * World axis (0/1/2) the dimension named `letter` is drawn along: its position
 * among the displayed dims (matched case-insensitively), else the letter's own
 * world axis.
 */
function worldAxisForLetter(letter: string, displayedNames: readonly string[]): number {
  const index = displayedNames.findIndex((name) => name.toLowerCase() === letter);
  if (index >= 0 && index < 3) return index;
  return WORLD_LETTERS.indexOf(letter as (typeof WORLD_LETTERS)[number]);
}

function unitAxis(axis: number): THREE.Vector3 {
  const v = new THREE.Vector3();
  v.setComponent(axis, 1);
  return v;
}

/** +1 or -1: whether the scene up points along (+) or against (-) `vertical`. */
function verticalSign(up: THREE.Vector3, vertical: number): number {
  const along = up.getComponent(vertical);
  if (Math.abs(along) > 1e-6) return Math.sign(along);
  const components = [up.x, up.y, up.z];
  const dominant = components.reduce((best, c) => (Math.abs(c) > Math.abs(best) ? c : best), 0);
  return dominant < 0 ? -1 : 1;
}

/**
 * Resolve `viewAxis` to a camera pose. Returns null when the preset's two axes
 * land on the same world axis (dimension names that cannot form the plane).
 *
 * @param viewAxis The preset.
 * @param sceneUp The scene's up vector (authored `camera.up`, else world +Y).
 * @param displayedNames Names of the displayed dimensions, in world-axis order.
 */
export function resolveViewAxisPose(
  viewAxis: ViewAxis,
  sceneUp: THREE.Vector3,
  displayedNames: readonly string[]
): ViewAxisPose | null {
  const horizontal = worldAxisForLetter(viewAxis[0], displayedNames);
  const vertical = worldAxisForLetter(viewAxis[1], displayedNames);
  if (horizontal === vertical || horizontal < 0 || vertical < 0) return null;
  const right = unitAxis(horizontal);
  const up = unitAxis(vertical).multiplyScalar(verticalSign(sceneUp, vertical));
  // right = forward × up  ⇔  forward = up × right (orthonormal right-handed basis).
  const forward = new THREE.Vector3().crossVectors(up, right);
  return { forward, up };
}
