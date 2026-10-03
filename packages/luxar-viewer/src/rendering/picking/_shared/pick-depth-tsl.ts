/**
 * The real fragment depth a TSL pick graph writes under the surface depth
 * convention (opaque / normal: front-most wins). GLSL twin: `gl_FragCoord.z`.
 *
 * Not three's `depth` node. That node picks `viewZToPerspectiveDepth` or
 * `viewZToOrthographicDepth` from `camera.isPerspectiveCamera` when the graph is
 * BUILT, and three's node-build cache key does not include the camera kind, so a
 * build is reused across cameras. A graph first built under a perspective camera
 * then wrote perspective depth under an orthographic one. Pick graphs built under
 * different kinds compared different formulas in one depth buffer, and front-most
 * picking broke after a perspective/ortho switch, or for a node first drawn after
 * one.
 *
 * Here the formula is chosen per draw from the projection matrix
 * (`isOrthoProjectionTSL()`, as every other projection-dependent term reads it).
 * Each branch is three's own expression over the same `positionView.z`, so the
 * value equals what `depth` gives in a build made under the drawn camera.
 *
 * @module rendering/picking/_shared/pick-depth-tsl
 */
import {
  cameraFar,
  cameraNear,
  int,
  positionView,
  viewZToOrthographicDepth,
  viewZToPerspectiveDepth,
} from 'three/tsl';
import { isOrthoProjectionTSL, type TSLNode } from '../../materials/_shared/tsl-helpers';

/** The drawn fragment's depth in [0, 1], for either projection kind (see the module doc). */
export function pickFragmentDepthTSL(): TSLNode {
  const viewZ: TSLNode = positionView.z;
  return isOrthoProjectionTSL()
    .equal(int(1))
    .select(
      viewZToOrthographicDepth(viewZ, cameraNear, cameraFar),
      viewZToPerspectiveDepth(viewZ, cameraNear, cameraFar)
    );
}
