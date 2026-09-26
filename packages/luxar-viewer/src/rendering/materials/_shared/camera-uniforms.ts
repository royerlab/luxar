/**
 * Pure camera-uniform math shared by Point materials and their picking
 * counterparts.
 *
 * The projection-derived terms (the perspective/ortho size scale, the focal
 * length) are no longer computed here: every shader reads them from the
 * projection matrix three binds per draw (see `projection-math.ts` for the
 * identities). What is left is viewport-only math.
 */

/**
 * Maximum point size in pixels — half the viewport height. This tracks
 * the hardware-typical `gl_PointSize` ceiling and keeps points from
 * blowing out when the camera gets very close.
 */
export function computeMaxPointSize(resY: number): number {
  return resY * 0.5;
}
