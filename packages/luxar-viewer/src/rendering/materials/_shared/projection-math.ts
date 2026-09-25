/**
 * Camera-projection math as the shaders derive it from the projection matrix.
 *
 * The point, line and splat shaders need a handful of view-dependent scales:
 * a point's pixel size per world radius, a line's pixel width per world width,
 * a splat's screen centre and the Jacobian that projects its covariance. All of
 * them are functions of the projection matrix P and the viewport, and the
 * shaders read P directly (GLSL `projectionMatrix`, TSL
 * `cameraProjectionMatrix`), which three sets per camera per draw. Deriving
 * them there, instead of pushing CPU-computed copies as uniforms, keeps them
 * correct for every camera the scene is drawn with: a cube-capture face (fov
 * −90, so P is flipped), an asymmetric frustum (`setViewOffset`, an XR eye), a
 * zoomed perspective camera, or an embedder's own camera.
 *
 * This module is the CPU mirror of that shader math, for tests: it states the
 * formulas once, in a form the unit tests can check against the historical
 * fov-based helpers (`camera-uniforms.ts`) and against finite differences of
 * the projection itself.
 *
 * Conventions: `P` is a THREE.Matrix4 `elements` array (column-major, so row r
 * of column c is `P[c * 4 + r]`); `c` is a view-space point; `res` is the
 * viewport size in pixels. Size terms use |P11| because a flipped projection
 * (CubeCamera) mirrors positions but must not negate sizes; positions and the
 * Jacobian keep the sign, and a splat's 2D covariance J Σ Jᵀ is invariant under
 * that flip.
 *
 * @module rendering/materials/_shared/projection-math
 */

/** A 4×4 matrix in THREE's column-major `elements` layout. */
export type Mat4Elements = ArrayLike<number>;

/** Element at row `r`, column `c` of a column-major 4×4 matrix. */
function at(P: Mat4Elements, r: number, c: number): number {
  return P[c * 4 + r];
}

/**
 * Whether P is an orthographic projection: its last row is (0, 0, 0, 1), so
 * P[3][3] is 1, where a perspective projection's is 0.
 */
export function isOrthoProjection(P: Mat4Elements): boolean {
  return at(P, 3, 3) === 1;
}

/**
 * Pixels per view-space unit at unit depth, vertically: |P11|·resY/2 for
 * perspective (the focal length in pixels) and for orthographic (pixels per
 * world unit). Every CPU-side size scale is a constant multiple of this.
 */
export function focalLengthFromProjection(P: Mat4Elements, resY: number): number {
  return 0.5 * Math.abs(at(P, 1, 1)) * resY;
}

/** Point size factor (the historical `pointSizeFactor`): 2·|P11|·resY. */
export function pointSizeFactorFromProjection(P: Mat4Elements, resY: number): number {
  return 2 * Math.abs(at(P, 1, 1)) * resY;
}

/** Line pixel-width scale (the historical `uPerspectiveLineScale` / `uOrthoLineScale`): |P11|·resY. */
export function lineScaleFromProjection(P: Mat4Elements, resY: number): number {
  return Math.abs(at(P, 1, 1)) * resY;
}

/** Clip-space position P·(c, 1). */
export function projectToClip(
  P: Mat4Elements,
  c: readonly [number, number, number]
): [number, number, number, number] {
  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) {
    out[r] = at(P, r, 0) * c[0] + at(P, r, 1) * c[1] + at(P, r, 2) * c[2] + at(P, r, 3);
  }
  return out;
}

/**
 * Screen position of a view-space point, in pixels from the bottom-left
 * corner of the viewport (the `gl_FragCoord` convention).
 */
export function projectCenterPx(
  P: Mat4Elements,
  c: readonly [number, number, number],
  res: readonly [number, number]
): [number, number] {
  const clip = projectToClip(P, c);
  return [
    (clip[0] / clip[3]) * 0.5 * res[0] + 0.5 * res[0],
    (clip[1] / clip[3]) * 0.5 * res[1] + 0.5 * res[1],
  ];
}

/**
 * Jacobian of `projectCenterPx` with respect to the view-space point: the 2×3
 * matrix that maps a splat's view-space covariance to its screen covariance.
 * Returned as three columns (one per view-space axis), each a screen vector,
 * matching the shader's `mat3x2 J` / `J[0..2]` layout.
 *
 * General form, valid for any projection: with clip = P·(c, 1) and w = clip.w,
 * `J[i][k] = ½·res_i·(P[i][k]/w − clip_i·P[3][k]/w²)`. For the symmetric
 * perspective P three builds this reduces to the historical
 * `[[fx/z, 0], [0, fy/z], [fx·x/z², fy·y/z²]]`, and for an orthographic P
 * (w = 1, P[3][k] = 0) to `[[fx, 0], [0, fy], [0, 0]]`.
 */
export function gsplatJacobian(
  P: Mat4Elements,
  c: readonly [number, number, number],
  res: readonly [number, number]
): [[number, number], [number, number], [number, number]] {
  const clip = projectToClip(P, c);
  const w = clip[3];
  const col = (k: number): [number, number] => [
    0.5 * res[0] * (at(P, 0, k) / w - (clip[0] * at(P, 3, k)) / (w * w)),
    0.5 * res[1] * (at(P, 1, k) / w - (clip[1] * at(P, 3, k)) / (w * w)),
  ];
  return [col(0), col(1), col(2)];
}
