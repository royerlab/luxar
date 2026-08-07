/**
 * The ONE geometric scenario that reproduces the asymmetric line-join miter of
 * issue #1346, shared by the vitest unit test
 * (`unit/rendering/line-join-near-plane-symmetry.test.ts`) and the Playwright
 * parity fixtures (`line-join-nearplane-miter` / `-none` in
 * `e2e/harnesses/tsl-harness/lines.ts`). One module so the two are provably the
 * same configuration rather than two hand-tuned copies that drift apart.
 *
 * THE BUG. The shared join block used to gate the miter on whether the
 * PARTNER's far endpoint sits in front of the near-cull plane. Each side of a
 * joint tests a DIFFERENT point, so the two segments meeting there could take
 * different branches: `A` (front → shared vertex) sees `B`'s far endpoint
 * behind the plane and falls back to the plain perpendicular, while `B`
 * (shared vertex → behind the plane) sees `A`'s far endpoint in front, finds
 * its own shared endpoint unclipped, and mitres ALONE. A wedge opens on one
 * side of the joint and `B`'s rotated edge protrudes on the other.
 *
 * THE GEOMETRY, and why every number matters:
 *
 * - Camera: `PerspectiveCamera(fov 60, aspect 1, near 0.1, far 10)` at
 *   `(0, 0, 1)` looking down −Z, so a world point at `z` has view depth
 *   `1 − z` — depths are readable straight off the world coordinates. Every one
 *   of those numbers lives in {@link NEAR_PLANE_CAMERA}, and the harness
 *   CONSTRUCTS its camera from them (`buildNearPlaneJoinCamera` in
 *   `tsl-harness/lines.ts`) rather than borrowing the shared
 *   `buildBehindCamera`: that one is used by ~20 unrelated fixtures, so
 *   retuning it for a point or gsplat case would silently move this joint out
 *   of its reproducing configuration.
 * - Viewport 64×64 (every parity fixture's size). Note this one is the ONE
 *   constant the harness does not take from here — see
 *   `NEAR_PLANE_CAMERA.resolution` for why `uResolution` must stay owned by the
 *   harness's own `HARNESS_SIZE`. `uNearCull`, `uPerspectiveLineScale` and
 *   `uMaxLinePixelWidth` are all overridden from this module.
 * - `uNearCull = 0.5`, deliberately ABOVE the camera's own near plane (0.1):
 *   an endpoint can then sit behind the near-CULL plane while still being in
 *   front of the camera and inside the hardware frustum, so the guard under
 *   test is the only thing that can reject it. A `nearCull` below the camera
 *   near plane would let hardware clipping mask the bug.
 * - `uPerspectiveLineScale = resolutionY / tan(fov/2)` — the real formula the
 *   materials use (see the uniform's comment in `materials/line/shader-glsl.ts`
 *   and `updateCameraParams`) — ≈ 110.85 here. `uMaxLinePixelWidth = 32`.
 * - Segment A: `(-0.30, -0.15, 0)` → the shared vertex `V = (0, 0, 0)`. Both
 *   endpoints at depth 1.0, so A is never near-clipped and its shared corner's
 *   half-width is the un-clamped one.
 * - Segment B: `V` → `(0.35, -0.15, 0.8)`, i.e. depth 0.2 — BEHIND `uNearCull`.
 *   B's own shared endpoint is at depth 1.0 and stays unclipped (`tA == 0`), so
 *   `reachesVertex` passes on B's side: the near-plane guard is the only thing
 *   left that can stop B mitering.
 * - `width` 0.06 at every endpoint → half-width `0.06 × 110.85 / 1.0 ≈ 6.65 px`
 *   at the shared vertex, comfortably over the join block's 2 px
 *   rendered-width gate. A thinner line would skip the block entirely and the
 *   whole fixture would go vacuous.
 * - Joint codes per `compute_joint_codes`: A (slot 0)'s END meets B (slot 1)'s
 *   START, so `A.endJointCode = +(1 + 1) = 2` and
 *   `B.startJointCode = −(0 + 3) = −3`; the two outer ends are free (`0`).
 *
 * Derived quantities (recomputed by the unit test, not just asserted here):
 * `turn = 0.6459`, `grow = 1.1023 ≤ 2`, axial reach `3.0848 px` against a bound
 * of `9.2952 px` — so the miter limit and the overshoot guard both PASS on both
 * sides, and the near-plane conjunction is the sole decider.
 *
 * CAVEAT on that `turn`, because it is easy to attribute to the wrong cause: it
 * is NOT the canonical operand order that makes the two sides agree here (`dot`
 * is commutative, so the ordering buys nothing for this quantity). They agree
 * because the shared vertex sits at the NDC ORIGIN, which turns the `wGuard`
 * clamp on B's behind-plane far endpoint into a pure RADIAL scale about
 * `sharedPx` — A's view of B's direction and B's own clipped direction then
 * coincide exactly. Moving the shared vertex off the origin breaks that and
 * with it the unit test's both-sides `turn` equality in the relaxed one-sided
 * mode. The fixture is fine as authored; do not move the vertex.
 *
 * {@link NEAR_PLANE_CONTROL_B_FAR} is the control: B's far endpoint moved in
 * front of the plane, where both sides must still mitre after the fix.
 *
 * Pure data plus analytic projection — no THREE, no DOM — so the module is
 * importable from a jsdom unit test and from the browser harness alike.
 *
 * @module tests/helpers/line-join-near-plane-scenario
 */

/** World-space point, `[x, y, z]`. */
export type Vec3 = readonly [number, number, number];
/** Pixel-space point, `[x, y]`, in the shader's centre-origin convention. */
export type Vec2 = readonly [number, number];

/** Camera + viewport constants the scenario is authored against. */
export const NEAR_PLANE_CAMERA = {
  /** Vertical field of view in degrees — `PerspectiveCamera(60, ...)`. */
  fovDegrees: 60,
  /** Square viewport, so the X and Y projection scales coincide. */
  aspect: 1,
  /** Camera sits at `(0, 0, camZ)` looking down −Z. */
  camZ: 1,
  /**
   * Viewport edge in pixels. This one is a MIRROR, not an authority: the parity
   * harness renders into a fixed square target of `HARNESS_SIZE`
   * (`tsl-harness/render.ts`), and `uResolution` must equal that target or the
   * shader's pixel↔NDC conversion desynchronises from the viewport it is drawing
   * into. So the fixture must NOT override `uResolution` from this constant —
   * on a harness resize that would actively create the mismatch (shader says 64,
   * target says 128) instead of coupling them. The two are equal today; if
   * `HARNESS_SIZE` ever changes, this value and every derived number below
   * (half-widths, pixel coordinates, `PERSPECTIVE_LINE_SCALE`) must be
   * recomputed to match it.
   */
  resolution: 64,
  /** The camera's own near/far planes. `nearCull` below sits ABOVE `near`. */
  near: 0.1,
  far: 10,
  /**
   * Scene-relative near-cull distance. ABOVE the camera's own near plane (0.1)
   * on purpose — see the module doc.
   */
  nearCull: 0.5,
  /**
   * `uMaxLinePixelWidth`; well above the ~6.65 px half-width here, so the clamp
   * is inert. Unlike `resolution` this is a pure fixture knob with no tie to the
   * render target, so the harness DOES override the uniform from it.
   */
  maxLinePixelWidth: 32,
} as const;

/**
 * `m11` of the perspective projection matrix, `1 / tan(fov/2)`. THREE's
 * `makePerspective` produces exactly this, with `clip.w = -mv.z`, which is what
 * lets the projection below be analytic.
 */
export const PROJ_SCALE_Y = 1 / Math.tan(((NEAR_PLANE_CAMERA.fovDegrees / 2) * Math.PI) / 180);

/**
 * `m00 = m11 / aspect`, written out so the aspect factor is explicit rather
 * than folded away — it is the value the camera is actually constructed with.
 *
 * It buys no generality, and should not be read as if it did: this module is
 * authored for a SQUARE viewport throughout. `resolution` is a single scalar
 * used as the pixel half-extent on both axes AND as `resY` in
 * {@link PERSPECTIVE_LINE_SCALE}, so a non-square viewport cannot be expressed
 * here at all. At `aspect === 1` this equals {@link PROJ_SCALE_Y} exactly.
 */
export const PROJ_SCALE_X = PROJ_SCALE_Y / NEAR_PLANE_CAMERA.aspect;

/**
 * `uPerspectiveLineScale = resolution.y / tan(fov * 0.5)` ≈ 110.85 — the same
 * expression `LineMaterial.updateCameraParams` precomputes on the CPU. Vertical
 * by definition, so it takes `m11`.
 */
export const PERSPECTIVE_LINE_SCALE = NEAR_PLANE_CAMERA.resolution * PROJ_SCALE_Y;

/**
 * Authored per-endpoint width in world units. Chosen so the shared vertex
 * renders at ≈ 6.65 px half-width — over the join block's 2 px gate with room
 * to spare, and well under the 32 px clamp.
 */
export const NEAR_PLANE_WIDTH = 0.06;

/** Segment A's free (outer) endpoint — depth 1.0, comfortably in front. */
export const NEAR_PLANE_A_START: Vec3 = [-0.3, -0.15, 0];
/** The shared vertex both segments meet at — depth 1.0. */
export const NEAR_PLANE_SHARED: Vec3 = [0, 0, 0];
/** Segment B's free (outer) endpoint at depth 0.2 — BEHIND `nearCull = 0.5`. */
export const NEAR_PLANE_B_FAR: Vec3 = [0.35, -0.15, 0.8];
/** Control variant: the same direction, but at depth 1.0 — in FRONT. */
export const NEAR_PLANE_CONTROL_B_FAR: Vec3 = [0.35, -0.15, 0];

/**
 * `compute_joint_codes`' output for this pair. A is storage slot 0, B slot 1;
 * A's END meets B's START, so A names `+(1 + 1)` and B names `−(0 + 3)`.
 */
export const NEAR_PLANE_JOINT_CODES = {
  aStart: 0,
  aEnd: 2,
  bStart: -3,
  bEnd: 0,
} as const;

/** View-space depth (`-mv.z`) of a world point under this camera. */
export function viewDepth(world: Vec3): number {
  return NEAR_PLANE_CAMERA.camZ - world[2];
}

/**
 * Project a world point into the shader's pixel space, INCLUDING the
 * `wGuard = max(clip.w, nearCull)` clamp the vertex stages and
 * `luxarLinePixelPos` both apply.
 *
 * The `+0.5` of `(ndc * 0.5 + 0.5) * resolution` is omitted because every
 * consumer takes differences, exactly as the shaders' comment notes.
 */
export function projectToPixels(world: Vec3): { px: Vec2; depth: number } {
  const mv: Vec3 = [world[0], world[1], world[2] - NEAR_PLANE_CAMERA.camZ];
  return { px: projectViewToPixels(mv), depth: -mv[2] };
}

/** The same projection from an already view-space position. */
export function projectViewToPixels(mv: Vec3): Vec2 {
  const clipW = -mv[2];
  const wGuard = Math.max(clipW, NEAR_PLANE_CAMERA.nearCull);
  const half = 0.5 * NEAR_PLANE_CAMERA.resolution;
  return [(PROJ_SCALE_X * mv[0] * half) / wGuard, (PROJ_SCALE_Y * mv[1] * half) / wGuard];
}

/**
 * The vertex stages' near-plane SEGMENT clip: `[tA, tB]` is the sub-range of
 * the segment that survives, and an endpoint with `tA > 0` / `tB < 1` was moved
 * off its source vertex (which is what makes `reachesVertex` false there).
 * Mirrors the `if (startDepth < nearCull && endDepth >= nearCull) ...` block.
 */
export function nearPlaneClipRange(
  startDepth: number,
  endDepth: number
): { tA: number; tB: number } {
  const { nearCull } = NEAR_PLANE_CAMERA;
  if (startDepth < nearCull && endDepth >= nearCull) {
    return { tA: (nearCull - startDepth) / (endDepth - startDepth), tB: 1 };
  }
  if (endDepth < nearCull && startDepth >= nearCull) {
    return { tA: 0, tB: (startDepth - nearCull) / (startDepth - endDepth) };
  }
  return { tA: 0, tB: 1 };
}

/**
 * The per-vertex rendered half-width the join block gates on:
 * `clamp(width * uPerspectiveLineScale / max(depth, nearCull), 1.5, maxPW)`.
 */
export function halfWidthAtDepth(depth: number, width = NEAR_PLANE_WIDTH): number {
  const raw = (width * PERSPECTIVE_LINE_SCALE) / Math.max(depth, NEAR_PLANE_CAMERA.nearCull);
  return Math.min(Math.max(raw, 1.5), Math.max(NEAR_PLANE_CAMERA.maxLinePixelWidth, 2.5));
}

/** Linear interpolation of two world/view-space points. */
export function mixVec3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/** World-space Euclidean length, for the fixture's `segmentLengths` column. */
export function worldLength(a: Vec3, b: Vec3): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
}

/**
 * The two segments of the scenario, in storage-slot order, with `bFar` chosen
 * by the caller: {@link NEAR_PLANE_B_FAR} for the reproducing case,
 * {@link NEAR_PLANE_CONTROL_B_FAR} for the control.
 */
export function nearPlaneSegments(bFar: Vec3 = NEAR_PLANE_B_FAR): ReadonlyArray<{
  readonly start: Vec3;
  readonly end: Vec3;
  readonly startJointCode: number;
  readonly endJointCode: number;
}> {
  return [
    {
      start: NEAR_PLANE_A_START,
      end: NEAR_PLANE_SHARED,
      startJointCode: NEAR_PLANE_JOINT_CODES.aStart,
      endJointCode: NEAR_PLANE_JOINT_CODES.aEnd,
    },
    {
      start: NEAR_PLANE_SHARED,
      end: bFar,
      startJointCode: NEAR_PLANE_JOINT_CODES.bStart,
      endJointCode: NEAR_PLANE_JOINT_CODES.bEnd,
    },
  ];
}
