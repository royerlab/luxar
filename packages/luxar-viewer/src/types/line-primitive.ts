/**
 * Line rendering primitive — which fragment model draws a line segment
 * (issue #1352).
 *
 * | primitive      | model                                                    |
 * | -------------- | -------------------------------------------------------- |
 * | `screen-space` | today's flat quad: perpendicular super-Gaussian profile, |
 * |                | screen-space width, degenerate end-on                    |
 * | `volumetric`   | segment ⊛ isotropic 3D Gaussian: the quad is only a      |
 * |                | rasterization stencil; every shading quantity is solved  |
 * |                | per-fragment against the true camera-space segment, so   |
 * |                | end-on viewing is exact and a joint's two cells partition |
 * |                | the bend at a bisector plane instead of overlapping      |
 * | `capsule`      | gaussian-like profile of the 2D point-to-segment distance |
 * |                | in pixel space: direction-stable near-axial (end-on is a  |
 * |                | radial disc), 2D bisector-cut joins, quad-class cost —    |
 * |                | see `_shared/line-capsule.ts` for the model + the three   |
 * |                | deliberate exactness relaxations                          |
 *
 * `volumetric` and `capsule` are calibrated so the side-on appearance
 * matches `screen-space` by construction (the shared Gaussian-equivalent
 * truncation T relates drawn width to σ; see `_shared/line-volumetric.ts`
 * and `_shared/line-capsule.ts`), which is what makes a session-wide A/B
 * meaningful.
 *
 * Unlike `lineJoin` this is NOT an authorable node attribute: the primitive is
 * a renderer implementation choice, not scene content, and the flip to a new
 * default (#1352) must not leave authored attributes
 * behind. It is a session-wide toggle only, set once from `?linePrimitive=`.
 *
 * It lives here, in the layer-neutral `types/`, rather than in `rendering/`
 * because `config/url-params.ts` must both parse it and install it, and
 * `config/` may not import from `rendering/` (see `.dependency-cruiser.cjs`).
 * Session-constant, set once at startup — the same shape as `?lineJoin=`.
 *
 * @module types/line-primitive
 */

/** Selectable line primitives. */
export type LinePrimitive = 'screen-space' | 'volumetric' | 'capsule';

/**
 * The default when nothing is overridden. Flipped to `capsule` after the
 * #1352 re-gate passed (2026-08-11: ≤1.09× the quad on the 10M worst
 * case, parity at vsync on fills, visual sign-off on the QA grid at any
 * zoom). `screen-space` and `volumetric` remain selectable via
 * `?linePrimitive=` until their scheduled deletion.
 */
export const DEFAULT_LINE_PRIMITIVE: LinePrimitive = 'capsule';

/** Every valid primitive, for validation and for error messages. */
export const LINE_PRIMITIVES: readonly LinePrimitive[] = ['screen-space', 'volumetric', 'capsule'];

/**
 * Parse a primitive from untrusted text (the `?linePrimitive=` URL
 * parameter). Returns `null` for anything unrecognised so the caller can
 * decide between falling back and warning — an unknown value must never
 * silently select a primitive.
 */
export function parseLinePrimitive(raw: string | null | undefined): LinePrimitive | null {
  if (raw === null || raw === undefined) return null;
  const v = raw.trim().toLowerCase();
  return (LINE_PRIMITIVES as readonly string[]).includes(v) ? (v as LinePrimitive) : null;
}

/**
 * Session-wide override, set once from `?linePrimitive=`. `null` means "no
 * override" (use {@link DEFAULT_LINE_PRIMITIVE}).
 *
 * Installed by `core/bootstrap.ts` before any line material is constructed —
 * both backends bake the primitive at material build time (the GLSL factory
 * picks a shader-source pair, the TSL factory a graph), so a late install
 * would silently miss already-built materials.
 */
let sessionOverride: LinePrimitive | null = null;

/** Install the session override (idempotent; call once from URL-param apply). */
export function setLinePrimitiveOverride(primitive: LinePrimitive | null): void {
  sessionOverride = primitive;
}

/**
 * Resolve the primitive to build, applying `?linePrimitive=` >
 * {@link DEFAULT_LINE_PRIMITIVE}.
 *
 * @param explicit - a caller-supplied primitive that bypasses the session
 *   override. Test harnesses (the GLSL/TSL parity page) never run
 *   `bootstrap.ts`, so fixtures pass the primitive explicitly instead of
 *   relying on module state.
 */
export function resolveLinePrimitive(explicit?: LinePrimitive | null): LinePrimitive {
  return explicit ?? sessionOverride ?? DEFAULT_LINE_PRIMITIVE;
}
