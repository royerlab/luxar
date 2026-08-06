/**
 * Line join style — the strategy the vertex stage uses at a degree-2 polyline
 * joint (issue #790).
 *
 * Without join geometry, two segments meeting at a turn of angle θ leave an
 * uncovered circular sector of that angle on the OUTSIDE of the bend and
 * double-cover a lens on the inside: measurably, dark ticks along the convex
 * edge of a thick curve (worst −199/255 on a 2-world-unit tube) and bright
 * ticks along the concave one. No per-endpoint intensity scalar can close the
 * outer wedge — nothing rasterises there to shade — so it needs geometry.
 *
 * | style   | per-vertex cost             | wedge      | blending modes    |
 * | ------- | --------------------------- | ---------- | ----------------- |
 * | `none`  | zero                        | left open  | n/a               |
 * | `miter` | + 1 texel fetch, 1 project  | **exact**  | all six, by construction |
 *
 * `miter` is the only style that fits the existing 4-vertices-per-segment quad
 * exactly: the mitred trapezoid's edges stay on the segment's own ±R offset
 * lines, so the perpendicular super-Gaussian cross-section is untouched.
 * `round` and `bevel` both need extra geometry, and an `overlap` variant (extend
 * the quads so they genuinely overlap, cross-fading the axial ramps) was
 * considered and dropped: it additionally needs the fragment stage (a signed cap
 * ramp and `vT` reparameterised over the extended span), it is exact only under
 * the sum-like blending modes, and it would lengthen every free polyline end.
 *
 * Both styles read the same per-endpoint joint code (`compute_joint_codes` in
 * `wasm/rust/src/lines_clipping.rs`); `none` simply ignores it.
 *
 * @module types/line-join
 */

/** Authored / selectable join styles. */
export type LineJoinStyle = 'none' | 'miter';

/**
 * The default when nothing is authored or overridden.
 *
 * `miter` is exact and blending-mode agnostic, leaves free polyline ends
 * untouched, and is gated in-shader by both a rendered-width threshold and a
 * miter limit — so every existing scene improves without re-authoring. Measured
 * overhead is 0 on thin-line scenes (the width gate skips them, and they are
 * the million-segment ones) and +0.1–0.2 ms/frame at 800k thick segments.
 */
export const DEFAULT_LINE_JOIN: LineJoinStyle = 'miter';

/**
 * Numeric encoding handed to the `uLineJoin` uniform. Kept in lockstep with the
 * comparison in the shader join block (`> 0.5` selects miter), and identical
 * across the GLSL and TSL backends so the parity harness compares like with
 * like.
 */
export const LINE_JOIN_UNIFORM: Readonly<Record<LineJoinStyle, number>> = {
  none: 0,
  miter: 1,
};

/** Every valid style, for validation and for error messages. */
export const LINE_JOIN_STYLES: readonly LineJoinStyle[] = ['none', 'miter'];

/**
 * Parse a join style from untrusted text (a URL parameter or an authored zarr
 * attribute). Returns `null` for anything unrecognised so the caller can decide
 * between falling back and warning — an unknown value must never silently mean
 * "no joins".
 */
export function parseLineJoinStyle(raw: string | null | undefined): LineJoinStyle | null {
  if (raw === null || raw === undefined) return null;
  const v = raw.trim().toLowerCase();
  return (LINE_JOIN_STYLES as readonly string[]).includes(v) ? (v as LineJoinStyle) : null;
}

/**
 * Session-wide override, set once from `?lineJoin=`. `null` means "no override",
 * which is distinct from `'none'` (an explicit request for no join geometry).
 *
 * A session override rather than a per-node value because its job is debugging
 * and workaround: it must be able to force a style onto a scene whose author
 * chose otherwise. Precedence is therefore
 * `?lineJoin=` > authored node attribute > {@link DEFAULT_LINE_JOIN}.
 *
 * It lives here, in the layer-neutral `types/`, rather than in `rendering/`
 * because `config/url-params.ts` must both parse it and install it, and
 * `config/` may not import from `rendering/` (see `.dependency-cruiser.cjs`).
 * Session-constant, set once at startup — the same shape as the `?dpr=` pin.
 */
let sessionOverride: LineJoinStyle | null = null;

/** Install the session override (idempotent; call once from URL-param apply). */
export function setLineJoinOverride(style: LineJoinStyle | null): void {
  sessionOverride = style;
}

/**
 * Resolve the style for one node, applying the documented precedence, and
 * return the uniform value the shader consumes.
 *
 * @param authored - the node's authored style, if the scene specified one
 */
export function resolveLineJoin(authored?: LineJoinStyle | null): number {
  return LINE_JOIN_UNIFORM[sessionOverride ?? authored ?? DEFAULT_LINE_JOIN];
}

/**
 * Invert {@link LINE_JOIN_UNIFORM} — recover the style from a `uLineJoin`
 * uniform value.
 *
 * The two backends encode the style differently on purpose: GLSL keeps it a
 * runtime uniform (a `?lineJoin=` override must not recompile a program) while
 * TSL bakes it into the graph, the same asymmetry the line factories already
 * have for `uIsOrtho`. A `ShaderSource.webgpu` factory is handed the GLSL-shaped
 * uniform RECORD, so it needs this to pick the matching graph variant.
 *
 * `undefined` (no such uniform in the record) returns `undefined`, which the
 * factories treat as "unauthored" and resolve through the normal precedence —
 * NOT as `'none'`.
 */
export function lineJoinStyleFromUniform(value: number | undefined): LineJoinStyle | undefined {
  if (value === undefined) return undefined;
  return value > 0.5 ? 'miter' : 'none';
}
