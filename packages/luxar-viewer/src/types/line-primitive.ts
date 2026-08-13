/**
 * Line rendering primitive — which fragment model draws a line segment
 * (issue #1352).
 *
 * | primitive      | model                                                    |
 * | -------------- | -------------------------------------------------------- |
 * | `screen-space` | today's flat quad: perpendicular super-Gaussian profile, |
 * |                | screen-space width, degenerate end-on                    |
 * | `capsule`      | gaussian-like profile of the 2D point-to-segment distance |
 * |                | in pixel space: direction-stable near-axial (end-on is a  |
 * |                | radial disc), 2D bisector-cut joins, quad-class cost —    |
 * |                | see `_shared/line-capsule.ts` for the model + the three   |
 * |                | deliberate exactness relaxations                          |
 *
 * `capsule` is calibrated so the side-on appearance matches
 * `screen-space` by construction (the Gaussian-equivalent truncation T in
 * `_shared/line-capsule.ts` relates drawn width to σ), which is what makes
 * a session-wide A/B meaningful.
 *
 * A third primitive, `volumetric` (segment ⊛ isotropic 3D Gaussian, solved
 * per-fragment against the true camera-space segment), shipped behind this
 * toggle during #1352 and was deleted after the capsule flip: the capsule
 * matched or beat it visually — including near-axial, its signature case —
 * at quad-class cost. Its closed-form ray-integral math survives in git
 * history (branch point `1481995d9`).
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
export type LinePrimitive = 'screen-space' | 'capsule';

/**
 * The default when nothing is overridden. Flipped to `capsule` after the
 * #1352 re-gate passed (2026-08-11: ≤1.09× the quad on the 10M worst
 * case, parity at vsync on fills, visual sign-off on the QA grid at any
 * zoom). `screen-space` remains selectable via `?linePrimitive=`.
 */
export const DEFAULT_LINE_PRIMITIVE: LinePrimitive = 'capsule';

/** Every valid primitive, for validation and for error messages. */
export const LINE_PRIMITIVES: readonly LinePrimitive[] = ['screen-space', 'capsule'];

/**
 * Session-wide primitive POLICY (the `Advanced → Line primitive` user
 * setting, #1352 follow-up). Deliberately a separate vocabulary from
 * {@link LinePrimitive}: the policy names the user-facing choice
 * (`quad` reads better than the internal `screen-space`, and `auto` is
 * not a primitive at all). `quad` maps to `'screen-space'` at
 * resolution; `parseLinePrimitive` still deliberately rejects `quad` for
 * `?linePrimitive=` — the URL parameter speaks the primitive vocabulary.
 *
 * - `auto` (default): capsule, except nodes whose effective segment
 *   load is very large, which build the cheaper quad — see
 *   {@link resolveLinePrimitiveForNode} for the measured rationale.
 * - `capsule` / `quad`: force one primitive for every line node.
 */
export type LinePrimitivePolicy = 'auto' | 'capsule' | 'quad';

/** Every valid policy value, for settings validation. */
export const LINE_PRIMITIVE_POLICIES: readonly LinePrimitivePolicy[] = ['auto', 'capsule', 'quad'];

/**
 * Auto-policy switch point, in EFFECTIVE segments (= authored segments ×
 * the width factor below). Measured 2026-08-13 on a discrete NVIDIA GPU
 * (RTX PRO 6000, WebGPU timestamp-query, A/A-replicated): the capsule
 * costs ~1.5× the quad's GPU pass at every thin-line count — parallel
 * curves, no crossover — and 3.2–3.4× on wide lines. Past ~2 M thin
 * segments the capsule's EXTRA cost alone exceeds a quarter of a 60 fps
 * frame budget on that class of GPU (4.6 ms vs 3.0 ms), while an Apple
 * GPU barely registers the difference (1.04–1.11× at 10 M, G1′). 2 M is
 * therefore a budget choice, not a crossover reading (verdict archived
 * in perf-results/1352-campaign/quad-campaign/).
 */
export const AUTO_QUAD_EFFECTIVE_SEGMENTS = 2_000_000;

/**
 * Width-factor normalization for the auto rule. Fill cost scales with
 * RENDERED width, and authored `max_width` is in world units — not
 * comparable across scenes (a nanometre scene authors 500, a normalized
 * one 0.5). The factor therefore estimates the width in PIXELS at the
 * opening framing (node extent fitted to a nominal viewport):
 *
 *   openingPx  = max_width / bboxDiagonal × NOMINAL_VIEWPORT_PX
 *   widthFactor = max(1, openingPx / MIN_RENDERED_WIDTH_PX)
 *
 * floored at 1 because the shader clamps thinner lines to
 * `minPixelWidth` (1.5 px in shader-glsl.ts) — below the clamp, fill
 * cost stops shrinking with width. Measured: capsule GPU cost grew
 * ~2.8× from the thin bench arms to the width-3 arms at equal count,
 * i.e. ~linearly in rendered width, which is what a linear factor
 * models. Nodes without projection bounds fall back to factor 1
 * (count-only) rather than guessing.
 */
export const NOMINAL_VIEWPORT_PX = 1024;
/** Shader-side minimum rendered pixel width (mirrors `minPixelWidth`). */
export const MIN_RENDERED_WIDTH_PX = 1.5;

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
 * Session-wide policy, installed once by `core/bootstrap.ts` from the
 * `advanced.linePrimitivePolicy` user setting — BEFORE any line material
 * is constructed, same ordering contract as the override above. `auto`
 * is both the setting default and the uninstalled default, so harnesses
 * that never run bootstrap behave exactly as before this setting existed.
 */
let sessionPolicy: LinePrimitivePolicy = 'auto';

/** Install the session policy (call once from bootstrap). */
export function setLinePrimitivePolicy(policy: LinePrimitivePolicy): void {
  sessionPolicy = policy;
}

/** Parse a policy from untrusted text (settings storage). */
export function parseLinePrimitivePolicy(
  raw: string | null | undefined
): LinePrimitivePolicy | null {
  if (raw === null || raw === undefined) return null;
  const v = raw.trim().toLowerCase();
  return (LINE_PRIMITIVE_POLICIES as readonly string[]).includes(v)
    ? (v as LinePrimitivePolicy)
    : null;
}

/**
 * Per-node size information for the `auto` policy, gathered where the
 * material is BUILT (`createLinesNode`). Fields come from the authored
 * `.zattrs` — `n_segments` is the stable authored total (the streaming
 * path's `processed.segmentCount` is 0 at build time and only grows
 * afterwards, so it must never be the policy input), `max_width` /
 * `bboxDiagonal` feed the width factor. All optional: a caller with no
 * size information (harnesses, the debug HUD) gets the plain
 * session-wide resolution.
 */
export interface LineNodeLoad {
  /** Authored segment total (`attrs.n_segments`). */
  nSegments?: number;
  /** Authored maximum line width in world units (`attrs.max_width`). */
  maxWidth?: number;
  /** Diagonal of the node's own bounding box, world units. */
  bboxDiagonal?: number;
}

/**
 * The `auto` rule's effective segment load: authored segments × a
 * rendered-width factor (see {@link NOMINAL_VIEWPORT_PX} for the
 * normalization and its measured basis). Exported for tests.
 */
export function effectiveSegmentLoad(load: LineNodeLoad): number {
  const segments = load.nSegments ?? 0;
  if (!Number.isFinite(segments) || segments <= 0) return 0;
  const { maxWidth, bboxDiagonal } = load;
  let widthFactor = 1;
  if (
    maxWidth !== undefined &&
    bboxDiagonal !== undefined &&
    Number.isFinite(maxWidth) &&
    Number.isFinite(bboxDiagonal) &&
    maxWidth > 0 &&
    bboxDiagonal > 0
  ) {
    const openingPx = (maxWidth / bboxDiagonal) * NOMINAL_VIEWPORT_PX;
    widthFactor = Math.max(1, openingPx / MIN_RENDERED_WIDTH_PX);
  }
  return segments * widthFactor;
}

/**
 * Resolve the primitive to build, applying `?linePrimitive=` >
 * {@link DEFAULT_LINE_PRIMITIVE}. A forced (non-`auto`) policy replaces
 * the default; the URL parameter stays the strongest override (the
 * explicit A/B escape hatch).
 *
 * @param explicit - a caller-supplied primitive that bypasses the session
 *   override. Test harnesses (the GLSL/TSL parity page) never run
 *   `bootstrap.ts`, so fixtures pass the primitive explicitly instead of
 *   relying on module state.
 */
export function resolveLinePrimitive(explicit?: LinePrimitive | null): LinePrimitive {
  return explicit ?? sessionOverride ?? forcedPolicyPrimitive() ?? DEFAULT_LINE_PRIMITIVE;
}

/** A forced policy's primitive, or `null` when the policy is `auto`. */
function forcedPolicyPrimitive(): LinePrimitive | null {
  if (sessionPolicy === 'capsule') return 'capsule';
  if (sessionPolicy === 'quad') return 'screen-space';
  return null;
}

/**
 * Resolve the primitive for ONE lines node, with its size in hand — the
 * seam every production material build goes through (visual + picking,
 * both backends), so the two footprints agree by construction.
 *
 * Precedence: `?linePrimitive=` (explicit escape hatch) > forced policy
 * (`capsule` / `quad` setting) > the `auto` rule (quad when
 * {@link effectiveSegmentLoad} ≥ {@link AUTO_QUAD_EFFECTIVE_SEGMENTS},
 * capsule otherwise).
 *
 * The result must be resolved ONCE per node, at first material build,
 * and carried on `userData.linePrimitive` from then on (clones, the
 * node-factory retro picking pass, TSL graph rebuilds): the TSL wrappers
 * rebuild their graphs on camera-mode flips, and a re-run of a policy
 * that read live state could silently swap a node's primitive
 * mid-session.
 */
export function resolveLinePrimitiveForNode(load: LineNodeLoad): LinePrimitive {
  const forced = sessionOverride ?? forcedPolicyPrimitive();
  if (forced) return forced;
  return effectiveSegmentLoad(load) >= AUTO_QUAD_EFFECTIVE_SEGMENTS
    ? 'screen-space'
    : DEFAULT_LINE_PRIMITIVE;
}
