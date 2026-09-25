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
 * default (#1352) must not leave authored attributes behind. It is selected
 * per session — `?linePrimitive=` override, then the `Advanced → Line
 * primitive` policy setting — with the `auto` policy additionally sizing
 * the scene once before material build ({@link sceneEffectiveLineLoad}) and
 * each node once at material build ({@link resolveLinePrimitiveForNode}).
 *
 * It lives here, in the layer-neutral `types/`, rather than in `rendering/`
 * because `config/url-params.ts` must both parse it and install it, and
 * `config/` may not import from `rendering/` (see `.dependency-cruiser.cjs`).
 * The override and policy are installed once at startup; the aggregate load
 * is replaced for each scene before any line material is constructed.
 *
 * @module types/line-primitive
 */

import type { LinesMetadata } from './lines';

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
 * - `auto` (default): capsule, except scenes whose aggregate effective
 *   segment load is very large, which build the cheaper quad — see
 *   {@link resolveLinePrimitiveForNode} for the measured rationale.
 * - `capsule` / `quad`: force one primitive for every line node.
 */
export type LinePrimitivePolicy = 'auto' | 'capsule' | 'quad';

/**
 * Every valid policy value. The single vocabulary: the settings
 * sanitizer (`config/user-settings.ts`) validates the stored value
 * against THIS list, so storage and the resolver below can never
 * disagree about what a policy is.
 */
export const LINE_PRIMITIVE_POLICIES: readonly LinePrimitivePolicy[] = ['auto', 'capsule', 'quad'];

/**
 * Auto-policy switch point, in EFFECTIVE segments (= authored segments ×
 * the width factor below), summed over concurrently drawn scene nodes.
 * Measured 2026-08-13 on a discrete NVIDIA GPU
 * (RTX PRO 6000, WebGPU timestamp-query, A/A-replicated): the capsule
 * costs ~1.5× the quad's GPU pass at every thin-line count — parallel
 * curves, no crossover — and 3.16–3.38× on wide lines. Past ~2 M thin
 * segments the capsule's GPU pass alone costs over a quarter of a 60 fps
 * frame budget on that class of GPU (4.6 ms of 16.7 ms, vs the quad's
 * 3.0 ms), while an Apple
 * GPU barely registers the difference (1.04–1.11× at 10 M, G1′). 2 M is
 * therefore a budget choice, not a crossover reading (verdict archived
 * in perf-results/1352-campaign/quad-campaign/).
 */
export const AUTO_QUAD_EFFECTIVE_SEGMENTS = 2_000_000;

/**
 * Width-factor normalization for the auto rule. Fill cost scales with
 * RENDERED width, and authored `max_width` is in world units — not
 * comparable across scenes (a nanometre scene authors 500, a normalized
 * one 0.5). The factor is therefore PROPORTIONAL to the on-screen width
 * at the opening framing (node extent fitted to a nominal viewport):
 *
 *   openingPx  = max_width / bboxDiagonal × NOMINAL_VIEWPORT_PX
 *   widthFactor = max(1, openingPx / MIN_RENDERED_WIDTH_PX)
 *
 * Deliberately order-of-magnitude, not exact: `openingPx` is the
 * authored width times a nominal px-per-unit, but the line shaders draw
 * about FOUR times that. Two factors of 2 stack — the line scale
 * `luxarLineScale` is `res.y · |P11| = res.y / tan(fov/2)`, i.e. twice the true px-per-unit conversion,
 * and the shader then consumes the result as the quad's HALF-extent
 * (`aQuadCorner.y ∈ {-1,+1}` in `shader-glsl.ts`) — on top of which each
 * primitive draws its own support multiple (see
 * `_shared/line-capsule.ts`). Left uncorrected on purpose: a ~4× small
 * factor keeps the capsule longer, which is the right bias for a quality
 * default. An nD bounds diagonal (extra non-spatial dims) only grows the
 * denominator, which is conservative in the same direction. A node
 * TRANSFORM, on the other hand, does NOT cancel: both terms are
 * authored, so the ratio itself is transform-free — but the rendered
 * width is not. The shader converts `width` against the VIEW-space depth
 * (`width * luxarLineScale / dist` in `shader-glsl.ts`) without
 * the model matrix, while the extent that sets that depth carries it. A
 * node scaled by s therefore draws s× thinner, relative to its own
 * extent, than this estimate says: a scaled-up node can reach the
 * threshold earlier than its true rendered width warrants (the ~4×
 * slack above absorbs the first two octaves of that), a scaled-down one
 * keeps the capsule longer. Order-of-magnitude, as stated.
 *
 * The width term is the node's MAXIMUM authored width, and that is the
 * one deviation biased the other way: a node of a million hairlines
 * carrying one fat line reads as a million fat lines and can flip to the
 * quad on fill cost it never pays. Deliberate — `.zattrs` records
 * `max_width` and no width distribution — and the cost of being wrong
 * here is the cheaper primitive on an unusual scene, never a broken one.
 *
 * Floored at 1 because the shader clamps thin lines to `minPixelWidth`
 * (1.5 px in shader-glsl.ts) — below the clamp, fill cost stops
 * shrinking with width. With the 4× above, the floor in fact holds until
 * a line draws roughly 6 px rather than releasing exactly at the clamp;
 * same conservative direction. Measured: capsule GPU cost grew
 * ~2.8× from the thin bench arms to the width-3 arms at equal count,
 * i.e. ~linearly in rendered width, which is what a linear factor
 * models. Nodes without authored bounds fall back to factor 1
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

/**
 * Aggregate effective line load for the scene currently being constructed.
 * Production owns one loader under the `default` id. Multi-loader embeddings
 * are therefore last-load-wins, and disposing any loader resets this shared
 * value.
 */
let sessionSceneLineLoad = 0;

/** Install the session policy (call once from bootstrap). */
export function setLinePrimitivePolicy(policy: LinePrimitivePolicy): void {
  sessionPolicy = policy;
}

/** Install the current scene's aggregate effective line load. */
export function setSceneLineLoad(load: number): void {
  sessionSceneLineLoad = Number.isFinite(load) && load > 0 ? load : 0;
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

/** Scene-tree shape needed by {@link sceneEffectiveLineLoad}. */
export interface SceneLineLoadNode {
  type: string;
  attrs: Record<string, unknown>;
  children?: readonly SceneLineLoadNode[];
}

/** Diagonal of one authored min/max pair, or `undefined` when unusable. */
function boundsDiagonal(min: unknown, max: unknown): number | undefined {
  if (!Array.isArray(min) || !Array.isArray(max) || min.length !== max.length || !min.length) {
    return undefined;
  }
  const diag = Math.hypot(...max.map((hi, i) => hi - min[i]));
  return Number.isFinite(diag) && diag > 0 ? diag : undefined;
}

/**
 * Extract the stable authored load inputs used by both scene aggregation and
 * per-node material resolution. Vertex ordering is the tight D-space extent;
 * segment ordering is a conservative √2-large fallback; `position_bounds`
 * keeps width normalization available for unindexed nodes. Never substitute
 * the worker's projected bounds: the streaming path builds the material on an
 * empty placeholder mesh before those bounds exist.
 */
export function lineNodeLoadFromAttrs(attrs: Partial<LinesMetadata>): LineNodeLoad {
  const ordering = attrs.vertex_ordering ?? attrs.segment_ordering;
  const indexed = ordering && boundsDiagonal(ordering.ordering_min, ordering.ordering_max);
  return {
    nSegments: attrs.n_segments,
    maxWidth: attrs.max_width,
    bboxDiagonal: indexed ?? boundsDiagonal(attrs.position_bounds?.min, attrs.position_bounds?.max),
  };
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
 * Fold a scene graph into the effective line load that can be drawn
 * concurrently. Plain groups and partitions sum their children; LOD groups
 * take the maximum because their levels are substitutive. Two levels can
 * overlap briefly during a cross-fade; accepting that at-most-2× transient
 * undercount keeps the frozen default biased toward capsule quality. The max
 * also treats every independent ladder as if it reached its finest level at
 * once, which can over-count a normally framed scene and choose the lower-
 * quality quad. Using each ladder's `default_level` instead was rejected
 * because the decision must remain safe after view-driven level changes. A
 * lines node is a leaf for this purpose: additive line ladders already
 * advertise their summed total in the parent node's authored `n_segments`.
 */
export function sceneEffectiveLineLoad(node: SceneLineLoadNode): number {
  if (node.type === 'lines') {
    return effectiveSegmentLoad(lineNodeLoadFromAttrs(node.attrs as Partial<LinesMetadata>));
  }

  const childLoads = node.children?.map(sceneEffectiveLineLoad) ?? [];
  if (node.type === 'group' && node.attrs.kind === 'lod') {
    return childLoads.reduce((maximum, load) => Math.max(maximum, load), 0);
  }
  return childLoads.reduce((total, load) => total + load, 0);
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
 * (`capsule` / `quad` setting) > the `auto` rule (quad when the greater of
 * this node's {@link effectiveSegmentLoad} and the installed scene load is ≥
 * {@link AUTO_QUAD_EFFECTIVE_SEGMENTS}, capsule otherwise).
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
  return Math.max(effectiveSegmentLoad(load), sessionSceneLineLoad) >= AUTO_QUAD_EFFECTIVE_SEGMENTS
    ? 'screen-space'
    : DEFAULT_LINE_PRIMITIVE;
}
