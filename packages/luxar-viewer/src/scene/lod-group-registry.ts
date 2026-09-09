/**
 * Per-frame LOD-group selector.
 *
 * Tracks every `lod_group` scene-graph node currently loaded. For each
 * one, every frame:
 *
 *   1. Fold each child's raw nD ``positionBounds`` into a cached per-entry
 *      **local-space** :type:`BoundingBox`, using the current ``displayDims``
 *      to map nD axes onto X/Y/Z, then transform it to world space for the
 *      frustum gate. Eviction uses the same full-geometry box.
 *   2. When any child publishes optional robust ``lodBounds``, fold them the
 *      same way (falling back per child to ``positionBounds``) for metric
 *      sizing only, so excluded outliers remain visible and resident.
 *   3. Transform the metric box into world space via
 *      :func:`transformBoundingBox` and the lod_group's ``matrixWorld``.
 *   4. Project the 8 corners through the camera and reduce them to the
 *      dimensionless **coverage metric**, on whichever scale the entry's
 *      ``selector`` names — the two branches of ``evaluateEntry``:
 *      - ``'screen-area'`` (what every derived ladder stamps): the fraction of
 *        the viewport the projected AABB covers by AREA, via
 *        {@link projectBoxAreaFraction}. Aspect-free, tops out at 1.0.
 *      - ``'coverage'`` (legacy): back to pixel coordinates, then the diagonal
 *        of the screen-space AABB divided by ``FILL_FACTOR × fittedAxisPx``
 *        (``fittedAxisPx`` is ``min(viewport.width, viewport.height)`` — the
 *        extent ``calculateCameraDistance`` actually fits; see the
 *        ``FILL_FACTOR`` doc), so 1.0 == the object's projected diagonal has
 *        reached ``FILL_FACTOR`` of the fitted axis.
 *   5. Pick the **finest** child whose ``coverage_fraction`` threshold is
 *      satisfied by that coverage metric, with 10% asymmetric hysteresis on
 *      the downgrade direction to suppress threshold-edge flicker.
 *   6. If the desired child differs from the current active one, swap
 *      visibility atomically — gated by the **never-downgrade display
 *      gate**: a fresh aspiration whose additive ladder is still streaming
 *      is not shown while the previously-displayed level looks strictly
 *      better (see ``shouldHoldPreviousDisplay`` in ``lod-display-gate.ts``).
 *
 * The atomic-swap invariant on initial load is realized by
 * ``loadLodGroupNode`` (sequential awaits + ``visible=false`` after
 * attach + a single ``register()`` call at the end) — no per-child
 * ``ready`` gate is needed in the registry.
 *
 * **Partition frustum gating** (``registerPartition`` / ``evaluatePartitionEntry``):
 * a ``kind=partition`` group's parts are tested every frame against a frustum
 * padded by ``PARTITION_FRUSTUM_MARGIN``; a part outside it is hidden and
 * stamped ``userData.partitionFrustumVisible = false``, which the scene
 * loader's sweep and refinement read to skip its loaders. Because a culled part
 * misses slice updates, its RE-ENTRY requests a resync of exactly that part's
 * loaders (``deps.requestReprocess(partPaths)``, coalesced per wrapper across
 * frames and gated on ``isUpdateInProgress``). The resync re-sweeps under the
 * UNCHANGED view version — bumping it here would read every lazy fine level
 * scene-wide as stale and drop all groups to coarse on camera motion.
 *
 * The bbox infrastructure is shared with the scene-bounds cache and
 * camera framing — ``projectBoundsToDisplayDims`` /
 * ``transformBoundingBox`` live in
 * ``scene-manager/clipping/bounds-math.ts`` and are reused here.
 *
 * Wiring: the SceneLoader instantiates one registry per scene; the
 * pipeline hooks ``evaluatePerFrame`` into ``AnimationController``
 * alongside the dynamic-clipping callback. Manual override
 * (``setSelectorMode(path, { lockLevel: i })``) bypasses the auto
 * selector — driven by the layers-panel dropdown.
 *
 * @module scene/lod-group-registry
 */

import * as THREE from 'three';

import type { BoundingBox } from './scene-manager/clipping/bounds-math';
import { log, Modules } from '../utils/log';
import { isEffectivelyVisible } from '../utils/object-visibility';
import type { LODGroupSelectorMode } from '../types/lod-group';
import {
  isFresh,
  isReady,
  isTrackedLeaf,
  SettleTracker,
  visibleElementCount,
} from './lod-freshness';
import {
  shouldHoldPreviousDisplay,
  subtreeDisplayProgress,
  type ProgressNode,
} from './lod-display-gate';
import { coverageBlendPlan } from './lod-blend';
import { applyLodFade, FADE_EPSILON, isBlendableSubtree } from './lod-fade';
import {
  computeEntryWorldBox,
  pickChildWithHysteresis,
  projectBoxAreaFraction,
  projectBoxDiagonalPx,
  type WorldBoxOptions,
} from './lod-selector-math';
import { enforceResidentByteBudget } from './lod-eviction';

// The selector math (box projection + hysteresis pick) lives in
// `lod-selector-math.ts`; re-exported here so existing importers (the
// selector unit tests) keep their import site.
export {
  pickChildWithHysteresis,
  projectBoxAreaFraction,
  projectBoxDiagonalPx,
} from './lod-selector-math';

/**
 * Cross-fade band half-width as a FRACTION of the local inter-level gap. The two
 * levels straddling a boundary blend while the coverage metric is within
 * `±fraction·min(adjacent gaps)` of it; outside, a single level renders. Because
 * the `coverage_fraction` thresholds are geometrically spaced, a proportional
 * band keeps the dissolve the same fraction of a step at every level (a constant
 * width would be over-wide at the coarse end). ~0.4 → the middle ~20% of each
 * step is crisp single-level, the rest a dissolve; `< 0.5` guarantees no
 * overlapping bands. See `coverageBlendPlan`.
 */
const CROSSFADE_BAND_FRACTION = 0.4;

/**
 * Frames the view-update version must hold steady before the registry reloads a
 * stale fine level (the settle debounce — see `maybeKickReload`). While the
 * user is actively scrubbing (version changes every frame) only the cheap
 * coarse level shows; the fine level reloads once they pause for ~this many
 * frames. ~8 frames ≈ 130 ms at 60 fps.
 */
const FINE_RELOAD_SETTLE_TICKS = 8;

/**
 * Milliseconds the registry will keep a STALE previously-displayed level on screen,
 * rather than dropping to a much coarser fresh one, while the aspiration
 * re-commits for a new slice (see ``staleHoldDisplayIndex``).
 *
 * The slice-aware fallback below shows the coarsest FRESH level the instant a
 * scrub invalidates the aspiration. That is right when the aspiration is
 * seconds away, and wrong when it is milliseconds away from a warm cache: on a
 * 151-timepoint gsplat timelapse the coarse level re-commits in ~10 ms and the
 * finest in ~70 ms, so every single step of the Time slider flashed 6,900
 * splats down to 108 and back — 1.6% of the detail, for four frames. A video
 * player holds the previous frame until the next one decodes; so does this.
 *
 * Bounded, because holding is only better while the wait is short. The budget
 * is spent from when the hold STARTS and is not refreshed by further version
 * bumps, so a continuous drag (a new version every frame, the aspiration never
 * committing) exhausts it once and then shows live coarse geometry exactly as
 * before.
 *
 * In MILLISECONDS, deliberately, unlike the frame-counted debounce above: this
 * is a tolerance for how long a viewer may show the previous slice, which is a
 * wall-clock judgement, not a "has the user stopped moving" one. Sizing it in
 * frames makes it display-dependent — the first version of this fix used 8
 * frames and worked on a 60 Hz panel while still flashing on a 165 Hz one,
 * where 8 frames is 48 ms and the re-commit needs ~70.
 *
 * 250 ms is comfortably above the ~70 ms a warm re-slice takes on a 1.6 M-splat
 * timelapse and well below the point where a frozen frame reads as a hang.
 */
const STALE_HOLD_MS = 250;

/**
 * How much worse the coarse fresh fallback must be, as a fraction of the held
 * level's committed element count, before holding a STALE finer level is worth
 * it (see ``staleHoldDisplayIndex``).
 *
 * Freshness normally wins: showing the right slice matters more than showing
 * more geometry. The exception this ratio carves out is the case where the
 * fallback is not a slightly coarser view of the new slice but a token of it —
 * the 108-of-6,900-splat drop that made a timelapse step read as a flash. At
 * half the detail or better the fallback is taken immediately, as before.
 */
const STALE_HOLD_MIN_RATIO = 0.5;

/**
 * Unit anchor for the LEGACY ``selector: 'coverage'`` thresholds (older
 * stores, and explicitly authored ``coverage_fractions=[...]`` lists — derived
 * ladders now use ``selector: 'screen-area'``, whose metric is
 * ``projectBoxAreaFraction`` and does not involve this constant): a threshold
 * of 1.0 is satisfied once the group's projected bbox diagonal reaches
 * ``FILL_FACTOR × fittedAxisPx`` pixels — i.e. HALF of the **fitted screen
 * axis** (``fittedAxisPx = min(viewport.width, viewport.height)``), which a
 * normal full-frame view already exceeds. Coarser children (smaller
 * fractions) step in as the object shrinks below that. Lowering the factor
 * shows finer levels sooner, raising it later.
 *
 * **Why the fitted axis, not the viewport diagonal (#1410).** The previous
 * anchor normalised by ``hypot(viewport.width, viewport.height)``, which grows
 * with width regardless of aspect, while ``calculateCameraDistance`` fits the
 * VERTICAL fov for aspect ≥ 1 (camera distance has NO aspect dependence there)
 * and the HORIZONTAL fov for aspect < 1. So a wide-but-not-tall canvas grew the
 * denominator without the framing showing any more of the object, and the raw
 * ratio collapsed as the canvas widened — a cube's opening-framing metric fell
 * from 3.43 at 1:1 to 1.31 at 32:9 under the old scheme, and thinner shapes (an
 * in-plane rod, a flat pancake) dropped BELOW the finest threshold entirely on
 * an ultrawide monitor (#1361's blur, returning at wide aspects).
 *
 * ``fittedAxisPx`` is exactly the extent ``calculateCameraDistance`` fits in
 * each regime — ``height`` for aspect ≥ 1, ``width`` for aspect < 1. The fit
 * uses the larger X/Y extent at the box's nearest face: ``halfDepth +
 * inPlane/(2·fitRatio·tan(halfFov))`` (and divides the second term by aspect in
 * portrait). The near-face distance is therefore proportional to the fitted
 * axis in both regimes, so the projected pixel diagonal divided by
 * ``fittedAxisPx`` is EXACTLY invariant across aspect and absolute viewport
 * size for the default centre fit modelled here. Preserving an authored
 * off-centre controls target changes which depth face bounds each side of the
 * projected rectangle. The identity remains exact while the target lies inside
 * the box's screen-plane footprint and at or behind its near face (``target.z
 * <= box.max.z``). It degrades when either condition is violated: for an
 * 8×8×100 box, a target 20 units off-axis drifts 23.8%, while a centred target
 * 10 units in front of the near face drifts 50.8%. The test matrix verifies the
 * centre-fit identity to 9 decimal digits for a cube, pancake, in-plane rod,
 * UMAP-like box, and a 1×1×100 view-axis rod from 1:4 portrait through 32:9
 * ultrawide.
 *
 * Why 0.5 and not 1.0: at 1.0 the finest level only activates once the object
 * OVERFILLS the fitted axis, reproducing the original #1361 symptom at the
 * default opening framing. Measured opening-framing ``diagonalPx /
 * fittedAxisPx`` across all five shapes and seven aspect ratios in the test
 * matrix ranges **0.750 (in-plane rod) – 1.061 (cube, pancake, and view-axis
 * rod)**. Dividing by 0.5 turns that into a metric of **1.50 – 2.12** — past
 * the finest threshold of 1.0 with **50% headroom in the worst case**. The
 * factor remains necessary: at 1.0 the in-plane rod would still open below the
 * finest rung even though the framing itself is now aspect-exact.
 *
 * **Coupled constant.** Python's ``MAX_COVERAGE_FRACTION`` (the upper bound on
 * any ``coverage_fraction``, authored or derived — a partition-bound ladder
 * derives exactly this value; see ``core/group/lod/group.py``) stays ``4.0``,
 * re-expressed as ``SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR`` rather than
 * ``1 / FILL_FACTOR``: a screen-filling object's projected diagonal is no
 * longer exactly the fitted axis — that identity only held for the OLD
 * diagonal normalisation, where a screen-filling box's diagonal trivially
 * equals the viewport's own diagonal. Under the fitted-axis normalisation it is
 * instead ``hypot(aspect, 1) / min(aspect, 1)`` times the fitted axis —
 * aspect-DEPENDENT, not a constant — measured **1.41 at 1:1, 1.67 at 4:3, 1.80
 * at 3:2, 1.89 at 16:10, 2.04 at 16:9, 2.57 at a real 21:9 panel (2560×1080),
 * 3.69 at 32:9**. ``SCREEN_FILL_DIAGONAL_RATIO`` does NOT pin an average across
 * that spread — it is anchored specifically at **16:9, the reference aspect**
 * (2.04, rounded to a plain ``2``), the same reference the ``FILL_FACTOR``
 * value 0.5 above is calibrated at. Away from 16:9 the identity is
 * increasingly approximate: 21:9 alone is ~28% off the round value. So
 * ``MAX_COVERAGE_FRACTION = 2 / 0.5 = 4.0`` is unchanged in VALUE, and the
 * switch point it anchors (reaching metric 4.0) still needs
 * ``diagonalPx ≈ 2 × fittedAxisPx`` at 16:9, matching what
 * ``diagonalPx ≈ viewportDiagonal`` (metric 4.0 under the OLD scheme) meant
 * there — but away from 16:9 this is a real, accepted behavioural trade, not
 * just a units relabelling. Because a screen-filling object's diagonal is
 * *narrower* than 2·fittedAxisPx at square-ish aspects and *wider* at
 * ultrawide ones, a partition-bound ladder's finest level (which derives its
 * anchor from ``MAX_COVERAGE_FRACTION``) now needs a tile to grow LARGER on
 * screen before showing its finest level at square-ish/portrait-ish windows,
 * and SMALLER at ultrawide ones, than it did before #1410 — measured as the
 * ratio of the new required projected diagonal to the old one: **×1.41 at
 * 1:1, ×1.20 at 4:3, ×1.06 at 16:10, ~×1.0 at 16:9 (by construction), ×0.78 at
 * a real 21:9 panel, ×0.54 at 32:9**. This is the accepted cost of the fix: the
 * WHOLE-OBJECT ladder (``coverage_fractions``, anchored at 1.0) is what a
 * normal full-frame opening view hits, and that is exactly where #1410's
 * wide-aspect blur showed up — making that anchor aspect-exact was the goal.
 * A tiled layer's per-tile anchor was already only a heuristic (each tile's
 * own on-screen footprint already varies with camera distance and framing,
 * partition shape, etc.), so it is the right place to absorb the residual
 * aspect dependence rather than the whole-object case. A Python test
 * (``test_max_coverage_fraction_matches_the_viewer_fill_factor``) reads this
 * file and asserts the ``MAX_COVERAGE_FRACTION`` relation holds, and
 * separately that ``SCREEN_FILL_DIAGONAL_RATIO`` itself stays close to the
 * 16:9 geometric value it stands for (so the two constants can't silently
 * compensate for each other).
 *
 * **View-axis depth fix (#1543).** A 1×1×100 cloud previously measured only
 * ~0.024 at the default 16:9 framing because the distance was sized from its
 * pure-depth dimension plus a hardcoded 20% margin. The exact near-face fit
 * above raises it to **2.121**, so it opens on the finest rung like the other
 * full-scene shapes. The same fit removes the margin: keeping both would count
 * depth twice and pull ordinary 3D scenes unnecessarily far back.
 *
 * **Known limitation: resize without a re-fit (out of scope here).**
 * ``updateCameraAspect`` (``utils/camera-utils.ts``) only updates
 * ``camera.aspect`` (perspective) / the horizontal frustum extent
 * (orthographic) on a window resize — it preserves the VERTICAL fov / ortho
 * extent, and the viewer never re-fits the camera distance afterwards. Both
 * screen-space pixel extents of a projected box then depend on viewport
 * HEIGHT alone (not width): with vertical fov and distance unchanged, the
 * horizontal pixel extent is ``(world extent / (z·tan(vFov/2))) ·
 * (height/2)`` — width cancels out of it algebraically — so narrowing the
 * window width with height held fixed leaves the projected diagonal in
 * pixels completely UNCHANGED while ``fittedAxisPx`` (now ``width``, once
 * width < height) keeps shrinking, inflating the metric by ``height/width``.
 * Measured (a 100×100×100 cube, default opening framing, 1600×900 baseline
 * narrowed with height fixed at 900): 1600×900 → 500×900 inflates the metric
 * ×1.80 (the pre-#1410 diagonal normalisation also inflated here, ×1.78 — a
 * wash), but 1600×900 → 200×900 inflates ×4.50 vs only ×1.99 under the old
 * normalisation — because the old denominator (``hypot(width, height)``) is
 * bounded below by ``height`` as width → 0, while the new one
 * (``min(width, height)``) is not. This is the flip side of the #1410 fix:
 * WIDENING a viewport (the actual #1410 symptom) is now exactly stable
 * (proven above) where it used to decay. NARROWING one is not symmetric: with
 * ``diagonalPx`` held fixed, the new metric only overtakes the old one PAST a
 * crossover at ``width = height² / width0`` (506px / aspect ≈ 0.56 for this
 * baseline) — down to that point the new normalisation is actually LESS
 * inflated than the old one (e.g. 1600×900 → 500×900, aspect 0.56: ×1.80 new
 * vs ×1.78 old, already a near-wash), and only below it does narrowing
 * inflate the metric faster than before (1600×900 → 200×900, aspect 0.22:
 * ×4.50 new vs ×1.99 old). Not fixed here: re-fitting the camera on resize is
 * a separate, larger change (it would also move the FRAMING, not just the
 * LOD selection) and out of scope for this normalisation fix.
 *
 * The selector normalises the projected diagonal by this to a dimensionless
 * coverage metric, so the same thresholds behave (near-)identically at any
 * aspect ratio and any pixel size of the viewport.
 *
 * Exported so tests can pin behaviour against the real constant instead of
 * hard-coding 0.5.
 */
export const FILL_FACTOR = 0.5;

/**
 * A screen-filling object's projected pixel diagonal, expressed as a multiple
 * of the fitted screen axis (``fittedAxisPx``): ``hypot(aspect, 1) / min(aspect,
 * 1)``. This is aspect-DEPENDENT (1.41 at 1:1 up to 3.69 at 32:9 — see the
 * ``FILL_FACTOR`` doc above for the full spread); the constant below is
 * anchored at the **16:9 reference aspect** (2.04, rounded to a plain ``2``),
 * the same aspect ``FILL_FACTOR`` is calibrated at, NOT an average or a fit
 * across the mainstream range — it is exact (to ~2%) only near 16:9 and gets
 * markedly less accurate at more extreme aspects (e.g. ~28% off at a real
 * 21:9 panel). Named so the ``MAX_COVERAGE_FRACTION`` coupling reads as what
 * it is — ``coverage metric of a screen-filling object ≈
 * SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR`` (approximately, at mainstream
 * aspect ratios) — rather than an unexplained ``1 / FILL_FACTOR``, which was
 * only exact under the old (diagonal) normalisation, at every aspect ratio.
 */
export const SCREEN_FILL_DIAGONAL_RATIO = 2;

/**
 * Frames a lazy level stays in the ``failed`` state before the registry
 * retries its deferred load. ~2 s at 60 fps — long enough to avoid
 * per-frame retry storms after a hard failure, short enough that a
 * transient (network blip) failure self-heals once the camera revisits
 * the level. See ``LODGroupRegistry.maybeKickLoad``.
 */
const FAILED_RETRY_FRAMES = 120;

/** One LOD-group child as tracked by the registry. */
export interface LODGroupChild {
  /**
   * The leaf THREE node (gsplats / points / lines / group). For a
   * lazily-loaded child this is the empty placeholder mesh attached at
   * registration; geometry is committed into it by ``ensureLoaded``.
   */
  object: THREE.Object3D;
  /** Authored scene-node path, including for anonymous deferred-group placeholders. */
  nodePath?: string;
  /**
   * Viewport-relative LOD-switch threshold, strictly monotonic increasing in
   * coarsest→finest order (coarsest 0.0). Its UNITS — and so its finest anchor —
   * come from the entry's ``selector``:
   *
   * - ``'screen-area'`` (every ladder the producer derives today): a literal
   *   fraction of the viewport AREA, compared against
   *   ``projectBoxAreaFraction``. A whole-object ladder anchors its finest at
   *   0.5 (full detail while the object covers at least half the screen); one
   *   bound to a spatial partition anchors at 1.0 (the tile alone fills the
   *   screen), which the producer derives for it automatically because a tile
   *   projects to only a fraction of the whole object's rect.
   * - ``'coverage'`` (legacy stores, and explicitly authored
   *   ``coverage_fractions=[...]`` lists): multiplied by
   *   ``FILL_FACTOR × fittedAxisPx`` at selection time and compared against the
   *   group's projected bbox DIAGONAL in pixels, so 1.0 activates once that
   *   diagonal reaches half of the fitted screen axis. An author may go up to
   *   ``SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR`` (4.0, roughly a
   *   screen-filling object) to hold a level until later than that.
   *
   * No upper bound is enforced here.
   */
  coverageFraction: number;
  /**
   * Raw nD position bounds (from the child's ``position_bounds`` zarr
   * attribute). Stored unprojected because ``displayDims`` can change
   * at runtime (user picks different dimensions to display) — the
   * selector re-projects each frame.
   */
  positionBounds: { min: readonly number[]; max: readonly number[] };
  /**
   * Optional robust nD bounds from the child's ``lod_bounds`` zarr attribute.
   * The selector uses these only to size the node for either LOD metric;
   * frustum gating and eviction keep the full ``positionBounds`` so visible
   * outliers are never treated as absent. Missing bounds fall back to
   * ``positionBounds`` for legacy stores.
   */
  lodBounds?: { min: readonly number[]; max: readonly number[] };
  /**
   * Lazy-loading readiness. ``undefined`` means "always ready" (eagerly
   * loaded — the default for callers that don't opt into lazy loading,
   * including unit tests that construct children directly). ``false``
   * means the child's geometry has not been committed yet, so the
   * selector must not make it visible; it fires ``ensureLoaded`` instead
   * and waits for a later frame to swap once the thunk sets this true.
   */
  ready?: boolean;
  /** Set by the registry when it fires ``ensureLoaded``; cleared by the thunk. */
  loading?: boolean;
  /** Set by the thunk on load failure to stop per-frame retry storms. */
  failed?: boolean;
  /**
   * Marks the lazy branch that observed a latched archive fault, so monitor
   * rows and explicit Retry can target it even when its THREE placeholder is
   * anonymous. The owning loader's archive latch is the shared dataset-fault
   * oracle; this branch marker is cleared by an explicit retry.
   */
  permanentlyFailed?: boolean;
  /** Human-readable reason retained for monitor rows after the loader latch clears. */
  failureReason?: string;
  /**
   * Registry tick when ``failed`` was first observed. Drives the
   * transient-failure retry cooldown (``FAILED_RETRY_FRAMES``): once it
   * elapses the registry clears ``failed`` and retries the load, so a
   * level that fails on *reload* (after a successful load + byte-eviction)
   * is not stuck on its placeholder forever. Cleared alongside ``failed``.
   */
  failedTick?: number;
  /**
   * Idempotent fire-and-forget loader for a lazy child. Kicks the
   * deferred geometry load; on success sets ``ready=true`` and clears
   * ``loading``; on failure sets ``failed=true`` and clears ``loading``. A
   * container fault may additionally set ``permanentlyFailed``.
   * Must not touch ``object.visible`` — the registry owns the swap.
   */
  ensureLoaded?: () => void;
  /**
   * Release this lazy level's GPU geometry back to the evictable pool
   * and reset its readiness so a later selection reloads it. Called by
   * the registry's LRU eviction when GPU memory is over budget — never
   * eagerly on swap (retention keeps re-shows free). Absent on
   * eagerly-loaded children (e.g. the coarsest default level), which
   * therefore stay resident as the always-available fallback and are
   * never evicted.
   */
  release?: () => void;
  /**
   * Registry tick when this level was last the visible/active child.
   * The eviction LRU evicts the coldest (lowest tick) loaded levels
   * first. Undefined ⇒ never been shown, so not an eviction candidate
   * (avoids evicting a just-loaded level in the 1-frame gap before its
   * swap).
   */
  lastVisibleTick?: number;
  /**
   * For a lazy level backed by a PROGRESSIVE loader (a substitutive level whose
   * geometry is itself an additive ladder, e.g. the `pyramid` recipe): reports
   * whether more additive LODs remain to stream for the current view. The
   * registry treats "ready & fresh but hasMoreLODs" like a not-yet-final state
   * and re-fires `ensureLoaded` (settle-gated) to advance the ladder until it
   * completes — driving progressive refinement entirely from the registry, since
   * lazy levels no longer ride the per-slice sweep. Absent / `() => false` on a
   * single-LOD level (the common case) ⇒ no extra refinement passes.
   */
  hasMoreLODs?: () => boolean;
  /**
   * `true` for a deferred GROUP child (a `kind=partition` / nested `kind=lod`
   * level — the `overview` recipe's fine branch), whose `ensureLoaded` runs
   * `loadChildren` once to attach the whole subtree. The registry never
   * re-fires such a child for staleness or refinement: its leaves are
   * sweep-registered and re-stamp themselves, and a second activation would
   * attach a second copy of the subtree. Set by `load-lod-group-node`.
   */
  deferredGroup?: boolean;
}

/** One LOD-group entry tracked by the registry. */
export interface LODGroupEntry {
  /** Scene path (for diagnostics + UI lookup). */
  path: string;
  /** The lod_group's THREE container. World matrix lives here. */
  groupObject: THREE.Object3D;
  /**
   * Children in coarsest→finest order (== insertion order on disk,
   * == ascending ``coverageFraction``).
   */
  children: LODGroupChild[];
  /**
   * Units of the children's ``coverageFraction`` thresholds (the on-disk
   * group's ``selector`` attr): ``'screen-area'`` compares them against the
   * projected bbox rect's fraction of the viewport AREA
   * (``projectBoxAreaFraction``); ``'coverage'`` — the legacy diagonal metric
   * (``projectBoxDiagonalPx / (FILL_FACTOR × min(viewport.width,
   * viewport.height))``, the fitted screen axis) — is the default when
   * absent, so older stores and test-constructed entries keep their
   * behaviour.
   */
  selector?: 'coverage' | 'screen-area';
  /** Current selector mode (``'auto'`` or ``{ lockLevel: i }``). */
  selectorMode: LODGroupSelectorMode;
  /** Initial active level, used when nothing else has selected yet. */
  defaultLevel: number;
  /**
   * Index into ``children`` of the screen-DESIRED level (the aspiration). This
   * is the hysteresis anchor and the level the auto-selector wants on screen.
   * It is NOT necessarily what is displayed: when its committed geometry is
   * stale for the current view version, the registry shows a coarser fresh
   * level (``displayedChildIndex``) until the aspiration commits.
   */
  activeChildIndex: number;
  /**
   * Index into ``children`` of the level ACTUALLY visible this frame. Equals
   * ``activeChildIndex`` in steady state; during a re-slice it transiently
   * points at the coarsest fresh level while the aspiration reloads, and
   * during a never-downgrade hold it can also point at a FINER
   * previously-displayed level while a coarser streaming aspiration catches
   * up. During a stale hold it can instead remain on a finer STALE level while
   * the next slice decodes. A per-frame transient written by ``evaluateEntry``
   * and read by ``enforceByteBudget`` (same synchronous ``evaluatePerFrame`` pass)
   * so eviction never releases the on-screen level. ``undefined`` before the
   * first evaluation ⇒ treated as ``activeChildIndex``. Tracks what is ACTUALLY
   * on screen every frame — including the coarse level shown while the group is
   * off-screen — which is what eviction needs, but is therefore NOT the
   * never-downgrade gate's memory (that is ``heldDisplayChildIndex``).
   */
  displayedChildIndex?: number;
  /**
   * The last level displayed while the group was ON SCREEN — shared memory for
   * the never-downgrade gate and stale hold. Distinct from
   * ``displayedChildIndex`` because the off-screen gate transiently displays
   * (and would otherwise record) the coarsest ready level; folding that into
   * the gate memory would let a mere look-away-and-back clobber a held finer
   * level and re-pop it to chunk-1 on return. Written by ``evaluateEntry``
   * only on frames where the group is on screen. ``undefined`` before the
   * first on-screen evaluation ⇒ neither hold policy has a prior level.
   */
  heldDisplayChildIndex?: number;
  /**
   * Wall-clock ms at which the current **stale-hold budget** started — see
   * ``staleHoldDisplayIndex``. Set on the first eligible hold and retained when
   * a later ratio check declines to hold, so the budget cannot restart during
   * the same scrub. Cleared only when the aspiration recommits fresh or the
   * budget is exhausted.
   */
  staleHoldSinceMs?: number;
  /**
   * True once a stale hold has exhausted `STALE_HOLD_MS` without the
   * aspiration recommitting. Latches the coarse fallback for the rest of this
   * scrub; cleared when the aspiration finally lands fresh.
   */
  staleHoldExhausted?: boolean;
  /**
   * Whether the auto-selector is currently holding this group at its
   * coarsest-ready level because its world bounds are outside the camera
   * frustum (the off-screen gate). ``false`` when on-screen or when a
   * level is explicitly locked. Surfaced in the layers-panel readout as an
   * "(off-screen)" hint so a coarse level on close inspection isn't
   * mistaken for a selection bug. Updated each ``evaluatePerFrame``.
   */
  offScreen?: boolean;
  /**
   * The level index the selector WANTED this frame, recorded BEFORE the
   * ready/freshness gates below it get a say. Written by ``evaluateEntry``
   * once a ``desired`` has been computed — the explicit lock and all three
   * auto branches (off-screen hold, screen-area pick, legacy coverage pick).
   * ``undefined`` (never evaluated) ⇒ read it as ``activeChildIndex``.
   *
   * It is NOT rewritten on every frame: ``evaluateEntry`` returns before
   * computing a ``desired`` when the entry has no registration cache or no
   * world box, and ``evaluatePerFrame`` returns before reaching the entries at
   * all on a zero-sized viewport or with fewer than two display dims. The
   * field then keeps its previous value. That is benign — both early returns
   * are stable properties of the entry/viewport rather than transient states,
   * so a stale value cannot describe a level the selector has since moved off,
   * and an entry that never got one reads as ``activeChildIndex`` (i.e.
   * "nothing pending"), which is the right answer for a group the selector has
   * never been able to evaluate.
   *
   * Purely diagnostic for the renderer — nothing about display reads it. It
   * exists so an OFFLINE CAPTURE can tell "the selector wants a finer level it
   * has not got yet" apart from "settled" (see
   * {@link LODGroupRegistry.isCaptureQuiescent}). ``activeChildIndex`` alone
   * cannot express that: the aspiration only ever advances ONTO A READY LEVEL,
   * so in the frame where a lazy fine level's async load lands (the thunk sets
   * ``ready=true`` and clears ``loading``) the registry has not swapped yet —
   * that happens on the NEXT selector pass. A quiescence predicate reading only
   * ``loading`` / ``ready`` / ``displayedChildIndex`` would call that window
   * "settled" and the capture would film the coarse level one frame before the
   * swap, which is exactly the LOD pop this field exists to close.
   */
  desiredChildIndex?: number;
}

export interface PartitionGroupChild {
  /** Stable node path for a part that may emit more than one scene object. */
  path: string;
  objects: THREE.Object3D[];
  positionBounds: { min: readonly number[]; max: readonly number[] };
}

export interface PartitionGroupEntry {
  path: string;
  groupObject: THREE.Object3D;
  children: PartitionGroupChild[];
}

/**
 * Internal per-entry cache populated by ``register()``. Lets
 * ``evaluateEntry`` run without allocating on the hot path: the
 * threshold list is rebuilt once at registration and the scratch
 * boxes are reused every frame.
 *
 * Kept separate from the public ``LODGroupEntry`` interface so test
 * code (which constructs entries directly) doesn't have to populate
 * caches — ``register()`` does it for them.
 */
interface LODGroupEntryCache {
  /**
   * Per-child ``coverage_fraction`` thresholds (dimensionless, ascending from a
   * 0.0 coarsest floor to whatever finest anchor the entry's ``selector`` units
   * imply — see ``LODGroupChild.coverageFraction``), rebuilt once at
   * registration. The selector compares them against ``projectBoxAreaFraction``
   * under ``'screen-area'``, or against the projected bbox diagonal normalised
   * by ``FILL_FACTOR × fittedAxisPx`` under the legacy ``'coverage'``. Either
   * way the list is viewport-independent and needs no per-frame rebuild.
   */
  thresholds: number[];
  /** Whether any child needs the optional robust-bounds metric fold. */
  hasLodBounds: boolean;
  localBoxScratch: BoundingBox;
  worldBoxOptions: WorldBoxOptions;
  metricWorldBoxOptions: WorldBoxOptions;
}

/**
 * Module-scope scratch for the per-frame frustum gate.
 * ``evaluatePerFrame`` is the single per-frame entry point (no re-entrancy),
 * so these are safe to share across all entries within one frame:
 *   - ``FRUSTUM_SCRATCH`` — rebuilt once per frame from the camera.
 *   - ``FRUSTUM_MATRIX_SCRATCH`` — projection × view product feeding it.
 *   - ``PARTITION_FRUSTUM_SCRATCH`` — partition fetch frustum with screen margin.
 *   - ``PARTITION_FRUSTUM_MATRIX_SCRATCH`` — padded projection × view product.
 *   - ``WORLD_BOX3_SCRATCH`` — a ``THREE.Box3`` view of a group's world bbox
 *     for ``frustum.intersectsBox`` (our ``BoundingBox`` is a plain object).
 *   - ``FOOTPRINT_BOX3_SCRATCH`` — one object's rendered footprint on its way
 *     into a part's cached footprint box.
 * (The eviction pass keeps its own scratches in ``lod-eviction.ts``.)
 */
const FRUSTUM_SCRATCH = new THREE.Frustum();
const FRUSTUM_MATRIX_SCRATCH = new THREE.Matrix4();
const PARTITION_FRUSTUM_SCRATCH = new THREE.Frustum();
const PARTITION_FRUSTUM_MATRIX_SCRATCH = new THREE.Matrix4();
// Cold parts have no loaded footprint to union, so pad x/y symmetrically to
// preload them before entry and keep entry/exit behavior from becoming asymmetric.
const PARTITION_FRUSTUM_MARGIN = 0.1;
const PARTITION_FRUSTUM_SCALE = new THREE.Matrix4().makeScale(
  1 / (1 + PARTITION_FRUSTUM_MARGIN),
  1 / (1 + PARTITION_FRUSTUM_MARGIN),
  1
);
const WORLD_BOX3_SCRATCH = new THREE.Box3();
const FOOTPRINT_BOX3_SCRATCH = new THREE.Box3();
// Bit flags returned by evaluatePartitionEntry so one child scan reports both effects.
const PARTITION_VISIBILITY_CHANGED = 1;
const PARTITION_BECAME_VISIBLE = 2;
// Part paths that re-entered the frustum THIS frame, collected by
// ``evaluatePartitionEntry`` and merged into ``partitionResyncPending`` only
// on a rising edge (rare), so the steady-state per-frame path allocates nothing.
const RISING_PARTS_SCRATCH = new Set<string>();

/**
 * Record a part that just re-entered the frustum, by its registered node path
 * (``PartitionGroupChild.path`` — the loader-registry key for a leaf part and
 * the prefix of a nested ``kind=lod`` part's level loaders). A part with no
 * path cannot be targeted, so it is recorded as the WRAPPER path: resync the
 * whole partition rather than silently miss it.
 */
function noteRisingPart(sink: Set<string>, partPath: string, wrapperPath: string): void {
  sink.add(partPath || wrapperPath);
}

function unionPartitionFootprints(objects: readonly THREE.Object3D[], target: THREE.Box3): void {
  for (const object of objects) {
    FOOTPRINT_BOX3_SCRATCH.setFromObject(object);
    if (!FOOTPRINT_BOX3_SCRATCH.isEmpty()) target.union(FOOTPRINT_BOX3_SCRATCH);
  }
}

/**
 * Whether any object of a part has moved since its footprint box was captured.
 * ``cached`` holds one 16-element world-matrix block per object, in
 * ``objects`` order, sized by ``registerPartition``. A part that later gained an
 * object reads ``undefined`` past the end and so counts as moved, which
 * recaptures and grows the array; one that lost an object simply stops
 * comparing the trailing blocks.
 *
 * Only the part's own objects are compared. A transform on a DESCENDANT of one
 * of them is not detected — descendant transforms are applied at node-creation
 * time and never animated, and every geometry commit dirties the part
 * explicitly. Anything that starts moving a descendant later has to call
 * ``invalidatePartitionFootprint``.
 */
function partitionFootprintMatricesDiffer(
  objects: readonly THREE.Object3D[],
  cached: readonly number[]
): boolean {
  for (let index = 0; index < objects.length; index++) {
    const elements = objects[index].matrixWorld.elements;
    const offset = index * 16;
    for (let element = 0; element < 16; element++) {
      if (cached[offset + element] !== elements[element]) return true;
    }
  }
  return false;
}

/** Rebuild a part's cached footprint box and the world matrices it was taken at. */
function capturePartitionFootprint(
  objects: readonly THREE.Object3D[],
  target: THREE.Box3,
  cached: number[]
): void {
  target.makeEmpty();
  unionPartitionFootprints(objects, target);
  // Snapshot AFTER the union: ``Box3.expandByObject`` refreshes each object's
  // world matrix, so a snapshot taken first would lag by one recompute and make
  // every following frame look like a move.
  for (let index = 0; index < objects.length; index++) {
    objects[index].matrixWorld.toArray(cached, index * 16);
  }
}

/**
 * Dirty every part the committed ``nodePath`` belongs to — the part itself, or
 * anything under it (an LOD level or an additive rung commits at a deeper path
 * than the part it renders into). Returns whether any part matched, so the
 * caller can fall back to dirtying the whole partition.
 */
function invalidateMatchingPartitionChildren(
  entry: PartitionGroupEntry,
  children: Array<{ footprintDirty: boolean }>,
  nodePath: string
): boolean {
  let matched = false;
  for (let index = 0; index < entry.children.length; index++) {
    const childPath = entry.children[index].path;
    if (nodePath === childPath || nodePath.startsWith(`${childPath}/`)) {
      children[index].footprintDirty = true;
      matched = true;
    }
  }
  return matched;
}

function updatePartitionObjectVisibility(
  objects: readonly THREE.Object3D[],
  visible: boolean
): number {
  // Any previously culled object makes the whole part a rising edge.
  let wasVisible = true;
  let changed = false;
  for (const object of objects) {
    if (object.userData.partitionFrustumVisible === false) wasVisible = false;
    object.userData.partitionFrustumVisible = visible;
    if (object.visible !== visible) {
      object.visible = visible;
      changed = true;
    }
  }
  return (
    (changed ? PARTITION_VISIBILITY_CHANGED : 0) |
    (visible && !wasVisible ? PARTITION_BECAME_VISIBLE : 0)
  );
}

/**
 * Injected view-state accessors. Lets the registry stay test-friendly
 * (mock the camera, the viewport, the slice) without coupling to the
 * SceneManager singleton.
 */
export interface LODGroupRegistryDeps {
  getCamera(): THREE.Camera;
  /** Viewport size in CSS pixels (matches the renderer canvas). */
  getViewportSize(): { width: number; height: number };
  /** Which dimensions of the data are being projected to screen. */
  getDisplayDims(): readonly number[];
  /** Whether the owning loader has latched an archive fault. */
  hasArchiveFault?: () => boolean;
  /** Whether a loader under this LOD group has a recorded network failure. */
  hasNetworkFailureUnder?: (path: string) => boolean;
  /**
   * Resident-byte budget (the ceiling). The single, adaptive VRAM budget
   * shared with the GPU buffer pool — one authority, not a competing one.
   * Omitted ⇒ no eviction (pure retention), the default for unit tests
   * that construct the registry directly.
   */
  getResidentByteBudget?: () => number;
  /**
   * Measured total resident VRAM bytes (active + pooled) from the GPU
   * buffer pool — the single accounting truth. The registry compares this
   * to the budget to decide when to demote cold levels; it no longer keeps
   * its own per-level byte estimate. Omitted ⇒ no eviction (pure
   * retention), the default for unit tests.
   */
  getResidentBytes?: () => number;
  /**
   * Current view-update version (``SceneLoader.currentViewVersion``). Used for
   * the slice-aware fallback: a gsplats level whose committed geometry was
   * stamped with an older version still shows a previous slice/displayDims, so
   * the registry treats it as stale and displays the coarsest level that IS
   * fresh until the re-slice commits. Omitted ⇒ no freshness tracking (every
   * ready level is treated as fresh — identical to the pre-feature behaviour;
   * the default for unit tests that don't exercise scrubbing).
   */
  getViewVersion?: () => number;
  /**
   * Monotonic wall clock in milliseconds, for the stale-hold budget (see
   * `STALE_HOLD_MS`). Injectable so tests can advance it deterministically;
   * omitted ⇒ ``performance.now()``.
   */
  now?: () => number;
  /**
   * Keep the render loop alive (the viewer is on-demand and idles after ~2s).
   * Called each frame while a lazy level is loading so a deferred fine reload
   * — which commits OUTSIDE the per-slice sweep and can take longer than the
   * idle timeout — still triggers the per-frame swap-up to the fresh level when
   * it lands, instead of waiting for the next user interaction. Wired to
   * ``AnimationController.startAnimation`` (resets the idle timeout). Omitted ⇒
   * no-op (unit tests don't run a loop).
   */
  requestRender?: () => void;
  /**
   * Re-run the current view update for the partition parts that just
   * re-entered the frustum. Culled children skip event-driven slice updates, so
   * the rising edge must resync any geometry that missed the latest view state.
   * ``paths`` are the re-entering PART node paths (``child.path``), or
   * the partition wrapper path when a part has none (= resync the whole
   * partition); the loader prefix-matches its sweep loaders under them, so a
   * nested ``kind=lod`` part's eager level is covered. The view state is
   * unchanged, so the loader MUST NOT bump the view version for this pass —
   * otherwise every lazy fine level scene-wide reads stale and drops to coarse.
   */
  requestReprocess?: (paths: readonly string[]) => void;
  /**
   * Whether the owning loader has a view PASS in flight or queued. Rising edges
   * are held (and coalesced) while this is true so a resync never lands on top
   * of a pass. A refinement hold deliberately does NOT count: the loader parks
   * a resync that arrives during one and cancels into its own pass, so
   * re-entering parts do not sit on a stale slice until the ladders finish.
   */
  isUpdateInProgress?: () => boolean;
  /**
   * Whether the LOD cross-fade is enabled (ON by default; `?no-lod-fade`
   * disables). When true and a blendable (additive/luminous/volumetric — see
   * `BLENDABLE_MODES` in `scene/lod-fade.ts`) group is zooming across a LOD
   * boundary, the registry
   * blends the two straddling levels' opacity — the finer at
   * `smoothstep(coverage metric across a ±band around the boundary)`, the
   * coarser at the complement — instead of a hard visibility swap. Distance-
   * driven (a function of the coverage metric), independent of additive
   * streaming. Omitted / false ⇒ the pre-cross-fade hard swap, byte-identical.
   * Read live so the flag applies without a reload. The default for unit tests
   * (off).
   */
  getCrossFadeEnabled?: () => boolean;
  /**
   * Whether streaming brightness compensation is enabled: as a blendable
   * (additive/luminous/volumetric)
   * leaf's ladder streams in, scale its opacity by `1/e(k)` so the partial prefix
   * renders at the full-level energy (no brightening pop). Distinct axis from the
   * cross-fade (time, not distance) and independently gated; either flag on
   * enables the registry's per-frame opacity management. Omitted / false ⇒
   * byte-identical (no material writes). Read live so the flag applies without a
   * reload. The default for unit tests (off).
   */
  getEnergyCompEnabled?: () => boolean;
  /**
   * Force the finest LOD level regardless of projected screen coverage (and
   * never coarsen off-screen) — for high-quality still/video capture (the
   * gallery harness), where a coarse level looks blurry even when the
   * subject is small in frame. Wired from `LuxarAppOptions.lodFinest`
   * (the `?lod-finest` URL flag, threaded through the standalone
   * bootstrap); omitted / false ⇒ normal coverage-driven selection. Read
   * live, like the sibling flags above.
   */
  getForceFinestLOD?: () => boolean;
  /**
   * Register a clone-on-first-fade material with the material manager so it keeps
   * receiving per-frame camera-uniform updates (the fade clones the shared cached
   * material to fade one level independently; an unregistered gsplat clone would
   * project with stale camera params). Wired to `materialManager.register`;
   * omitted in unit tests (no camera loop).
   */
  registerMaterial?: (material: THREE.Material) => void;
}

/**
 * Tracks loaded ``lod_group`` and ``kind=partition`` nodes in a scene;
 * evaluates per-frame to pick the active LOD and frustum-visible parts.
 */
export class LODGroupRegistry {
  private entries: Map<string, LODGroupEntry> = new Map();
  private partitionEntries: Map<string, PartitionGroupEntry> = new Map();
  private partitionCaches: Map<
    string,
    {
      children: Array<{
        source: PartitionGroupEntry;
        localBoxScratch: BoundingBox;
        worldBoxOptions: WorldBoxOptions;
        footprintBox: THREE.Box3;
        /**
         * World transforms ``footprintBox`` was captured at — one flattened
         * 16-element block per object of the part, in ``children[i].objects`` order.
         */
        footprintMatrixWorld: number[];
        footprintDirty: boolean;
      }>;
    }
  > = new Map();
  /**
   * Per-entry register-time cache (parallel to ``entries`` by path).
   * Populated by :meth:`register`. Stored on the side so the public
   * ``LODGroupEntry`` interface stays test-friendly (callers don't
   * have to compute thresholds or allocate scratch boxes).
   */
  private caches: Map<string, LODGroupEntryCache> = new Map();
  /** Reused per frame to feed ``transformBoundingBox``'s matrix arg. */
  private readonly matrixScratch: number[] = new Array(16).fill(0);
  /**
   * Monotonic per-frame counter. Each frame the active (visible) child of
   * every entry is stamped with the current tick; the resident-byte
   * eviction LRU evicts the lowest-tick (coldest) loaded levels first.
   */
  private tick = 0;
  /**
   * Entry paths already warned about a missing-ready-child invariant break in
   * ``coarsestReadyIndex``. The off-screen gate calls that method every frame,
   * so without a dedupe a genuinely-stuck group (eager default failed to
   * attach) would log at frame rate. One warning per entry surfaces the break
   * without the flood.
   */
  private warnedNoReadyChild: Set<string> = new Set();

  /**
   * Entry paths already warned about the fresh-but-empty display guard
   * firing (see ``evaluateEntry``). The guard is evaluated every frame, so
   * without a dedupe a persistently inconsistent dataset would warn at
   * frame rate.
   */
  private warnedEmptyLevel: Set<string> = new Set();

  /**
   * Tracks when the global view-update version last changed (in ticks) so the
   * selector can defer a stale fine level's reload until the scrub settles —
   * the debounce behind ``maybeKickReload``. See ``scene/lod-freshness.ts``.
   */
  private settleTracker = new SettleTracker();

  /**
   * Whether fade management (cross-fade and/or energy compensation) was ON
   * during the previous ``evaluatePerFrame``. Falling-edge detector for the
   * one-shot residual-opacity restore in ``evaluateEntry``: toggling BOTH
   * anti-popping flags off MID-fade would otherwise strand a half-faded
   * level's opacity forever (``manageFade === false`` skips the per-frame
   * restore branch). Updated once per frame after all entries are evaluated;
   * one restore pass on the edge keeps the both-flags-off steady state
   * byte-identical (no material writes, no subtree traversal).
   */
  private fadeWasManaged = false;
  /**
   * Partition rising edges waiting for their wrapper to be visible and the
   * loader to be idle: wrapper path → the re-entering PART paths. A set that
   * contains the wrapper path itself means "resync the whole partition" (a
   * pathless part). Coalesced across frames; flushed as ONE
   * ``requestReprocess(paths)`` call.
   */
  private readonly partitionResyncPending = new Map<string, Set<string>>();

  constructor(private deps: LODGroupRegistryDeps) {}

  /** Register a newly-loaded lod_group (called by the scene loader). */
  register(entry: LODGroupEntry): void {
    this.entries.set(entry.path, entry);
    this.caches.set(entry.path, {
      thresholds: entry.children.map((c) => c.coverageFraction),
      hasLodBounds: entry.children.some((c) => c.lodBounds != null),
      localBoxScratch: {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0, y: 0, z: 0 },
      },
      worldBoxOptions: {
        worldBoxScratch: {
          min: { x: 0, y: 0, z: 0 },
          max: { x: 0, y: 0, z: 0 },
        },
      },
      metricWorldBoxOptions: {
        worldBoxScratch: {
          min: { x: 0, y: 0, z: 0 },
          max: { x: 0, y: 0, z: 0 },
        },
        useLodBounds: true,
      },
    });
    // Apply initial visibility: only the active child is visible, and
    // only if its geometry is ready. A lazily-loaded active child that
    // isn't ready yet stays hidden until its thunk completes — the
    // per-frame ``evaluateEntry`` self-heals by kicking the active child's
    // load and showing it once ready, so a fallback that pins a not-ready
    // lazy level as active (e.g. when the eager default failed to attach)
    // can never leave the group permanently blank.
    for (let i = 0; i < entry.children.length; i++) {
      const child = entry.children[i];
      child.object.visible = i === entry.activeChildIndex && isReady(child);
    }
    // The initially-shown level is the active default; ``evaluateEntry`` may
    // transiently move the displayed level to a coarser fresh one during a
    // re-slice, but it starts equal to the aspiration. The gate memory starts
    // there too (the group is presumed on-screen until the first evaluation).
    entry.displayedChildIndex = entry.activeChildIndex;
    entry.heldDisplayChildIndex = entry.activeChildIndex;
  }

  /** Register a partition whose children are independently frustum-gated. */
  registerPartition(entry: PartitionGroupEntry): void {
    this.partitionEntries.set(entry.path, entry);
    this.partitionCaches.set(entry.path, {
      children: entry.children.map((child) => ({
        source: { path: entry.path, groupObject: entry.groupObject, children: [child] },
        localBoxScratch: {
          min: { x: 0, y: 0, z: 0 },
          max: { x: 0, y: 0, z: 0 },
        },
        worldBoxOptions: {
          worldBoxScratch: {
            min: { x: 0, y: 0, z: 0 },
            max: { x: 0, y: 0, z: 0 },
          },
        },
        footprintBox: new THREE.Box3(),
        footprintMatrixWorld: new Array<number>(child.objects.length * 16).fill(0),
        footprintDirty: true,
      })),
    });
    for (const child of entry.children) {
      // Each loader path resolves to its emitted object, so stamping them all
      // lets the loader gate use its normal ancestor walk for multi-object parts.
      for (const object of child.objects) object.userData.partitionFrustumVisible = true;
    }
  }

  /**
   * Mark the owning partition part's rendered footprint stale after a geometry commit.
   *
   * ``SceneLoader.updatePointsGeometry`` and the three ``commit*Geometry`` methods
   * are the complete geometry-attach funnels, including lazy LOD children and
   * additive rungs. They dirty the part before writing, so a commit that hands off
   * geometry and then throws cannot leave the previous footprint cached.
   * ``registerPartition`` starts every part dirty, which also covers a commit that
   * races registration. A path under a partition that matches no registered child
   * dirties the whole partition conservatively rather than allowing an
   * under-covering stale box. Footprints are cached in world space; the per-frame
   * gate separately detects transform changes.
   */
  invalidatePartitionFootprint(nodePath: string): void {
    for (const [entryPath, cache] of this.partitionCaches) {
      if (nodePath !== entryPath && !nodePath.startsWith(`${entryPath}/`)) continue;
      const entry = this.partitionEntries.get(entryPath);
      if (!entry) continue;
      const matched = invalidateMatchingPartitionChildren(entry, cache.children, nodePath);
      if (!matched) {
        for (const childCache of cache.children) childCache.footprintDirty = true;
      }
    }
  }

  /** Drop an lod_group from the registry (called on scene teardown). */
  unregister(path: string): void {
    const partition = this.partitionEntries.get(path);
    if (partition) this.restorePartitionChildren(partition);
    this.partitionResyncPending.delete(path);
    this.entries.delete(path);
    this.caches.delete(path);
    this.partitionEntries.delete(path);
    this.partitionCaches.delete(path);
    this.warnedNoReadyChild.delete(path);
    this.warnedEmptyLevel.delete(path);
  }

  /** Clear all entries (called on full scene tear-down). */
  clear(): void {
    for (const partition of this.partitionEntries.values()) {
      this.restorePartitionChildren(partition);
    }
    this.entries.clear();
    this.caches.clear();
    this.partitionEntries.clear();
    this.partitionCaches.clear();
    // Reset the monotonic tick so a reused registry (shared-registry
    // refactor) starts cold rather than inheriting stale LRU ordering — and
    // the settle clock with it: it is keyed on the tick, and a leftover
    // `lastChangeTick` from the old scene would read "not settled" for that
    // many frames, freezing every stale reload after a dataset switch.
    this.tick = 0;
    this.settleTracker.reset();
    this.warnedNoReadyChild.clear();
    this.warnedEmptyLevel.clear();
    this.fadeWasManaged = false;
    this.partitionResyncPending.clear();
  }

  private restorePartitionChildren(entry: PartitionGroupEntry): void {
    for (const child of entry.children) {
      for (const object of child.objects) {
        object.visible = true;
        delete object.userData.partitionFrustumVisible;
      }
    }
  }

  /** Number of registered lod_groups (mainly for tests / diagnostics). */
  size(): number {
    return this.entries.size;
  }

  /** Number of registered groups that can require an offline-capture drain. */
  captureSize(): number {
    return this.entries.size + this.partitionEntries.size;
  }

  /** Lookup an entry by path (mainly for tests / UI). */
  get(path: string): LODGroupEntry | undefined {
    return this.entries.get(path);
  }

  /** All registered entries (mainly for the layers panel UI). */
  list(): LODGroupEntry[] {
    return Array.from(this.entries.values());
  }

  /** Authored paths of lazy levels currently latched on an archive fault. */
  getFailedLazyChildPaths(): string[] {
    const paths: string[] = [];
    for (const entry of this.entries.values()) {
      for (const child of entry.children) {
        if (child.permanentlyFailed && child.nodePath) paths.push(child.nodePath);
      }
    }
    return paths;
  }

  /** Failure reason for a latched lazy level, if that path is still failed. */
  getFailedLazyChildReason(path: string): string | undefined {
    for (const entry of this.entries.values()) {
      const child = entry.children.find((candidate) => candidate.nodePath === path);
      if (child?.permanentlyFailed) return child.failureReason;
    }
    return undefined;
  }

  /**
   * Whether every lod_group and partition part that contributes pixels to the
   * CURRENT view is already at final committed quality — i.e. one more frame
   * of waiting would not improve what is on screen.
   *
   * **Why this exists.** An offline turntable capture (``OfflineCaptureStrategy``)
   * takes exactly one ``requestAnimationFrame`` per exported frame. Since the
   * rAF loop runs for the whole sweep, the auto-selector is live and
   * frustum-aware, so a tile that leaves the frustum mid-orbit is demoted to
   * its coarsest ready level and the resident-byte budget may release its fine
   * one. When it swings back into view the fine level reloads ASYNCHRONOUSLY —
   * and without a wait those frames go into the ZIP/MP4 at the coarse level and
   * pop back a few frames later. The capture loop therefore drains on this
   * predicate (bounded) before grabbing each frame. Forcing finest instead was
   * deliberately rejected: a capture visits the whole scene, so peak residency
   * would be the entire dataset.
   *
   * Partition parts block while a visible rising edge is pending, while any load
   * pass is queued or committing and any part is visible, or while a visible
   * stamped leaf is stale / still climbing its additive ladder. Hidden or
   * re-culled parts are excluded because they contribute no pixels. Unstamped
   * leaves carry no freshness or ladder signal and remain non-blocking, matching
   * the lod-group subtree fold below.
   *
   * - **A latched archive fault skips all work that could start or wait for new
   *   loads, but still waits for loads already in flight to finish committing.**
   *   The latch prevents any further automatic kick, so every other unmet
   *   condition would be permanently false until an explicit or connectivity
   *   retry clears it. An already-started load still clears ``loading`` in its
   *   ``finally`` block and may commit geometry, so releasing the capture frame
   *   before that transition would allow a one-frame pop.
   * - **A partition leaf load failure that does not latch an archive fault has no
   *   success commit to refresh its stale stamp.** The predicate remains false;
   *   the capture drain's bounded timeout and consecutive-timeout latch are the
   *   escape hatch for that failed part.
   *
   * Per entry, in order:
   *
   * - **Off-screen entries are skipped entirely.** ``offScreen`` means the
   *   selector is deliberately holding the group coarse *because it draws
   *   nothing this frame* — blocking on it would wait for a level that will
   *   never be selected while it is culled.
   * - **Entries that are not EFFECTIVELY VISIBLE are skipped too** (a layer
   *   toggled off in the panel, or authored ``visible=false``, anywhere up the
   *   ancestor chain). They draw nothing, and — decisively —
   *   {@link kickDeferredLoadIfVisible} refuses to START a deferred load while
   *   the group is hidden, whereas the selector's frustum test is purely
   *   geometric and still records a fine ``desiredChildIndex`` for it. Without
   *   this skip such an entry has ``desired !== active`` with nothing ever
   *   loading, failing or becoming ready, so the predicate would be
   *   PERMANENTLY false and every capture frame would burn the full drain
   *   budget before giving up.
   * - **An entry with no child at the aspiration index is skipped** —
   *   ``children`` can legitimately be EMPTY (every level failed its
   *   ``getObjectByName`` attach in ``load-lod-group-node``, which warns and
   *   carries on). Nothing at a non-existent index can ever become ready, so
   *   blocking on it is the permanently-false trap again: the drain would burn
   *   its whole budget on every frame and then report a degraded-LOD verdict
   *   the scene never earned. An out-of-range ``desiredChildIndex`` is the same
   *   shape but is NOT skipped — the rest of the entry is still checked and
   *   only the missing ``desired`` is let through (see its bullet below).
   * - ``displayed !== activeChildIndex`` ⇒ not quiescent. A stale slice
   *   fallback or a never-downgrade hold is on screen instead of the
   *   aspiration, so what renders is not what the selector settled on.
   * - ``desired !== activeChildIndex`` ⇒ not quiescent — UNLESS that desired
   *   child is ``failed``, or absent (an out-of-range ``desired``, handled
   *   right here rather than by skipping the entry). The aspiration only
   *   advances onto a READY level, so this is the one-frame window after a lazy
   *   load lands but before the next selector pass swaps (see
   *   ``LODGroupEntry.desiredChildIndex``); it is also the whole in-flight
   *   load. A ``failed`` level can never become ready this frame, so blocking
   *   on it only buys a timeout — treat it as the best available and keep
   *   checking the rest.
   * - The aspiration must be ``isReady``.
   * - When freshness is tracked (``getViewVersion`` wired), the aspiration must
   *   be FRESH for the current view version — via the group-aware
   *   {@link childFreshAndCount}, not the leaf-only ``isFresh``, so a deferred
   *   ``kind=partition`` subtree stamped for an older slice counts as stale.
   * - The aspiration's additive ladder must be complete, on BOTH available
   *   signals. A lazy child's live ``hasMoreLODs()`` thunk answers first:
   *   still true means only a prefix of the level has committed. Then
   *   {@link childFreshAndCount}'s ``subtreeLadderComplete`` — the
   *   commit-time ``committedLadderComplete`` stamp, taken from the child
   *   itself for a tracked LEAF and folded over the visible stamped leaves of
   *   a deferred GROUP child (a nested ``kind=partition`` / ``kind=lod``
   *   subtree — the ``overview`` recipe's fine branch). Neither alone is
   *   enough: a group child carries no thunk, so without the fold the drain
   *   released the frame the moment such a branch became ready, with its part
   *   leaves at chunk-1 by construction; and only the DEFERRED path gets a
   *   thunk, so without the stamp the eagerly-loaded default level — still
   *   climbing its ladder under the sweep-driven refinement loop — read as
   *   complete. Anything with no stamp at all (never committed, or a
   *   non-progressive loader) carries no signal and counts as complete.
   * - No child of the entry may be ``loading`` — an in-flight commit can change
   *   what renders on a later frame.
   *
   * An empty registry (and an entry-free scene) is quiescent: there is nothing
   * to wait for.
   *
   * Called from the capture drain — once per drain rAF, so up to the drain's
   * own frame cap (``LOD_SETTLE_MAX_FRAMES``, which is itself INCLUSIVE of the
   * mandatory catch-up tick) plus one for the strategy's opening tri-state
   * probe, per exported frame in the worst case; twice on a scene that is
   * already settled (probe + one poll), and once on a latched frame (the
   * re-arm probe alone). Either way NOT the rAF hot path, so unlike the rest of this file
   * it does not avoid allocation: it walks the entry Map with ``for…of`` and
   * resolves freshness through ``childFreshAndCount``, which returns a fresh
   * object per entry. That cost is genuinely irrelevant here.
   */
  isCaptureQuiescent(): boolean {
    // Wired to SceneLoader.isLoadPassInProgress: any pass can still change a
    // visible partition part before this fixed-pose capture frame is exported.
    if (this.anyVisiblePartitionPart() && this.deps.isUpdateInProgress?.() === true) return false;
    if (this.deps.hasArchiveFault?.()) return !this.anyChildLoading();
    const version = this.deps.getViewVersion?.();
    if (!this.partitionsCaptureQuiescent(version ?? null)) return false;
    for (const entry of this.entries.values()) {
      // Deliberately excluded: an off-screen group is held coarse on purpose
      // and contributes no pixels to the frame being captured.
      if (entry.offScreen === true) continue;
      // Likewise excluded, and this one is load-bearing rather than merely an
      // optimisation: a hidden group draws nothing AND cannot start a deferred
      // load (``kickDeferredLoadIfVisible``), while the selector — whose
      // frustum test is pure geometry — happily records a fine
      // ``desiredChildIndex`` for it. Blocking on that combination never
      // resolves.
      if (!isEffectivelyVisible(entry.groupObject)) continue;

      const active = entry.activeChildIndex;
      const desired = entry.desiredChildIndex ?? active;
      const displayed = entry.displayedChildIndex ?? active;

      const aspiration = entry.children[active];
      // Degenerate entry — skipped, not blocked on. ``children`` is empty when
      // every level failed its ``getObjectByName`` attach at load (the loader
      // warns and continues), and an aspiration index can otherwise point past
      // the end. There is no child to become ready, so returning false here
      // would make the predicate PERMANENTLY false: every capture frame would
      // spend the full drain budget and the run would end claiming frames were
      // filmed at a coarse LOD, on a scene that has no level to wait for.
      if (!aspiration) continue;

      if (displayed !== active) return false;
      if (desired !== active) {
        const target = entry.children[desired];
        // A failed level never becomes ready, so waiting on it only times out.
        // An out-of-range ``desired`` (no child there at all) is the same trap
        // as the missing aspiration above and likewise must not block.
        if (target && target.failed !== true) return false;
      }

      if (!isReady(aspiration)) return false;
      // One fold for both group-aware answers: per-slice freshness AND — for a
      // deferred GROUP child, which has no ``hasMoreLODs`` thunk — whether its
      // subtree's committed additive ladders are complete.
      const progress = this.childFreshAndCount(aspiration, version ?? null);
      if (version != null && !progress.fresh) return false;
      if (aspiration.hasMoreLODs?.() === true) return false;
      if (!progress.subtreeLadderComplete) return false;

      for (let i = 0; i < entry.children.length; i++) {
        if (entry.children[i].loading) return false;
      }
    }
    return true;
  }

  /** Whether visible partition parts have no pending resync or incomplete commit. */
  private partitionsCaptureQuiescent(version: number | null): boolean {
    if (!this.pendingPartitionResyncsQuiescent()) return false;
    for (const entry of this.partitionEntries.values()) {
      if (!this.partitionEntryCaptureQuiescent(entry, version)) return false;
    }
    return true;
  }

  private pendingPartitionResyncsQuiescent(): boolean {
    for (const [path, parts] of this.partitionResyncPending) {
      const entry = this.partitionEntries.get(path);
      if (!entry) {
        continue;
      }
      if (this.pendingPartitionResyncContributes(entry, parts)) return false;
    }
    return true;
  }

  private pendingPartitionResyncContributes(
    entry: PartitionGroupEntry,
    parts: ReadonlySet<string>
  ): boolean {
    if (!isEffectivelyVisible(entry.groupObject)) return false;
    if (parts.has(entry.path)) return this.partitionHasVisiblePart(entry);
    return entry.children.some(
      (child) => parts.has(child.path) && this.partitionChildIsVisible(child)
    );
  }

  private partitionEntryCaptureQuiescent(
    entry: PartitionGroupEntry,
    version: number | null
  ): boolean {
    if (!isEffectivelyVisible(entry.groupObject)) return true;
    for (const child of entry.children) {
      if (!this.partitionChildCaptureQuiescent(child, version)) return false;
    }
    return true;
  }

  private partitionChildCaptureQuiescent(
    child: PartitionGroupChild,
    version: number | null
  ): boolean {
    if (!this.partitionChildIsVisible(child)) return true;
    for (const object of child.objects) {
      const progress = subtreeDisplayProgress(object as unknown as ProgressNode, version);
      if (progress && (!progress.fresh || !progress.complete)) return false;
    }
    return true;
  }

  /** Whether any registered partition part contributes pixels to this frame. */
  private anyVisiblePartitionPart(): boolean {
    for (const entry of this.partitionEntries.values()) {
      if (!isEffectivelyVisible(entry.groupObject)) continue;
      if (this.partitionHasVisiblePart(entry)) return true;
    }
    return false;
  }

  private partitionHasVisiblePart(entry: PartitionGroupEntry): boolean {
    return entry.children.some((child) => this.partitionChildIsVisible(child));
  }

  private partitionChildIsVisible(child: PartitionGroupChild): boolean {
    return child.objects.some((object) => object.userData.partitionFrustumVisible !== false);
  }

  /**
   * Whether any lazy LOD-group level has an `ensureLoaded` fetch in flight.
   * These promotions run outside every `updateView` cycle, so neither
   * `isUpdateInProgress()` nor `getState().isLoading` sees them; the perf
   * snapshot's `isSettled` does. (Deferred partition parts load through
   * `updateView` and are covered by the update lock instead.)
   */
  isAnyLevelLoading(): boolean {
    return this.anyChildLoading();
  }

  /**
   * Whether a visible partition has a rising-edge resync waiting for the
   * owning loader to become idle. Pending work retained under a hidden wrapper
   * is not actionable and must not keep wide settledness false indefinitely;
   * neither can work when no resync dispatcher is wired.
   */
  hasVisiblePendingPartitionResync(): boolean {
    if (!this.deps.requestReprocess) return false;
    for (const path of this.partitionResyncPending.keys()) {
      const entry = this.partitionEntries.get(path);
      if (entry && isEffectivelyVisible(entry.groupObject)) return true;
    }
    return false;
  }

  private anyChildLoading(): boolean {
    for (const entry of this.entries.values()) {
      for (const child of entry.children) {
        if (child.loading) return true;
      }
    }
    return false;
  }

  /**
   * Retry a LAZY lod_group level by its authored scene-node path. The path is
   * stored beside the placeholder in ``LODGroupChild`` so anonymous deferred
   * GROUP placeholders remain addressable without duplicating scene identity
   * onto the THREE object.
   *
   * Clears the failure cooldown (``failed``/``failedTick``) through the shared
   * ``kickDeferredLoad`` gate, which owns setting ``loading`` before firing
   * ``ensureLoaded`` (the thunk itself never sets ``loading`` — only the
   * registry does; keep that invariant here).
   *
   * Returns ``true`` when a retry was kicked OR one is already in flight
   * (``loading``), ``false`` when no retryable lazy child with that path
   * exists. Explicit retries bypass the owning loader's automatic archive-fault
   * gate; the per-child marker keeps concurrent failed branches independently
   * targetable.
   * Fire-and-forget semantics: ``true`` means "retry started", not "retry
   * succeeded" — the thunk owns the ready/failed outcome, and a repeat
   * failure re-enters the normal cooldown cycle.
   */
  retryLazyChildByNodePath(path: string): boolean {
    if (!path) return false;
    for (const entry of this.entries.values()) {
      for (const child of entry.children) {
        if (child.nodePath !== path || !child.ensureLoaded) continue;
        if (child.loading) return true; // retry already in flight
        return this.kickDeferredLoad(child, true);
      }
    }
    return false;
  }

  /**
   * Update an lod_group's selector mode. ``'auto'`` re-enables
   * view-driven selection; ``{ lockLevel: i }`` pins the lod_group to
   * child index ``i`` (0-based in coarsest→finest order). An
   * out-of-range ``lockLevel`` is **clamped** into ``[0, n-1]`` with a
   * warning — throwing here would force every UI caller to guard
   * against stale registry state.
   *
   * Visibility is *not* swapped synchronously — the next
   * ``evaluatePerFrame()`` call will pick the new desired child. (This
   * matches the per-frame contract for the auto path; a synchronous
   * swap would diverge.)
   */
  setSelectorMode(path: string, mode: LODGroupSelectorMode): void {
    const entry = this.entries.get(path);
    if (!entry) return;
    if (mode === 'auto') {
      entry.selectorMode = mode;
      return;
    }
    const n = entry.children.length;
    if (n === 0) {
      entry.selectorMode = mode;
      return;
    }
    const clamped = Math.max(0, Math.min(mode.lockLevel, n - 1));
    if (clamped !== mode.lockLevel) {
      log.warning(
        Modules.SCENE_LOADER,
        `lod_group ${path}: lockLevel ${mode.lockLevel} out of range ` +
          `[0, ${n - 1}], clamped to ${clamped}`
      );
    }
    entry.selectorMode = { lockLevel: clamped };
  }

  /**
   * Per-frame evaluation. Wired through
   * ``AnimationController.addPerFrameCallback`` by the app pipeline.
   *
   * Returns ``true`` when at least one lod_group swapped its active
   * child this frame, so the caller can refresh anything that depends
   * on which level renders (e.g. the data-monitor's visible-element
   * tally). Returns ``false`` on a no-op frame (the common case), which
   * keeps the per-frame cost to the projection math alone.
   */
  evaluatePerFrame(): boolean {
    if (this.entries.size === 0 && this.partitionEntries.size === 0) return false;
    const camera = this.deps.getCamera();
    const viewport = this.deps.getViewportSize();
    const displayDims = this.deps.getDisplayDims();
    if (viewport.width === 0 || viewport.height === 0) return false;
    if (displayDims.length < 2) return false;

    this.tick++;
    // Build the camera frustum once per frame. ``camera.matrixWorldInverse`` /
    // ``projectionMatrix`` are current here (``controls.update()`` →
    // ``updateMatrixWorld()`` runs before per-frame callbacks), and the default
    // WebGL coordinate system matches the NDC convention used by the manual
    // projection×view divide inside ``projectBoxDiagonalPx``. The projection×view
    // product (``FRUSTUM_MATRIX_SCRATCH``) is shared three ways: it seeds the
    // frustum for the off-screen LOD gate and eviction ranking, and is passed
    // into ``projectBoxDiagonalPx`` so the per-group pixel-diagonal reuses it.
    FRUSTUM_MATRIX_SCRATCH.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    FRUSTUM_SCRATCH.setFromProjectionMatrix(FRUSTUM_MATRIX_SCRATCH);
    PARTITION_FRUSTUM_MATRIX_SCRATCH.copy(FRUSTUM_MATRIX_SCRATCH).premultiply(
      PARTITION_FRUSTUM_SCALE
    );
    PARTITION_FRUSTUM_SCRATCH.setFromProjectionMatrix(PARTITION_FRUSTUM_MATRIX_SCRATCH);

    // Track whether the (global) view version has settled, to gate deferred
    // fine-level reloads (see ``evaluateEntry``). ``undefined`` view version
    // (no wiring / tests) ⇒ treat as settled so the trigger is inert.
    const version = this.deps.getViewVersion?.();
    if (version != null) this.settleTracker.observe(version, this.tick);
    const settled =
      version == null || this.settleTracker.isSettled(this.tick, FINE_RELOAD_SETTLE_TICKS);

    let changed = false;
    let anyLoading = false;
    let hasVisiblePendingResync = false;
    for (const entry of this.partitionEntries.values()) {
      if (!isEffectivelyVisible(entry.groupObject)) continue;
      RISING_PARTS_SCRATCH.clear();
      const result = this.evaluatePartitionEntry(
        entry,
        displayDims,
        PARTITION_FRUSTUM_SCRATCH,
        RISING_PARTS_SCRATCH
      );
      if ((result & PARTITION_VISIBILITY_CHANGED) !== 0) changed = true;
      if ((result & PARTITION_BECAME_VISIBLE) !== 0) {
        this.notePartitionRisingEdge(entry.path, RISING_PARTS_SCRATCH);
      }
      if (this.partitionResyncPending.has(entry.path)) hasVisiblePendingResync = true;
    }
    RISING_PARTS_SCRATCH.clear();
    if (this.partitionResyncPending.size > 0 && this.deps.isUpdateInProgress?.() !== true) {
      this.flushPartitionResyncs();
    }
    if (hasVisiblePendingResync && this.partitionResyncPending.size > 0) {
      this.deps.requestRender?.();
    }
    for (const entry of this.entries.values()) {
      if (this.evaluateEntry(entry, camera, viewport, displayDims, FRUSTUM_SCRATCH, settled)) {
        changed = true;
      }
      // A lazy level loading (initial or a settled fine reload) commits
      // asynchronously OUTSIDE the per-slice sweep. Keep the on-demand render
      // loop alive so the per-frame swap-up to the fresh level fires when the
      // load lands, rather than waiting for the next user interaction. Plain
      // loop (not ``.some``) to preserve this file's no-per-frame-allocation
      // hot-path invariant.
      if (!anyLoading) {
        for (const c of entry.children) {
          if (c.loading) {
            anyLoading = true;
            break;
          }
        }
      }
    }
    if (anyLoading) this.deps.requestRender?.();
    // Record whether fade management was ON this frame — the falling-edge
    // detector behind ``evaluateEntry``'s one-shot residual-opacity restore
    // (see ``fadeWasManaged``). Written AFTER the entry loop so every entry in
    // one frame sees the same previous-frame value.
    this.fadeWasManaged =
      this.deps.getCrossFadeEnabled?.() === true || this.deps.getEnergyCompEnabled?.() === true;
    // Bound resident LOD geometry against the shared GPU-pool byte budget
    // (one VRAM authority). Retention keeps loaded levels resident so
    // re-shows are free; this LRU-evicts only hidden levels when over budget —
    // hidden-layer (undrawable) first, then off-screen / furthest-from-camera —
    // so no per-swap release, hence no reload churn.
    this.enforceByteBudget(camera, FRUSTUM_SCRATCH, displayDims);
    return changed;
  }

  /**
   * Rising edge (rare): remember WHICH parts of ``wrapperPath`` came back so
   * the resync can be targeted at their loaders instead of re-sweeping the
   * whole scene. Coalesces with parts already pending for the same wrapper.
   */
  private notePartitionRisingEdge(wrapperPath: string, parts: ReadonlySet<string>): void {
    let pending = this.partitionResyncPending.get(wrapperPath);
    if (!pending) {
      pending = new Set<string>();
      this.partitionResyncPending.set(wrapperPath, pending);
    }
    for (const partPath of parts) pending.add(partPath);
  }

  /**
   * Hand every pending rising edge whose wrapper is visible to the loader as
   * ONE ``requestReprocess(paths)`` call; hidden wrappers stay pending, dropped
   * wrappers are forgotten. A set that contains its own wrapper path (a
   * pathless part) collapses to the wrapper path alone — it already covers
   * every part.
   */
  private flushPartitionResyncs(): void {
    const requestReprocess = this.deps.requestReprocess;
    if (!requestReprocess) return;
    let resyncPaths: string[] | null = null;
    for (const [path, parts] of this.partitionResyncPending) {
      const entry = this.partitionEntries.get(path);
      if (!entry) {
        this.partitionResyncPending.delete(path);
        continue;
      }
      if (!isEffectivelyVisible(entry.groupObject)) continue;
      this.partitionResyncPending.delete(path);
      resyncPaths ??= [];
      if (parts.has(path)) resyncPaths.push(path);
      else resyncPaths.push(...parts);
    }
    if (resyncPaths) requestReprocess(resyncPaths);
  }

  /**
   * Frustum-gate one partition's parts. Returns the ``PARTITION_*`` bit flags;
   * parts that re-entered this frame are added to ``risingParts`` by node path
   * (``child.path``), or as the WRAPPER path when a part has none so the caller
   * resyncs the whole partition rather than missing it.
   */
  private evaluatePartitionEntry(
    entry: PartitionGroupEntry,
    displayDims: readonly number[],
    frustum: THREE.Frustum,
    risingParts: Set<string>
  ): number {
    const cache = this.partitionCaches.get(entry.path);
    if (!cache) return 0;
    let result = 0;
    for (let index = 0; index < entry.children.length; index++) {
      const child = entry.children[index];
      const childCache = cache.children[index];
      const worldBox = computeEntryWorldBox(
        childCache.source,
        displayDims,
        childCache.localBoxScratch,
        this.matrixScratch,
        childCache.worldBoxOptions
      );
      let visible = true;
      if (worldBox) {
        WORLD_BOX3_SCRATCH.min.set(worldBox.min.x, worldBox.min.y, worldBox.min.z);
        WORLD_BOX3_SCRATCH.max.set(worldBox.max.x, worldBox.max.y, worldBox.max.z);
        for (const object of child.objects) object.updateWorldMatrix(false, false);
        if (
          childCache.footprintDirty ||
          partitionFootprintMatricesDiffer(child.objects, childCache.footprintMatrixWorld)
        ) {
          capturePartitionFootprint(
            child.objects,
            childCache.footprintBox,
            childCache.footprintMatrixWorld
          );
          childCache.footprintDirty = false;
        }
        if (!childCache.footprintBox.isEmpty()) {
          WORLD_BOX3_SCRATCH.union(childCache.footprintBox);
        }
        visible = frustum.intersectsBox(WORLD_BOX3_SCRATCH);
      }
      const flags = updatePartitionObjectVisibility(child.objects, visible);
      result |= flags;
      if ((flags & PARTITION_BECAME_VISIBLE) !== 0) {
        noteRisingPart(risingParts, child.path, entry.path);
      }
    }
    return result;
  }

  /** Returns ``true`` if this entry's displayed child changed. */
  private evaluateEntry(
    entry: LODGroupEntry,
    camera: THREE.Camera,
    viewport: { width: number; height: number },
    displayDims: readonly number[],
    frustum: THREE.Frustum,
    settled: boolean
  ): boolean {
    // Pick the desired child index.
    let desired: number;
    // The dimensionless coverage metric for this frame (projected diagonal ÷
    // FILL_FACTOR·fittedAxisPx), hoisted so the coverage-band cross-fade below
    // can blend around a boundary. -1 ⇒ not computed (locked / off-screen).
    let coverageMetric = -1;
    if (entry.selectorMode !== 'auto') {
      // Explicit lock bypasses the off-screen gate: a user who pins a level
      // keeps it whether or not the group is on screen.
      desired = entry.selectorMode.lockLevel;
      entry.offScreen = false;
    } else {
      const cache = this.caches.get(entry.path);
      if (!cache) return false; // shouldn't happen — register() populates this.

      const worldBox = this.computeWorldBox(entry, displayDims);
      if (!worldBox) return false;

      // Off-screen gate: if the group's world bounds are entirely outside the
      // camera frustum, hold it at the coarsest *ready* level instead of
      // selecting — and lazily loading — a fine level the renderer will
      // frustum-cull anyway. This turns frustum culling into a LOD/loading
      // input, not just a draw-time skip. ``coarsestReadyIndex`` never targets
      // a not-ready lazy child, so no load is kicked while off-screen, and the
      // outgoing fine levels fall out of the visible tally and become eviction
      // candidates. On re-entry, retention usually makes the upgrade a free
      // visibility toggle rather than a reload.
      WORLD_BOX3_SCRATCH.min.set(worldBox.min.x, worldBox.min.y, worldBox.min.z);
      WORLD_BOX3_SCRATCH.max.set(worldBox.max.x, worldBox.max.y, worldBox.max.z);
      const forceFinest = this.deps.getForceFinestLOD?.() === true;
      if (!forceFinest && !frustum.intersectsBox(WORLD_BOX3_SCRATCH)) {
        desired = this.coarsestReadyIndex(entry);
        entry.offScreen = true;
      } else {
        if (forceFinest) {
          coverageMetric = Infinity;
        } else {
          const metricWorldBox = cache.hasLodBounds
            ? (this.computeWorldBox(entry, displayDims, true) ?? worldBox)
            : worldBox;
          if (entry.selector === 'screen-area') {
            // Screen-area selector: the metric IS the fraction of the viewport
            // area the group's projected bbox rect covers (viewport-size
            // independent by construction — see projectBoxAreaFraction). The
            // thresholds are literal area fractions ([0, …, 1/4, 1/2] whole-object;
            // a partition tile anchors at 1.0), so no FILL_FACTOR normalisation.
            // Camera inside the box → +Infinity → finest, same as the diagonal path.
            coverageMetric = projectBoxAreaFraction(metricWorldBox, camera, FRUSTUM_MATRIX_SCRATCH);
            if (cache.hasLodBounds) {
              // The thin-rectangle ramp is not monotone under box containment:
              // trimming the thin axis can increase the robust metric. Robust
              // bounds may only keep or reduce the raw-bounds selection.
              coverageMetric = Math.min(
                coverageMetric,
                projectBoxAreaFraction(worldBox, camera, FRUSTUM_MATRIX_SCRATCH)
              );
            }
          } else {
            // Legacy 'coverage' selector (the default for older stores).
            // Reuse the per-frame projection×view product (FRUSTUM_MATRIX_SCRATCH,
            // built in evaluatePerFrame) instead of recomputing it per group.
            const diagonalPx = projectBoxDiagonalPx(
              metricWorldBox,
              camera,
              viewport,
              FRUSTUM_MATRIX_SCRATCH
            );
            // Normalise the projected pixel diagonal to a dimensionless **coverage
            // metric** (1.0 == the projected diagonal has reached FILL_FACTOR of the
            // FITTED AXIS) so the viewport-relative coverage_fraction thresholds
            // anchor the finest at half the fitted screen axis — any normal
            // full-frame view — on any monitor OR aspect ratio (see the
            // ``FILL_FACTOR`` doc for why this denominator, unlike the viewport
            // diagonal it replaces, stays invariant across aspect ratio).
            // diagonalPx == +Infinity (camera inside the box) → Infinity →
            // finest, unchanged. fittedAxisPx is > 0 here (evaluatePerFrame guards
            // width/height == 0).
            //
            // fittedAxisPx mirrors calculateCameraDistance's own fit selection
            // (bounds-math.ts): that function fits the VERTICAL fov for aspect >= 1
            // (distance independent of width) and the HORIZONTAL fov for aspect < 1
            // (distance ∝ 1/aspect) — i.e. ``min(width, height)`` in pixel space is
            // exactly the extent the opening framing fits, on both sides of aspect 1.
            const fittedAxisPx = Math.min(viewport.width, viewport.height);
            coverageMetric = diagonalPx / (FILL_FACTOR * fittedAxisPx);
          }
        }
        desired = pickChildWithHysteresis(cache.thresholds, entry.activeChildIndex, coverageMetric);
        entry.offScreen = false;
      }
    }

    // Record what the selector WANTS this frame, before any of the ready /
    // freshness gates below can veto it. Read by ``isCaptureQuiescent`` only —
    // see ``LODGroupEntry.desiredChildIndex`` for why the aspiration index
    // cannot answer the same question.
    entry.desiredChildIndex = desired;

    // ── Advance the aspiration (``activeChildIndex``) toward ``desired`` ──
    // The aspiration is the hysteresis anchor and only moves onto a READY level;
    // a not-ready desired kicks its deferred loader and we keep aspiring to the
    // current level until it commits. Visibility is NOT touched here — the
    // display-resolution pass below is the single owner of ``object.visible``.
    if (desired !== entry.activeChildIndex) {
      const target = entry.children[desired];
      // `undefined` when a lock index outlives its children (an empty entry is
      // registered on purpose and `setSelectorMode` skips clamping for it):
      // nothing to aspire to, nothing to kick — never throw per frame.
      if (!target) return false;
      if (isReady(target)) {
        entry.activeChildIndex = desired;
      } else {
        this.maybeKickLoad(entry, target);
      }
    }
    // Self-heal a NOT-ready aspiration (eager default failed to attach, or a
    // fallback pinned a not-ready lazy level) so the group can never be stuck.
    // A ready-but-STALE aspiration is deliberately NOT kicked here:
    // ``maybeKickLoad`` early-returns on ready children, and a stale level's
    // re-slice reload is driven by the scene-loader update sweep (every
    // registered child loader re-queries on a view change), not by the registry
    // — we just wait for that commit to re-stamp it fresh.
    const aspiration = entry.children[entry.activeChildIndex];
    if (aspiration && !isReady(aspiration)) this.maybeKickLoad(entry, aspiration);

    // ── Slice-aware DISPLAY resolution ──
    // Show the aspiration when its committed geometry is fresh for the current
    // view version; while it is stale (a time/displayDims scrub reloaded it in
    // place without flipping ``ready``) show the coarsest FRESH level so the new
    // slice appears immediately at low detail, then swap up once the aspiration
    // recommits. ``getViewVersion`` undefined ⇒ freshness untracked ⇒
    // display == aspiration (identical to the pre-feature behaviour).
    const version = this.deps.getViewVersion?.();
    const aspirationReady = !!aspiration && isReady(aspiration);
    // Freshness resolves a GROUP-typed aspiration (deferred kind=partition /
    // nested lod subtree — the overview recipe) through the subtree aggregate,
    // not the leaf-only stamp: a bare THREE.Group has no leaf nodeType, so
    // ``isFresh`` would call it unconditionally fresh and a re-slice would show
    // the stale subtree with no coarse fallback. Leaf aspirations are unchanged.
    const aspirationFresh =
      version == null || (!!aspiration && this.childFreshAndCount(aspiration, version).fresh);
    // Preserve this across the fresh-aspiration branch below, which re-arms
    // the hold state before the never-downgrade gate evaluates the handoff.
    const staleHoldEnded = aspirationReady && aspirationFresh && entry.staleHoldSinceMs != null;
    let displayIdx: number;
    if (aspirationReady && aspirationFresh) {
      // Aspiration is committed and fresh (or freshness untracked) → show it.
      // The wait is over, so a spent stale-hold budget is re-armed for the
      // NEXT slice change (see ``staleHoldExhausted``).
      entry.staleHoldSinceMs = undefined;
      entry.staleHoldExhausted = false;
      displayIdx = entry.activeChildIndex;
    } else if (version != null) {
      // Stale or not-yet-ready aspiration, freshness tracked → display the
      // coarsest fresh level (the slice-aware fallback; falls back to the
      // coarsest ready level if none is fresh yet, so it never goes blank)...
      const fallbackIdx = this.coarsestFreshOrReadyIndex(entry, version);
      // ...unless what is already on screen is far better and the aspiration
      // is about to land, in which case hold it for a few frames instead of
      // flashing down and back up (``staleHoldDisplayIndex``).
      const held = this.staleHoldDisplayIndex(entry, fallbackIdx, version);
      displayIdx = held ?? fallbackIdx;
    } else {
      // Freshness untracked and the aspiration isn't ready (a lazy level still
      // loading): keep the previously-displayed level if it's still ready,
      // otherwise show nothing until the load commits — the legacy behaviour.
      const prev = entry.displayedChildIndex ?? -1;
      displayIdx = prev >= 0 && isReady(entry.children[prev]) ? prev : -1;
    }

    // ── Fresh-but-EMPTY display guard ──
    // With consistent LOD data a finer level can never be empty where a
    // coarser one is not (coarse levels are derived from fine), so a fresh
    // level that committed 0 elements while another fresh level holds visible
    // geometry signals inconsistent/corrupt data (e.g. a stale cache serving
    // an old layout whose chunk queries zero-fill). Displaying the empty level
    // would silently blank the group; redirect to the coarsest fresh NON-empty
    // level and warn once so the inconsistency is visible instead of black.
    // A genuinely empty slice (every fresh level empty) is unchanged.
    if (version != null && displayIdx >= 0) {
      const chosen = entry.children[displayIdx];
      // Group-aware: a deferred kind=partition / nested lod subtree whose visible
      // stamped leaves are all fresh-but-empty (poisoned/stale cache serving an
      // old layout) would otherwise slip past the leaf-only ``visibleElementCount
      // === 0`` check and blank the group. ``childFreshAndCount`` folds the
      // subtree so the guard fires for a group chosen too; the redirect target
      // stays the coarsest fresh non-empty leaf level.
      const chosenProgress = chosen ? this.childFreshAndCount(chosen, version) : undefined;
      if (chosen && chosenProgress?.fresh && chosenProgress.count === 0) {
        const fallback = this.coarsestFreshNonEmptyIndex(entry, version);
        if (fallback >= 0 && fallback !== displayIdx) {
          if (!this.warnedEmptyLevel.has(entry.path)) {
            this.warnedEmptyLevel.add(entry.path);
            const recovery = this.deps.hasNetworkFailureUnder?.(entry.path)
              ? 'A network load failed under this group; use the monitor Retry action.'
              : 'This usually means inconsistent/stale data (e.g. a dataset regenerated at ' +
                'the same URL with a poisoned cache); try reloading with ?clear-cache.';
            log.warning(
              Modules.SCENE_LOADER,
              `lod_group ${entry.path}: level ${displayIdx} is fresh but committed 0 ` +
                `elements while level ${fallback} has visible geometry — showing level ` +
                `${fallback} instead. ${recovery}`
            );
          }
          displayIdx = fallback;
        }
      }
    }

    // ── Never-downgrade display gate ──
    // A lazy level flips ``ready`` after its FIRST additive chunk commits, so
    // an ungated swap to a fresh-but-still-streaming aspiration pops displayed
    // quality down to chunk-1 and climbs back. Hold the previously-displayed
    // level while the streaming aspiration is strictly worse than what is on
    // screen; release on ladder completion, committed-count crossover, ladder
    // failure, or the previous level losing freshness. Bypassed for an explicit
    // lock and while off-screen. This is a STREAMING/loading concern (WHEN a
    // just-loaded level is good enough to show) — orthogonal to the
    // distance-driven cross-fade below, and stays a hard hold.
    if (
      displayIdx === entry.activeChildIndex &&
      entry.selectorMode === 'auto' &&
      !entry.offScreen
    ) {
      // Read the gate's memory (last ON-SCREEN displayed level), NOT
      // ``displayedChildIndex`` — the latter is clobbered to the coarse level
      // during an off-screen excursion, which would defeat the hold on return.
      let prevIdx = entry.heldDisplayChildIndex;
      // A stale hold keeps the aspiration itself in the display memory. When
      // its first fresh prefix lands, compare that prefix against the fresh
      // fallback the hold displaced; otherwise prevIdx === displayIdx skips
      // the never-downgrade gate and can reveal less geometry than fallback.
      if (prevIdx === displayIdx && staleHoldEnded && version != null) {
        prevIdx = this.coarsestFreshOrReadyIndex(entry, version);
      }
      if (prevIdx != null && prevIdx !== displayIdx) {
        const prev = entry.children[prevIdx];
        // Children are coarsest→finest, so displayIdx (== activeChildIndex) being
        // FINER than the held prev means an upgrade (zoom-in); the gate's early
        // energy-release is sound only then (downgrade would pop below the held).
        const isUpgrade = displayIdx > prevIdx;
        if (shouldHoldPreviousDisplay(aspiration!, prev, version ?? null, isUpgrade)) {
          displayIdx = prevIdx;
          aspiration!.lastVisibleTick = this.tick;
        }
      }
    }

    // ── Coverage-band cross-fade (distance-driven) ──
    // As the camera zooms across a LOD boundary, blend the two levels straddling
    // it — coarser at (1−w), finer at w = smoothstep of the coverage metric
    // across a ±band around the boundary — so the substitutive switch dissolves
    // instead of popping. Purely a function of DISTANCE (the coverage metric),
    // independent of additive streaming; brightness is preserved by the levels'
    // build-time mass conservation (both integrate to the same DC). Blendable
    // modes only (BLENDABLE_MODES = additive/luminous/volumetric — energy sums
    // linearly, or opacity linearly scales optical depth τ so the pair
    // interpolates monotonically between the two levels' absorptions; see that
    // set's doc for what volumetric does NOT guarantee). For two mid-fade
    // volumetric siblings the mesh draw order may come from the render-order
    // containment rule (near-identical bounds); acceptable because combined
    // TRANSMITTANCE is order-independent (transmittances multiply), so occlusion
    // of content behind the pair is exact at every weight. If strict containment
    // does not fire for near-co-located siblings, the view-z fallback chooses
    // their order; in the thick regime camera motion can therefore flip the
    // mid-fade draw order and emission result. Emission's ordering residual is
    // the product of the levels' w/(1−w)-scaled per-fragment alphas and their
    // local radiance difference: always first order in that difference, and
    // second order in the alphas only in the optically thin regime. Individually
    // thick fragments can saturate both alphas and remove that suppression, but
    // the residual magnitude remains bounded by the local inter-level radiance
    // difference, not by the rendered hard-swap pop.
    // Off / non-blendable /
    // off-screen / locked / a held-stale display ⇒ no blend (byte-identical hard
    // swap). The finer partner must be resident to fade against; if it is not,
    // kick its load so the NEXT crossing blends (the first hard-swaps meanwhile).
    let blendPartnerIdx: number | null = null;
    let primaryWeight = 1;
    if (
      this.deps.getCrossFadeEnabled?.() === true &&
      entry.selectorMode === 'auto' &&
      !entry.offScreen &&
      aspirationFresh &&
      displayIdx === entry.activeChildIndex &&
      coverageMetric >= 0
    ) {
      const cache = this.caches.get(entry.path);
      const plan = cache
        ? coverageBlendPlan(cache.thresholds, coverageMetric, CROSSFADE_BAND_FRACTION)
        : null;
      if (
        plan &&
        plan.hiWeight > FADE_EPSILON &&
        plan.hiWeight < 1 - FADE_EPSILON &&
        (displayIdx === plan.lo || displayIdx === plan.hi)
      ) {
        const partnerIdx = displayIdx === plan.hi ? plan.lo : plan.hi;
        const partner = entry.children[partnerIdx];
        const partnerFresh = version == null || this.childFreshAndCount(partner, version).fresh;
        const aspBlendable = this.isBlendable(aspiration!);
        if (
          partner &&
          isReady(partner) &&
          partnerFresh &&
          aspBlendable &&
          this.isBlendable(partner)
        ) {
          blendPartnerIdx = partnerIdx;
          // primaryWeight is the OPACITY of the primary (displayIdx); the plan's
          // hiWeight is the FINER level's opacity, mapped to whichever is primary.
          primaryWeight = displayIdx === plan.hi ? plan.hiWeight : 1 - plan.hiWeight;
          // Keep both warm in the eviction LRU (both are on screen). The blend
          // weight is a function of the coverage metric (camera distance), so
          // camera motion already keeps the on-demand loop awake through the
          // band; a parked camera settles on the correct static blended frame.
          partner.lastVisibleTick = this.tick;
          aspiration!.lastVisibleTick = this.tick;
        } else if (partner && !isReady(partner) && aspBlendable) {
          // Approaching a not-yet-resident finer level: load it so the next
          // crossing can blend (this crossing hard-swaps while it loads).
          this.maybeKickLoad(entry, partner);
        }
      }
    }

    // ── Settle-gated reload / progressive refinement of the lazy aspiration ──
    // A lazy level no longer joins the per-slice sweep (see load-lod-group-node.ts),
    // so the registry drives its (re)loading HERE, settle-gated: during active
    // scrubbing (version changing every frame) nothing fires, so only the cheap
    // coarse level (still sweep-driven) shows the new slice; once the user pauses
    // we (a) reload a STALE level for the new slice, and (b) advance a PROGRESSIVE
    // level that is fresh but still has additive LODs to stream — both by
    // re-firing the same ``ensureLoaded``, until the level is ready, fresh, AND
    // complete. Eager (coarse) levels have no ``ensureLoaded`` and stay
    // sweep-driven, so this only ever targets lazy levels.
    // Never for a deferred GROUP child (the overview recipe's fine partition /
    // nested lod branch): it has an ``ensureLoaded`` too, but its expensive
    // step is ``loadChildren`` — re-running it attaches a SECOND copy of the
    // whole subtree under the placeholder (double-drawn geometry, duplicate
    // names, re-registered loaders, leaked buffers). Its leaves are
    // sweep-registered and re-stamp themselves, so staleness needs no kick.
    // Keyed on the explicit ``deferredGroup`` flag, not on the leaf's
    // ``nodeType`` stamp: a lazy leaf whose placeholder is not stamped yet
    // must still drain its ladder here.
    const needsReloadOrRefine =
      aspirationReady &&
      aspiration!.deferredGroup !== true &&
      (!aspirationFresh || (aspiration!.hasMoreLODs?.() ?? false));
    if (settled && needsReloadOrRefine && aspiration!.ensureLoaded) {
      this.maybeKickReload(entry, aspiration!);
    }

    // ── Apply visibility (single owner) ──
    // At most the display child plus its cross-fade partner is visible
    // (``displayIdx`` / ``blendPartnerIdx``, each only if READY — never
    // force-show a not-ready placeholder; outside a cross-fade band it's the
    // classic single visible level). ``changed`` flips when a SHOWN level
    // changes so ``evaluatePerFrame`` refreshes the monitor's visible
    // tally, which counts the displayed level, not the aspiration.
    let changed = false;
    // Opacity is managed only while at least one anti-popping feature is on: the
    // coverage cross-fade (blends the two levels straddling a distance boundary)
    // and/or the streaming energy compensation (per-leaf `1/e(k)` on the displayed
    // streaming level). When BOTH are off, `manageFade` is false and this reduces
    // to the original single-level visibility swap with no material writes —
    // byte-identical to before.
    const energyComp = this.deps.getEnergyCompEnabled?.() === true;
    const manageFade = this.deps.getCrossFadeEnabled?.() === true || energyComp;
    // Falling edge of fade management (both flags just toggled OFF, possibly
    // MID-fade): restore every child's authored opacity ONCE so a half-faded
    // level (e.g. opacity 0.5 from an in-flight cross-fade) doesn't stay dim
    // forever. ``fadeWasManaged`` is updated per-frame in ``evaluatePerFrame``
    // after all entries run, so the edge fires exactly one frame for each
    // entry; afterwards the both-flags-off path is byte-identical again.
    const restoreResidualFade = !manageFade && this.fadeWasManaged;
    for (let i = 0; i < entry.children.length; i++) {
      const child = entry.children[i];
      const isPrimary = i === displayIdx;
      const shouldShow = (isPrimary || i === blendPartnerIdx) && isReady(child);
      if (child.object.visible !== shouldShow) {
        child.object.visible = shouldShow;
        if (shouldShow) changed = true; // a new level became visible
      }
      if (manageFade) {
        if (shouldShow) {
          // Cross-fade weight only when a partner is in flight (primary at α,
          // partner at 1−α); otherwise no coverage weight (null ⇒ 1). Energy
          // compensation is folded in PER-LEAF inside applyChildFade, so it also
          // covers a plainly-displayed streaming level with no cross-fade partner.
          const coverageWeight =
            blendPartnerIdx != null ? (isPrimary ? primaryWeight : 1 - primaryWeight) : null;
          this.applyChildFade(child, coverageWeight, energyComp);
        } else {
          // Hidden / left the plan: restore authored opacity if we faded it
          // (idempotent — a no-op on any never-faded child).
          this.applyChildFade(child, null, false);
        }
      } else if (restoreResidualFade) {
        // One-shot restore on the fade-management falling edge (see above):
        // idempotent no-op on never-faded children, so the pass writes only
        // where a residual fade opacity actually lingers.
        this.applyChildFade(child, null, false);
      }
    }
    // Mark the on-screen level most-recently-used and record it for the eviction
    // pass, which must never release the level currently displayed. Only update
    // when a ready level is actually shown — otherwise keep the last shown index
    // so eviction still protects whatever the user last saw.
    const shown = displayIdx >= 0 ? entry.children[displayIdx] : undefined;
    if (shown && isReady(shown)) {
      shown.lastVisibleTick = this.tick;
      entry.displayedChildIndex = displayIdx;
      // The gate's memory tracks only what was shown ON SCREEN, so an
      // off-screen excursion (which displays the coarse fallback) cannot
      // clobber a held finer level and re-pop it on camera return.
      if (!entry.offScreen) entry.heldDisplayChildIndex = displayIdx;
    }

    // Stamp any already-ready child that has never been shown so it ages into
    // the eviction LRU. Without this, a lazy level that finished loading but was
    // never displayed (camera/slice moved away mid-load) keeps
    // ``lastVisibleTick == null`` and is permanently exempt from eviction,
    // leaking VRAM.
    // A SENTINEL, not the current tick: the eviction LRU's final tiebreak is
    // coldest-first on this field, so stamping "now" would make the one level
    // the user never saw the HOTTEST in its group and evict levels they looked
    // at moments ago before it. 0 is older than any real tick (ticks start at 1).
    for (const child of entry.children) {
      if (isReady(child) && child.lastVisibleTick == null) {
        child.lastVisibleTick = 0;
      }
    }

    return changed;
  }

  /**
   * Fold an entry's children nD bounds into one world-space
   * :type:`BoundingBox` (see {@link computeEntryWorldBox} in
   * ``lod-selector-math.ts`` for the math). The default uses raw
   * ``positionBounds`` for frustum gating and eviction; ``useLodBounds`` uses
   * robust bounds with a per-child raw fallback for selector metrics. This
   * wrapper supplies per-entry local/world scratch boxes and the registry's
   * ``matrixScratch``. Raw and robust metric bounds use distinct world boxes
   * because both remain live during one selector evaluation.
   */
  private computeWorldBox(
    entry: LODGroupEntry,
    displayDims: readonly number[],
    useLodBounds: boolean = false
  ): BoundingBox | null {
    const cache = this.caches.get(entry.path);
    if (!cache) return null;
    return computeEntryWorldBox(
      entry,
      displayDims,
      cache.localBoxScratch,
      this.matrixScratch,
      useLodBounds ? cache.metricWorldBoxOptions : cache.worldBoxOptions
    );
  }

  /**
   * Freshness + committed element count of a child, resolving a GROUP-typed LOD
   * child (a deferred ``kind=partition`` / nested ``lod`` subtree — the
   * ``overview`` recipe) through {@link subtreeDisplayProgress} rather than the
   * leaf-only stamps. A bare ``THREE.Group`` carries no leaf ``nodeType``, so
   * ``isFresh`` would report it unconditionally fresh and ``visibleElementCount``
   * would return ``null`` — hiding a stale re-slice and defeating the empty
   * guard. Mirrors the never-downgrade gate's ``sideProgress`` so both paths
   * agree on what "fresh" means for a group. Leaf children (a direct count
   * stamp) keep the exact pre-existing behaviour.
   *
   * **``fresh`` implies ``ready``** for every child shape: the leaf branch's
   * ``isFresh`` is ready-gated, and a NOT-ready group child (a deferred
   * placeholder whose subtree never committed, or a released level awaiting
   * reload) reports ``fresh: false`` regardless of any stamps its subtree may
   * retain — it cannot draw, so no display path (slice-aware fallback,
   * empty-guard redirect, blend pairing) may ever elect it. A READY group with
   * no stamped leaf (nested group with no slice-dependent geometry) carries no
   * per-slice staleness signal and reports ``fresh: true, count: null``.
   *
   * ``subtreeLadderComplete`` is the third answer, folded from the same walk
   * (``SubtreeDisplayProgress.complete``): false when any visible stamped leaf
   * under the subtree has committed only a prefix of its additive ladder. A
   * tracked LEAF has no subtree to fold, so it answers with its OWN
   * ``committedLadderComplete`` stamp.
   *
   * That stamp rather than the child's ``hasMoreLODs()`` thunk, because the
   * thunk does not exist on every leaf: ``load-lod-group-node`` attaches it
   * only on the DEFERRED path, so the eagerly-loaded default level — whose
   * ladder is advanced by the sweep-driven background refinement loop — has
   * none, and reporting an unconditional ``true`` here declared a still-
   * streaming coarse level complete. The stamp is also the safer of the two
   * where both exist (see ``lod-display-gate``'s "committed state only" note:
   * a live getter flips when the last fetch resolves, frames before the commit
   * lands). Callers still read ``hasMoreLODs()`` directly on top of this, since
   * it is what re-fires ``ensureLoaded`` to advance a lazy ladder. Only
   * {@link isCaptureQuiescent} consults this field; the display paths ignore
   * it.
   *
   * ``version === null`` means no view-version tracking is wired: the per-slice
   * staleness test is skipped and every READY child reads fresh — which is
   * exactly what the ``version != null`` guards at the display call sites
   * already assume, so those are unaffected.
   */
  private childFreshAndCount(
    child: LODGroupChild,
    version: number | null
  ): { fresh: boolean; count: number | null; subtreeLadderComplete: boolean } {
    // Leaf detection is by tracked nodeType, NOT by "has a count stamp": a leaf
    // that has not committed a count yet is still a leaf whose freshness is its
    // own ``loadedViewVersion`` stamp. Only a genuine group subtree folds.
    if (isTrackedLeaf(child)) {
      return {
        fresh: version == null ? isReady(child) : isFresh(child, version),
        count: visibleElementCount(child),
        // The leaf's own commit stamp — absent (never committed, or a
        // non-progressive loader) reads as complete, so an unstamped leaf
        // never blocks. See the doc above for why not ``hasMoreLODs()``.
        subtreeLadderComplete: child.object.userData?.committedLadderComplete !== false,
      };
    }
    // Ready gate for group children (the leaf branch gets it from ``isFresh``).
    // Without it, a not-ready deferred-group placeholder (no stamped leaves →
    // ``!aggregate`` below) would read fresh-with-unknown-count and the
    // empty-level guard could redirect display onto a level that CANNOT draw,
    // blanking the group permanently. Nothing has committed, so no completeness
    // can be claimed either.
    if (!isReady(child)) return { fresh: false, count: null, subtreeLadderComplete: false };
    const aggregate = subtreeDisplayProgress(child.object as unknown as ProgressNode, version);
    // Ready, but no stamped leaf under the subtree (nested group with no
    // slice-dependent geometry): no per-slice staleness signal, so treat as
    // fresh — exactly the pre-existing ``isFresh`` behaviour for a ready
    // non-leaf. Only a subtree that DOES carry stamped-but-stale leaves (a
    // non-null aggregate with ``fresh === false``) triggers the coarse fallback.
    // No stamped leaf likewise means no ladder to be waiting on: complete.
    if (!aggregate) return { fresh: true, count: null, subtreeLadderComplete: true };
    return {
      fresh: aggregate.fresh,
      count: aggregate.count,
      subtreeLadderComplete: aggregate.complete,
    };
  }

  /**
   * Index of the coarsest child that is fresh for ``version`` AND has a
   * non-zero committed element count — or ``-1`` when none qualifies. The
   * group-aware counterpart of the empty-level display guard's fallback: it
   * resolves each child through {@link childFreshAndCount}, so a fresh-but-empty
   * GROUP child (a deferred ``kind=partition`` subtree whose visible leaves all
   * committed 0) is correctly skipped rather than treated as non-empty (a bare
   * ``THREE.Group`` has no leaf count stamp). A READY child with an UNTRACKED
   * count (``null`` — group with no stamped leaf) is accepted, matching the
   * leaf-only helper it replaced: the guard only redirects away from KNOWN-empty
   * levels. Because ``childFreshAndCount``'s ``fresh`` implies ``ready``, a
   * NOT-ready placeholder can never be returned — the guard must only redirect
   * to a level that can actually draw. When nothing qualifies (``-1``) the
   * caller keeps the fresh-but-empty current level: an empty-but-real level
   * beats a blank placeholder.
   */
  private coarsestFreshNonEmptyIndex(entry: LODGroupEntry, version: number): number {
    for (let i = 0; i < entry.children.length; i++) {
      const p = this.childFreshAndCount(entry.children[i], version);
      if (!p.fresh) continue;
      if (p.count === 0) continue;
      return i;
    }
    return -1;
  }

  /**
   * Whether every fadeable leaf material under ``child.object`` uses a blend
   * mode that cross-fades correctly — see {@link isBlendableSubtree}
   * (``lod-fade.ts``) for the criteria.
   */
  private isBlendable(child: LODGroupChild): boolean {
    return isBlendableSubtree(child.object);
  }

  /**
   * Apply the per-leaf LOD anti-popping opacity (cross-fade weight ×
   * streaming `1/e(k)` energy compensation) to a child's leaf materials, or
   * restore the authored opacity — see {@link applyLodFade}
   * (``lod-fade.ts``) for the full mechanics. This wrapper supplies the
   * registry's ``registerMaterial`` dep so a clone-on-first-fade material
   * keeps receiving per-frame camera-uniform updates.
   */
  private applyChildFade(child: LODGroupChild, weight: number | null, energyComp: boolean): void {
    applyLodFade(child.object, weight, energyComp, this.deps.registerMaterial);
  }

  /**
   * Coarsest child that is ready AND fresh for ``version``, falling back to the
   * coarsest READY level when none is fresh yet (the ≤1-frame window right after
   * a re-slice) so the group shows stale-but-ready geometry rather than going
   * blank. GROUP-AWARE: each child resolves through ``childFreshAndCount``, the
   * same freshness the sibling display paths (aspiration check, empty guard,
   * blend pairing) use — so a ready GROUP child whose subtree leaves are stamped
   * for an older slice is correctly skipped. The leaf-only
   * ``coarsestFreshIndex`` it replaced treated any non-leaf as unconditionally
   * fresh, which displayed the OLD slice from a stale partition/overview branch
   * after a re-slice even while a genuinely fresh level was resident.
   */
  private coarsestFreshOrReadyIndex(entry: LODGroupEntry, version: number): number {
    for (let i = 0; i < entry.children.length; i++) {
      if (this.childFreshAndCount(entry.children[i], version).fresh) return i;
    }
    return this.coarsestReadyIndex(entry);
  }

  /**
   * Should a STALE previously-displayed level be kept on screen for a few more
   * frames instead of dropping to ``fallbackIdx``, the coarsest fresh level?
   *
   * Returns the index to hold, or ``undefined`` to take the fallback.
   *
   * The slice-aware fallback exists so a scrub shows the new slice immediately
   * at low detail. It becomes a defect when the aspiration is only a few frames
   * behind: stepping a 4D timelapse one timepoint made the display drop from
   * the finest level to the coarsest and climb back within ~70 ms, every step —
   * a flash to 1.6% of the geometry while the finest level's data was already
   * cached and decoding. What is on screen is the PREVIOUS slice, which for a
   * timelapse step is the previous frame: the same thing a video player leaves
   * up while the next frame decodes, and far closer to the truth than 108 of
   * 6,900 splats.
   *
   * Held only when ALL of:
   *   - the previous display is still ready and is FINER than the fallback
   *     (holding something coarser than the fallback would be a downgrade),
   *   - the fallback is a SEVERE downgrade — below
   *     `STALE_HOLD_MIN_RATIO` of the held level's committed count —
   *     so a fallback that is nearly as good is taken immediately (it is
   *     fresh, and freshness wins whenever quality is comparable),
   *   - the hold has not exhausted its `STALE_HOLD_MS` budget.
   *
   * The budget is deliberately spent from when the hold STARTS and is not
   * refreshed by later version bumps, and once exhausted it latches until the
   * aspiration commits fresh. So a continuous drag degrades to exactly the
   * pre-existing behaviour after `STALE_HOLD_MS`, rather than freezing on
   * one frame for as long as the user keeps dragging.
   */
  private staleHoldDisplayIndex(
    entry: LODGroupEntry,
    fallbackIdx: number,
    version: number
  ): number | undefined {
    if (entry.selectorMode !== 'auto' || entry.offScreen) return undefined;
    if (entry.staleHoldExhausted) return undefined;

    // The gate's memory (last ON-SCREEN level), for the reason the
    // never-downgrade gate uses it: ``displayedChildIndex`` is clobbered to the
    // coarse level during an off-screen excursion.
    const prevIdx = entry.heldDisplayChildIndex;
    if (prevIdx == null || prevIdx <= fallbackIdx) return undefined;
    const prev = entry.children[prevIdx];
    if (!prev || !isReady(prev)) return undefined;

    // Compare committed counts.
    const prevCount = this.childFreshAndCount(prev, version).count;
    const fallback = entry.children[fallbackIdx];
    const fallbackCount = fallback ? this.childFreshAndCount(fallback, version).count : null;
    // An unknown count on either side means the comparison cannot be made, so
    // there is no evidence the fallback is severe — take it, as before.
    if (prevCount == null || fallbackCount == null || prevCount <= 0) return undefined;
    if (fallbackCount >= prevCount * STALE_HOLD_MIN_RATIO) return undefined;

    const now = this.deps.now?.() ?? performance.now();
    if (entry.staleHoldSinceMs == null) entry.staleHoldSinceMs = now;
    if (now - entry.staleHoldSinceMs >= STALE_HOLD_MS) {
      entry.staleHoldExhausted = true;
      entry.staleHoldSinceMs = undefined;
      return undefined;
    }
    // Keep the held level warm so eviction does not reclaim it mid-hold.
    prev.lastVisibleTick = this.tick;
    return prevIdx;
  }

  /**
   * Index of the coarsest currently-ready child. Children are stored
   * coarsest→finest, so the first ready index is the coarsest available
   * (resident) level. Used by the off-screen gate to hold a culled group on
   * geometry that is already loaded — never a not-ready lazy level, so it
   * cannot kick a load. Falls back to the current active index when nothing is
   * ready (shouldn't happen: the eager default level is always ready).
   */
  private coarsestReadyIndex(entry: LODGroupEntry): number {
    for (let i = 0; i < entry.children.length; i++) {
      if (isReady(entry.children[i])) return i;
    }
    // Invariant: a substitutive lod_group's default level is loaded eagerly
    // (committed before ``register`` and never evicted), so at least one child
    // is always ready. If that ever breaks (e.g. the eager default failed to
    // attach, or a future change makes the default lazy too), the off-screen
    // gate pins whatever ``activeChildIndex`` happens to be — surface it rather
    // than fail silently, but only ONCE per entry: this method runs every frame
    // while the group is off-screen, so an unguarded log would flood at frame
    // rate for a genuinely-stuck group.
    if (!this.warnedNoReadyChild.has(entry.path)) {
      this.warnedNoReadyChild.add(entry.path);
      log.warning(
        Modules.SCENE_LOADER,
        `lod_group ${entry.path}: no ready child for off-screen gate; ` +
          `holding active index ${entry.activeChildIndex}`
      );
    }
    return entry.activeChildIndex;
  }

  /**
   * Kick a lazy child's deferred loader if eligible: not ready, has an
   * ``ensureLoaded`` thunk, not already loading, and not inside an
   * un-expired failure cooldown. Centralises the lazy-load gate used by
   * both the desired-target swap path and the active-child self-heal so
   * the failure/cooldown logic lives in exactly one place.
   *
   * A freshly-failed child is stamped with the current tick; once
   * ``FAILED_RETRY_FRAMES`` elapse the ``failed`` flag is cleared and the
   * load retried — recovering a level that failed on *reload* (after a
   * successful load + byte-eviction), which the old "failed until
   * released" behaviour left permanently stuck (a failed child is never an
   * eviction candidate).
   */
  private maybeKickLoad(entry: LODGroupEntry, child: LODGroupChild): void {
    if (isReady(child)) return; // not-ready-only: a ready level needs no initial load
    this.kickDeferredLoadIfVisible(entry, child);
  }

  /**
   * Reload a READY-but-STALE lazy fine level for the current view — the sibling
   * of ``maybeKickLoad`` for a child whose geometry is committed but reflects an
   * older slice/displayDims version. A fine level leaves the per-slice sweep
   * once loaded (see ``load-lod-group-node.ts``), so the registry — not the
   * sweep — drives its reload, gated on settle by the caller. Re-fires
   * ``ensureLoaded``, which re-runs the expensive loader (overwrites the
   * geometry in place, commits independently, re-stamps ``loadedViewVersion``
   * fresh). Unlike ``maybeKickLoad`` it does NOT early-return on ``isReady`` —
   * refreshing a ready level is the whole point. The stale level stays hidden
   * behind the coarse fallback meanwhile (the display pass), so this never
   * blanks the screen, and the shared ``loading``/cooldown guards make
   * re-calling it every settled frame safe.
   */
  private maybeKickReload(entry: LODGroupEntry, child: LODGroupChild): void {
    this.kickDeferredLoadIfVisible(entry, child);
  }

  /**
   * **Effective-visibility gate** — the single place the per-frame paths
   * (``maybeKickLoad`` / ``maybeKickReload``) decide whether a deferred load is
   * worth STARTING at all.
   *
   * A layer authored ``visible=false`` (or toggled off in the layers panel)
   * hides the LAYER object; the lod_group and its levels underneath keep their
   * own ``visible`` flags, so the selector happily kept aspiring to — and
   * lazily loading — fine levels that cannot be drawn. Those loads compete for
   * the shared fetch gate, the worker pool, and VRAM with the layer the user is
   * actually looking at (measured: a hidden 9.75M-point level finished FIRST,
   * roughly doubling scene load time). So: no group visible ⇒ no new loads.
   * The same gate refuses every automatic kick while the owning loader has a
   * latched archive fault; explicit retry remains the only bypass.
   *
   * Scope is deliberately narrow — this only stops STARTING work:
   *   - it never hides or unloads anything already resident (a hidden layer
   *     draws nothing anyway, and retention keeps a re-show free);
   *   - the eager ``default_level`` is loaded by ``loadLodGroupNode``, not from
   *     here, so a hidden layer still has its cheap coarse level ready to
   *     display the instant the panel toggles it on;
   *   - the walk is ancestor-aware via ``entry.groupObject`` (the hidden flag
   *     usually sits on an ANCESTOR layer/group, not on the lod_group itself);
   *   - the gate is re-evaluated every frame, so toggling the layer back on
   *     (``LayerApplyEngine.applyVisibility`` → ``requestRender`` →
   *     ``AnimationController`` → ``evaluatePerFrame``) resumes loading on the
   *     very next frame with no extra wiring.
   *
   * ``retryLazyChildByNodePath`` (an explicit user retry of a FAILED level)
   * deliberately bypasses this and calls ``kickDeferredLoad`` directly: an
   * explicit request for a retryable child is honoured whatever the layer's
   * visibility or the current archive-fault latch.
   */
  private kickDeferredLoadIfVisible(entry: LODGroupEntry, child: LODGroupChild): void {
    if (this.deps.hasArchiveFault?.() || !isEffectivelyVisible(entry.groupObject)) return;
    this.kickDeferredLoad(child);
  }

  /**
   * Shared lazy-load gate for ``maybeKickLoad`` (initial load of a not-ready
   * level) and ``maybeKickReload`` (refresh of a ready-but-stale level): fire
   * ``ensureLoaded`` unless already loading or inside the failure cooldown. A
   * freshly-failed child is stamped with the current tick; once
   * ``FAILED_RETRY_FRAMES`` elapse the ``failed`` flag clears and the load
   * retries — recovering a level that failed on reload (after a successful load
   * + byte-eviction), which the old "failed until released" behaviour left stuck.
   * The automatic caller separately gates loader-level archive faults. The
   * per-child latch keeps concurrent failed branches individually addressable
   * by Retry; an explicit retry clears that selected branch as it starts.
   */
  private kickDeferredLoad(child: LODGroupChild, explicitRetry = false): boolean {
    if (!child.ensureLoaded || child.loading || (!explicitRetry && child.permanentlyFailed)) {
      return false;
    }
    if (child.failed) {
      if (explicitRetry) {
        child.failed = false;
        child.failedTick = undefined;
      } else {
        if (child.failedTick == null) {
          // First frame we observe the failure — start the cooldown clock.
          child.failedTick = this.tick;
          return false;
        }
        if (this.tick - child.failedTick < FAILED_RETRY_FRAMES) return false;
        // Cooldown elapsed — clear the failure and fall through to retry.
        child.failed = false;
        child.failedTick = undefined;
      }
    }
    if (explicitRetry) {
      child.permanentlyFailed = false;
      child.failureReason = undefined;
    }
    child.loading = true;
    child.ensureLoaded();
    return true;
  }

  /**
   * Bound resident LOD geometry to the GPU-pool byte budget — see
   * {@link enforceResidentByteBudget} (``lod-eviction.ts``) for the full
   * policy. This wrapper supplies the registry's entries, the pool-accounting
   * deps, and the raw per-entry world-box fold, so eviction matches the
   * selector's frustum gate rather than its optional robust metric bounds.
   */
  private enforceByteBudget(
    camera: THREE.Camera,
    frustum: THREE.Frustum,
    displayDims: readonly number[]
  ): void {
    enforceResidentByteBudget({
      entries: this.entries.values(),
      camera,
      frustum,
      getResidentByteBudget: this.deps.getResidentByteBudget,
      getResidentBytes: this.deps.getResidentBytes,
      computeWorldBox: (entry) => this.computeWorldBox(entry, displayDims),
    });
  }
}
