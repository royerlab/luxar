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
 * **Partition frustum gating** (``registerPartition``): a ``kind=partition``
 * group's parts are frustum- and slice-culled every frame, and a part that
 * re-enters the frustum is resynced — see ``partition-gate.ts``, which the
 * registry drives before and after its LOD pass.
 *
 * **Collaborators.** The registry is the orchestrator; the stateful pieces
 * live in their own modules, each stating until when it needs the loop to keep
 * ticking (``tick-demand.ts``), which ``evaluatePerFrame`` folds into one
 * ``requestTick``:
 *   - ``lod-dissolve.ts`` — the time-driven level dissolve (state machine);
 *   - ``playback-aspiration.ts`` — reload timing and the playback level cap;
 *   - ``partition-gate.ts`` — partition culling, resync and lazy activation;
 *   - ``capture-quiescence.ts`` — the offline-capture "frame is final" test;
 *   - ``lod-selector-math.ts`` / ``lod-display-gate.ts`` / ``lod-fade.ts`` /
 *     ``lod-eviction.ts`` — the pure metric, display, opacity and eviction
 *     policies.
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

import { bumpFailedLoadsVersion } from '../utils/failed-loads-version';
import { clearChildFailure } from '../utils/lod-child-failure';
import * as THREE from 'three';

import type { BoundingBox } from './scene-manager/clipping/bounds-math';
import { ViewContextProvider, type ViewContext } from './view-context';
import { log, Modules } from '../utils/log';
import { isEffectivelyVisible } from '../utils/object-visibility';
import type { LODGroupSelectorMode } from '../types/lod-group';
import type { LodSelectorName } from '../types/format-contract';
import { isReady, SettleTracker } from './lod-freshness';
import { childFreshAndCount, shouldHoldPreviousDisplay } from './lod-display-gate';
import { smoothstep } from './lod-blend';
import { config } from '../config';
import { applyLodFade, isBlendableSubtree } from './lod-fade';
import {
  computeEntryWorldBox,
  pickChildByFootprintWithHysteresis,
  pickChildWithHysteresis,
  preloadNeighbourIndex,
  projectBoxAreaFraction,
  projectBoxDiagonalPx,
  projectWorldRadiusPx,
  type WorldBoxOptions,
} from './lod-selector-math';
import { enforceResidentByteBudget } from './lod-eviction';
import { foldLoadTime, PlaybackAspiration } from './playback-aspiration';
import { LodDissolves, type DissolveHost, type LevelFade } from './lod-dissolve';
import { PartitionGate, type PartitionGroupEntry } from './partition-gate';
import { isCaptureQuiescent, type CaptureQuiescenceSource } from './capture-quiescence';
import { NO_TICK, UNTIL_RESOLVED } from './tick-demand';
import { RetryWakes } from './retry-wakes';
import { perfCounters } from '../profiling/perf-counters';
import type { PartitionSliceView } from '../data/scene-loader/view-state/partition-slice-gate';

// Partition gating lives in `partition-gate.ts`; its types are re-exported here
// so the loader-side importers keep their import site.
export type {
  LazyPartState,
  PartitionChildCache,
  PartitionGroupChild,
  PartitionGroupEntry,
  PartitionPartRef,
} from './partition-gate';

// The selector math (box projection + hysteresis pick) lives in
// `lod-selector-math.ts`; re-exported here so existing importers (the
// selector unit tests) keep their import site.
export {
  pickChildWithHysteresis,
  projectBoxAreaFraction,
  projectBoxDiagonalPx,
} from './lod-selector-math';

/**
 * Band preload (see ``LODGroupRegistry.preloadNeighbour``): the level a group is
 * making resident, hidden, while its selector metric sits in the band of the
 * threshold between it and the displayed level. ``sawReady`` records that it
 * landed during this visit, so a release under VRAM pressure is not followed by
 * a reload while the camera stays put (load → evict → load …).
 */
export interface PreloadVisit {
  idx: number;
  sawReady: boolean;
}

/** Perf counter: displayed-level changes of a lod group (one per group per frame). */
const S_LOD_LEVEL_SWAPS = perfCounters.slot('lod.levelSwaps');
/** Perf counter: group-frames drawing two levels dissolving (one per group per frame). */
const S_LOD_BLEND_FRAMES = perfCounters.slot('lod.blendFrames');

/**
 * Tally this frame's display outcome for one group (perf counters only): a
 * level swap when the shown level differs from the last one displayed, and a
 * blend frame when the primary and its dissolve partner are both drawn.
 * Must run BEFORE ``displayedChildIndex`` is updated.
 */
function countLodDisplay(
  entry: LODGroupEntry,
  displayIdx: number,
  blendPartnerIdx: number | null
): void {
  const shown = displayIdx >= 0 ? entry.children[displayIdx] : undefined;
  if (!shown || !isReady(shown)) return;
  if (entry.displayedChildIndex !== undefined && entry.displayedChildIndex !== displayIdx) {
    perfCounters.add(S_LOD_LEVEL_SWAPS);
  }
  if (blendPartnerIdx != null && isReady(entry.children[blendPartnerIdx])) {
    perfCounters.add(S_LOD_BLEND_FRAMES);
  }
}

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
  /** Median element footprint in node-local scene units. */
  medianFootprint?: number;
  /** Data columns included in the median footprint measurement. */
  footprintDims?: readonly number[];
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
  /**
   * Registry clock (ms) at which the in-flight ``ensureLoaded`` was fired, until
   * the registry observes ``loading`` cleared. Registry-owned.
   */
  loadStartMs?: number;
  /**
   * Registry clock (ms) at which the load started at ``loadStartMs`` settled
   * (``onLoadSettled``). Registry-owned.
   */
  loadEndMs?: number;
  /**
   * Exponentially weighted mean of this level's measured load+commit time, in
   * ms (fire → ``loading`` cleared, failures excluded). Registry-owned; during
   * playback it decides which level can keep up with the period.
   */
  loadEwmaMs?: number;
  /**
   * Successful timed loads folded into ``loadEwmaMs``. Registry-owned. The
   * first is the level's cold load (cache miss, connection and decoder
   * warm-up): it is recorded, but playback treats the level as unmeasured
   * until a second load reseeds the average in its place.
   */
  loadSamples?: number;
  /** Last playback probe of a level whose measured reload exceeded the budget. */
  lastPlaybackProbeMs?: number;
  /**
   * Set when a playback probe's reload STARTS (a refused kick sets nothing);
   * that timed load then REPLACES ``loadEwmaMs`` instead of blending into it.
   * Cleared when it is folded and when playback stops. Registry-owned.
   */
  playbackProbePending?: boolean;
  /**
   * A probe may move the aspiration, but cannot grant the 1.0 keep budget.
   * Set on the probe frame, cleared once the level is admitted or playback
   * stops. Registry-owned.
   */
  playbackProbeAdmissionPending?: boolean;
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
   * Registry clock (``frame.nowMs``) when ``failed`` was first observed. Drives
   * the transient-failure retry cooldown (starting at ``config.lod.failedRetryMs``,
   * backed off per consecutive failure — ``retry-wakes.ts``): once it
   * elapses the registry clears ``failed`` and retries the load, so a
   * level that fails on *reload* (after a successful load + byte-eviction)
   * is not stuck on its placeholder forever. Cleared alongside ``failed``.
   */
  failedAtMs?: number;
  /**
   * Idempotent fire-and-forget loader for a lazy child. Kicks the
   * deferred geometry load; on success sets ``ready=true`` and clears
   * ``loading``; on failure sets ``failed=true`` and clears ``loading``. A
   * container fault may additionally set ``permanentlyFailed``.
   * Must not touch ``object.visible`` — the registry owns the swap.
   * Calls ``onLoadSettled`` right after clearing ``loading``.
   */
  ensureLoaded?: () => void;
  /**
   * Registry-owned hook, set each time the registry fires ``ensureLoaded``;
   * the thunk calls it once the load has settled (``loading`` cleared). It
   * stamps ``loadEndMs`` so the measured ``loadEwmaMs`` is the load itself,
   * not the wait for the next evaluated frame (a hidden tab runs none).
   */
  onLoadSettled?: () => void;
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
   * projected bbox's fraction of the viewport AREA, measured through its
   * inscribed ellipsoid sized by its view-axis half-chord
   * (``projectBoxAreaFraction``); ``'coverage'`` — the legacy diagonal metric
   * (``projectBoxDiagonalPx / (FILL_FACTOR × min(viewport.width,
   * viewport.height))``, the fitted screen axis) — is the default when
   * absent, so older stores and test-constructed entries keep their
   * behaviour.
   */
  selector?: LodSelectorName;
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
   * True once a stale hold has exhausted `config.lod.staleHoldMs` without the
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
  medianFootprints: number[] | null;
  footprintDims: readonly number[] | null;
  footprintPx: number[];
  /** Whether any child needs the optional robust-bounds metric fold. */
  hasLodBounds: boolean;
  localBoxScratch: BoundingBox;
  /** Local (group-space) box of the robust metric bounds; distinct from the raw one. */
  metricLocalBoxScratch: BoundingBox;
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
const FOOTPRINT_CENTER_SCRATCH = new THREE.Vector3();
const FOOTPRINT_VIEW_CENTER_SCRATCH = new THREE.Vector3();

function resolveLodBias(value: number | undefined): number {
  return value != null && Number.isFinite(value) && value > 0 ? value : 1;
}

interface FootprintPickOptions {
  viewportHeight: number;
  lodBias: number;
  displayDims: readonly number[];
}

function pickStampedFootprintChild(
  entry: LODGroupEntry,
  cache: LODGroupEntryCache,
  worldBox: BoundingBox,
  view: ViewContext,
  options: FootprintPickOptions
): number | null {
  if (!cache.medianFootprints || !cache.footprintDims) return null;
  if (
    cache.footprintDims.length !== options.displayDims.length ||
    cache.footprintDims.some((dim) => !options.displayDims.includes(dim))
  ) {
    return null;
  }
  FOOTPRINT_CENTER_SCRATCH.set(
    (worldBox.min.x + worldBox.max.x) / 2,
    (worldBox.min.y + worldBox.max.y) / 2,
    (worldBox.min.z + worldBox.max.z) / 2
  );
  // The bbox centre is a stable single depth sample, but can under-estimate a
  // deep node; max-axis scale conservatively over-estimates anisotropic nodes.
  const worldScale = entry.groupObject.matrixWorld.getMaxScaleOnAxis();
  for (let i = 0; i < cache.medianFootprints.length; i++) {
    const projected = projectWorldRadiusPx(
      cache.medianFootprints[i] * worldScale,
      FOOTPRINT_CENTER_SCRATCH,
      view,
      options.viewportHeight,
      FOOTPRINT_VIEW_CENTER_SCRATCH
    );
    if (projected == null || !Number.isFinite(projected)) return null;
    cache.footprintPx[i] = projected;
  }
  return pickChildByFootprintWithHysteresis(
    cache.footprintPx,
    entry.activeChildIndex,
    config.lod.maxMedianFootprintPx / Math.sqrt(options.lodBias)
  );
}
const WORLD_BOX3_SCRATCH = new THREE.Box3();
/** projection × view × a group's matrixWorld: local box corners → clip space. */
const LOCAL_PROJ_SCRATCH = new THREE.Matrix4();

/** The per-frame inputs every entry's evaluation shares (``evaluatePerFrame``). */
export interface EntryFrame {
  view: ViewContext;
  viewport: { width: number; height: number };
  displayDims: readonly number[];
  frustum: THREE.Frustum;
  /** The view version has held steady for ``config.lod.fineReloadSettleMs``. */
  settled: boolean;
}

/** ``LODGroupRegistry.pickDesired`` outcomes. */
const PICK_OK = 0;
/** No registration cache: evaluate nothing (shouldn't happen). */
const PICK_SKIP = 1;
/** No world box: drop the dissolve, change nothing. */
const PICK_DROP = 2;

/**
 * The stages of one entry's evaluation hand their results on through these
 * module scratches (``evaluatePerFrame`` is the single, non-reentrant entry
 * point), so the per-frame path allocates nothing.
 */
const PICK = {
  desired: 0,
  /** The metric the threshold pick used, for the band preload; null on every
   * other path (lock, off-screen, force-finest, footprint pick). */
  preloadMetric: null as number | null,
  /** The projected-footprint pick decided (no dissolve, no band preload). */
  usedFootprint: false,
};
/** ``applyVisibility``'s opacity mode this frame (see ``applyChildOpacity``). */
const FADE_MODE = { manage: false, energyComp: false, restoreResidual: false };
/** ``aspirationState``: the aspiration's readiness and freshness this frame. */
const ASPIRATION = { ready: false, fresh: false };
const BLEND = {
  /** The dissolve's outgoing level drawn with the displayed one, if any. */
  partner: null as number | null,
  /** The displayed (incoming) level's opacity; the partner gets the complement. */
  primaryWeight: 1,
  /** Last frame's outgoing level of an in-flight dissolve, or -1. */
  fadingFromIdx: -1,
  /** The outgoing level of a dissolve dropped since the last visibility pass, or -1. */
  droppedFromIdx: -1,
};

/**
 * What one ``evaluatePerFrame`` call changed, split by kind because the two
 * kinds mean different things downstream:
 *
 * - ``levelChanged`` — a ``kind=lod`` group swapped its active level. That is
 *   a CONTENT change (a different level costs differently to render), so the
 *   app notifies adaptive DPR, whose learned bounds described the old level.
 * - ``cullChanged`` — a ``kind=partition`` part entered or left the frustum.
 *   That is the same content seen from elsewhere, and it flips constantly
 *   while the camera orbits a partitioned scene: it must redraw, but must NOT
 *   reset adaptive-DPR learning.
 *
 * The returned objects are shared frozen constants, so the per-frame path
 * allocates nothing; compare fields, not identity.
 */
export interface LODFrameChanges {
  readonly levelChanged: boolean;
  readonly cullChanged: boolean;
}

/** Nothing changed this frame (the common case). */
export const LOD_FRAME_UNCHANGED: LODFrameChanges = Object.freeze({
  levelChanged: false,
  cullChanged: false,
});
const LOD_FRAME_LEVEL: LODFrameChanges = Object.freeze({ levelChanged: true, cullChanged: false });
const LOD_FRAME_CULL: LODFrameChanges = Object.freeze({ levelChanged: false, cullChanged: true });
const LOD_FRAME_BOTH: LODFrameChanges = Object.freeze({ levelChanged: true, cullChanged: true });

/** The shared constant for a (level, cull) pair — no per-frame allocation. */
function lodFrameChanges(levelChanged: boolean, cullChanged: boolean): LODFrameChanges {
  if (levelChanged) return cullChanged ? LOD_FRAME_BOTH : LOD_FRAME_LEVEL;
  return cullChanged ? LOD_FRAME_CULL : LOD_FRAME_UNCHANGED;
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
  /**
   * The frame's shared camera snapshot. When omitted the registry builds its
   * own from ``getCamera`` / ``getViewportSize``, refreshed every evaluation.
   */
  getViewContext?(): ViewContext;
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
   * `config.lod.staleHoldMs`). Injectable so tests can advance it deterministically;
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
   * Keep the loop TICKING without asking for a redraw — the render-on-change
   * counterpart of {@link requestRender} for the two keep-alive calls above
   * (a lazy level loading, a pending partition resync): nothing drawn has
   * changed yet, the loop just has to keep running the per-frame selector so
   * the swap fires when the load lands (a swap marks the frame dirty through
   * {@link LODGroupRegistry.takeDrawnStateChanged}). Wired to
   * ``AnimationController.requestTick``. Omitted ⇒ those calls fall back to
   * ``requestRender`` (an embedder whose host renders on every wake).
   */
  requestTick?: () => void;
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
   * The view the drawn geometry was last COMMITTED for (the owning loader's
   * last committed pass). Partition parts whose bounds miss its hidden-dim
   * slice draw nothing, so they are hidden and kept out of refinement (B4) —
   * against the COMMITTED view, not the requested one, so a part leaving the
   * slice disappears in the same frame its successors are committed rather
   * than when the pass that replaces it starts. Omitted / `undefined` ⇒ no
   * slice gating (every part is in slice — the pre-B4 behaviour).
   */
  getCommittedViewState?: () => PartitionSliceView | undefined;
  /**
   * Whether the owning loader has a view PASS in flight or queued. Rising edges
   * are held (and coalesced) while this is true so a resync never lands on top
   * of a pass. A refinement hold deliberately does NOT count: the loader parks
   * a resync that arrives during one and cancels into its own pass, so
   * re-entering parts do not sit on a stale slice until the ladders finish.
   */
  isUpdateInProgress?: () => boolean;
  /**
   * Whether the LOD level dissolve is enabled (ON by default; `?noLodFade`
   * disables; the name predates the time-driven dissolve). When true and a blendable (additive/luminous/volumetric — see
   * `BLENDABLE_MODES` in `scene/lod-fade.ts`) group changes its displayed
   * level, the registry dissolves from the outgoing level to the incoming one
   * over `config.lod.fadeMs` — the incoming at `smoothstep(elapsed / fadeMs)`,
   * the outgoing at the complement — instead of a hard visibility swap.
   * Time-driven, so a parked camera always settles on ONE level, and
   * independent of additive streaming. Omitted / false ⇒ the hard swap,
   * byte-identical.
   * Read live so the flag applies without a reload. The default for unit tests
   * (off).
   */
  getCrossFadeEnabled?: () => boolean;
  /**
   * The period (ms) of the fastest dimension currently PLAYING, or ``null``
   * when nothing plays (`DimensionAnimationManager.getPlaybackPeriodMs`).
   * While non-null, lazy levels reload on every timepoint without the settle
   * debounce and the aspiration is capped at the finest level whose measured
   * load time fits (see `config.lod.playbackLoadBudgetFraction`). Omitted ⇒ never
   * playing — identical to the pre-feature behaviour.
   */
  getPlaybackPeriodMs?: () => number | null;
  /**
   * Whether streaming brightness compensation is enabled: as a blendable
   * (additive/luminous/volumetric)
   * leaf's ladder streams in, scale its opacity by `1/e(k)` so the partial prefix
   * renders at the full-level energy (no brightening pop). Distinct from the
   * level dissolve (within one level's stream, not between levels) and
   * independently gated; either flag on
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
   * (the `?lodFinest` URL flag, threaded through the standalone
   * bootstrap); omitted / false ⇒ normal coverage-driven selection. Read
   * live, like the sibling flags above.
   */
  getForceFinestLOD?: () => boolean;
  /**
   * Replacement-LOD bias in screen-area units. `2` advances one level on an
   * occupancy-halved ladder. Legacy diagonal coverage receives `sqrt(bias)`
   * so both selectors shift by the same area factor. Since finite screen-area
   * coverage is at most `1`, bias below `1` makes a partition-anchored finest
   * level unreachable and bias below `0.5` does the same for a whole-object
   * finest level. Invalid values are neutral.
   */
  getLodBias?: () => number | undefined;
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
 * Whether the fresh ``fallback`` holds less than ``config.lod.staleHoldMinRatio``
 * of the ``held`` level's committed elements. An unknown count on either side
 * means the comparison cannot be made, so there is no evidence the fallback is
 * severe — take it.
 */
function fallbackIsSevere(
  held: LODGroupChild,
  fallback: LODGroupChild | undefined,
  version: number
): boolean {
  const heldCount = childFreshAndCount(held, version).count;
  const fallbackCount = fallback ? childFreshAndCount(fallback, version).count : null;
  if (heldCount == null || fallbackCount == null || heldCount <= 0) return false;
  return fallbackCount < heldCount * config.lod.staleHoldMinRatio;
}

/**
 * The never-downgrade gate judges only the aspiration itself being displayed,
 * auto-selected and on screen (an explicit lock and an off-screen hold bypass
 * it).
 */
function neverDowngradeApplies(entry: LODGroupEntry, displayIdx: number): boolean {
  return displayIdx === entry.activeChildIndex && entry.selectorMode === 'auto' && !entry.offScreen;
}

function bumpForFailedEntryReplacement(
  previous: LODGroupEntry | undefined,
  next: LODGroupEntry
): void {
  if (
    previous?.children.some((child) => child.permanentlyFailed) ||
    next.children.some((child) => child.permanentlyFailed)
  ) {
    bumpFailedLoadsVersion();
  }
}

/**
 * Tracks loaded ``lod_group`` and ``kind=partition`` nodes in a scene;
 * evaluates per-frame to pick the active LOD and frustum-visible parts.
 */
export class LODGroupRegistry {
  private entries: Map<string, LODGroupEntry> = new Map();
  /** The ``kind=partition`` groups and their per-part gating (``partition-gate.ts``). */
  private readonly partitions: PartitionGate;
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
   * Whether fade management (level dissolve and/or energy compensation) was ON
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
   * Set when a per-frame evaluation changed what the next frame DRAWS — any
   * level shown OR hidden, a fade opacity written with a new value. Broader
   * than {@link evaluatePerFrame}'s return (which reports only a newly shown
   * level, for the monitor tally): hiding a level, a time-driven stale-hold
   * expiry, or a dissolve weight step all change pixels too. Read and
   * cleared by {@link takeDrawnStateChanged} — the render-on-change loop's
   * signal to redraw.
   */
  private drawnStateChanged = false;
  /** In-flight level dissolves, by group path (``lod-dissolve.ts``). */
  private readonly dissolves = new LodDissolves<LODGroupEntry>();
  /** Asks this registry whether a pair of levels can dissolve (no per-frame closure). */
  private readonly dissolveHost: DissolveHost<LODGroupEntry> = {
    canDissolve: (entry, fromIdx, toIdx, version) =>
      this.canDissolve(entry, fromIdx, toIdx, version),
  };
  /** The per-frame inputs of ``evaluateEntry``, refilled by ``beginFrame``. */
  private readonly entryFrame = {} as EntryFrame;
  /** Reused argument of {@link LodDissolves.advance}. */
  private readonly dissolveFrame = { nowMs: 0, fadeMs: 0, version: null as number | null };
  /** Band preloads in progress, by group path (see {@link preloadNeighbour}). */
  private readonly preloads = new Map<string, PreloadVisit>();
  /** This frame's clock reading and playback period (see ``evaluatePerFrame``). */
  private readonly frame: { nowMs: number; playbackPeriodMs: number | null } = {
    nowMs: 0,
    playbackPeriodMs: null,
  };
  /** Playback aspiration: reload timing, admission, probes (``playback-aspiration.ts``). */
  private readonly playback = new PlaybackAspiration();
  /** One-shot wakes for failure cooldowns and unanswered part requests (``retry-wakes.ts``). */
  private readonly retryWakes = new RetryWakes(
    () => this.deps.requestTick !== undefined || this.deps.requestRender !== undefined,
    () => this.keepTicking()
  );
  /** The registry clock, bound once (``foldLoadTime`` reads it lazily). */
  private readonly clock = (): number => this.nowMs();

  /** What ``isCaptureQuiescent`` reads (``capture-quiescence.ts``); built once. */
  private readonly captureSource: CaptureQuiescenceSource = {
    isAnimating: () => this.isAnimating(),
    anyVisiblePartitionPart: () => this.partitions.anyVisiblePartitionPart(),
    isUpdateInProgress: () => this.deps.isUpdateInProgress?.() === true,
    hasArchiveFault: () => this.deps.hasArchiveFault?.() === true,
    anyChildLoading: () => this.anyChildLoading(),
    viewVersion: () => this.deps.getViewVersion?.() ?? null,
    partitionsCaptureQuiescent: (version) => this.partitions.partitionsCaptureQuiescent(version),
    entries: () => this.entries.values(),
    preloadIdx: (path) => this.preloads.get(path)?.idx,
  };

  /** Snapshot source when no shared ``getViewContext`` is injected. */
  private readonly ownViews: ViewContextProvider | null;

  constructor(private deps: LODGroupRegistryDeps) {
    this.partitions = new PartitionGate(deps, {
      view: () => this.view(),
      keepTicking: () => this.keepTicking(),
      markDrawn: () => {
        this.drawnStateChanged = true;
      },
      frame: this.frame,
      retryWakes: this.retryWakes,
    });
    this.ownViews = deps.getViewContext
      ? null
      : new ViewContextProvider({
          getCamera: () => deps.getCamera(),
          getViewportCss: () => deps.getViewportSize(),
          getDrawingBuffer: () => null,
        });
  }

  private view(): ViewContext {
    if (this.deps.getViewContext) return this.deps.getViewContext();
    const own = this.ownViews as ViewContextProvider;
    own.invalidate();
    return own.get();
  }

  /** Register a newly-loaded lod_group (called by the scene loader). */
  register(entry: LODGroupEntry): void {
    this.cancelEntryRetryWakes(entry.path);
    const footprintDims = entry.children[0]?.footprintDims;
    bumpForFailedEntryReplacement(this.entries.get(entry.path), entry);
    this.entries.set(entry.path, entry);
    this.caches.set(entry.path, {
      thresholds: entry.children.map((c) => c.coverageFraction),
      medianFootprints: entry.children.every(
        (child) => child.medianFootprint != null && child.medianFootprint > 0
      )
        ? entry.children.map((child) => child.medianFootprint!)
        : null,
      footprintDims:
        footprintDims != null &&
        footprintDims.length > 0 &&
        entry.children.every(
          (child) =>
            child.footprintDims != null &&
            child.footprintDims.length === footprintDims.length &&
            child.footprintDims.every(
              (dim, index) => Number.isInteger(dim) && dim === footprintDims[index]
            )
        )
          ? footprintDims
          : null,
      footprintPx: new Array<number>(entry.children.length),
      hasLodBounds: entry.children.some((c) => c.lodBounds != null),
      localBoxScratch: {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0, y: 0, z: 0 },
      },
      metricLocalBoxScratch: {
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
    this.partitions.registerPartition(entry);
  }

  /** See {@link PartitionGate.invalidatePartitionFootprint}. */
  invalidatePartitionFootprint(nodePath: string): void {
    this.partitions.invalidatePartitionFootprint(nodePath);
  }

  /** See {@link PartitionGate.applyCommittedSlice}. */
  applyCommittedSlice(): void {
    this.partitions.applyCommittedSlice();
  }

  /** See {@link PartitionGate.isPathInPartitionSlice}. */
  isPathInPartitionSlice(path: string, view: PartitionSliceView): boolean {
    return this.partitions.isPathInPartitionSlice(path, view);
  }

  /** See {@link PartitionGate.activatePartitionParts}. */
  activatePartitionParts(
    view: PartitionSliceView,
    targets?: ReadonlySet<string>,
    claim: AbortSignal | boolean = true
  ): Promise<string[]> {
    return this.partitions.activatePartitionParts(view, targets, claim);
  }

  /** See {@link PartitionGate.rankPartitionPartsForLoad}. */
  rankPartitionPartsForLoad(
    groupObject: THREE.Object3D,
    bounds: readonly { min: readonly number[]; max: readonly number[] }[]
  ): { inFrustum: boolean[]; order: number[] } | null {
    return this.partitions.rankPartitionPartsForLoad(groupObject, bounds);
  }

  /** See {@link PartitionGate.hasVisiblePendingPartitionResync}. */
  hasVisiblePendingPartitionResync(): boolean {
    return this.partitions.hasVisiblePendingPartitionResync();
  }

  /** Drop an lod_group from the registry (called on scene teardown). */
  unregister(path: string): void {
    this.cancelEntryRetryWakes(path);
    this.partitions.unregister(path);
    if (this.entries.get(path)?.children.some((child) => child.permanentlyFailed)) {
      bumpFailedLoadsVersion();
    }
    this.entries.delete(path);
    this.caches.delete(path);
    this.warnedNoReadyChild.delete(path);
    this.warnedEmptyLevel.delete(path);
    this.dissolves.forget(path);
    this.preloads.delete(path);
  }

  /** Clear all entries (called on full scene tear-down). */
  clear(): void {
    this.retryWakes.cancelAll();
    this.partitions.clear();
    if (
      [...this.entries.values()].some((entry) =>
        entry.children.some((child) => child.permanentlyFailed)
      )
    ) {
      bumpFailedLoadsVersion();
    }
    this.entries.clear();
    this.caches.clear();
    this.dissolves.clear();
    this.preloads.clear();
    this.playback.endEntry();
    // Reset the monotonic tick so a reused registry (shared-registry
    // refactor) starts cold rather than inheriting stale LRU ordering — and
    // the settle clock with it, so the new scene's first observed version
    // starts its own debounce instead of inheriting the old scene's.
    this.tick = 0;
    this.settleTracker.reset();
    this.warnedNoReadyChild.clear();
    this.warnedEmptyLevel.clear();
    this.fadeWasManaged = false;
  }

  /** Number of registered lod_groups (mainly for tests / diagnostics). */
  size(): number {
    return this.entries.size;
  }

  /** Number of registered groups that can require an offline-capture drain. */
  captureSize(): number {
    return this.entries.size + this.partitions.size();
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
   * CURRENT view is already at final committed quality, so an offline capture
   * may grab the frame — see ``capture-quiescence.ts`` for the rules and why.
   */
  isCaptureQuiescent(): boolean {
    return isCaptureQuiescent(this.captureSource);
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
   * Clears the failure cooldown (``failed``/``failedAtMs``) through the shared
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
   *
   * A deferred partition part (B4) whose activation failed is retried by its
   * part path the same way: it is re-armed, and the per-frame gate asks for a
   * pass to activate it again while it is wanted.
   */
  retryLazyChildByNodePath(path: string): boolean {
    if (!path) return false;
    for (const entry of this.entries.values()) {
      for (const child of entry.children) {
        if (child.nodePath !== path || !child.ensureLoaded) continue;
        if (child.loading) return true; // retry already in flight
        return this.kickDeferredLoad(child, true, true);
      }
    }
    return this.partitions.rearmFailedPartitionPart(path);
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
   * Reports, separately, whether at least one lod_group swapped its active
   * child (``levelChanged``) and whether a partition part's frustum-cull
   * visibility flipped (``cullChanged``) this frame — see
   * {@link LODFrameChanges} for why the two must not be conflated. Either
   * one changes what the visible-element tally counts. A no-op frame (the
   * common case) returns {@link LOD_FRAME_UNCHANGED}, which keeps the
   * per-frame cost to the projection math alone.
   */
  evaluatePerFrame(): LODFrameChanges {
    if (this.entries.size === 0 && this.partitions.size() === 0) return LOD_FRAME_UNCHANGED;
    const displayDims = this.deps.getDisplayDims();
    if (displayDims.length < 2) return this.skipFrame();
    // The frame's view snapshot: the camera matrices as this frame renders
    // them, whichever callbacks ran before (fly controls do not refresh
    // ``matrixWorldInverse``; the snapshot derives the view itself). A collapsed
    // canvas has no viewport to select for, so the frame is skipped.
    const view = this.view();
    const viewport = view.viewportCss;
    if (viewport === null) return this.skipFrame();
    const frame = this.beginFrame(view, viewport, displayDims);

    let levelChanged = false;
    // Partitions first, so a lod_group nested in a part sees this frame's cull;
    // the wrappers the LOD pass reveals are gated after it (``endFrame``).
    let cullChanged = this.partitions.beginFrame(displayDims, FRUSTUM_MATRIX_SCRATCH, view.camera);
    let loadsTickUntil = NO_TICK;
    for (const entry of this.entries.values()) {
      if (this.evaluateEntry(entry, frame)) levelChanged = true;
      // A lazy level loading (initial or a settled fine reload) commits
      // asynchronously OUTSIDE the per-slice sweep. Keep the on-demand render
      // loop alive so the per-frame swap-up to the fresh level fires when the
      // load lands, rather than waiting for the next user interaction. Plain
      // loop (not ``.some``) to preserve this file's no-per-frame-allocation
      // hot-path invariant. The same walk times the loads that just landed.
      loadsTickUntil = Math.max(loadsTickUntil, this.observeLoads(entry));
    }
    this.playback.endEntry();
    if (this.partitions.endFrame(displayDims)) cullChanged = true;
    this.endFrame(loadsTickUntil, view.camera, displayDims);
    return lodFrameChanges(levelChanged, cullChanged);
  }

  /**
   * Start an evaluated frame: advance the tick, take the frame's clock reading
   * and playback period, and fill the shared {@link EntryFrame}.
   *
   * ``view.projView`` (projection×view, default WebGL coordinate system, the
   * NDC convention of the manual divide inside ``projectBoxDiagonalPx``) is
   * shared three ways: its frustum gates off-screen groups and ranks eviction,
   * and the per-group projections reuse the matrix. The settle tracker records
   * whether the (global) view version has settled, to gate deferred fine-level
   * reloads; an ``undefined`` view version (no wiring / tests) ⇒ settled, so
   * the trigger is inert.
   */
  private beginFrame(
    view: ViewContext,
    viewport: { width: number; height: number },
    displayDims: readonly number[]
  ): EntryFrame {
    this.tick++;
    FRUSTUM_MATRIX_SCRATCH.copy(view.projView);
    FRUSTUM_SCRATCH.copy(view.frustum);
    this.frame.nowMs = this.nowMs();
    this.frame.playbackPeriodMs = this.deps.getPlaybackPeriodMs?.() ?? null;
    this.playback.beginFrame(this.frame.playbackPeriodMs, this.entries.values());
    const version = this.deps.getViewVersion?.();
    if (version != null) this.settleTracker.observe(version, this.frame.nowMs);
    const frame = this.entryFrame;
    frame.view = view;
    frame.viewport = viewport;
    frame.displayDims = displayDims;
    frame.frustum = FRUSTUM_SCRATCH;
    frame.settled =
      version == null ||
      this.settleTracker.isSettled(this.frame.nowMs, config.lod.fineReloadSettleMs);
    return frame;
  }

  /**
   * End an evaluated frame.
   *
   * Liveness (``tick-demand.ts``): keep the on-demand loop ticking while any
   * component still has work only a later frame can do — a load landing (the
   * swap fires on that frame), a dissolve advancing (a function of TIME: each
   * step writes a new opacity, which marks the frame dirty through
   * ``takeDrawnStateChanged``), a partition resync waiting for the loader. A
   * failed level's cooldown and an unanswered part activation request need no
   * ticks: each wakes the loop once at its expiry (``retry-wakes.ts``).
   *
   * Record whether fade management was ON this frame — the falling-edge
   * detector behind the one-shot residual-opacity restore (see
   * ``fadeWasManaged``), written AFTER the entry loop so every entry in one
   * frame sees the same previous-frame value. Then bound resident LOD geometry
   * against the shared GPU-pool byte budget (one VRAM authority): retention
   * keeps loaded levels resident so re-shows are free; this LRU-evicts only
   * hidden levels when over budget — hidden-layer (undrawable) first, then
   * off-screen / furthest-from-camera — so no per-swap release, hence no reload
   * churn.
   */
  private endFrame(
    loadsTickUntil: number,
    camera: THREE.Camera,
    displayDims: readonly number[]
  ): void {
    const tickUntil = Math.max(
      loadsTickUntil,
      this.dissolves.tickUntilMs(),
      this.partitions.tickUntilMs()
    );
    if (tickUntil > this.frame.nowMs) this.keepTicking();
    this.fadeWasManaged =
      this.deps.getCrossFadeEnabled?.() === true || this.deps.getEnergyCompEnabled?.() === true;
    this.enforceByteBudget(camera, FRUSTUM_SCRATCH, displayDims);
  }

  /**
   * A frame the registry cannot select for (fewer than two display dims, a
   * collapsed canvas). No group is evaluated, so no dissolve can land: drop
   * them all rather than leave {@link isAnimating} true until the view returns
   * (the next evaluated frame then draws each group's level alone).
   */
  private skipFrame(): LODFrameChanges {
    this.dissolves.dropAll();
    return LOD_FRAME_UNCHANGED;
  }

  /**
   * Whether a level dissolve is still in flight: the frames it spans are
   * animation frames, drawn although neither the camera nor the data moves.
   *
   * Read-only. Only ``evaluatePerFrame`` retires a dissolve, on the one end
   * rule it draws by (``levelFade``), so "not animating" means the last
   * evaluated frame drew one level. A poll between evaluates (the offline
   * capture drain through {@link isCaptureQuiescent}, the debug settle probe)
   * used to delete a fade whose wall-clock end had passed while the drawn frame
   * still showed both levels — releasing the capture on a mid-dissolve frame
   * and swallowing the landing's ``levelChanged``. Every evaluated entry reaches
   * ``levelFade`` or drops its fade, so an entry that stops dissolving never
   * holds this true.
   */
  isAnimating(): boolean {
    return this.dissolves.isAnimating();
  }

  /**
   * Until when ``entry``'s children need the loop ticking: while one is loading
   * (its swap fires on the frame it lands). A child cooling down after a
   * failure needs no ticks — one wake at the cooldown's end lets a parked
   * camera retry it (#2944 A6) — and a child neither failed nor loading drops
   * its retry backoff. For each child whose ``ensureLoaded`` has finished since
   * it was fired, fold the measured load+commit time into its ``loadEwmaMs`` (a
   * failure is not a timing).
   */
  private observeLoads(entry: LODGroupEntry): number {
    let until = NO_TICK;
    for (const c of entry.children) {
      if (c.loading) until = UNTIL_RESOLVED;
      else foldLoadTime(c, this.clock);
      if (c.failed === true && c.failedAtMs !== undefined) {
        const retryAtMs = c.failedAtMs + this.retryWakes.delay(c, config.lod.failedRetryMs);
        if (retryAtMs > this.frame.nowMs) this.retryWakes.schedule(c, retryAtMs - this.frame.nowMs);
      } else if (c.failed !== true && !c.loading) {
        this.retryWakes.reset(c);
      }
    }
    return until;
  }

  /** Cancel the retry wakes of the lod_group registered at ``path``. */
  private cancelEntryRetryWakes(path: string): void {
    for (const child of this.entries.get(path)?.children ?? []) this.retryWakes.cancel(child);
  }

  /**
   * During playback, the finest level at or below ``desired`` whose reload fits
   * the period (see ``playback-aspiration.ts``). ``desired`` is unchanged when
   * not playing, locked, or forced finest.
   */
  private playbackAspiration(entry: LODGroupEntry, desired: number): number {
    this.playback.endEntry();
    if (entry.selectorMode !== 'auto') return desired;
    if (this.deps.getForceFinestLOD?.() === true) return desired;
    return this.playback.aspire(entry, desired, this.frame.playbackPeriodMs, this.frame.nowMs);
  }

  /** Keep the loop ticking (see {@link LODGroupRegistryDeps.requestTick}). */
  private keepTicking(): void {
    (this.deps.requestTick ?? this.deps.requestRender)?.();
  }

  /**
   * Whether anything this registry drives changed what the next frame draws
   * since the last call (level visibility either way, a fade opacity step, a
   * partition part culled or restored) — then clear it. The app pipeline
   * returns it from the `'lod-group-selector'` per-frame callback so the
   * render-on-change loop redraws exactly then.
   */
  takeDrawnStateChanged(): boolean {
    const changed = this.drawnStateChanged;
    this.drawnStateChanged = false;
    return changed;
  }

  /**
   * One group's frame, in stages: pick the level the selector wants and
   * advance the aspiration toward it (``selectAspiration``), resolve what can
   * be DISPLAYED (``resolveDisplay``), advance the dissolve, drive the
   * aspiration's reload / refinement and the band preload, then apply
   * visibility and opacity (the single owner of ``object.visible``). Returns
   * ``true`` if this entry's displayed child changed.
   */
  private evaluateEntry(entry: LODGroupEntry, frame: EntryFrame): boolean {
    const desired = this.selectAspiration(entry, frame);
    if (desired === null) return false;
    const version = this.deps.getViewVersion?.();
    const displayIdx = this.resolveDisplay(entry, version);
    this.dissolveStep(entry, displayIdx, ASPIRATION.fresh, version ?? null);
    if (ASPIRATION.ready) {
      const aspiration = entry.children[entry.activeChildIndex];
      this.refreshAspiration(entry, aspiration, ASPIRATION.fresh, frame.settled);
    }
    this.preloadNeighbour(entry, desired, PICK.preloadMetric, version ?? null, frame.settled);
    const changed = this.applyVisibility(entry, displayIdx);
    this.recordDisplay(entry, displayIdx);
    return changed;
  }

  /**
   * Pick the desired level (``pickDesired``), cap it during playback, record
   * it, and advance the aspiration toward it. Returns the desired index, or
   * ``null`` when the entry cannot be evaluated this frame (its dissolve, if
   * any, is dropped where that applies).
   */
  private selectAspiration(entry: LODGroupEntry, frame: EntryFrame): number | null {
    const pick = this.pickDesired(entry, frame);
    if (pick === PICK_SKIP) return null; // no registration cache — shouldn't happen.
    if (pick === PICK_DROP) return this.dropFadeNull(entry);

    // During playback, no finer than what can reload within the period.
    const desired = this.playbackAspiration(entry, PICK.desired);

    // Record what the selector WANTS this frame, before any of the ready /
    // freshness gates below can veto it. Read by ``isCaptureQuiescent`` only —
    // see ``LODGroupEntry.desiredChildIndex`` for why the aspiration index
    // cannot answer the same question.
    entry.desiredChildIndex = desired;

    if (!this.advanceAspiration(entry, desired)) return this.dropFadeNull(entry);
    // Self-heal a NOT-ready aspiration (eager default failed to attach, or a
    // fallback pinned a not-ready lazy level) so the group can never be stuck.
    // A ready-but-STALE aspiration is deliberately NOT kicked here:
    // ``maybeKickLoad`` early-returns on ready children, and a stale level's
    // re-slice reload is driven by the scene-loader update sweep (every
    // registered child loader re-queries on a view change), not by the registry
    // — we just wait for that commit to re-stamp it fresh.
    const aspiration = entry.children[entry.activeChildIndex];
    if (aspiration && !isReady(aspiration)) this.maybeKickLoad(entry, aspiration);
    return desired;
  }

  /** {@link dropFade}, for a stage that answers ``null`` (no evaluation). */
  private dropFadeNull(entry: LODGroupEntry): null {
    this.dropFade(entry);
    return null;
  }

  /**
   * What the group displays this frame: the slice-aware fallback / stale hold,
   * then the fresh-but-empty guard, then the never-downgrade gate. Fills
   * ``ASPIRATION`` for the later stages.
   */
  private resolveDisplay(entry: LODGroupEntry, version: number | undefined): number {
    const { ready, fresh } = this.aspirationState(entry, version);
    // Preserve this across the fresh-aspiration branch of ``slicedDisplayIndex``,
    // which re-arms the hold state before the never-downgrade gate evaluates the
    // handoff.
    const staleHoldEnded = ready && fresh && entry.staleHoldSinceMs != null;
    let displayIdx = this.slicedDisplayIndex(entry, ready && fresh, version);
    if (version != null && displayIdx >= 0) {
      displayIdx = this.nonEmptyDisplayIndex(entry, displayIdx, version);
    }
    return this.neverDowngradeIndex(entry, displayIdx, staleHoldEnded, version);
  }

  /**
   * Whether the aspiration is ready and fresh for ``version`` (always fresh
   * when freshness is untracked), into ``ASPIRATION``. Freshness resolves a
   * GROUP-typed aspiration (deferred kind=partition / nested lod subtree — the
   * overview recipe) through the subtree aggregate, not the leaf-only stamp: a
   * bare THREE.Group has no leaf nodeType, so ``isFresh`` would call it
   * unconditionally fresh and a re-slice would show the stale subtree with no
   * coarse fallback. Leaf aspirations are unchanged.
   */
  private aspirationState(entry: LODGroupEntry, version: number | undefined): typeof ASPIRATION {
    const aspiration = entry.children[entry.activeChildIndex];
    ASPIRATION.ready = !!aspiration && isReady(aspiration);
    ASPIRATION.fresh =
      version == null || (!!aspiration && childFreshAndCount(aspiration, version).fresh);
    return ASPIRATION;
  }

  /**
   * Pick the desired child index into ``PICK`` (``desired``, the band-preload
   * metric, whether the footprint pick decided). An explicit lock bypasses the
   * off-screen gate: a user who pins a level keeps it whether or not the group
   * is on screen.
   *
   * Off-screen gate: if the group's world bounds are entirely outside the
   * camera frustum, hold it at the coarsest *ready* level instead of selecting
   * — and lazily loading — a fine level the renderer will frustum-cull anyway.
   * This turns frustum culling into a LOD/loading input, not just a draw-time
   * skip. ``coarsestReadyIndex`` never targets a not-ready lazy child, so no
   * load is kicked while off-screen, and the outgoing fine levels fall out of
   * the visible tally and become eviction candidates. On re-entry, retention
   * usually makes the upgrade a free visibility toggle rather than a reload.
   *
   * @returns ``PICK_OK``; ``PICK_SKIP`` when the entry has no registration
   *   cache; ``PICK_DROP`` when its bounds fold to no world box.
   */
  private pickDesired(entry: LODGroupEntry, frame: EntryFrame): number {
    PICK.preloadMetric = null;
    PICK.usedFootprint = false;
    if (entry.selectorMode !== 'auto') {
      PICK.desired = entry.selectorMode.lockLevel;
      entry.offScreen = false;
      return PICK_OK;
    }
    const cache = this.caches.get(entry.path);
    if (!cache) return PICK_SKIP;
    const worldBox = this.computeWorldBox(entry, frame.displayDims);
    if (!worldBox) return PICK_DROP;
    WORLD_BOX3_SCRATCH.min.set(worldBox.min.x, worldBox.min.y, worldBox.min.z);
    WORLD_BOX3_SCRATCH.max.set(worldBox.max.x, worldBox.max.y, worldBox.max.z);
    const forceFinest = this.deps.getForceFinestLOD?.() === true;
    if (!forceFinest && !frame.frustum.intersectsBox(WORLD_BOX3_SCRATCH)) {
      PICK.desired = this.coarsestReadyIndex(entry);
      entry.offScreen = true;
      return PICK_OK;
    }
    this.pickOnScreen(entry, cache, worldBox, frame, forceFinest);
    entry.offScreen = false;
    return PICK_OK;
  }

  /**
   * The on-screen pick: the threshold pick on the (biased) coverage metric, or
   * — for a ``screen-area`` group whose levels carry complete footprint stamps —
   * the projected-footprint pick, which then decides alone (and leaves the band
   * preload and the dissolve out, see ``dissolveStep``). Not when the metric
   * saturated (the camera is inside the box, or ``forceFinest``): the footprint
   * is sized at the box-centre depth and says nothing about the splats at the
   * eye, so the finest level stands.
   */
  private pickOnScreen(
    entry: LODGroupEntry,
    cache: LODGroupEntryCache,
    worldBox: BoundingBox,
    frame: EntryFrame,
    forceFinest: boolean
  ): void {
    let coverageMetric = forceFinest ? Infinity : this.coverageMetric(entry, cache, frame);
    const lodBias = resolveLodBias(this.deps.getLodBias?.());
    coverageMetric *= entry.selector === 'screen-area' ? lodBias : Math.sqrt(lodBias);
    const footprintDesired =
      entry.selector !== 'screen-area' || !Number.isFinite(coverageMetric)
        ? null
        : pickStampedFootprintChild(entry, cache, worldBox, frame.view, {
            viewportHeight: frame.viewport.height,
            lodBias,
            displayDims: frame.displayDims,
          });
    if (footprintDesired == null) {
      PICK.desired = pickChildWithHysteresis(
        cache.thresholds,
        entry.activeChildIndex,
        coverageMetric
      );
      if (!forceFinest) PICK.preloadMetric = coverageMetric;
    } else {
      PICK.desired = footprintDesired;
      PICK.usedFootprint = true;
    }
  }

  /**
   * The dimensionless coverage metric for this frame, in the units the
   * entry's ``selector`` names: the screen-area fraction, or the projected
   * diagonal ÷ ``FILL_FACTOR``·fittedAxisPx (legacy ``coverage``).
   *
   * The screen metrics project the group's LOCAL box through
   * projView × matrixWorld, i.e. the 8 corners of the box as oriented on
   * screen. Projecting the corners of its world AABB instead (the frustum
   * gate's box) inflated a rotated group twice and picked too fine a level.
   * ``computeWorldBox`` refreshed matrixWorld in ``pickDesired``.
   */
  private coverageMetric(
    entry: LODGroupEntry,
    cache: LODGroupEntryCache,
    frame: EntryFrame
  ): number {
    const rawLocal = cache.localBoxScratch;
    const metricLocal =
      cache.hasLodBounds && this.computeWorldBox(entry, frame.displayDims, true)
        ? cache.metricLocalBoxScratch
        : rawLocal;
    LOCAL_PROJ_SCRATCH.multiplyMatrices(FRUSTUM_MATRIX_SCRATCH, entry.groupObject.matrixWorld);
    const camera = frame.view.camera;
    if (entry.selector !== 'screen-area') {
      // Legacy 'coverage' selector (the default for older stores).
      const diagonalPx = projectBoxDiagonalPx(
        metricLocal,
        camera,
        frame.viewport,
        LOCAL_PROJ_SCRATCH
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
      const fittedAxisPx = Math.min(frame.viewport.width, frame.viewport.height);
      return diagonalPx / (FILL_FACTOR * fittedAxisPx);
    }
    // Screen-area selector: the metric IS the fraction of the viewport area the
    // group's projected inscribed ellipsoid (sized by its view-axis half-chord)
    // covers, in rect units (orientation-stable, viewport-size independent by
    // construction — see projectBoxAreaFraction). The thresholds are literal
    // area fractions ([0, …, 1/4, 1/2] whole-object; a partition tile anchors
    // at 1.0), so no FILL_FACTOR normalisation. Camera inside the box →
    // +Infinity → finest, same as the diagonal path.
    const metric = projectBoxAreaFraction(metricLocal, camera, LOCAL_PROJ_SCRATCH);
    // The thin-rectangle ramp is not monotone under box containment: trimming
    // the thin axis can increase the robust metric. Robust bounds may only keep
    // or reduce the raw-bounds selection.
    return cache.hasLodBounds
      ? Math.min(metric, projectBoxAreaFraction(rawLocal, camera, LOCAL_PROJ_SCRATCH))
      : metric;
  }

  /**
   * Advance the aspiration (``activeChildIndex``) toward ``desired``. The
   * aspiration is the hysteresis anchor and only moves onto a READY level; a
   * not-ready desired kicks its deferred loader and we keep aspiring to the
   * current level until it commits. Visibility is NOT touched here — the
   * display-resolution pass is the single owner of ``object.visible``.
   *
   * @returns ``false`` when a lock index outlives its children (an empty entry
   *   is registered on purpose and ``setSelectorMode`` skips clamping for it):
   *   nothing to aspire to, nothing to kick — never throw per frame.
   */
  private advanceAspiration(entry: LODGroupEntry, desired: number): boolean {
    if (desired === entry.activeChildIndex) return true;
    const target = entry.children[desired];
    if (!target) return false;
    if (isReady(target)) entry.activeChildIndex = desired;
    else this.maybeKickLoad(entry, target);
    return true;
  }

  /**
   * **Slice-aware display resolution.** Show the aspiration when its committed
   * geometry is fresh for the current view version (``showable``); while it is
   * stale (a time/displayDims scrub reloaded it in place without flipping
   * ``ready``) show the coarsest FRESH level so the new slice appears
   * immediately at low detail — falling back to the coarsest ready level if
   * none is fresh yet, so it never goes blank — unless what is already on
   * screen is far better and the aspiration is about to land
   * (``staleHoldDisplayIndex``). ``getViewVersion`` undefined ⇒ freshness
   * untracked: a not-ready aspiration (a lazy level still loading) keeps the
   * previously-displayed level if it is still ready, otherwise nothing until
   * the load commits — the legacy behaviour.
   */
  private slicedDisplayIndex(
    entry: LODGroupEntry,
    showable: boolean,
    version: number | undefined
  ): number {
    if (showable) {
      // The wait is over, so a spent stale-hold budget is re-armed for the
      // NEXT slice change (see ``staleHoldExhausted``).
      entry.staleHoldSinceMs = undefined;
      entry.staleHoldExhausted = false;
      return entry.activeChildIndex;
    }
    if (version != null) {
      const fallbackIdx = this.coarsestFreshOrReadyIndex(entry, version);
      return this.staleHoldDisplayIndex(entry, fallbackIdx, version) ?? fallbackIdx;
    }
    const prev = entry.displayedChildIndex ?? -1;
    return prev >= 0 && isReady(entry.children[prev]) ? prev : -1;
  }

  /**
   * **Fresh-but-EMPTY display guard.** If the chosen display level committed 0
   * elements, prefer the coarsest fresh NON-empty level no finer than the
   * chosen display or the selector's aspiration, whichever is finer. An
   * intermediate level can legitimately fill a stale fallback gap; only
   * redirecting to a level coarser than the chosen display signals
   * inconsistent/corrupt data and warrants a warning. Finer levels must not
   * override the selector, even when a coarse slice is legitimately empty (see
   * #1600). Group-aware: a deferred kind=partition / nested lod subtree whose
   * visible stamped leaves are all fresh-but-empty (poisoned/stale cache
   * serving an old layout) would otherwise slip past a leaf-only count check
   * and blank the group; the redirect target stays the coarsest fresh
   * non-empty leaf level.
   */
  private nonEmptyDisplayIndex(entry: LODGroupEntry, displayIdx: number, version: number): number {
    const chosen = entry.children[displayIdx];
    const chosenProgress = chosen ? childFreshAndCount(chosen, version) : undefined;
    if (!chosen || !chosenProgress?.fresh || chosenProgress.count !== 0) return displayIdx;
    const fallbackLimit = Math.max(displayIdx, entry.activeChildIndex);
    const fallback = this.coarsestFreshNonEmptyIndex(entry, version, fallbackLimit);
    if (fallback < 0 || fallback === displayIdx) return displayIdx;
    if (fallback < displayIdx) this.warnEmptyLevel(entry, displayIdx, fallback);
    return fallback;
  }

  /** Warn (once per group) that a fresh level committed nothing while a coarser one has geometry. */
  private warnEmptyLevel(entry: LODGroupEntry, displayIdx: number, fallback: number): void {
    if (this.warnedEmptyLevel.has(entry.path)) return;
    this.warnedEmptyLevel.add(entry.path);
    const recovery = this.deps.hasNetworkFailureUnder?.(entry.path)
      ? 'A network load failed under this group; use the monitor Retry action.'
      : 'This usually means inconsistent/stale data (e.g. a dataset regenerated at ' +
        'the same URL with a poisoned cache); try reloading with ?clearCache.';
    log.warning(
      Modules.SCENE_LOADER,
      `lod_group ${entry.path}: level ${displayIdx} is fresh but committed 0 ` +
        `elements while level ${fallback} has visible geometry — showing level ` +
        `${fallback} instead. ${recovery}`
    );
  }

  /**
   * **Never-downgrade display gate.** A lazy level flips ``ready`` after its
   * FIRST additive chunk commits, so an ungated swap to a fresh-but-still-
   * streaming aspiration pops displayed quality down to chunk-1 and climbs
   * back. Hold the previously-displayed level while the streaming aspiration
   * is strictly worse than what is on screen; release on ladder completion,
   * committed-count crossover, ladder failure, or the previous level losing
   * freshness. Bypassed for an explicit lock and while off-screen. This is a
   * STREAMING/loading concern (WHEN a just-loaded level is good enough to
   * show) — orthogonal to the time-driven level dissolve, and stays a hard
   * hold.
   */
  private neverDowngradeIndex(
    entry: LODGroupEntry,
    displayIdx: number,
    staleHoldEnded: boolean,
    version: number | undefined
  ): number {
    if (!neverDowngradeApplies(entry, displayIdx)) return displayIdx;
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
    if (prevIdx == null || prevIdx === displayIdx) return displayIdx;
    const aspiration = entry.children[displayIdx];
    // Children are coarsest→finest, so displayIdx (== activeChildIndex) being
    // FINER than the held prev means an upgrade (zoom-in); the gate's early
    // energy-release is sound only then (downgrade would pop below the held).
    const isUpgrade = displayIdx > prevIdx;
    const prev = entry.children[prevIdx];
    if (!shouldHoldPreviousDisplay(aspiration, prev, version ?? null, isUpgrade)) {
      return displayIdx;
    }
    aspiration.lastVisibleTick = this.tick;
    return prevIdx;
  }

  /**
   * **Level dissolve (time-driven)** into ``BLEND``. When the DISPLAYED level
   * changes, dissolve from the outgoing level to the incoming one over
   * ``config.lod.fadeMs`` — incoming at w = smoothstep of the elapsed fraction,
   * outgoing at (1−w) — so the substitutive switch dissolves instead of
   * popping. A function of TIME since the change, never of the camera's
   * distance to a threshold: a parked camera always settles on ONE level at
   * full weight (#2925 — a distance band drew two levels for as long as the
   * camera stayed inside it). Brightness is preserved by the levels'
   * build-time mass conservation (both integrate to the same DC). Blendable
   * modes only (BLENDABLE_MODES = additive/luminous/volumetric — energy sums
   * linearly, or opacity linearly scales optical depth τ so the pair
   * interpolates monotonically between the two levels' absorptions; see that
   * set's doc for what volumetric does NOT guarantee). For two mid-fade
   * volumetric siblings the mesh draw order may come from the render-order
   * containment rule (near-identical bounds); acceptable because combined
   * TRANSMITTANCE is order-independent (transmittances multiply), so
   * occlusion of content behind the pair is exact at every weight. The
   * emission ordering residual stays bounded by the local inter-level radiance
   * difference, not by the rendered hard-swap pop.
   *
   * Off / non-blendable / off-screen / locked / a held-stale display ⇒ no
   * dissolve (byte-identical hard swap). Nor for a level picked by the GSplat
   * projected-FOOTPRINT rule (``PICK.usedFootprint``). That rule picks the
   * coarsest level whose median splat stays under
   * ``config.lod.maxMedianFootprintPx``, so its swaps happen where splats are
   * about a pixel across and the two levels are expected to read alike — the
   * case the dissolve exists for (a visible pop) should not arise, and drawing
   * both levels for ``fadeMs`` would add a second level's overdraw. The
   * footprint path leaves ``preloadMetric`` null too, so it gets no band
   * preload: the band is measured against coverage thresholds, and its switch
   * point is a pixel-size threshold. (A design reading, not a measured
   * guarantee: if footprint swaps are seen to pop, this is the gate to open.)
   */
  private dissolveStep(
    entry: LODGroupEntry,
    displayIdx: number,
    aspirationFresh: boolean,
    version: number | null
  ): void {
    BLEND.partner = null;
    BLEND.primaryWeight = 1;
    const fadeEligible =
      this.deps.getCrossFadeEnabled?.() === true &&
      entry.selectorMode === 'auto' &&
      !entry.offScreen &&
      aspirationFresh &&
      displayIdx === entry.activeChildIndex &&
      !PICK.usedFootprint;
    // Last frame's outgoing level, if a dissolve was in flight (see the hide
    // edge in the visibility pass).
    BLEND.fadingFromIdx = this.dissolves.fadingFrom(entry.path);
    BLEND.droppedFromIdx = this.dissolves.droppedFromIdx(entry.path);
    const fade = fadeEligible
      ? this.levelFade(entry, displayIdx, version)
      : this.dissolves.end(entry.path);
    if (!fade) return;
    BLEND.partner = fade.fromIdx;
    // primaryWeight is the OPACITY of the primary (displayIdx = the incoming
    // level); the outgoing level gets the complement.
    BLEND.primaryWeight = smoothstep(0, 1, fade.progress);
    // Keep both warm in the eviction LRU (both are on screen).
    entry.children[fade.fromIdx].lastVisibleTick = this.tick;
    entry.children[entry.activeChildIndex].lastVisibleTick = this.tick;
  }

  /**
   * **Settle-gated reload / progressive refinement of the (ready) lazy
   * aspiration.** A lazy level no longer joins the per-slice sweep (see
   * load-lod-group-node.ts), so the registry drives its (re)loading HERE,
   * settle-gated: during active scrubbing (version changing every frame)
   * nothing fires, so only the cheap coarse level (still sweep-driven) shows
   * the new slice; once the user pauses we (a) reload a STALE level for the
   * new slice, and (b) advance a PROGRESSIVE level that is fresh but still has
   * additive LODs to stream — both by re-firing the same ``ensureLoaded``,
   * until the level is ready, fresh, AND complete. Eager (coarse) levels have
   * no ``ensureLoaded`` and stay sweep-driven, so this only ever targets lazy
   * levels. Only (a) is a reload TIMING; a ladder refinement step is not.
   *
   * Never for a deferred GROUP child (the overview recipe's fine partition /
   * nested lod branch): it has an ``ensureLoaded`` too, but its expensive step
   * is ``loadChildren`` — re-running it attaches a SECOND copy of the whole
   * subtree under the placeholder (double-drawn geometry, duplicate names,
   * re-registered loaders, leaked buffers). Its leaves are sweep-registered
   * and re-stamp themselves, so staleness needs no kick. Keyed on the explicit
   * ``deferredGroup`` flag, not on the leaf's ``nodeType`` stamp: a lazy leaf
   * whose placeholder is not stamped yet must still drain its ladder here.
   *
   * During playback a STALE aspiration reloads on every timepoint (the version
   * never settles there, and ``playbackAspiration`` kept it to a level that
   * fits the period); refinement of a fresh one still waits.
   */
  private refreshAspiration(
    entry: LODGroupEntry,
    aspiration: LODGroupChild,
    aspirationFresh: boolean,
    settled: boolean
  ): void {
    if (aspiration.deferredGroup === true || !aspiration.ensureLoaded) return;
    if (aspirationFresh && !(aspiration.hasMoreLODs?.() ?? false)) return;
    const playingStale = this.frame.playbackPeriodMs !== null && !aspirationFresh;
    if (settled || playingStale) this.maybeKickReload(entry, aspiration, !aspirationFresh);
  }

  /**
   * **Apply visibility (single owner).** At most the display child plus its
   * dissolve partner is visible (``displayIdx`` / ``BLEND.partner``, each only
   * if READY — never force-show a not-ready placeholder). Returns whether a
   * SHOWN level changed, so ``evaluatePerFrame`` refreshes the monitor's
   * visible tally, which counts the displayed level, not the aspiration.
   *
   * Opacity is managed only while at least one anti-popping feature is on: the
   * level dissolve and/or the streaming energy compensation (per-leaf
   * `1/e(k)` on the displayed streaming level). When BOTH are off this reduces
   * to the original single-level visibility swap with no material writes —
   * byte-identical. On the falling edge of fade management (both flags just
   * toggled OFF, possibly MID-fade) every child's authored opacity is restored
   * ONCE so a half-faded level does not stay dim forever (``fadeWasManaged`` is
   * updated per frame after all entries run, so the edge fires exactly one
   * frame for each entry).
   */
  private applyVisibility(entry: LODGroupEntry, displayIdx: number): boolean {
    FADE_MODE.energyComp = this.deps.getEnergyCompEnabled?.() === true;
    FADE_MODE.manage = this.deps.getCrossFadeEnabled?.() === true || FADE_MODE.energyComp;
    FADE_MODE.restoreResidual = !FADE_MODE.manage && this.fadeWasManaged;
    let changed = false;
    for (let i = 0; i < entry.children.length; i++) {
      const child = entry.children[i];
      const isPrimary = i === displayIdx;
      const shouldShow = (isPrimary || i === BLEND.partner) && isReady(child);
      if (this.setChildVisible(child, i, shouldShow)) changed = true;
      this.applyChildOpacity(child, isPrimary, shouldShow);
    }
    this.dissolves.settled(entry.path);
    return changed;
  }

  /**
   * One child's opacity under ``FADE_MODE``. Dissolve weight only when a
   * partner is in flight (primary at α, partner at 1−α); otherwise no coverage
   * weight (null ⇒ 1). Energy compensation is folded in PER-LEAF inside
   * applyChildFade, so it also covers a plainly-displayed streaming level with
   * no cross-fade partner. A hidden child gets its authored opacity back
   * (idempotent), as does every child once on the fade-management falling edge.
   */
  private applyChildOpacity(child: LODGroupChild, isPrimary: boolean, shown: boolean): void {
    if (FADE_MODE.manage) {
      const weight = shown ? this.blendWeight(isPrimary) : null;
      this.applyChildFade(child, weight, shown && FADE_MODE.energyComp);
    } else if (FADE_MODE.restoreResidual) {
      this.applyChildFade(child, null, false);
    }
  }

  /**
   * Show or hide one child. Returns whether that changes the visible tally: a
   * new level became visible, or a dissolve's outgoing level left on its own
   * (its partner has been on screen since the dissolve started) — the monitor
   * must recount then too, like the swap it completes.
   */
  private setChildVisible(child: LODGroupChild, index: number, shouldShow: boolean): boolean {
    if (child.object.visible === shouldShow) return false;
    child.object.visible = shouldShow;
    this.drawnStateChanged = true;
    return shouldShow || index === BLEND.fadingFromIdx || index === BLEND.droppedFromIdx;
  }

  /** A shown child's dissolve weight: ``null`` (⇒ 1) unless a partner is in flight. */
  private blendWeight(isPrimary: boolean): number | null {
    if (BLEND.partner == null) return null;
    return isPrimary ? BLEND.primaryWeight : 1 - BLEND.primaryWeight;
  }

  /**
   * Mark the on-screen level most-recently-used and record it for the eviction
   * pass, which must never release the level currently displayed. Only update
   * when a ready level is actually shown — otherwise keep the last shown index
   * so eviction still protects whatever the user last saw.
   *
   * Then stamp any already-ready child that has never been shown so it ages
   * into the eviction LRU. Without this, a lazy level that finished loading but
   * was never displayed (camera/slice moved away mid-load) keeps
   * ``lastVisibleTick == null`` and is permanently exempt from eviction,
   * leaking VRAM. A SENTINEL, not the current tick: the eviction LRU's final
   * tiebreak is coldest-first on this field, so stamping "now" would make the
   * one level the user never saw the HOTTEST in its group and evict levels they
   * looked at moments ago before it. 0 is older than any real tick (ticks start
   * at 1).
   */
  private recordDisplay(entry: LODGroupEntry, displayIdx: number): void {
    countLodDisplay(entry, displayIdx, BLEND.partner);
    const shown = displayIdx >= 0 ? entry.children[displayIdx] : undefined;
    if (shown && isReady(shown)) {
      shown.lastVisibleTick = this.tick;
      entry.displayedChildIndex = displayIdx;
      // The gate's memory tracks only what was shown ON SCREEN, so an
      // off-screen excursion (which displays the coarse fallback) cannot
      // clobber a held finer level and re-pop it on camera return.
      if (!entry.offScreen) entry.heldDisplayChildIndex = displayIdx;
    }
    for (const child of entry.children) {
      if (isReady(child) && child.lastVisibleTick == null) child.lastVisibleTick = 0;
    }
  }

  /**
   * **Band preload.** While the selector metric sits within
   * ``config.lod.preloadBandFraction`` of the smaller adjacent inter-threshold
   * gap from a threshold of the selected level (``desired``), make the level
   * across that threshold resident in the background, WITHOUT drawing it: the
   * visibility pass shows only the displayed level and its dissolve partner,
   * so a preloaded level stays hidden, at its authored opacity, and outside
   * every drawn tally until the selector picks it. Crossing the threshold then
   * finds it ready and the dissolve starts that frame, instead of after its
   * load (the band the retired distance cross-fade DREW both levels in).
   *
   * It drives the same ``ensureLoaded`` the selection would, through the same
   * gates (hidden layer, archive fault, failure cooldown, one load in flight),
   * and the same settle-gated reload / ladder refinement the aspiration gets,
   * so a level that becomes the aspiration is already as far along as it would
   * have been. It never runs:
   *   - with the dissolve off (``?noLodFade``) — a hard swap was never
   *     preceded by a band, and that mode stays byte-identical;
   *   - during playback, where ``playbackAspiration`` alone decides what fits;
   *   - while the selected level is not ready yet, so the load being waited
   *     for is not slowed by a speculative one;
   *   - for a deferred GROUP level, whose activation attaches a subtree
   *     (``loadChildren``) with loaders of its own;
   *   - off-screen, under a lock, or for a footprint- or force-finest pick
   *     (``metric`` null), or while the group's layer is hidden.
   * Any of these ends the visit. The fetches ride the same fetch class as any
   * lazy level load; a load in flight when the camera leaves the band simply
   * lands hidden, as one does when the selection moves away mid-load.
   *
   * Residency is otherwise left to the byte budget: a preloaded level is an
   * ordinary hidden eviction candidate. Once it has landed, a release is not
   * followed by a reload until the metric leaves the wider exit band
   * (``config.lod.preloadExitBandFraction``) and comes back, so VRAM pressure cannot
   * turn a parked camera into a load/evict loop.
   */
  private preloadNeighbour(
    entry: LODGroupEntry,
    desired: number,
    metric: number | null,
    version: number | null,
    settled: boolean
  ): void {
    const thresholds = metric === null ? null : this.preloadThresholds(entry, desired);
    const visit =
      thresholds === null || metric === null
        ? this.endPreload(entry.path)
        : this.preloadVisit(entry.path, thresholds, desired, metric);
    if (visit) this.advancePreload(entry, entry.children[visit.idx], visit, version, settled);
  }

  /**
   * The group's thresholds when a band preload may run this frame (see
   * {@link preloadNeighbour} for the conditions), else ``null``.
   */
  private preloadThresholds(entry: LODGroupEntry, desired: number): readonly number[] | null {
    if (this.deps.getCrossFadeEnabled?.() !== true) return null;
    if (this.frame.playbackPeriodMs !== null || desired !== entry.activeChildIndex) return null;
    const selected = entry.children[desired];
    if (!selected || !isReady(selected)) return null;
    // A hidden layer ends the visit: nothing starts loading under it anyway
    // (``kickDeferredLoadIfVisible``), and a level released while it was hidden
    // must be preloadable again once it is shown.
    if (!isEffectivelyVisible(entry.groupObject)) return null;
    return this.caches.get(entry.path)?.thresholds ?? null;
  }

  /** Forget a group's preload visit; always ``null`` (nothing to advance). */
  private endPreload(path: string): null {
    this.preloads.delete(path);
    return null;
  }

  /**
   * The preload visit to advance this frame, or ``null``. A metric inside the
   * entry band of the neighbour already being preloaded continues that visit;
   * inside another neighbour's band starts a new one; between the entry and
   * the exit band of the current visit keeps it without advancing it; anywhere
   * else ends it.
   */
  private preloadVisit(
    path: string,
    thresholds: readonly number[],
    desired: number,
    metric: number
  ): PreloadVisit | null {
    const visit = this.preloads.get(path);
    const idx = preloadNeighbourIndex(thresholds, desired, metric, config.lod.preloadBandFraction);
    if (idx >= 0 && visit?.idx === idx) return visit;
    if (idx < 0) {
      const exitIdx = preloadNeighbourIndex(
        thresholds,
        desired,
        metric,
        config.lod.preloadExitBandFraction
      );
      return visit !== undefined && exitIdx === visit.idx ? null : this.endPreload(path);
    }
    const started: PreloadVisit = { idx, sawReady: false };
    this.preloads.set(path, started);
    return started;
  }

  /**
   * One frame of a preload visit: load the level if it is not resident (unless
   * it already landed this visit and was released since — no reload loop);
   * once resident, reload it for a new slice or advance its ladder on the same
   * settle gate as the aspiration. Deferred GROUP levels are never preloaded.
   */
  private advancePreload(
    entry: LODGroupEntry,
    child: LODGroupChild,
    visit: PreloadVisit,
    version: number | null,
    settled: boolean
  ): void {
    if (child.deferredGroup === true || !child.ensureLoaded) return;
    if (!isReady(child)) {
      if (!visit.sawReady) this.maybeKickLoad(entry, child);
      return;
    }
    visit.sawReady = true;
    if (!settled) return;
    const stale = version !== null && !childFreshAndCount(child, version).fresh;
    if (stale || child.hasMoreLODs?.() === true) this.maybeKickReload(entry, child, stale);
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
      useLodBounds ? cache.metricLocalBoxScratch : cache.localBoxScratch,
      this.matrixScratch,
      useLodBounds ? cache.metricWorldBoxOptions : cache.worldBoxOptions
    );
  }

  /**
   * Index of the coarsest child strictly before ``beforeIndex`` that is fresh
   * for ``version`` AND has a non-zero committed element count — or ``-1`` when
   * none qualifies. The group-aware counterpart of the empty-level display
   * guard's fallback: it resolves each child through {@link childFreshAndCount},
   * so a fresh-but-empty GROUP child (a deferred ``kind=partition`` subtree
   * whose visible leaves all committed 0) is correctly skipped rather than
   * treated as non-empty (a bare ``THREE.Group`` has no leaf count stamp). A
   * READY child with an UNTRACKED count (``null`` — group with no stamped leaf)
   * is accepted: the guard only redirects away from KNOWN-empty levels. The
   * caller bounds the search at the chosen display level or the selector's
   * aspiration, whichever is finer, so no finer level can override it. Because
   * ``childFreshAndCount``'s ``fresh`` implies ``ready``, a NOT-ready placeholder
   * can never be returned — the guard must only redirect to a level that can
   * actually draw. When nothing qualifies (``-1``) the caller keeps the
   * fresh-but-empty current level: an empty-but-real level beats a blank
   * placeholder.
   */
  private coarsestFreshNonEmptyIndex(
    entry: LODGroupEntry,
    version: number,
    beforeIndex: number
  ): number {
    for (let i = 0; i < entry.children.length && i < beforeIndex; i++) {
      const p = childFreshAndCount(entry.children[i], version);
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
   * Advance (or start) the dissolve of ``entry`` toward ``displayIdx`` on this
   * frame's clock (``LodDissolves.advance``), or ``null`` when the group should
   * draw ``displayIdx`` alone.
   */
  private levelFade(
    entry: LODGroupEntry,
    displayIdx: number,
    version: number | null
  ): LevelFade | null {
    const frame = this.dissolveFrame;
    frame.nowMs = this.frame.nowMs;
    frame.fadeMs = config.lod.fadeMs;
    frame.version = version;
    return this.dissolves.advance(entry, displayIdx, frame, this.dissolveHost);
  }

  /**
   * Forget ``entry``'s dissolve on an ``evaluateEntry`` path that returns before
   * ``levelFade`` (so {@link isAnimating} cannot stay true for a group that is no
   * longer dissolving). Returns ``false``: no level change this frame.
   */
  private dropFade(entry: LODGroupEntry): false {
    this.dissolves.drop(entry.path);
    return false;
  }

  /** The registry's wall clock, in milliseconds (``deps.now`` in tests). */
  private nowMs(): number {
    return this.deps.now?.() ?? performance.now();
  }

  /** Whether ``fromIdx`` can be drawn dissolving against ``toIdx`` this frame. */
  private canDissolve(
    entry: LODGroupEntry,
    fromIdx: number,
    toIdx: number,
    version: number | null
  ): boolean {
    const from = entry.children[fromIdx];
    const to = entry.children[toIdx];
    if (!from || !to || !isReady(from)) return false;
    if (version != null && !childFreshAndCount(from, version).fresh) return false;
    return this.isBlendable(from) && this.isBlendable(to);
  }

  /**
   * Apply the per-leaf LOD anti-popping opacity (dissolve weight ×
   * streaming `1/e(k)` energy compensation) to a child's leaf materials, or
   * restore the authored opacity — see {@link applyLodFade}
   * (``lod-fade.ts``) for the full mechanics. This wrapper supplies the
   * registry's ``registerMaterial`` dep so a clone-on-first-fade material
   * keeps receiving per-frame camera-uniform updates.
   */
  private applyChildFade(child: LODGroupChild, weight: number | null, energyComp: boolean): void {
    if (applyLodFade(child.object, weight, energyComp, this.deps.registerMaterial)) {
      this.drawnStateChanged = true;
    }
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
      if (childFreshAndCount(entry.children[i], version).fresh) return i;
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
   *     `config.lod.staleHoldMinRatio` of the held level's committed count —
   *     so a fallback that is nearly as good is taken immediately (it is
   *     fresh, and freshness wins whenever quality is comparable),
   *   - the hold has not exhausted its `config.lod.staleHoldMs` budget.
   *
   * The budget is deliberately spent from when the hold STARTS and is not
   * refreshed by later version bumps, and once exhausted it latches until the
   * aspiration commits fresh. So a continuous drag degrades to exactly the
   * pre-existing behaviour after `config.lod.staleHoldMs`, rather than freezing on
   * one frame for as long as the user keeps dragging. The one exception is
   * playback with the held level's own reload in flight: that hold lasts
   * until the reload lands, and its budget restarts on each such frame.
   */
  private staleHoldDisplayIndex(
    entry: LODGroupEntry,
    fallbackIdx: number,
    version: number
  ): number | undefined {
    const prevIdx = this.staleHoldCandidate(entry, fallbackIdx, version);
    if (prevIdx === undefined) return undefined;
    const prev = entry.children[prevIdx];
    const now = this.frame.nowMs;
    if (entry.staleHoldSinceMs == null) entry.staleHoldSinceMs = now;
    // During playback the held level's own reload for the new timepoint is in
    // flight: keep it until that replacement lands rather than dropping to a
    // token of the new frame for the rest of the wait.
    // The budget counts from the last frame of that exemption, so pausing
    // mid-reload keeps the hold for a full config.lod.staleHoldMs instead of dropping
    // to the coarse level on the first paused frame.
    const replacementInFlight = this.frame.playbackPeriodMs !== null && prev.loading === true;
    if (replacementInFlight) {
      entry.staleHoldSinceMs = now;
    } else if (now - entry.staleHoldSinceMs >= config.lod.staleHoldMs) {
      entry.staleHoldExhausted = true;
      entry.staleHoldSinceMs = undefined;
      return undefined;
    }
    // Keep the held level warm so eviction does not reclaim it mid-hold.
    prev.lastVisibleTick = this.tick;
    return prevIdx;
  }

  /**
   * The previously-displayed level a stale hold could keep, or ``undefined``:
   * the group is auto-selected, on screen and has budget left; the level is
   * ready and FINER than the fallback; and the fallback is a SEVERE downgrade
   * (see ``staleHoldDisplayIndex``).
   */
  private staleHoldCandidate(
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
    return fallbackIsSevere(prev, entry.children[fallbackIdx], version) ? prevIdx : undefined;
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
   * A freshly-failed child is stamped with the frame clock; once its cooldown
   * (``config.lod.failedRetryMs``, backed off per consecutive failure) elapses
   * the ``failed`` flag is cleared and the load retried — recovering a level that failed on *reload* (after a
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
   * re-calling it every settled frame safe. ``timed`` is false for a ladder
   * refinement step of a FRESH level: only a full reload is a reload timing
   * for ``loadEwmaMs``.
   */
  private maybeKickReload(entry: LODGroupEntry, child: LODGroupChild, timed: boolean): void {
    this.kickDeferredLoadIfVisible(entry, child, timed);
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
  private kickDeferredLoadIfVisible(
    entry: LODGroupEntry,
    child: LODGroupChild,
    timed = true
  ): void {
    if (this.deps.hasArchiveFault?.() || !isEffectivelyVisible(entry.groupObject)) return;
    this.kickDeferredLoad(child, false, timed);
  }

  /**
   * Shared lazy-load gate for ``maybeKickLoad`` (initial load of a not-ready
   * level) and ``maybeKickReload`` (refresh of a ready-but-stale level): fire
   * ``ensureLoaded`` unless already loading or inside the failure cooldown. A
   * freshly-failed child is stamped with the frame clock; once its cooldown
   * (``config.lod.failedRetryMs``, backed off) elapses the ``failed`` flag clears
   * and the load retries — recovering a level that failed on reload (after a successful load
   * + byte-eviction), which the old "failed until released" behaviour left stuck.
   * The automatic caller separately gates loader-level archive faults. The
   * per-child latch keeps concurrent failed branches individually addressable
   * by Retry; an explicit retry clears that selected branch as it starts.
   *
   * A ``timed`` load (every full load or reload) is measured into
   * ``loadEwmaMs``; the kick that starts this frame's playback probe marks it
   * ``playbackProbePending``. An untimed one (a ladder refinement step) is not
   * measured.
   */
  private kickDeferredLoad(child: LODGroupChild, explicitRetry: boolean, timed: boolean): boolean {
    if (!child.ensureLoaded || child.loading || (!explicitRetry && child.permanentlyFailed)) {
      return false;
    }
    if (child.failed) {
      // A cooldown that has not elapsed refuses an automatic retry; an explicit
      // one clears the failure at once.
      if (explicitRetry) this.retryWakes.reset(child);
      else if (this.failureCooledDown(child)) this.retryWakes.backOff(child);
      else return false;
      child.failed = false;
      child.failedAtMs = undefined;
    }
    if (explicitRetry) {
      clearChildFailure(child);
    }
    // A load that finished this frame, before the post-evaluation walk saw it.
    foldLoadTime(child, this.clock);
    child.loading = true;
    this.startLoadTiming(child, timed);
    child.ensureLoaded();
    return true;
  }

  /**
   * Stamp the start of a ``timed`` load (its end is stamped by
   * ``onLoadSettled``) and let the playback probe claim it; an untimed one (a
   * ladder refinement step) is not measured.
   */
  private startLoadTiming(child: LODGroupChild, timed: boolean): void {
    if (!timed) {
      child.onLoadSettled = undefined;
      return;
    }
    const startMs = this.nowMs();
    child.loadStartMs = startMs;
    child.onLoadSettled = () => {
      // Only the load this kick started (a release + re-kick restarts it).
      if (child.loadStartMs === startMs) child.loadEndMs = this.nowMs();
    };
    this.playback.noteTimedLoadStarted(child);
  }

  /**
   * Whether a failed child's retry cooldown has elapsed. The first frame that
   * observes the failure starts the cooldown clock (and so answers no).
   */
  private failureCooledDown(child: LODGroupChild): boolean {
    const cooldownMs = this.retryWakes.delay(child, config.lod.failedRetryMs);
    if (child.failedAtMs == null) {
      child.failedAtMs = this.frame.nowMs;
      // Nothing else may wake a parked camera: one wake at the cooldown's end.
      this.retryWakes.schedule(child, cooldownMs);
      return false;
    }
    return this.frame.nowMs - child.failedAtMs >= cooldownMs;
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
