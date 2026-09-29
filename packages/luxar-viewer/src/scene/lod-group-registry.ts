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
 * frames and gated on ``isUpdateInProgress``) — unless every leaf of the part is
 * already committed for the current view with a complete ladder, in which case
 * nothing was missed and the re-entry only re-shows it. The resync re-sweeps under the
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

import { bumpFailedLoadsVersion } from '../utils/failed-loads-version';
import { clearChildFailure } from '../utils/lod-child-failure';
import * as THREE from 'three';

import type { BoundingBox } from './scene-manager/clipping/bounds-math';
import { ViewContextProvider, type ViewContext } from './view-context';
import { log, Modules } from '../utils/log';
import { isEffectivelyVisible, isPartitionFrustumCulled } from '../utils/object-visibility';
import type { LODGroupSelectorMode } from '../types/lod-group';
import type { LodSelectorName } from '../types/format-contract';
import {
  isFresh,
  isReady,
  isTrackedLeaf,
  SettleTracker,
  subtreeSweepSettled,
  visibleElementCount,
  type SweepNode,
} from './lod-freshness';
import {
  shouldHoldPreviousDisplay,
  subtreeDisplayProgress,
  type ProgressNode,
} from './lod-display-gate';
import { smoothstep } from './lod-blend';
import { config } from '../config';
import { applyLodFade, FADE_EPSILON, isBlendableSubtree } from './lod-fade';
import {
  computeEntryWorldBox,
  MAX_MEDIAN_FOOTPRINT_PX,
  pickChildByFootprintWithHysteresis,
  pickChildWithHysteresis,
  preloadNeighbourIndex,
  projectBoxAreaFraction,
  projectBoxDiagonalPx,
  projectWorldRadiusPx,
  type WorldBoxOptions,
} from './lod-selector-math';
import { enforceResidentByteBudget } from './lod-eviction';
import { perfCounters } from '../profiling/perf-counters';
import {
  partBoundsIntersectSlice,
  type PartitionSliceView,
} from '../data/scene-loader/view-state/partition-slice-gate';

/** Perf counter: deferred partition parts initialised on activation (B4). */
const S_PARTS_ACTIVATED = perfCounters.slot('partition.partsActivated');

// The selector math (box projection + hysteresis pick) lives in
// `lod-selector-math.ts`; re-exported here so existing importers (the
// selector unit tests) keep their import site.
export {
  pickChildWithHysteresis,
  projectBoxAreaFraction,
  projectBoxDiagonalPx,
} from './lod-selector-math';

/**
 * An in-flight dissolve between two levels of one group (see
 * ``LODGroupRegistry.levelFade``). ``progress`` ∈ [0, 1) is the INCOMING level's
 * share of the dissolve; its opacity is ``smoothstep(progress)`` and the
 * outgoing level's the complement. It advances at ``1 / config.lod.fadeMs`` per
 * millisecond from ``startProgress`` at ``startMs``.
 */
interface LevelFade {
  /** The outgoing level (drawn at the complement weight). */
  fromIdx: number;
  /** The incoming level — the one displayed. */
  toIdx: number;
  startMs: number;
  startProgress: number;
  progress: number;
}

/**
 * The dissolve that starts when the displayed level becomes ``toIdx`` (the
 * previous frame displayed ``prevIdx``), given the dissolve in flight, if any.
 * It starts from what is on screen. With nothing in flight, from ``prevIdx`` at
 * progress 0. Reversing an in-flight dissolve (back to its outgoing level)
 * starts at ``1 − progress``, so each level keeps exactly its current opacity
 * and the reversal only undoes what happened. Retargeting to a THIRD level
 * keeps the more opaque of the two as the outgoing level, at its current
 * opacity (the other one, at most half-weight, drops out).
 */
function retargetedFade(
  inFlight: LevelFade | null,
  prevIdx: number,
  toIdx: number,
  startMs: number
): LevelFade {
  let fromIdx = prevIdx;
  let start = 0;
  if (inFlight !== null && inFlight.toIdx === prevIdx) {
    const p = inFlight.progress;
    if (toIdx !== inFlight.fromIdx && p < 0.5) {
      fromIdx = inFlight.fromIdx; // still the more opaque one, at 1 − w(p) = w(1 − p)
      start = p;
    } else {
      start = 1 - p; // the incoming-so-far becomes outgoing, at w(p) = 1 − w(1 − p)
    }
  }
  return { fromIdx, toIdx, startMs, startProgress: start, progress: start };
}

/** Perf counter: displayed-level changes of a lod group (one per group per frame). */
/**
 * Band preload (see ``LODGroupRegistry.preloadNeighbour``): the level a group is
 * making resident, hidden, while its selector metric sits in the band of the
 * threshold between it and the displayed level. ``sawReady`` records that it
 * landed during this visit, so a release under VRAM pressure is not followed by
 * a reload while the camera stays put (load → evict → load …).
 */
interface PreloadVisit {
  idx: number;
  sawReady: boolean;
}

/**
 * Exit half-width of the preload band, as a fraction of the smaller adjacent
 * inter-threshold gap: a visit ends only once the metric leaves this band, which
 * is wider than the entry band (``config.lod.preloadBandFraction``) so a camera
 * hovering at the entry edge does not start a new visit (and a reload) per
 * wobble. 0.5 is the widest band that cannot reach a neighbouring threshold's.
 */
const PRELOAD_EXIT_BAND_FRACTION = 0.5;

const S_LOD_LEVEL_SWAPS = perfCounters.slot('lod.levelSwaps');
/** Perf counter: group-frames drawing two levels cross-faded (one per group per frame). */
const S_LOD_BLEND_FRAMES = perfCounters.slot('lod.blendFrames');

/**
 * Tally this frame's display outcome for one group (perf counters only): a
 * level swap when the shown level differs from the last one displayed, and a
 * blend frame when the primary and its cross-fade partner are both drawn.
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
 * Milliseconds the view-update version must hold steady before the registry
 * reloads a stale fine level (the settle debounce — see `maybeKickReload`).
 * While the user is actively scrubbing (version changes every frame) only the
 * cheap coarse level shows; the fine level reloads once they pause this long.
 * In milliseconds, like `STALE_HOLD_MS`, so it means the same on any display
 * (it was 8 frames: 130 ms at 60 Hz, 48 ms at 165 Hz).
 *
 * Playback does not wait for it (see {@link PLAYBACK_LOAD_BUDGET_FRACTION}): a
 * playing timelapse bumps the version every period, so it never settles.
 */
const FINE_RELOAD_SETTLE_MS = 130;

/**
 * During playback the registry aspires to the finest level whose measured
 * load+commit time (``LODGroupChild.loadEwmaMs``) is at most this fraction of
 * the playback period, and reloads it on every timepoint without waiting for
 * the settle debounce. A level the period cannot carry would be stale on
 * every frame and drop the display to the coarsest fresh level; the finest
 * one that CAN keep up is what a video player's "auto quality" would pick.
 * An unmeasured lazy level counts as fitting, so it gets measured.
 */
const PLAYBACK_LOAD_BUDGET_FRACTION = 0.8;

/** Weight of a new sample in ``LODGroupChild.loadEwmaMs``. */
const LOAD_EWMA_ALPHA = 0.3;

/** Retry the next capped playback level once per second to measure warm loads. */
const PLAYBACK_PROBE_INTERVAL_MS = 1000;

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
 * In MILLISECONDS, deliberately, like the settle debounce above: this is a
 * tolerance for how long a viewer may show the previous slice, which is a
 * wall-clock judgement. Sizing it in frames makes it display-dependent — the
 * first version of this fix used 8 frames and worked on a 60 Hz panel while
 * still flashing on a 165 Hz one, where 8 frames is 48 ms and the re-commit
 * needs ~70.
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
  /** Last playback probe of a level whose measured reload exceeded the budget. */
  lastPlaybackProbeMs?: number;
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
   * inscribed ellipsoid sized at its nearest corner (``projectBoxAreaFraction``); ``'coverage'`` — the legacy diagonal metric
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
  /**
   * Deferred part (B4): the part's subtree has NOT been loaded — `objects` is
   * the empty slot it will load into. The registry runs this once, inside a
   * loader pass ({@link LODGroupRegistry.activatePartitionParts}), when the
   * part is in the padded frustum and in the pass's slice. Absent ⇒ loaded.
   */
  activate?: () => Promise<void>;
  /**
   * `true` ⇒ never slice-gate this part (its bounds are not in the space of the
   * world slice, e.g. an `nd_transform` on its path). See `partition-slice-gate.ts`.
   */
  sliceExempt?: boolean;
  /** Names of the dimensions the part extends across (`extend_to_all`): never gated. */
  extendDims?: readonly string[];
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
    MAX_MEDIAN_FOOTPRINT_PX / Math.sqrt(options.lodBias)
  );
}
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
/** projection × view × a group's matrixWorld: local box corners → clip space. */
const LOCAL_PROJ_SCRATCH = new THREE.Matrix4();
const FOOTPRINT_BOX3_SCRATCH = new THREE.Box3();
// Bit flags returned by evaluatePartitionEntry so one child scan reports both effects.
const PARTITION_VISIBILITY_CHANGED = 1;
const PARTITION_BECAME_VISIBLE = 2;
// Part paths that re-entered the frustum THIS frame, collected by
// ``evaluatePartitionEntry`` and merged into ``partitionResyncPending`` only
// on a rising edge (rare), so the steady-state per-frame path allocates nothing.
const RISING_PARTS_SCRATCH = new Set<string>();
/** Scratch for {@link LODGroupRegistry.rankPartitionPartsForLoad} (load time, not per frame). */
const LOAD_RANK_LOCAL_BOX: BoundingBox = { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
const LOAD_RANK_OPTIONS: WorldBoxOptions = {
  worldBoxScratch: { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } },
};

/**
 * Whether a resync naming `targets` covers the part keyed `partKey` of the
 * partition at `wrapperPath`: the wrapper itself, the part, or an ancestor of
 * the part (a nested partition's outer part). Node paths are `/`-separated.
 */
function isPartTargeted(
  partKey: string,
  wrapperPath: string,
  targets: ReadonlySet<string>
): boolean {
  if (targets.has(wrapperPath) || targets.has(partKey)) return true;
  for (const target of targets) {
    if (partKey.startsWith(target.endsWith('/') ? target : `${target}/`)) return true;
  }
  return false;
}

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

/** `'/'` — the path segment separator {@link LODGroupRegistry.invalidatePartitionFootprint} splits on. */
const SLASH = 0x2f;

/**
 * One registered part, as the footprint-invalidation path index stores it: the
 * partition it belongs to and its index in that partition's children. A part's
 * path may be shared (duplicates) or empty (a pathless part, keyed by `''`, which
 * is a segment prefix of every absolute path — the same match the linear rule
 * `nodePath.startsWith(childPath + '/')` gave it).
 */
interface PartitionPartRef {
  entryPath: string;
  index: number;
}

/**
 * A deferred part's activation state (B4). `requested` is set when the
 * per-frame gate asked the loader for a pass to activate it (a pending
 * resync), `running` while that pass's activation is in flight; both clear
 * when the pass declines it, so a part still wanted is asked for again.
 */
interface LazyPartState {
  requested: boolean;
  /** The in-flight activation, if one is running. */
  running: Promise<void> | null;
  /**
   * The loader passes awaiting the running activation, each of which sweeps
   * the part's loaders itself once it settles (see
   * {@link LODGroupRegistry.activatePartitionParts}); `true` is a claim that
   * cannot be withdrawn. A claim only counts while its pass is alive: a
   * superseded pass commits nothing, so an activation settling with no live
   * claim resyncs the part itself. Empty ⇒ started ahead of any pass
   * (`prefetchSlice`).
   */
  claims: Array<AbortSignal | true>;
  /**
   * The last activation rejected (its leaves recorded a retryable failure):
   * the per-frame gate does not ask again until a Retry re-arms it
   * ({@link LODGroupRegistry.retryLazyChildByNodePath}).
   */
  failed: boolean;
}

/** Per-part per-frame state of a registered partition (parallel to its children). */
interface PartitionChildCache {
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
  /** Last frustum test (`true` until the first evaluation). */
  inFrustum: boolean;
  /** Deferred part state (B4): see {@link PartitionGroupChild.activate}. */
  lazy: LazyPartState | null;
}

/** Whether a part can draw anything for `view` (always, when there is no view). */
function partitionChildInSlice(
  child: PartitionGroupChild,
  view: PartitionSliceView | undefined
): boolean {
  if (!view || child.sliceExempt === true) return true;
  return partBoundsIntersectSlice(child.positionBounds, view, child.extendDims);
}

/**
 * Stamp one part's objects. `partitionFrustumVisible` (what draws, and what
 * refinement / capture read) is `inFrustum && inSlice`; `partitionInFrustum`
 * is the frustum test alone, which a view pass reads with its OWN slice.
 * Returns the ``PARTITION_*`` flags; a rising edge is a FRUSTUM re-entry only —
 * a slice entry is committed by the pass that moved the slice, which swept it.
 */
function updatePartitionObjectVisibility(
  objects: readonly THREE.Object3D[],
  inFrustum: boolean,
  inSlice: boolean
): number {
  const visible = inFrustum && inSlice;
  // Any previously culled object makes the whole part a rising edge.
  let wasInFrustum = true;
  let changed = false;
  for (const object of objects) {
    if (isPartitionFrustumCulled(object)) wasInFrustum = false;
    object.userData.partitionInFrustum = inFrustum;
    object.userData.partitionFrustumVisible = visible;
    if (object.visible !== visible) {
      object.visible = visible;
      changed = true;
    }
  }
  return (
    (changed ? PARTITION_VISIBILITY_CHANGED : 0) |
    (inFrustum && !wasInFrustum ? PARTITION_BECAME_VISIBLE : 0)
  );
}

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
   * Whether the LOD cross-fade is enabled (ON by default; `?noLodFade`
   * disables). When true and a blendable (additive/luminous/volumetric — see
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
   * load time fits (see `PLAYBACK_LOAD_BUDGET_FRACTION`). Omitted ⇒ never
   * playing — identical to the pre-feature behaviour.
   */
  getPlaybackPeriodMs?: () => number | null;
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
  private partitionEntries: Map<string, PartitionGroupEntry> = new Map();
  private partitionCaches: Map<string, { children: PartitionChildCache[] }> = new Map();
  /**
   * Part path → every registered part carrying it (B9c): the index that makes
   * {@link invalidatePartitionFootprint} O(path depth) instead of a scan over
   * every part of every partition. Maintained by ``registerPartition`` /
   * ``unregister`` / ``clear``; ``partitionPartKeys`` remembers which keys each
   * partition added so it can take exactly those back out.
   */
  private readonly partitionPartsByPath = new Map<string, PartitionPartRef[]>();
  private readonly partitionPartKeys = new Map<string, string[]>();
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
   * Set when a per-frame evaluation changed what the next frame DRAWS — any
   * level shown OR hidden, a fade opacity written with a new value. Broader
   * than {@link evaluatePerFrame}'s return (which reports only a newly shown
   * level, for the monitor tally): hiding a level, a time-driven stale-hold
   * expiry, or a cross-fade weight step all change pixels too. Read and
   * cleared by {@link takeDrawnStateChanged} — the render-on-change loop's
   * signal to redraw.
   */
  private drawnStateChanged = false;
  /** In-flight level dissolves, by group path (see {@link levelFade}). */
  private readonly fades = new Map<string, LevelFade>();
  /** Outgoing levels whose dissolve was dropped before its visibility pass. */
  private readonly droppedFadeFrom = new Map<string, number>();
  /** Band preloads in progress, by group path (see {@link preloadNeighbour}). */
  private readonly preloads = new Map<string, PreloadVisit>();
  /** This frame's clock reading and playback period (see ``evaluatePerFrame``). */
  private readonly frame: { nowMs: number; playbackPeriodMs: number | null } = {
    nowMs: 0,
    playbackPeriodMs: null,
  };
  /**
   * Partition rising edges waiting for their wrapper to be visible and the
   * loader to be idle: wrapper path → the re-entering PART paths. A set that
   * contains the wrapper path itself means "resync the whole partition" (a
   * pathless part). Coalesced across frames; flushed as ONE
   * ``requestReprocess(paths)`` call.
   */
  private readonly partitionResyncPending = new Map<string, Set<string>>();

  /** Snapshot source when no shared ``getViewContext`` is injected. */
  private readonly ownViews: ViewContextProvider | null;

  constructor(private deps: LODGroupRegistryDeps) {
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
    this.forgetPartitionParts(entry.path);
    this.partitionEntries.set(entry.path, entry);
    this.indexPartitionParts(entry);
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
        inFrustum: true,
        lazy: child.activate
          ? { requested: false, running: null, claims: [], failed: false }
          : null,
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
    if (this.partitionCaches.size === 0) return;
    // A partition or part is affected exactly when its path is ``nodePath`` or a
    // segment prefix of it, so enumerate those prefixes (O(depth)) and look them
    // up, instead of scanning every registered part — a commit per part on a
    // 2000-part partition made the scan O(parts²) per pass (B9c).
    const touched: string[] = [];
    const hits: PartitionPartRef[] = [];
    for (let end = nodePath.length; end >= 0; end--) {
      if (end !== nodePath.length && nodePath.charCodeAt(end) !== SLASH) continue;
      const prefix = nodePath.slice(0, end);
      if (this.partitionEntries.has(prefix)) touched.push(prefix);
      const parts = this.partitionPartsByPath.get(prefix);
      if (parts) hits.push(...parts);
    }
    for (const entryPath of touched) this.dirtyPartitionParts(entryPath, hits);
  }

  /**
   * Dirty the ``hits`` belonging to partition ``entryPath`` — or, when none
   * does (a commit under the partition matching no registered part), every part
   * of it, conservatively, rather than keep an under-covering stale box.
   */
  private dirtyPartitionParts(entryPath: string, hits: readonly PartitionPartRef[]): void {
    const cache = this.partitionCaches.get(entryPath);
    if (!cache) return;
    let matched = false;
    for (const hit of hits) {
      if (hit.entryPath !== entryPath) continue;
      cache.children[hit.index].footprintDirty = true;
      matched = true;
    }
    if (matched) return;
    for (const childCache of cache.children) childCache.footprintDirty = true;
  }

  /** Index a registered partition's parts by path for {@link invalidatePartitionFootprint}. */
  private indexPartitionParts(entry: PartitionGroupEntry): void {
    const keys = entry.children.map((child) => child.path);
    keys.forEach((key, index) => {
      let refs = this.partitionPartsByPath.get(key);
      if (!refs) {
        refs = [];
        this.partitionPartsByPath.set(key, refs);
      }
      refs.push({ entryPath: entry.path, index });
    });
    this.partitionPartKeys.set(entry.path, keys);
  }

  /** Drop a partition's parts from the path index. */
  private forgetPartitionParts(entryPath: string): void {
    const keys = this.partitionPartKeys.get(entryPath);
    if (!keys) return;
    this.partitionPartKeys.delete(entryPath);
    for (const key of new Set(keys)) {
      const refs = this.partitionPartsByPath.get(key);
      if (!refs) continue;
      const kept = refs.filter((ref) => ref.entryPath !== entryPath);
      if (kept.length > 0) this.partitionPartsByPath.set(key, kept);
      else this.partitionPartsByPath.delete(key);
    }
  }

  /**
   * Re-stamp every partition part against the committed view NOW (B4). The
   * owning loader calls this right after a pass commits, so a part that left
   * the slice is hidden, and one that entered it is shown and refinable, in
   * the same frame as the commit — not one evaluation later (refinement,
   * scheduled at the pass tail, reads these stamps before the next frame).
   * The frustum half is the last evaluation's; nothing here is a rising edge.
   */
  applyCommittedSlice(): void {
    const committed = this.deps.getCommittedViewState?.();
    for (const entry of this.partitionEntries.values()) {
      const cache = this.partitionCaches.get(entry.path);
      if (!cache) continue;
      for (let index = 0; index < entry.children.length; index++) {
        const child = entry.children[index];
        const inSlice = partitionChildInSlice(child, committed);
        const flags = updatePartitionObjectVisibility(
          child.objects,
          cache.children[index].inFrustum,
          inSlice
        );
        if ((flags & PARTITION_VISIBILITY_CHANGED) !== 0) this.drawnStateChanged = true;
      }
    }
  }

  /**
   * Whether the loader at `path` can draw anything for `view` — `false` when
   * any partition part enclosing it is provably empty on a discrete hidden
   * dimension (see `partition-slice-gate.ts`). A view pass skips such loaders:
   * the part is hidden from the commit on ({@link applyCommittedSlice}), so its
   * previous geometry is never drawn for `view`. O(path depth).
   */
  isPathInPartitionSlice(path: string, view: PartitionSliceView): boolean {
    if (this.partitionPartsByPath.size === 0) return true;
    for (let end = path.length; end >= 0; end--) {
      if (end !== path.length && path.charCodeAt(end) !== SLASH) continue;
      const refs = this.partitionPartsByPath.get(path.slice(0, end));
      if (refs && !refs.every((ref) => this.partRefInSlice(ref, view))) return false;
    }
    return true;
  }

  private partRefInSlice(ref: PartitionPartRef, view: PartitionSliceView): boolean {
    const child = this.partitionEntries.get(ref.entryPath)?.children[ref.index];
    return child === undefined || partitionChildInSlice(child, view);
  }

  /**
   * Activate the deferred parts a loader pass for `view` needs (B4): every
   * deferred part in the padded frustum (last evaluation), under an effectively
   * visible wrapper and in `view`'s slice — restricted to `targets` (a targeted
   * resync) when given. An activation attaches and REGISTERS the part's
   * loaders without loading their data: resolves, once every activation it
   * started (or joined) has settled, with the keys of those parts, whose new
   * loaders the pass then sweeps itself and commits with everything else.
   * `claim` is the pass's abort signal (`true`: a claim that is never
   * withdrawn); `false` (`prefetchSlice`, activating ahead of the next slice)
   * claims nothing: that slice's pass finds the loaders registered. A deferred
   * part under `targets` that does not qualify has its request cleared, so the
   * per-frame gate asks again while it is still wanted.
   */
  activatePartitionParts(
    view: PartitionSliceView,
    targets?: ReadonlySet<string>,
    claim: AbortSignal | boolean = true
  ): Promise<string[]> {
    const runs: Promise<void>[] = [];
    const keys: string[] = [];
    for (const entry of this.partitionEntries.values()) {
      const cache = this.partitionCaches.get(entry.path);
      if (!cache) continue;
      for (let index = 0; index < entry.children.length; index++) {
        const run = this.activatePart(entry, index, cache.children[index], view, targets);
        if (!run) continue;
        runs.push(run.promise);
        keys.push(run.key);
        if (claim !== false) run.lazy.claims.push(claim);
      }
    }
    return runs.length === 0 ? Promise.resolve(keys) : Promise.all(runs).then(() => keys);
  }

  private activatePart(
    entry: PartitionGroupEntry,
    index: number,
    childCache: PartitionChildCache,
    view: PartitionSliceView,
    targets: ReadonlySet<string> | undefined
  ): { promise: Promise<void>; key: string; lazy: LazyPartState } | null {
    const child = entry.children[index];
    const lazy = childCache.lazy;
    if (!lazy || !child.activate || lazy.failed) return null;
    const partKey = child.path || entry.path;
    if (targets && !isPartTargeted(partKey, entry.path, targets)) return null;
    if (!lazy.running) lazy.requested = false;
    if (!this.partActivationWanted(entry, child, childCache, view)) return null;
    // Already loading (activated ahead of this view): the caller still waits for it.
    lazy.running ??= this.runActivation(entry, partKey, child, childCache, lazy);
    return { promise: lazy.running, key: partKey, lazy };
  }

  /** In the padded frustum (last evaluation), in `view`'s slice, under a visible wrapper. */
  private partActivationWanted(
    entry: PartitionGroupEntry,
    child: PartitionGroupChild,
    childCache: PartitionChildCache,
    view: PartitionSliceView
  ): boolean {
    return (
      childCache.inFrustum &&
      partitionChildInSlice(child, view) &&
      isEffectivelyVisible(entry.groupObject)
    );
  }

  private runActivation(
    entry: PartitionGroupEntry,
    partKey: string,
    child: PartitionGroupChild,
    childCache: PartitionChildCache,
    lazy: LazyPartState
  ): Promise<void> {
    const activate = child.activate as () => Promise<void>;
    return activate().then(
      () => {
        perfCounters.add(S_PARTS_ACTIVATED);
        this.settleActivation(entry, partKey, child, childCache, lazy);
      },
      (error: unknown) => {
        log.warning(
          Modules.SCENE_LOADER,
          `partition part ${child.path} failed to activate: ${String(error)}`
        );
        // Re-armable: its loaders recorded a retryable failure, and a Retry
        // clears `failed` (see `retryLazyChildByNodePath`).
        lazy.running = null;
        lazy.requested = false;
        lazy.claims = [];
        lazy.failed = true;
      }
    );
  }

  /**
   * A part's activation registered its loaders: never activate the slot again.
   * Nothing drawn changed — its placeholders are empty until a pass commits
   * them. With no LIVE claim (started ahead of any pass, or every pass that
   * awaited it was superseded) and the COMMITTED view showing it, nothing else
   * will sweep the new loaders for that view: resync it. A part of a
   * registration that has since been cleared or replaced (dataset switch) is
   * left alone.
   */
  private settleActivation(
    entry: PartitionGroupEntry,
    partKey: string,
    child: PartitionGroupChild,
    childCache: PartitionChildCache,
    lazy: LazyPartState
  ): void {
    child.activate = undefined;
    if (childCache.lazy === lazy) childCache.lazy = null;
    childCache.footprintDirty = true;
    if (this.partitionEntries.get(entry.path) !== entry) return;
    const claimLive = lazy.claims.some((claim) => claim === true || !claim.aborted);
    const committed = this.deps.getCommittedViewState?.();
    if (
      !claimLive &&
      childCache.inFrustum &&
      partitionChildInSlice(child, committed) &&
      isEffectivelyVisible(entry.groupObject)
    ) {
      this.notePartitionRisingEdge(entry.path, new Set([partKey]));
      this.keepTicking();
    }
  }

  /**
   * Load-time ranking of a partition's parts against the CURRENT camera (B4):
   * which intersect the padded partition frustum, and a nearest-first load
   * order. `null` when there is no usable view, or when no part is in the
   * frustum at all — a camera that sees none of a partition has not been framed
   * on it yet (the scene frames the camera after the load), so gating on it
   * would defer everything the opening view is about to show.
   */
  rankPartitionPartsForLoad(
    groupObject: THREE.Object3D,
    bounds: readonly { min: readonly number[]; max: readonly number[] }[]
  ): { inFrustum: boolean[]; order: number[] } | null {
    const displayDims = this.deps.getDisplayDims();
    const view = this.view();
    if (displayDims.length < 2 || view.viewportCss === null) return null;
    PARTITION_FRUSTUM_MATRIX_SCRATCH.copy(view.projView).premultiply(PARTITION_FRUSTUM_SCALE);
    PARTITION_FRUSTUM_SCRATCH.setFromProjectionMatrix(PARTITION_FRUSTUM_MATRIX_SCRATCH);
    groupObject.updateWorldMatrix(true, false);
    const inFrustum: boolean[] = [];
    const distance: number[] = [];
    for (const positionBounds of bounds) {
      const box = computeEntryWorldBox(
        { groupObject, children: [{ positionBounds }] },
        displayDims,
        LOAD_RANK_LOCAL_BOX,
        this.matrixScratch,
        LOAD_RANK_OPTIONS
      );
      if (!box) {
        inFrustum.push(true);
        distance.push(0);
        continue;
      }
      WORLD_BOX3_SCRATCH.min.set(box.min.x, box.min.y, box.min.z);
      WORLD_BOX3_SCRATCH.max.set(box.max.x, box.max.y, box.max.z);
      inFrustum.push(PARTITION_FRUSTUM_SCRATCH.intersectsBox(WORLD_BOX3_SCRATCH));
      distance.push(WORLD_BOX3_SCRATCH.distanceToPoint(view.cameraWorldPosition));
    }
    if (!inFrustum.some(Boolean)) return null;
    const order = bounds.map((_, i) => i).sort((a, b) => distance[a] - distance[b] || a - b);
    return { inFrustum, order };
  }

  /** Drop an lod_group from the registry (called on scene teardown). */
  unregister(path: string): void {
    const partition = this.partitionEntries.get(path);
    if (partition) this.restorePartitionChildren(partition);
    this.partitionResyncPending.delete(path);
    if (this.entries.get(path)?.children.some((child) => child.permanentlyFailed)) {
      bumpFailedLoadsVersion();
    }
    this.entries.delete(path);
    this.caches.delete(path);
    this.partitionEntries.delete(path);
    this.partitionCaches.delete(path);
    this.forgetPartitionParts(path);
    this.warnedNoReadyChild.delete(path);
    this.warnedEmptyLevel.delete(path);
    this.fades.delete(path);
    this.droppedFadeFrom.delete(path);
    this.preloads.delete(path);
  }

  /** Clear all entries (called on full scene tear-down). */
  clear(): void {
    for (const partition of this.partitionEntries.values()) {
      this.restorePartitionChildren(partition);
    }
    if (
      [...this.entries.values()].some((entry) =>
        entry.children.some((child) => child.permanentlyFailed)
      )
    ) {
      bumpFailedLoadsVersion();
    }
    this.entries.clear();
    this.caches.clear();
    this.fades.clear();
    this.droppedFadeFrom.clear();
    this.preloads.clear();
    this.partitionEntries.clear();
    this.partitionCaches.clear();
    this.partitionPartsByPath.clear();
    this.partitionPartKeys.clear();
    // Reset the monotonic tick so a reused registry (shared-registry
    // refactor) starts cold rather than inheriting stale LRU ordering — and
    // the settle clock with it, so the new scene's first observed version
    // starts its own debounce instead of inheriting the old scene's.
    this.tick = 0;
    this.settleTracker.reset();
    this.warnedNoReadyChild.clear();
    this.warnedEmptyLevel.clear();
    this.fadeWasManaged = false;
    this.partitionResyncPending.clear();
  }

  private restorePartitionChildren(entry: PartitionGroupEntry): void {
    this.drawnStateChanged = true;
    for (const child of entry.children) {
      for (const object of child.objects) {
        object.visible = true;
        delete object.userData.partitionFrustumVisible;
        delete object.userData.partitionInFrustum;
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
   * - No level dissolve may be in flight ({@link isAnimating}). The dissolve
   *   is driven by WALL time, which an offline capture does not follow, so
   *   filming one mid-way would make each exported frame depend on how long
   *   the previous one took to render. Waiting turns it into a clean cut.
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
    if (this.isAnimating()) return false;
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

  /**
   * Whether visible partition parts have no pending resync or incomplete commit.
   * Unlike {@link hasVisiblePendingPartitionResync}, pending resyncs count here
   * only while their specific parts contribute pixels to the capture frame.
   */
  private partitionsCaptureQuiescent(version: number | null): boolean {
    if (!this.pendingPartitionResyncsQuiescent()) return false;
    for (const entry of this.partitionEntries.values()) {
      if (!this.partitionEntryCaptureQuiescent(entry, version)) return false;
    }
    return true;
  }

  private pendingPartitionResyncsQuiescent(): boolean {
    if (!this.deps.requestReprocess) return true;
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
   * owning loader to become idle. Unlike
   * {@link pendingPartitionResyncsQuiescent}, this deliberately mirrors
   * {@link flushPartitionResyncs}'s wrapper-level visibility gate: every queued
   * part under a visible wrapper is dispatched, even if that part re-exits
   * before the flush. Pending work retained under a hidden wrapper is not
   * actionable and must not keep wide settledness false indefinitely; neither
   * can work when no resync dispatcher is wired.
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
        return this.kickDeferredLoad(child, true);
      }
    }
    return this.rearmFailedPartitionPart(path);
  }

  /**
   * Re-arm a deferred partition part whose activation failed (B4): the
   * per-frame gate asks for a pass to activate it again while it is wanted.
   */
  private rearmFailedPartitionPart(path: string): boolean {
    for (const [entryPath, entry] of this.partitionEntries) {
      const cache = this.partitionCaches.get(entryPath);
      const index = entry.children.findIndex((child) => child.path === path);
      const lazy = index < 0 ? null : cache?.children[index]?.lazy;
      if (!lazy?.failed) continue;
      lazy.failed = false;
      this.keepTicking();
      return true;
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
   * Reports, separately, whether at least one lod_group swapped its active
   * child (``levelChanged``) and whether a partition part's frustum-cull
   * visibility flipped (``cullChanged``) this frame — see
   * {@link LODFrameChanges} for why the two must not be conflated. Either
   * one changes what the visible-element tally counts. A no-op frame (the
   * common case) returns {@link LOD_FRAME_UNCHANGED}, which keeps the
   * per-frame cost to the projection math alone.
   */
  evaluatePerFrame(): LODFrameChanges {
    if (this.entries.size === 0 && this.partitionEntries.size === 0) return LOD_FRAME_UNCHANGED;
    const displayDims = this.deps.getDisplayDims();
    if (displayDims.length < 2) return this.skipFrame();
    // The frame's view snapshot: the camera matrices as this frame renders
    // them, whichever callbacks ran before (fly controls do not refresh
    // ``matrixWorldInverse``; the snapshot derives the view itself). A collapsed
    // canvas has no viewport to select for, so the frame is skipped.
    const view = this.view();
    const viewport = view.viewportCss;
    if (viewport === null) return this.skipFrame();
    const camera = view.camera;

    this.tick++;
    // ``view.projView`` (projection×view, default WebGL coordinate system, the
    // NDC convention of the manual divide inside ``projectBoxDiagonalPx``) is
    // shared three ways: its frustum gates off-screen groups and ranks
    // eviction, and the per-group projections reuse the matrix.
    FRUSTUM_MATRIX_SCRATCH.copy(view.projView);
    FRUSTUM_SCRATCH.copy(view.frustum);
    PARTITION_FRUSTUM_MATRIX_SCRATCH.copy(FRUSTUM_MATRIX_SCRATCH).premultiply(
      PARTITION_FRUSTUM_SCALE
    );
    PARTITION_FRUSTUM_SCRATCH.setFromProjectionMatrix(PARTITION_FRUSTUM_MATRIX_SCRATCH);

    // Track whether the (global) view version has settled, to gate deferred
    // fine-level reloads (see ``evaluateEntry``). ``undefined`` view version
    // (no wiring / tests) ⇒ treat as settled so the trigger is inert.
    this.frame.nowMs = this.nowMs();
    this.frame.playbackPeriodMs = this.deps.getPlaybackPeriodMs?.() ?? null;
    const version = this.deps.getViewVersion?.();
    if (version != null) this.settleTracker.observe(version, this.frame.nowMs);
    const settled =
      version == null || this.settleTracker.isSettled(this.frame.nowMs, FINE_RELOAD_SETTLE_MS);

    let levelChanged = false;
    let cullChanged = false;
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
      if ((result & PARTITION_VISIBILITY_CHANGED) !== 0) {
        cullChanged = true;
        this.drawnStateChanged = true;
      }
      // Only parts with something stale to resync are recorded (a settled
      // re-entry just re-shows), so an empty set means no resync at all — never
      // an empty-path request, which the loader would read as a FULL re-sweep.
      if (RISING_PARTS_SCRATCH.size > 0) {
        this.notePartitionRisingEdge(entry.path, RISING_PARTS_SCRATCH);
      }
      if (this.partitionResyncPending.has(entry.path)) hasVisiblePendingResync = true;
    }
    RISING_PARTS_SCRATCH.clear();
    if (this.partitionResyncPending.size > 0 && this.deps.isUpdateInProgress?.() !== true) {
      this.flushPartitionResyncs();
    }
    if (hasVisiblePendingResync && this.partitionResyncPending.size > 0) {
      this.keepTicking();
    }
    for (const entry of this.entries.values()) {
      if (this.evaluateEntry(entry, view, viewport, displayDims, FRUSTUM_SCRATCH, settled)) {
        levelChanged = true;
      }
      // A lazy level loading (initial or a settled fine reload) commits
      // asynchronously OUTSIDE the per-slice sweep. Keep the on-demand render
      // loop alive so the per-frame swap-up to the fresh level fires when the
      // load lands, rather than waiting for the next user interaction. Plain
      // loop (not ``.some``) to preserve this file's no-per-frame-allocation
      // hot-path invariant. The same walk times the loads that just landed.
      if (this.observeLoads(entry)) anyLoading = true;
    }
    if (anyLoading) this.keepTicking();
    // A dissolve is a function of TIME, so the loop must keep drawing until it
    // lands even when nothing else moves (each step writes a new opacity,
    // which marks the frame dirty through ``takeDrawnStateChanged``).
    if (this.isAnimating()) this.keepTicking();
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
    return lodFrameChanges(levelChanged, cullChanged);
  }

  /**
   * A frame the registry cannot select for (fewer than two display dims, a
   * collapsed canvas). No group is evaluated, so no dissolve can land: drop
   * them all rather than leave {@link isAnimating} true until the view returns
   * (the next evaluated frame then draws each group's level alone).
   */
  private skipFrame(): LODFrameChanges {
    for (const [path, fade] of this.fades) this.droppedFadeFrom.set(path, fade.fromIdx);
    this.fades.clear();
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
    return this.fades.size > 0;
  }

  /**
   * Whether any child of ``entry`` is loading, and — for each child whose
   * ``ensureLoaded`` has finished since it was fired — fold the measured
   * load+commit time into its ``loadEwmaMs`` (a failure is not a timing).
   */
  private observeLoads(entry: LODGroupEntry): boolean {
    let anyLoading = false;
    for (const c of entry.children) {
      if (c.loading) anyLoading = true;
      else this.foldLoadTime(c);
    }
    return anyLoading;
  }

  /**
   * Fold a finished (``loading`` cleared) timed load of ``c`` into
   * ``loadEwmaMs``: start to its stamped end, on the registry clock that
   * stamped the start. Without an end stamp (a thunk that never calls
   * ``onLoadSettled``) the load is timed to now — the first moment it was
   * observed finished.
   */
  private foldLoadTime(c: LODGroupChild): void {
    if (c.loadStartMs === undefined || c.loading === true) return;
    if (c.failed !== true) {
      const ms = Math.max(0, (c.loadEndMs ?? this.nowMs()) - c.loadStartMs);
      c.loadEwmaMs =
        c.loadEwmaMs === undefined ? ms : c.loadEwmaMs + LOAD_EWMA_ALPHA * (ms - c.loadEwmaMs);
    }
    c.loadStartMs = undefined;
    c.loadEndMs = undefined;
  }

  /**
   * During playback, the finest level at or below ``desired`` that can reload
   * within ``PLAYBACK_LOAD_BUDGET_FRACTION`` of the period: an eager level
   * (no ``ensureLoaded`` — the per-slice sweep carries it), an unmeasured one,
   * or one whose ``loadEwmaMs`` fits. Once per second, the next finer capped
   * level is probed for a warm-load sample. ``desired`` is unchanged when not
   * playing, locked, or forced finest.
   */
  private playbackAspiration(entry: LODGroupEntry, desired: number): number {
    const period = this.frame.playbackPeriodMs;
    if (period === null || entry.selectorMode !== 'auto') return desired;
    if (this.deps.getForceFinestLOD?.() === true) return desired;
    const budget = PLAYBACK_LOAD_BUDGET_FRACTION * period;
    const affordable = this.affordablePlaybackLevel(entry, desired, budget);
    return this.probedPlaybackLevel(entry, desired, affordable);
  }

  /** Finest level whose measured reload fits this playback period. */
  private affordablePlaybackLevel(entry: LODGroupEntry, desired: number, budget: number): number {
    for (let i = desired; i > 0; i--) {
      const c = entry.children[i];
      if (!c.ensureLoaded || c.loadEwmaMs === undefined || c.loadEwmaMs <= budget) return i;
    }
    return 0;
  }

  /** Periodically re-measure the next finer capped level after a cold load. */
  private probedPlaybackLevel(entry: LODGroupEntry, desired: number, affordable: number): number {
    const next = entry.children[affordable + 1];
    if (affordable < desired && next?.loadEwmaMs !== undefined) {
      const now = this.frame.nowMs;
      if (
        now - (next.lastPlaybackProbeMs ?? Number.NEGATIVE_INFINITY) >=
        PLAYBACK_PROBE_INTERVAL_MS
      ) {
        next.lastPlaybackProbeMs = now;
        return affordable + 1;
      }
    }
    return affordable;
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
   * Whether a part that just re-entered the frustum has nothing to resync
   * (B9c): every tracked leaf under it already holds a commit for the CURRENT
   * view version with a complete ladder, so the view did not move while it was
   * culled and a targeted re-sweep would only re-derive what is on screen.
   * Unknown (no view-version wiring, or a leaf never committed) is NOT settled,
   * so such a part resyncs exactly as before.
   */
  private partitionPartSettled(child: PartitionGroupChild): boolean {
    const version = this.deps.getViewVersion?.();
    if (version == null) return false;
    return child.objects.every((object) =>
      subtreeSweepSettled(object as unknown as SweepNode, version)
    );
  }

  /**
   * Frustum- and slice-gate one partition's parts. Returns the ``PARTITION_*``
   * bit flags; parts that re-entered the frustum this frame are added to
   * ``risingParts`` by node path (``child.path``), or as the WRAPPER path when a
   * part has none so the caller resyncs the whole partition rather than missing
   * it — unless the part is already settled for the current view
   * ({@link partitionPartSettled}). A deferred part that is in the frustum and
   * the committed slice is added too: the resync pass is what activates it.
   */
  private evaluatePartitionEntry(
    entry: PartitionGroupEntry,
    displayDims: readonly number[],
    frustum: THREE.Frustum,
    risingParts: Set<string>
  ): number {
    const cache = this.partitionCaches.get(entry.path);
    if (!cache) return 0;
    const committed = this.deps.getCommittedViewState?.();
    let result = 0;
    for (let index = 0; index < entry.children.length; index++) {
      const child = entry.children[index];
      const childCache = cache.children[index];
      childCache.inFrustum = this.partitionPartInFrustum(child, childCache, displayDims, frustum);
      const inSlice = partitionChildInSlice(child, committed);
      const flags = updatePartitionObjectVisibility(child.objects, childCache.inFrustum, inSlice);
      result |= flags;
      if (childCache.lazy) {
        const wanted = childCache.inFrustum && inSlice;
        this.requestLazyActivation(risingParts, entry.path, child, childCache.lazy, wanted);
      } else if ((flags & PARTITION_BECAME_VISIBLE) !== 0 && !this.partitionPartSettled(child)) {
        noteRisingPart(risingParts, child.path, entry.path);
      }
    }
    return result;
  }

  /** Frustum test of one part: its stored bounds united with its rendered footprint. */
  private partitionPartInFrustum(
    child: PartitionGroupChild,
    childCache: PartitionChildCache,
    displayDims: readonly number[],
    frustum: THREE.Frustum
  ): boolean {
    const worldBox = computeEntryWorldBox(
      childCache.source,
      displayDims,
      childCache.localBoxScratch,
      this.matrixScratch,
      childCache.worldBoxOptions
    );
    if (!worldBox) return true;
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
    return frustum.intersectsBox(WORLD_BOX3_SCRATCH);
  }

  /**
   * Ask for a pass that activates a deferred part — once, while it is `wanted`
   * (in the frustum and the committed slice) and no activation is requested or
   * running. The request rides the rising-edge resync: the targeted pass it
   * triggers calls {@link activatePartitionParts} for the part.
   */
  private requestLazyActivation(
    risingParts: Set<string>,
    entryPath: string,
    child: PartitionGroupChild,
    lazy: LazyPartState,
    wanted: boolean
  ): void {
    if (!wanted || lazy.requested || lazy.running || lazy.failed) return;
    lazy.requested = true;
    noteRisingPart(risingParts, child.path, entryPath);
  }

  /** Returns ``true`` if this entry's displayed child changed. */
  private evaluateEntry(
    entry: LODGroupEntry,
    view: ViewContext,
    viewport: { width: number; height: number },
    displayDims: readonly number[],
    frustum: THREE.Frustum,
    settled: boolean
  ): boolean {
    // Pick the desired child index.
    let desired: number;
    // The dimensionless coverage metric for this frame (projected diagonal ÷
    // FILL_FACTOR·fittedAxisPx, or the screen-area fraction). Only computed
    // on the auto path.
    let coverageMetric: number;
    let usedFootprintSelection = false;
    // The metric the threshold pick used, for the band preload; stays null on
    // every other path (lock, off-screen, force-finest, footprint pick).
    let preloadMetric: number | null = null;
    if (entry.selectorMode !== 'auto') {
      // Explicit lock bypasses the off-screen gate: a user who pins a level
      // keeps it whether or not the group is on screen.
      desired = entry.selectorMode.lockLevel;
      entry.offScreen = false;
    } else {
      const cache = this.caches.get(entry.path);
      if (!cache) return false; // shouldn't happen — register() populates this.

      const worldBox = this.computeWorldBox(entry, displayDims);
      if (!worldBox) return this.dropFade(entry);

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
          // The screen metrics project the group's LOCAL box through
          // projView × matrixWorld, i.e. the 8 corners of the box as oriented
          // on screen. Projecting the corners of its world AABB instead (the
          // frustum gate's box) inflated a rotated group twice and picked too
          // fine a level. computeWorldBox refreshed matrixWorld above.
          const rawLocal = cache.localBoxScratch;
          const metricLocal =
            cache.hasLodBounds && this.computeWorldBox(entry, displayDims, true)
              ? cache.metricLocalBoxScratch
              : rawLocal;
          LOCAL_PROJ_SCRATCH.multiplyMatrices(
            FRUSTUM_MATRIX_SCRATCH,
            entry.groupObject.matrixWorld
          );
          if (entry.selector === 'screen-area') {
            // Screen-area selector: the metric IS the fraction of the viewport
            // area the group's projected inscribed ellipsoid (sized at its
            // nearest corner) covers, in rect units (orientation-stable, viewport-size
            // independent by construction — see projectBoxAreaFraction). The
            // thresholds are literal area fractions ([0, …, 1/4, 1/2] whole-object;
            // a partition tile anchors at 1.0), so no FILL_FACTOR normalisation.
            // Camera inside the box → +Infinity → finest, same as the diagonal path.
            coverageMetric = projectBoxAreaFraction(metricLocal, view.camera, LOCAL_PROJ_SCRATCH);
            if (cache.hasLodBounds) {
              // The thin-rectangle ramp is not monotone under box containment:
              // trimming the thin axis can increase the robust metric. Robust
              // bounds may only keep or reduce the raw-bounds selection.
              coverageMetric = Math.min(
                coverageMetric,
                projectBoxAreaFraction(rawLocal, view.camera, LOCAL_PROJ_SCRATCH)
              );
            }
          } else {
            // Legacy 'coverage' selector (the default for older stores).
            const diagonalPx = projectBoxDiagonalPx(
              metricLocal,
              view.camera,
              viewport,
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
            const fittedAxisPx = Math.min(viewport.width, viewport.height);
            coverageMetric = diagonalPx / (FILL_FACTOR * fittedAxisPx);
          }
        }
        const lodBias = resolveLodBias(this.deps.getLodBias?.());
        coverageMetric *= entry.selector === 'screen-area' ? lodBias : Math.sqrt(lodBias);
        const footprintDesired =
          forceFinest || entry.selector !== 'screen-area'
            ? null
            : pickStampedFootprintChild(entry, cache, worldBox, view, {
                viewportHeight: viewport.height,
                lodBias,
                displayDims,
              });
        if (footprintDesired == null) {
          desired = pickChildWithHysteresis(
            cache.thresholds,
            entry.activeChildIndex,
            coverageMetric
          );
          if (!forceFinest) preloadMetric = coverageMetric;
        } else {
          desired = footprintDesired;
          usedFootprintSelection = true;
        }
        entry.offScreen = false;
      }
    }

    // During playback, no finer than what can reload within the period.
    desired = this.playbackAspiration(entry, desired);

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
      if (!target) return this.dropFade(entry);
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
    // If the chosen display level committed 0 elements, prefer the coarsest
    // fresh NON-empty level no finer than the chosen display or the selector's
    // aspiration, whichever is finer. An intermediate level can legitimately
    // fill a stale fallback gap; only redirecting to a level coarser than the
    // chosen display signals inconsistent/corrupt data and warrants the warning
    // below. Finer levels must not override the selector, even when a coarse
    // slice is legitimately empty (see #1600).
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
        const fallbackLimit = Math.max(displayIdx, entry.activeChildIndex);
        const fallback = this.coarsestFreshNonEmptyIndex(entry, version, fallbackLimit);
        if (fallback >= 0 && fallback !== displayIdx) {
          if (fallback < displayIdx && !this.warnedEmptyLevel.has(entry.path)) {
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
    // time-driven level dissolve below, and stays a hard hold.
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

    // ── Level dissolve (time-driven) ──
    // When the DISPLAYED level changes, dissolve from the outgoing level to the
    // incoming one over ``config.lod.fadeMs`` — incoming at w = smoothstep of
    // the elapsed fraction, outgoing at (1−w) — so the substitutive switch
    // dissolves instead of popping. A function of TIME since the change, never
    // of the camera's distance to a threshold: a parked camera always settles
    // on ONE level at full weight (#2925 — a distance band drew two levels for
    // as long as the camera stayed inside it). Brightness is preserved by the
    // levels' build-time mass conservation (both integrate to the same DC).
    // Blendable modes only (BLENDABLE_MODES = additive/luminous/volumetric —
    // energy sums linearly, or opacity linearly scales optical depth τ so the
    // pair interpolates monotonically between the two levels' absorptions; see
    // that set's doc for what volumetric does NOT guarantee). For two mid-fade
    // volumetric siblings the mesh draw order may come from the render-order
    // containment rule (near-identical bounds); acceptable because combined
    // TRANSMITTANCE is order-independent (transmittances multiply), so
    // occlusion of content behind the pair is exact at every weight. The
    // emission ordering residual stays bounded by the local inter-level
    // radiance difference, not by the rendered hard-swap pop.
    // Off / non-blendable / off-screen / locked / a held-stale display ⇒ no
    // dissolve (byte-identical hard swap).
    let blendPartnerIdx: number | null = null;
    let primaryWeight = 1;
    const fadeEligible =
      this.deps.getCrossFadeEnabled?.() === true &&
      entry.selectorMode === 'auto' &&
      !entry.offScreen &&
      aspirationFresh &&
      displayIdx === entry.activeChildIndex &&
      !usedFootprintSelection;
    // Last frame's outgoing level, if a dissolve was in flight (see the hide
    // edge in the visibility pass below).
    const fadingFromIdx = this.fades.get(entry.path)?.fromIdx ?? -1;
    const droppedFromIdx = this.droppedFadeFrom.get(entry.path) ?? -1;
    const fade = fadeEligible ? this.levelFade(entry, displayIdx, version ?? null) : null;
    if (fade) {
      blendPartnerIdx = fade.fromIdx;
      // primaryWeight is the OPACITY of the primary (displayIdx = the incoming
      // level); the outgoing level gets the complement.
      primaryWeight = smoothstep(0, 1, fade.progress);
      // Keep both warm in the eviction LRU (both are on screen).
      entry.children[fade.fromIdx].lastVisibleTick = this.tick;
      aspiration!.lastVisibleTick = this.tick;
    } else {
      this.fades.delete(entry.path);
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
    // During playback a STALE aspiration reloads on every timepoint (the
    // version never settles there, and playbackAspiration kept it to a level
    // that fits the period); refinement of a fresh one still waits.
    const playingStale = this.frame.playbackPeriodMs !== null && !aspirationFresh;
    if ((settled || playingStale) && needsReloadOrRefine && aspiration!.ensureLoaded) {
      this.maybeKickReload(entry, aspiration!);
    }
    this.preloadNeighbour(entry, desired, preloadMetric, version ?? null, settled);

    // ── Apply visibility (single owner) ──
    // At most the display child plus its cross-fade partner is visible
    // (``displayIdx`` / ``blendPartnerIdx``, each only if READY — never
    // force-show a not-ready placeholder; outside a cross-fade band it's the
    // classic single visible level). ``changed`` flips when a SHOWN level
    // changes so ``evaluatePerFrame`` refreshes the monitor's visible
    // tally, which counts the displayed level, not the aspiration.
    let changed = false;
    // Opacity is managed only while at least one anti-popping feature is on: the
    // level dissolve (blends the outgoing and incoming level after a change)
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
        this.drawnStateChanged = true;
        if (shouldShow) {
          changed = true; // a new level became visible
        } else {
          // A dissolve's outgoing level leaving on its own (its partner has
          // been on screen since the dissolve started) changes the visible
          // tally, so the monitor must recount — like the swap it completes.
          if (i === fadingFromIdx || i === droppedFromIdx) changed = true;
        }
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
    this.droppedFadeFrom.delete(entry.path);
    // Mark the on-screen level most-recently-used and record it for the eviction
    // pass, which must never release the level currently displayed. Only update
    // when a ready level is actually shown — otherwise keep the last shown index
    // so eviction still protects whatever the user last saw.
    countLodDisplay(entry, displayIdx, blendPartnerIdx);
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
   * (``PRELOAD_EXIT_BAND_FRACTION``) and comes back, so VRAM pressure cannot
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
        PRELOAD_EXIT_BAND_FRACTION
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
    const stale = version !== null && !this.childFreshAndCount(child, version).fresh;
    if (stale || child.hasMoreLODs?.() === true) this.maybeKickReload(entry, child);
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
   * Advance (or start) the dissolve of ``entry`` toward ``displayIdx`` and
   * return it, or ``null`` when the group should draw ``displayIdx`` alone.
   *
   * A dissolve STARTS when the displayed level differs from the one displayed
   * last frame. It starts from what is on screen: when the previous level was
   * itself still dissolving in (a retarget mid-dissolve), the new one begins at
   * ``1 − progress``, which keeps the previous level at exactly the opacity it
   * had, so reversing a change only has to undo the part that happened. It
   * ENDS once ``progress`` reaches 1, or as soon as the outgoing level can no
   * longer be drawn against the incoming one (released, stale for the current
   * slice, or not blendable).
   */
  private levelFade(
    entry: LODGroupEntry,
    displayIdx: number,
    version: number | null
  ): LevelFade | null {
    const fadeMs = config.lod.fadeMs;
    const now = this.nowMs();
    let fade = this.fades.get(entry.path) ?? null;
    const prev = entry.displayedChildIndex;
    if (prev !== undefined && prev !== displayIdx && fadeMs > 0) {
      fade = retargetedFade(fade, prev, displayIdx, now);
      this.fades.set(entry.path, fade);
    }
    if (fade === null || fade.toIdx !== displayIdx) return null;
    fade.progress = Math.min(1, fade.startProgress + (now - fade.startMs) / fadeMs);
    // The last few percent end the dissolve: applyLodFade treats a weight within
    // FADE_EPSILON of 1 as the authored opacity, so the complement would be drawn
    // on top of a full-weight level.
    if (
      smoothstep(0, 1, fade.progress) >= 1 - FADE_EPSILON ||
      !this.canDissolve(entry, fade.fromIdx, displayIdx, version)
    ) {
      return null;
    }
    return fade;
  }

  /**
   * Forget ``entry``'s dissolve on an ``evaluateEntry`` path that returns before
   * ``levelFade`` (so {@link isAnimating} cannot stay true for a group that is no
   * longer dissolving). Returns ``false``: no level change this frame.
   */
  private dropFade(entry: LODGroupEntry): false {
    const fade = this.fades.get(entry.path);
    if (fade) this.droppedFadeFrom.set(entry.path, fade.fromIdx);
    this.fades.delete(entry.path);
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
    if (version != null && !this.childFreshAndCount(from, version).fresh) return false;
    return this.isBlendable(from) && this.isBlendable(to);
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
   * one frame for as long as the user keeps dragging. The one exception is
   * playback with the held level's own reload in flight: that hold lasts
   * until the reload lands, and its budget restarts on each such frame.
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
    // During playback the held level's own reload for the new timepoint is in
    // flight: keep it until that replacement lands rather than dropping to a
    // token of the new frame for the rest of the wait.
    // The budget counts from the last frame of that exemption, so pausing
    // mid-reload keeps the hold for a full STALE_HOLD_MS instead of dropping
    // to the coarse level on the first paused frame.
    const replacementInFlight = this.frame.playbackPeriodMs !== null && prev.loading === true;
    if (replacementInFlight) {
      entry.staleHoldSinceMs = now;
    } else if (now - entry.staleHoldSinceMs >= STALE_HOLD_MS) {
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
      clearChildFailure(child);
    }
    // A load that finished this frame, before the post-evaluation walk saw it.
    this.foldLoadTime(child);
    child.loading = true;
    const startMs = this.nowMs();
    child.loadStartMs = startMs;
    child.onLoadSettled = () => {
      // Only the load this kick started (a release + re-kick restarts it).
      if (child.loadStartMs === startMs) child.loadEndMs = this.nowMs();
    };
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
