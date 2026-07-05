/**
 * Per-frame LOD-group selector.
 *
 * Tracks every `lod_group` scene-graph node currently loaded. For each
 * one, every frame:
 *
 *   1. Fold each child's nD ``positionBounds`` directly into a cached
 *      per-entry **local-space** :type:`BoundingBox`, using the current
 *      ``displayDims`` to map nD axes onto X/Y/Z. (No intermediate
 *      per-child boxes — the union is computed in place.)
 *   2. Transform the local box into world space via
 *      :func:`transformBoundingBox` and the lod_group's ``matrixWorld``.
 *   3. Project the 8 corners through the camera to NDC and back to
 *      pixel coordinates; the diagonal of the screen-space AABB, divided
 *      by ``FILL_FACTOR × viewportDiagonal``, is the dimensionless
 *      **coverage metric** (fraction of a filled viewport).
 *   4. Pick the **finest** child whose ``coverage_fraction`` threshold is
 *      satisfied by that coverage metric, with 10% asymmetric hysteresis on
 *      the downgrade direction to suppress threshold-edge flicker.
 *   5. If the desired child differs from the current active one, swap
 *      visibility atomically — gated by the **never-downgrade display
 *      gate**: a fresh aspiration whose additive ladder is still streaming
 *      is not shown while the previously-displayed level looks strictly
 *      better (see ``shouldHoldPreviousDisplay`` in ``lod-freshness.ts``).
 *
 * The atomic-swap invariant on initial load is realized by
 * ``loadLodGroupNode`` (sequential awaits + ``visible=false`` after
 * attach + a single ``register()`` call at the end) — no per-child
 * ``ready`` gate is needed in the registry.
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

import { type BoundingBox, transformBoundingBox } from './scene-manager/clipping/bounds-math';
import { log, Modules } from '../utils/log';
import type { LODGroupSelectorMode } from '../types/lod-group';
import {
  coarsestFreshIndex,
  coarsestFreshNonEmptyIndex,
  isFresh,
  isReady,
  SettleTracker,
  visibleElementCount,
} from './lod-freshness';
import { shouldHoldPreviousDisplay } from './lod-display-gate';

/**
 * Frames the view-update version must hold steady before the registry reloads a
 * stale fine level (the settle debounce — see `maybeKickReload`). While the
 * user is actively scrubbing (version changes every frame) only the cheap
 * coarse level shows; the fine level reloads once they pause for ~this many
 * frames. ~8 frames ≈ 130 ms at 60 fps.
 */
const FINE_RELOAD_SETTLE_TICKS = 8;

/** Asymmetric hysteresis on the "downgrade to coarser" direction. */
const HYSTERESIS_RATIO = 0.1;

/**
 * Anchor for the viewport-relative ``coverage_fraction`` thresholds: the finest
 * child (coverage 1.0) activates when the group's projected bbox diagonal reaches
 * ``FILL_FACTOR × viewportDiagonal`` pixels — i.e. when the object roughly fills
 * the screen. Coarser children (smaller fractions) take over as it shrinks. 1.0 =
 * "finest at fills-screen"; lower shows finest a touch sooner, higher a touch
 * later. The selector normalises the projected diagonal by this to a dimensionless
 * coverage metric, so the same thresholds behave identically on any viewport size.
 */
const FILL_FACTOR = 1.0;

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
  /**
   * Viewport-relative LOD-switch threshold in [0, 1], strictly monotonic
   * increasing in coarsest→finest order (coarsest 0.0, finest 1.0). Multiplied
   * by ``FILL_FACTOR × viewportDiagonal`` at selection time to compare against the
   * group's projected bbox diagonal in pixels.
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
   * ``loading``; on failure sets ``failed=true`` and clears ``loading``.
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
   * up. A per-frame transient written by ``evaluateEntry`` and read by
   * ``enforceResidentByteBudget`` (same synchronous ``evaluatePerFrame`` pass)
   * so eviction never releases the on-screen level. ``undefined`` before the
   * first evaluation ⇒ treated as ``activeChildIndex``. Tracks what is ACTUALLY
   * on screen every frame — including the coarse level shown while the group is
   * off-screen — which is what eviction needs, but is therefore NOT the
   * never-downgrade gate's memory (that is ``heldDisplayChildIndex``).
   */
  displayedChildIndex?: number;
  /**
   * The last level displayed while the group was ON SCREEN — the
   * never-downgrade gate's "previously-displayed level" memory. Distinct from
   * ``displayedChildIndex`` because the off-screen gate transiently displays
   * (and would otherwise record) the coarsest ready level; folding that into
   * the gate memory would let a mere look-away-and-back clobber a held finer
   * level and re-pop it to chunk-1 on return. Written by ``evaluateEntry``
   * only on frames where the group is on screen. ``undefined`` before the
   * first on-screen evaluation ⇒ the gate has no prior level to hold.
   */
  heldDisplayChildIndex?: number;
  /**
   * Whether the auto-selector is currently holding this group at its
   * coarsest-ready level because its world bounds are outside the camera
   * frustum (the off-screen gate). ``false`` when on-screen or when a
   * level is explicitly locked. Surfaced in the layers-panel readout as an
   * "(off-screen)" hint so a coarse level on close inspection isn't
   * mistaken for a selection bug. Updated each ``evaluatePerFrame``.
   */
  offScreen?: boolean;
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
   * Per-child ``coverage_fraction`` thresholds (dimensionless, ascending,
   * coarsest 0.0 → finest 1.0), rebuilt once at registration. The selector
   * compares these against the projected bbox diagonal normalised by
   * ``FILL_FACTOR × viewportDiagonal`` (a dimensionless coverage metric), so the
   * list is viewport-independent and needs no per-frame rebuild.
   */
  thresholds: number[];
  localBoxScratch: BoundingBox;
}

/**
 * Module-scope scratch for ``projectBoxDiagonalPx``'s projection × view
 * product. Single-threaded — ``evaluatePerFrame`` is the only per-frame entry
 * point, so reusing one matrix across all entries within a frame is safe.
 */
const PROJ_VIEW_SCRATCH = new THREE.Matrix4();

/**
 * Module-scope scratch for the per-frame frustum gate and eviction ranking.
 * ``evaluatePerFrame`` is the single per-frame entry point (no re-entrancy),
 * so these are safe to share across all entries within one frame:
 *   - ``FRUSTUM_SCRATCH`` — rebuilt once per frame from the camera.
 *   - ``FRUSTUM_MATRIX_SCRATCH`` — projection × view product feeding it.
 *   - ``WORLD_BOX3_SCRATCH`` — a ``THREE.Box3`` view of a group's world bbox
 *     for ``frustum.intersectsBox`` (our ``BoundingBox`` is a plain object).
 *   - ``BOX_CENTER_SCRATCH`` / ``CAMERA_POS_SCRATCH`` — eviction distance math.
 */
const FRUSTUM_SCRATCH = new THREE.Frustum();
const FRUSTUM_MATRIX_SCRATCH = new THREE.Matrix4();
const WORLD_BOX3_SCRATCH = new THREE.Box3();
const BOX_CENTER_SCRATCH = new THREE.Vector3();
const CAMERA_POS_SCRATCH = new THREE.Vector3();

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
   * Keep the render loop alive (the viewer is on-demand and idles after ~2s).
   * Called each frame while a lazy level is loading so a deferred fine reload
   * — which commits OUTSIDE the per-slice sweep and can take longer than the
   * idle timeout — still triggers the per-frame swap-up to the fresh level when
   * it lands, instead of waiting for the next user interaction. Wired to
   * ``AnimationController.startAnimation`` (resets the idle timeout). Omitted ⇒
   * no-op (unit tests don't run a loop).
   */
  requestRender?: () => void;
}

/**
 * Saturation epsilon for ``projectBoxDiagonalPx``'s near-plane guard. When any
 * bbox corner's homogeneous ``w`` (clip-space, ≈ view-space depth in front of
 * the camera) falls to/below this, the perspective divide is already producing
 * exploding/flipped NDC, so the diagonal is meaningless. We trip *before* ``w``
 * crosses zero (hence 1e-6, not ``transformBoundingBox``'s singular-point
 * ``1e-12``) to eliminate the unstable near-plane regime, not just the literal
 * singularity. With identity matrices ``w == 1 ≫ 1e-6``, so this never fires in
 * the identity-camera unit tests.
 */
const W_EPSILON = 1e-6;

/**
 * Project a world-space :type:`BoundingBox` through the camera and
 * return the diagonal of the screen-space AABB in pixels.
 *
 * Treats the bbox's 8 corners independently (works for both
 * perspective and orthographic projection without a closed-form
 * radius). NDC → pixels assumes the viewport size matches the
 * renderer canvas.
 *
 * **Near-plane saturation.** Projects with an explicit homogeneous ``w`` (the
 * combined ``projectionMatrix * matrixWorldInverse``, not THREE's
 * ``Vector3.project`` which divides by ``w`` unguarded). If any corner has
 * ``w <= W_EPSILON`` — i.e. the camera is inside or straddling the box — the
 * group fills the screen, so we return ``+Infinity`` to saturate the selector
 * to its finest level (``pickChildWithHysteresis`` then picks the top index;
 * the value is never fed to finite arithmetic, so no NaN). This is the inverse
 * of the old behaviour, where a corner crossing behind the near plane
 * *collapsed* the diagonal and wrongly dropped to a coarse level on close
 * approach. Orthographic cameras keep ``w == 1`` and so never saturate.
 *
 * Exported for unit testing.
 */
export function projectBoxDiagonalPx(
  box: BoundingBox,
  camera: THREE.Camera,
  viewport: { width: number; height: number },
  precomputedProjView?: THREE.Matrix4
): number {
  // Combined projection × view. ``evaluatePerFrame`` already builds this product
  // once per frame (``FRUSTUM_MATRIX_SCRATCH``) and passes it in via
  // ``precomputedProjView`` so we don't recompute the 4×4 per group. Standalone
  // callers (unit tests) omit it and we fall back to a module-scope scratch (no
  // per-call allocation). Unlike THREE's ``Vector3.project`` this exposes ``w``
  // so we can guard the near plane. ``evaluatePerFrame`` is the single per-frame
  // entry point, so sharing the scratch is safe.
  const m =
    precomputedProjView ??
    PROJ_VIEW_SCRATCH.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const e = m.elements; // THREE.Matrix4 is column-major flat[16]
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < 8; i++) {
    const x = i & 1 ? box.max.x : box.min.x;
    const y = i & 2 ? box.max.y : box.min.y;
    const z = i & 4 ? box.max.z : box.min.z;
    // Same column-major indexing as ``transformBoundingBox`` (bounds-math.ts):
    // w = m[3]*x + m[7]*y + m[11]*z + m[15].
    const w = e[3] * x + e[7] * y + e[11] * z + e[15];
    if (w <= W_EPSILON) {
      // Camera inside / straddling the bbox near plane → group fills the
      // screen → saturate so the finest child is selected.
      return Number.POSITIVE_INFINITY;
    }
    const ndcX = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w;
    const ndcY = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w;
    if (ndcX < minX) minX = ndcX;
    if (ndcX > maxX) maxX = ndcX;
    if (ndcY < minY) minY = ndcY;
    if (ndcY > maxY) maxY = ndcY;
  }
  const widthPx = (maxX - minX) * 0.5 * viewport.width;
  const heightPx = (maxY - minY) * 0.5 * viewport.height;
  return Math.hypot(widthPx, heightPx);
}

/**
 * Pick the desired child index given a scalar view ``metric`` and the
 * current active index. Applies 10% asymmetric hysteresis on the
 * downgrade direction.
 *
 * ``metric`` is the dimensionless coverage metric (projected bbox diagonal ÷
 * ``FILL_FACTOR × viewportDiagonal``) and ``thresholds`` are the per-child
 * ``coverage_fraction`` values; both are in the same [0,1]-ish space. The
 * "natural" pick is the finest child whose ``coverageFraction`` is less than or
 * equal to ``metric``. Hysteresis only resists dropping back to a coarser level:
 * when downgrading from index ``currentIdx``, the metric must fall below the
 * current threshold by a margin that is ``hysteresisRatio`` (default 10%) of the
 * GAP to the adjacent coarser threshold — i.e. below
 * ``thresholds[currentIdx] - hysteresisRatio * (thresholds[currentIdx] -
 * thresholds[currentIdx - 1])``; otherwise we stay on the current level
 * even though the natural pick is coarser. (At the bottom level
 * ``thresholds[currentIdx - 1]`` is effectively 0, reducing the margin to
 * ``hysteresisRatio * thresholds[currentIdx]`` — the old isolated-threshold
 * form.) Upgrades to a finer level are immediate (no hysteresis).
 *
 * Exported for unit testing.
 */
export function pickChildWithHysteresis(
  thresholds: readonly number[],
  currentIdx: number,
  metric: number,
  hysteresisRatio: number = HYSTERESIS_RATIO
): number {
  if (thresholds.length === 0) return -1;

  // Natural pick: finest child with threshold ≤ metric. Thresholds
  // are monotonic increasing in coarsest→finest order, so scan upward
  // until the threshold exceeds the metric.
  let natural = 0;
  for (let i = 0; i < thresholds.length; i++) {
    if (thresholds[i] <= metric) natural = i;
    else break;
  }

  if (natural === currentIdx) return currentIdx;
  if (natural > currentIdx) return natural; // upgrade: literal threshold wins

  // Downgrade: require the metric to drop below the current threshold by a
  // hysteresis margin that is **spacing-aware** — a fraction
  // (``hysteresisRatio``) of the GAP to the adjacent coarser threshold,
  // rather than of the current threshold in isolation. For the bottom real
  // level (coarser threshold 0) the gap equals the threshold, so this
  // reduces to the original ``currentThreshold * (1 - ratio)`` behaviour.
  // For tightly-spaced levels (e.g. separated only by the
  // ``coverage_fractions`` ×1.1 monotonicity nudge) the band shrinks
  // proportionally, so the deadband never straddles the neighbour — every level still
  // renders on the way down and the selection can't flip-flop across a band
  // wider than the inter-level spacing.
  //
  // The margin guards only the immediate ``currentIdx → currentIdx - 1``
  // boundary, but ``natural`` may be several levels coarser. That is correct: a
  // multi-level drop means the metric fell well past the adjacent band, so the
  // hysteresis (sized to one inter-level gap) cannot suppress it and we snap
  // straight to ``natural`` — no flip-flop, because the metric is nowhere near
  // the band it would need to re-cross to come back up.
  const currentThreshold = thresholds[currentIdx];
  const prevThreshold = thresholds[currentIdx - 1]; // currentIdx >= 1 here
  const margin = hysteresisRatio * (currentThreshold - prevThreshold);
  if (metric < currentThreshold - margin) {
    return natural;
  }
  return currentIdx;
}

/**
 * Tracks all loaded ``lod_group`` nodes in a scene; evaluates per-frame
 * to pick which child renders.
 */
export class LODGroupRegistry {
  private entries: Map<string, LODGroupEntry> = new Map();
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

  constructor(private deps: LODGroupRegistryDeps) {}

  /** Register a newly-loaded lod_group (called by the scene loader). */
  register(entry: LODGroupEntry): void {
    this.entries.set(entry.path, entry);
    this.caches.set(entry.path, {
      thresholds: entry.children.map((c) => c.coverageFraction),
      localBoxScratch: {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0, y: 0, z: 0 },
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

  /** Drop an lod_group from the registry (called on scene teardown). */
  unregister(path: string): void {
    this.entries.delete(path);
    this.caches.delete(path);
    this.warnedNoReadyChild.delete(path);
    this.warnedEmptyLevel.delete(path);
  }

  /** Clear all entries (called on full scene tear-down). */
  clear(): void {
    this.entries.clear();
    this.caches.clear();
    // Reset the monotonic tick so a reused registry (shared-registry
    // refactor) starts cold rather than inheriting stale LRU ordering.
    this.tick = 0;
    this.warnedNoReadyChild.clear();
    this.warnedEmptyLevel.clear();
  }

  /** Number of registered lod_groups (mainly for tests / diagnostics). */
  size(): number {
    return this.entries.size;
  }

  /** Lookup an entry by path (mainly for tests / UI). */
  get(path: string): LODGroupEntry | undefined {
    return this.entries.get(path);
  }

  /** All registered entries (mainly for the layers panel UI). */
  list(): LODGroupEntry[] {
    return Array.from(this.entries.values());
  }

  /**
   * Retry a LAZY lod_group level by its LEAF path (the path of the level's
   * placeholder mesh — leaf lazy children are named with their node path by
   * the node factory; anonymous deferred-GROUP placeholders carry no name and
   * correctly never match). Used by ``SceneLoader.retryFailedLoader``: lazy
   * levels never join the update-sweep loader maps, so the map-based retry
   * cannot reach them — this is their retry entry point.
   *
   * Clears the failure cooldown (``failed``/``failedTick``) and routes
   * through the shared ``kickDeferredLoad`` gate, which owns setting
   * ``loading`` before firing ``ensureLoaded`` (the thunk itself never sets
   * ``loading`` — only the registry does; keep that invariant here).
   *
   * Returns ``true`` when a retry was kicked OR one is already in flight
   * (``loading``), ``false`` when no lazy child with that leaf path exists.
   * Fire-and-forget semantics: ``true`` means "retry started", not "retry
   * succeeded" — the thunk owns the ready/failed outcome, and a repeat
   * failure re-enters the normal cooldown cycle.
   */
  retryLazyChildByLeafPath(path: string): boolean {
    if (!path) return false; // anonymous (deferred-group) placeholders have name '' — never match
    for (const entry of this.entries.values()) {
      for (const child of entry.children) {
        if (child.object.name !== path || !child.ensureLoaded) continue;
        if (child.loading) return true; // retry already in flight
        child.failed = false;
        child.failedTick = undefined;
        this.kickDeferredLoad(child);
        return true;
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
    if (this.entries.size === 0) return false;
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

    // Track whether the (global) view version has settled, to gate deferred
    // fine-level reloads (see ``evaluateEntry``). ``undefined`` view version
    // (no wiring / tests) ⇒ treat as settled so the trigger is inert.
    const version = this.deps.getViewVersion?.();
    if (version != null) this.settleTracker.observe(version, this.tick);
    const settled =
      version == null || this.settleTracker.isSettled(this.tick, FINE_RELOAD_SETTLE_TICKS);

    let changed = false;
    let anyLoading = false;
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
    // Bound resident LOD geometry against the shared GPU-pool byte budget
    // (one VRAM authority). Retention keeps loaded levels resident so
    // re-shows are free; this LRU-evicts only hidden levels when over budget —
    // off-screen / furthest-from-camera first — so no per-swap release, hence
    // no reload churn.
    this.enforceResidentByteBudget(camera, FRUSTUM_SCRATCH, displayDims);
    return changed;
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
      if (!frustum.intersectsBox(WORLD_BOX3_SCRATCH)) {
        desired = this.coarsestReadyIndex(entry);
        entry.offScreen = true;
      } else {
        // Reuse the per-frame projection×view product (FRUSTUM_MATRIX_SCRATCH,
        // built in evaluatePerFrame) instead of recomputing it per group.
        const diagonalPx = projectBoxDiagonalPx(worldBox, camera, viewport, FRUSTUM_MATRIX_SCRATCH);
        // Normalise the projected pixel diagonal to a dimensionless **coverage
        // metric** (fraction of a filled viewport) so the viewport-relative
        // coverage_fraction thresholds anchor the finest at fills-screen on any
        // monitor. diagonalPx == +Infinity (camera inside the box) → Infinity →
        // finest, unchanged. viewportDiag is > 0 here (evaluatePerFrame guards
        // width/height == 0).
        const viewportDiag = Math.hypot(viewport.width, viewport.height);
        const coverageMetric = diagonalPx / (FILL_FACTOR * viewportDiag);
        desired = pickChildWithHysteresis(cache.thresholds, entry.activeChildIndex, coverageMetric);
        entry.offScreen = false;
      }
    }

    // ── Advance the aspiration (``activeChildIndex``) toward ``desired`` ──
    // The aspiration is the hysteresis anchor and only moves onto a READY level;
    // a not-ready desired kicks its deferred loader and we keep aspiring to the
    // current level until it commits. Visibility is NOT touched here — the
    // display-resolution pass below is the single owner of ``object.visible``.
    if (desired !== entry.activeChildIndex) {
      const target = entry.children[desired];
      if (isReady(target)) {
        entry.activeChildIndex = desired;
      } else {
        this.maybeKickLoad(target);
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
    if (aspiration && !isReady(aspiration)) this.maybeKickLoad(aspiration);

    // ── Slice-aware DISPLAY resolution ──
    // Show the aspiration when its committed geometry is fresh for the current
    // view version; while it is stale (a time/displayDims scrub reloaded it in
    // place without flipping ``ready``) show the coarsest FRESH level so the new
    // slice appears immediately at low detail, then swap up once the aspiration
    // recommits. ``getViewVersion`` undefined ⇒ freshness untracked ⇒
    // display == aspiration (identical to the pre-feature behaviour).
    const version = this.deps.getViewVersion?.();
    const aspirationReady = !!aspiration && isReady(aspiration);
    const aspirationFresh = version == null || isFresh(aspiration, version);
    let displayIdx: number;
    if (aspirationReady && aspirationFresh) {
      // Aspiration is committed and fresh (or freshness untracked) → show it.
      displayIdx = entry.activeChildIndex;
    } else if (version != null) {
      // Stale or not-yet-ready aspiration, freshness tracked → display the
      // coarsest fresh level (the slice-aware fallback; falls back to the
      // coarsest ready level if none is fresh yet, so it never goes blank).
      displayIdx = this.coarsestFreshOrReadyIndex(entry, version);
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
      if (chosen && isFresh(chosen, version) && visibleElementCount(chosen) === 0) {
        const fallback = coarsestFreshNonEmptyIndex(entry.children, version);
        if (fallback >= 0 && fallback !== displayIdx) {
          if (!this.warnedEmptyLevel.has(entry.path)) {
            this.warnedEmptyLevel.add(entry.path);
            log.warning(
              Modules.SCENE_LOADER,
              `lod_group ${entry.path}: level ${displayIdx} is fresh but committed 0 ` +
                `elements while level ${fallback} has visible geometry — showing level ` +
                `${fallback} instead. This usually means inconsistent/stale data ` +
                '(e.g. a dataset regenerated at the same URL with a poisoned cache); ' +
                'try reloading with ?clear-cache.'
            );
          }
          displayIdx = fallback;
        }
      }
    }

    // ── Never-downgrade display gate ──
    // A lazy level flips ``ready`` after its FIRST additive chunk commits, so
    // an ungated swap to a fresh-but-still-streaming aspiration pops displayed
    // quality down to chunk-1 (on zoom in, zoom out, or after a scrub settles)
    // and climbs back. Hold the previously-displayed level while the streaming
    // aspiration is strictly worse than what is on screen; release on ladder
    // completion (committed, not just fetched — see shouldHoldPreviousDisplay),
    // committed-count crossover (the rest of the ladder then streams VISIBLY),
    // ladder failure, or the previous level losing freshness. A group with
    // nothing better on screen swaps immediately (fast first paint preserved).
    // Bypassed for an explicit lock (the user wants that level now) and while
    // off-screen (frustum-culled: no visual pop, and holding would pin the
    // previous level's VRAM for nothing).
    if (
      displayIdx === entry.activeChildIndex &&
      entry.selectorMode === 'auto' &&
      !entry.offScreen
    ) {
      // Read the gate's memory (last ON-SCREEN displayed level), NOT
      // ``displayedChildIndex`` — the latter is clobbered to the coarse level
      // during an off-screen excursion, which would defeat the hold on return.
      const prevIdx = entry.heldDisplayChildIndex;
      if (prevIdx != null && prevIdx !== displayIdx) {
        const prev = entry.children[prevIdx];
        if (shouldHoldPreviousDisplay(aspiration!, prev, version ?? null)) {
          displayIdx = prevIdx;
          // The held aspiration is semantically in use — keep it warm in the
          // eviction LRU. The never-shown stamp below only fires once
          // (``lastVisibleTick == null``), so during a failure-cooldown window
          // (ready, not loading, not displayed) it would otherwise be the
          // globally coldest eviction candidate and churn release→re-stream.
          aspiration!.lastVisibleTick = this.tick;
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
    const needsReloadOrRefine =
      aspirationReady && (!aspirationFresh || (aspiration!.hasMoreLODs?.() ?? false));
    if (settled && needsReloadOrRefine && aspiration!.ensureLoaded) {
      this.maybeKickReload(aspiration!);
    }

    // ── Apply visibility (single owner) ──
    // At most one child visible (``displayIdx``, and only if it is READY — never
    // force-show a not-ready placeholder). ``changed`` flips when the SHOWN
    // level changes so ``evaluatePerFrame`` refreshes the monitor's visible
    // tally, which counts the displayed level, not the aspiration.
    let changed = false;
    for (let i = 0; i < entry.children.length; i++) {
      const child = entry.children[i];
      const shouldShow = i === displayIdx && isReady(child);
      if (child.object.visible !== shouldShow) {
        child.object.visible = shouldShow;
        if (shouldShow) changed = true; // a new level became visible
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
    for (const child of entry.children) {
      if (isReady(child) && child.lastVisibleTick == null) {
        child.lastVisibleTick = this.tick;
      }
    }

    return changed;
  }

  /**
   * Fold an entry's children nD ``positionBounds`` into a single world-space
   * :type:`BoundingBox`, mapping nD axes onto X/Y/Z via the current
   * ``displayDims`` and lifting through the group's ``matrixWorld``. Returns
   * ``null`` when no child has usable bounds (mismatched/empty min-max). Shared
   * by the auto selector (diagonal pick + frustum gate) and the eviction
   * ranking so both reason over identical geometry. Children with bogus bounds
   * are skipped. Uses the per-entry ``localBoxScratch`` and the registry's
   * ``matrixScratch``; ``transformBoundingBox`` allocates the returned box, so
   * it is independent of that scratch and safe to keep past the next call.
   */
  private computeWorldBox(
    entry: LODGroupEntry,
    displayDims: readonly number[]
  ): BoundingBox | null {
    const cache = this.caches.get(entry.path);
    if (!cache) return null;

    const local = cache.localBoxScratch;
    let any = false;
    for (let ci = 0; ci < entry.children.length; ci++) {
      const pb = entry.children[ci].positionBounds;
      if (pb.min.length === 0 || pb.max.length === 0 || pb.min.length !== pb.max.length) {
        continue;
      }
      // Project to X/Y/Z. Unmapped axes default to 0 (matches
      // ``projectBoundsToDisplayDims``'s defensive fallback).
      let x0 = 0;
      let x1 = 0;
      let y0 = 0;
      let y1 = 0;
      let z0 = 0;
      let z1 = 0;
      if (displayDims.length > 0) {
        const d0 = displayDims[0];
        if (d0 < pb.min.length) {
          x0 = pb.min[d0];
          x1 = pb.max[d0];
        }
      }
      if (displayDims.length > 1) {
        const d1 = displayDims[1];
        if (d1 < pb.min.length) {
          y0 = pb.min[d1];
          y1 = pb.max[d1];
        }
      }
      if (displayDims.length > 2) {
        const d2 = displayDims[2];
        if (d2 < pb.min.length) {
          z0 = pb.min[d2];
          z1 = pb.max[d2];
        }
      }
      if (!any) {
        local.min.x = x0;
        local.max.x = x1;
        local.min.y = y0;
        local.max.y = y1;
        local.min.z = z0;
        local.max.z = z1;
        any = true;
      } else {
        if (x0 < local.min.x) local.min.x = x0;
        if (x1 > local.max.x) local.max.x = x1;
        if (y0 < local.min.y) local.min.y = y0;
        if (y1 > local.max.y) local.max.y = y1;
        if (z0 < local.min.z) local.min.z = z0;
        if (z1 > local.max.z) local.max.z = z1;
      }
    }
    if (!any) return null;

    // Lift to world space. THREE updates matrix lazily; force a refresh before
    // reading — cheap and idempotent. Copy ``matrixWorld.elements`` into a
    // reusable array instead of allocating one via ``.toArray()`` every frame.
    entry.groupObject.updateWorldMatrix(true, false);
    const elements = entry.groupObject.matrixWorld.elements;
    const m = this.matrixScratch;
    for (let i = 0; i < 16; i++) m[i] = elements[i];
    return transformBoundingBox(local, m);
  }

  /**
   * Coarsest child that is ready AND fresh for ``version``, falling back to the
   * coarsest READY level when none is fresh yet (the ≤1-frame window right after
   * a re-slice) so the group shows stale-but-ready geometry rather than going
   * blank. Thin wrapper over the pure ``coarsestFreshIndex`` (lod-freshness.ts)
   * + ``coarsestReadyIndex``.
   */
  private coarsestFreshOrReadyIndex(entry: LODGroupEntry, version: number): number {
    const fresh = coarsestFreshIndex(entry.children, version);
    return fresh >= 0 ? fresh : this.coarsestReadyIndex(entry);
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
  private maybeKickLoad(child: LODGroupChild): void {
    if (isReady(child)) return; // not-ready-only: a ready level needs no initial load
    this.kickDeferredLoad(child);
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
  private maybeKickReload(child: LODGroupChild): void {
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
   */
  private kickDeferredLoad(child: LODGroupChild): void {
    if (!child.ensureLoaded || child.loading) return;
    if (child.failed) {
      if (child.failedTick == null) {
        // First frame we observe the failure — start the cooldown clock.
        child.failedTick = this.tick;
        return;
      }
      if (this.tick - child.failedTick < FAILED_RETRY_FRAMES) return;
      // Cooldown elapsed — clear the failure and fall through to retry.
      child.failed = false;
      child.failedTick = undefined;
    }
    child.loading = true;
    child.ensureLoaded();
  }

  /**
   * Bound resident LOD geometry to the GPU-pool byte budget. Runs once per
   * frame after all entries are evaluated. The registry is pure *policy*
   * here: it does not track bytes itself — it asks the pool for the live
   * resident total (``getResidentBytes``, the single accounting truth) and,
   * while over the budget ceiling, demotes evictable levels (loaded, has a
   * ``release`` thunk, not the visible child, shown at least once).
   *
   * Eviction order prioritises **what is furthest from the visible**: levels
   * whose group is entirely outside the camera frustum first, then by
   * descending camera distance, then coldest-last-visible-tick as the final
   * tiebreak (preserving the previous time-LRU behaviour when spatial keys
   * tie). This pairs with the off-screen selector gate: groups the camera
   * turned away from drop to coarsest *and* are the first to give back VRAM,
   * keeping the on-screen working set resident.
   *
   * Each ``release()`` moves that level's buffer active→pooled and
   * synchronously triggers the pool's byte-eviction pass, which disposes
   * pooled buffers (largest-first) until total resident is back under
   * budget — so the loop typically demotes one level then exits. The
   * visible level of each group and eager fallback levels (no ``release``)
   * are never demoted. Eviction is rare (only under genuine VRAM pressure),
   * preserving the no-churn retention property — the per-entry world-box /
   * frustum / distance math here only runs on that rare over-budget frame.
   */
  private enforceResidentByteBudget(
    camera: THREE.Camera,
    frustum: THREE.Frustum,
    displayDims: readonly number[]
  ): void {
    const budget = this.deps.getResidentByteBudget?.();
    const getResidentBytes = this.deps.getResidentBytes;
    // No budget or no measurement wired ⇒ pure retention.
    if (budget == null || budget <= 0 || !getResidentBytes) return;
    if (getResidentBytes() <= budget) return;

    camera.getWorldPosition(CAMERA_POS_SCRATCH);

    // Collect evictable levels, tagging each with its group's off-screen flag
    // and camera distance for spatial-priority ranking.
    const evictable: { child: LODGroupChild; offscreen: boolean; distance: number }[] = [];
    for (const entry of this.entries.values()) {
      const worldBox = this.computeWorldBox(entry, displayDims);
      let offscreen = false;
      let distance = 0;
      if (worldBox) {
        WORLD_BOX3_SCRATCH.min.set(worldBox.min.x, worldBox.min.y, worldBox.min.z);
        WORLD_BOX3_SCRATCH.max.set(worldBox.max.x, worldBox.max.y, worldBox.max.z);
        offscreen = !frustum.intersectsBox(WORLD_BOX3_SCRATCH);
        WORLD_BOX3_SCRATCH.getCenter(BOX_CENTER_SCRATCH);
        distance = BOX_CENTER_SCRATCH.distanceTo(CAMERA_POS_SCRATCH);
      }
      const children = entry.children;
      // Never evict the level currently DISPLAYED (which, during a re-slice, can
      // be a coarser fresh level rather than the aspiration ``activeChildIndex``)
      // — releasing it would blank the on-screen group. ``displayedChildIndex``
      // is written by ``evaluateEntry`` earlier in this same per-frame pass.
      const displayed = entry.displayedChildIndex ?? entry.activeChildIndex;
      for (let i = 0; i < children.length; i++) {
        const child = children[i];
        if (!isReady(child)) continue;
        // Skip a child mid-(re)load: ``release()`` resets ``loading=false`` and
        // ``ready=false``, so evicting one whose deferred reload is in flight
        // would let the registry kick a SECOND concurrent ``ensureLoaded`` for
        // the same loader. The not-ready guard above misses it because a stale
        // RELOAD keeps ``ready=true`` while ``loading=true``.
        if (i !== displayed && !child.loading && child.release && child.lastVisibleTick != null) {
          evictable.push({ child, offscreen, distance });
        }
      }
    }
    // Off-screen first, then furthest-first, then coldest-first.
    evictable.sort((a, b) => {
      if (a.offscreen !== b.offscreen) return a.offscreen ? -1 : 1;
      if (a.distance !== b.distance) return b.distance - a.distance;
      return (a.child.lastVisibleTick ?? 0) - (b.child.lastVisibleTick ?? 0);
    });

    // Demote ranked levels until the pool reports we are back under budget.
    // Bounded by the fixed `evictable` list (never re-collected), so it
    // always terminates even if `release()`'s pool eviction were to stop
    // freeing (defensive — today the byte pass is uncapped and frees fully).
    for (const { child } of evictable) {
      if (getResidentBytes() <= budget) break;
      child.release!(); // active→pooled; release's evictUnused() disposes the excess
    }
  }
}
