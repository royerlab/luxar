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
 *      better (see ``shouldHoldPreviousDisplay`` in ``lod-display-gate.ts``).
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
  projectBoxDiagonalPx,
} from './lod-selector-math';
import { enforceResidentByteBudget } from './lod-eviction';

// The selector math (box projection + hysteresis pick) lives in
// `lod-selector-math.ts`; re-exported here so existing importers (the
// selector unit tests) keep their import site.
export { pickChildWithHysteresis, projectBoxDiagonalPx } from './lod-selector-math';

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
   * ``enforceByteBudget`` (same synchronous ``evaluatePerFrame`` pass)
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
 * Module-scope scratch for the per-frame frustum gate.
 * ``evaluatePerFrame`` is the single per-frame entry point (no re-entrancy),
 * so these are safe to share across all entries within one frame:
 *   - ``FRUSTUM_SCRATCH`` — rebuilt once per frame from the camera.
 *   - ``FRUSTUM_MATRIX_SCRATCH`` — projection × view product feeding it.
 *   - ``WORLD_BOX3_SCRATCH`` — a ``THREE.Box3`` view of a group's world bbox
 *     for ``frustum.intersectsBox`` (our ``BoundingBox`` is a plain object).
 * (The eviction pass keeps its own scratches in ``lod-eviction.ts``.)
 */
const FRUSTUM_SCRATCH = new THREE.Frustum();
const FRUSTUM_MATRIX_SCRATCH = new THREE.Matrix4();
const WORLD_BOX3_SCRATCH = new THREE.Box3();

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
  /**
   * Whether the LOD cross-fade is enabled (ON by default; `?no-lod-fade`
   * disables). When true and an additive/luminous group is zooming across a LOD
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
   * Whether streaming brightness compensation is enabled: as an additive/luminous
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
    this.fadeWasManaged = false;
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
    // Record whether fade management was ON this frame — the falling-edge
    // detector behind ``evaluateEntry``'s one-shot residual-opacity restore
    // (see ``fadeWasManaged``). Written AFTER the entry loop so every entry in
    // one frame sees the same previous-frame value.
    this.fadeWasManaged =
      this.deps.getCrossFadeEnabled?.() === true || this.deps.getEnergyCompEnabled?.() === true;
    // Bound resident LOD geometry against the shared GPU-pool byte budget
    // (one VRAM authority). Retention keeps loaded levels resident so
    // re-shows are free; this LRU-evicts only hidden levels when over budget —
    // off-screen / furthest-from-camera first — so no per-swap release, hence
    // no reload churn.
    this.enforceByteBudget(camera, FRUSTUM_SCRATCH, displayDims);
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
    // The dimensionless coverage metric for this frame (projected diagonal ÷
    // FILL_FACTOR·viewportDiag), hoisted so the coverage-band cross-fade below
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
        // Reuse the per-frame projection×view product (FRUSTUM_MATRIX_SCRATCH,
        // built in evaluatePerFrame) instead of recomputing it per group.
        const diagonalPx = projectBoxDiagonalPx(worldBox, camera, viewport, FRUSTUM_MATRIX_SCRATCH);
        // Normalise the projected pixel diagonal to a dimensionless **coverage
        // metric** (fraction of a filled viewport) so the viewport-relative
        // coverage_fraction thresholds anchor the finest at fills-screen on any
        // monitor. diagonalPx == +Infinity (camera inside the box) → Infinity →
        // finest, unchanged. viewportDiag is > 0 here (evaluatePerFrame guards
        // width/height == 0). ``?lod-finest`` forces Infinity → always finest.
        const viewportDiag = Math.hypot(viewport.width, viewport.height);
        coverageMetric = forceFinest ? Infinity : diagonalPx / (FILL_FACTOR * viewportDiag);
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
      const prevIdx = entry.heldDisplayChildIndex;
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
    // build-time mass conservation (both integrate to the same DC). Additive/
    // luminous only (order-independent compositing). Off / non-blendable /
    // off-screen / locked / a held-stale display ⇒ no blend (byte-identical hard
    // swap). The finer partner must be resident to fade against; if it is not,
    // kick its load so the NEXT crossing blends (the first hard-swaps meanwhile).
    let blendPartnerIdx: number | null = null;
    let primaryWeight = 1;
    if (
      this.deps.getCrossFadeEnabled?.() === true &&
      entry.selectorMode === 'auto' &&
      !entry.offScreen &&
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
    const needsReloadOrRefine =
      aspirationReady && (!aspirationFresh || (aspiration!.hasMoreLODs?.() ?? false));
    if (settled && needsReloadOrRefine && aspiration!.ensureLoaded) {
      this.maybeKickReload(entry, aspiration!);
    }

    // ── Apply visibility (single owner) ──
    // At most one child visible (``displayIdx``, and only if it is READY — never
    // force-show a not-ready placeholder). ``changed`` flips when the SHOWN
    // level changes so ``evaluatePerFrame`` refreshes the monitor's visible
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
    for (const child of entry.children) {
      if (isReady(child) && child.lastVisibleTick == null) {
        child.lastVisibleTick = this.tick;
      }
    }

    return changed;
  }

  /**
   * Fold an entry's children nD ``positionBounds`` into a single world-space
   * :type:`BoundingBox` (see {@link computeEntryWorldBox} in
   * ``lod-selector-math.ts`` for the math). Shared by the auto selector
   * (diagonal pick + frustum gate) and the eviction ranking so both reason
   * over identical geometry. This wrapper supplies the per-entry
   * ``localBoxScratch`` and the registry's ``matrixScratch``;
   * ``transformBoundingBox`` allocates the returned box, so it is independent
   * of those scratches and safe to keep past the next call.
   */
  private computeWorldBox(
    entry: LODGroupEntry,
    displayDims: readonly number[]
  ): BoundingBox | null {
    const cache = this.caches.get(entry.path);
    if (!cache) return null;
    return computeEntryWorldBox(entry, displayDims, cache.localBoxScratch, this.matrixScratch);
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
   */
  private childFreshAndCount(
    child: LODGroupChild,
    version: number
  ): { fresh: boolean; count: number | null } {
    // Leaf detection is by tracked nodeType, NOT by "has a count stamp": a leaf
    // that has not committed a count yet is still a leaf whose freshness is its
    // own ``loadedViewVersion`` stamp. Only a genuine group subtree folds.
    if (isTrackedLeaf(child)) {
      return { fresh: isFresh(child, version), count: visibleElementCount(child) };
    }
    // Ready gate for group children (the leaf branch gets it from ``isFresh``).
    // Without it, a not-ready deferred-group placeholder (no stamped leaves →
    // ``!aggregate`` below) would read fresh-with-unknown-count and the
    // empty-level guard could redirect display onto a level that CANNOT draw,
    // blanking the group permanently.
    if (!isReady(child)) return { fresh: false, count: null };
    const aggregate = subtreeDisplayProgress(child.object as unknown as ProgressNode, version);
    // Ready, but no stamped leaf under the subtree (nested group with no
    // slice-dependent geometry): no per-slice staleness signal, so treat as
    // fresh — exactly the pre-existing ``isFresh`` behaviour for a ready
    // non-leaf. Only a subtree that DOES carry stamped-but-stale leaves (a
    // non-null aggregate with ``fresh === false``) triggers the coarse fallback.
    if (!aggregate) return { fresh: true, count: null };
    return { fresh: aggregate.fresh, count: aggregate.count };
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
   * ``retryLazyChildByLeafPath`` (an explicit user retry of a FAILED level)
   * deliberately bypasses this and calls ``kickDeferredLoad`` directly: an
   * explicit request is honoured whatever the layer's visibility.
   */
  private kickDeferredLoadIfVisible(entry: LODGroupEntry, child: LODGroupChild): void {
    if (!isEffectivelyVisible(entry.groupObject)) return;
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
   * Bound resident LOD geometry to the GPU-pool byte budget — see
   * {@link enforceResidentByteBudget} (``lod-eviction.ts``) for the full
   * policy. This wrapper supplies the registry's entries, the pool-accounting
   * deps, and the shared per-entry world-box fold (so eviction and the auto
   * selector reason over identical geometry).
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
