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
 *      pixel coordinates; the diagonal of the screen-space AABB is
 *      the selector metric.
 *   4. Pick the **finest** child whose ``min_pixel_size`` threshold is
 *      satisfied by that diagonal, with 10% asymmetric hysteresis on
 *      the downgrade direction to suppress threshold-edge flicker.
 *   5. If the desired child differs from the current active one, swap
 *      visibility atomically.
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

/** Asymmetric hysteresis on the "downgrade to coarser" direction. */
const HYSTERESIS_RATIO = 0.1;

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
  /** Strictly monotonic increasing in coarsest→finest order. */
  minPixelSize: number;
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
}

/** A child is renderable iff its geometry is committed. Absent flag ⇒ ready. */
function isReady(child: LODGroupChild): boolean {
  return child.ready !== false;
}

/** One LOD-group entry tracked by the registry. */
export interface LODGroupEntry {
  /** Scene path (for diagnostics + UI lookup). */
  path: string;
  /** The lod_group's THREE container. World matrix lives here. */
  groupObject: THREE.Object3D;
  /**
   * Children in coarsest→finest order (== insertion order on disk,
   * == ascending ``minPixelSize``).
   */
  children: LODGroupChild[];
  /** Current selector mode (``'auto'`` or ``{ lockLevel: i }``). */
  selectorMode: LODGroupSelectorMode;
  /** Initial active level, used when nothing else has selected yet. */
  defaultLevel: number;
  /** Index into ``children`` of the currently-visible child. */
  activeChildIndex: number;
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
  thresholds: number[];
  localBoxScratch: BoundingBox;
}

/** Module-scope scratch for ``projectBoxDiagonalPx``. Single-threaded. */
const CORNER_SCRATCH = new THREE.Vector3();

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
}

/**
 * Project a world-space :type:`BoundingBox` through the camera and
 * return the diagonal of the screen-space AABB in pixels.
 *
 * Treats the bbox's 8 corners independently (works for both
 * perspective and orthographic projection without a closed-form
 * radius). NDC → pixels assumes the viewport size matches the
 * renderer canvas.
 *
 * Exported for unit testing.
 */
export function projectBoxDiagonalPx(
  box: BoundingBox,
  camera: THREE.Camera,
  viewport: { width: number; height: number }
): number {
  // Reuse a module-scope Vector3. ``evaluatePerFrame`` is invoked from
  // a single per-frame callback, so this is safe — no concurrent
  // entries into this function.
  const corner = CORNER_SCRATCH;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < 8; i++) {
    corner.set(
      i & 1 ? box.max.x : box.min.x,
      i & 2 ? box.max.y : box.min.y,
      i & 4 ? box.max.z : box.min.z
    );
    corner.project(camera);
    if (corner.x < minX) minX = corner.x;
    if (corner.x > maxX) maxX = corner.x;
    if (corner.y < minY) minY = corner.y;
    if (corner.y > maxY) maxY = corner.y;
  }
  const widthPx = (maxX - minX) * 0.5 * viewport.width;
  const heightPx = (maxY - minY) * 0.5 * viewport.height;
  return Math.hypot(widthPx, heightPx);
}

/**
 * Pick the desired child index given a screen-space diagonal and the
 * current active index. Applies 10% asymmetric hysteresis on the
 * downgrade direction.
 *
 * The "natural" pick is the finest child whose ``minPixelSize`` is
 * less than or equal to ``diagonalPx``. Hysteresis only resists
 * dropping back to a coarser level: when downgrading from index
 * ``currentIdx``, the metric must fall below
 * ``thresholds[currentIdx] * (1 - 0.1)``; otherwise we stay on the
 * current level even though the natural pick is coarser.
 *
 * Exported for unit testing.
 */
export function pickChildWithHysteresis(
  thresholds: readonly number[],
  currentIdx: number,
  diagonalPx: number,
  hysteresisRatio: number = HYSTERESIS_RATIO
): number {
  if (thresholds.length === 0) return -1;

  // Natural pick: finest child with threshold ≤ diagonalPx. Thresholds
  // are monotonic increasing in coarsest→finest order, so scan upward
  // until the threshold exceeds the metric.
  let natural = 0;
  for (let i = 0; i < thresholds.length; i++) {
    if (thresholds[i] <= diagonalPx) natural = i;
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
  // ``derive_min_pixel_sizes`` ×1.1 nudge) the band shrinks proportionally,
  // so the deadband never straddles the neighbour — every level still
  // renders on the way down and the selection can't flip-flop across a band
  // wider than the inter-level spacing.
  const currentThreshold = thresholds[currentIdx];
  const prevThreshold = thresholds[currentIdx - 1]; // currentIdx >= 1 here
  const margin = hysteresisRatio * (currentThreshold - prevThreshold);
  if (diagonalPx < currentThreshold - margin) {
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

  constructor(private deps: LODGroupRegistryDeps) {}

  /** Register a newly-loaded lod_group (called by the scene loader). */
  register(entry: LODGroupEntry): void {
    this.entries.set(entry.path, entry);
    this.caches.set(entry.path, {
      thresholds: entry.children.map((c) => c.minPixelSize),
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
  }

  /** Drop an lod_group from the registry (called on scene teardown). */
  unregister(path: string): void {
    this.entries.delete(path);
    this.caches.delete(path);
  }

  /** Clear all entries (called on full scene tear-down). */
  clear(): void {
    this.entries.clear();
    this.caches.clear();
    // Reset the monotonic tick so a reused registry (shared-registry
    // refactor) starts cold rather than inheriting stale LRU ordering.
    this.tick = 0;
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
    // WebGL coordinate system matches the NDC convention used by
    // ``Vector3.project()`` inside ``projectBoxDiagonalPx``. The frustum is
    // shared by the off-screen LOD gate (per entry) and the eviction ranking.
    FRUSTUM_MATRIX_SCRATCH.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    FRUSTUM_SCRATCH.setFromProjectionMatrix(FRUSTUM_MATRIX_SCRATCH);

    let changed = false;
    for (const entry of this.entries.values()) {
      if (this.evaluateEntry(entry, camera, viewport, displayDims, FRUSTUM_SCRATCH)) {
        changed = true;
      }
    }
    // Bound resident LOD geometry against the shared GPU-pool byte budget
    // (one VRAM authority). Retention keeps loaded levels resident so
    // re-shows are free; this LRU-evicts only hidden levels when over budget —
    // off-screen / furthest-from-camera first — so no per-swap release, hence
    // no reload churn.
    this.enforceResidentByteBudget(camera, FRUSTUM_SCRATCH, displayDims);
    return changed;
  }

  /** Returns ``true`` if this entry's active child changed. */
  private evaluateEntry(
    entry: LODGroupEntry,
    camera: THREE.Camera,
    viewport: { width: number; height: number },
    displayDims: readonly number[],
    frustum: THREE.Frustum
  ): boolean {
    // Pick the desired child index.
    let desired: number;
    if (entry.selectorMode !== 'auto') {
      // Explicit lock bypasses the off-screen gate: a user who pins a level
      // keeps it whether or not the group is on screen.
      desired = entry.selectorMode.lockLevel;
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
      } else {
        const diagonalPx = projectBoxDiagonalPx(worldBox, camera, viewport);
        desired = pickChildWithHysteresis(cache.thresholds, entry.activeChildIndex, diagonalPx);
      }
    }

    let changed = false;
    if (desired !== entry.activeChildIndex) {
      const target = entry.children[desired];
      if (isReady(target)) {
        // Atomic swap: hide outgoing, show incoming. No release here —
        // retention keeps the outgoing level resident so swapping back is
        // a sub-millisecond visibility toggle, not a reload. Memory is
        // bounded by the byte-budget LRU (``enforceResidentByteBudget``),
        // not by releasing on every swap.
        entry.children[entry.activeChildIndex].object.visible = false;
        target.object.visible = true;
        entry.activeChildIndex = desired;
        changed = true;
      } else {
        // Lazy gate: desired level not committed yet — kick its deferred
        // loader (guarded, with failure cooldown) and keep the current
        // level visible. A later frame performs the swap once the thunk
        // sets ``ready=true``.
        this.maybeKickLoad(target);
      }
    }

    // Reconcile the active child every frame. The registry — not the
    // loader — guarantees the active level ends up loaded and visible:
    //   * ready  → ensure it is shown (a ``desired === active`` frame skips
    //     the swap block above, so a freshly-ready active child would
    //     otherwise stay hidden) and stamp it most-recently-used.
    //   * not ready → self-heal by kicking its load (covers a fallback that
    //     pinned a not-ready lazy level as active when the eager default
    //     failed to attach — otherwise that group renders blank forever).
    const active = entry.children[entry.activeChildIndex];
    if (active) {
      if (isReady(active)) {
        if (!active.object.visible) {
          active.object.visible = true;
          changed = true;
        }
        active.lastVisibleTick = this.tick;
      } else {
        this.maybeKickLoad(active);
      }
    }

    // Stamp any already-ready child that has never been shown so it ages
    // into the eviction LRU. Without this, a lazy level that finished
    // loading but was never swapped-to (camera moved away mid-load) keeps
    // ``lastVisibleTick == null`` and is permanently exempt from eviction,
    // leaking VRAM. Stamping "became ready" as a use sorts a just-loaded
    // level as most-recent (preserving the 1-frame swap-gap guard) while a
    // never-shown level ages out normally.
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
    if (isReady(child) || !child.ensureLoaded || child.loading) return;
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
      for (let i = 0; i < children.length; i++) {
        const child = children[i];
        if (!isReady(child)) continue;
        if (i !== entry.activeChildIndex && child.release && child.lastVisibleTick != null) {
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
