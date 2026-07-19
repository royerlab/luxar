/**
 * Resident-byte eviction policy for LOD-group levels.
 *
 * The VRAM-pressure half of `lod-group-registry.ts`, extracted as a pure
 * policy function: given the registry's entries and the GPU buffer pool's
 * live accounting (`getResidentBytes` — the single truth; the registry keeps
 * no per-level byte estimate), demote cold hidden levels while the pool is
 * over the shared budget ceiling. Ranking prioritises what is furthest from
 * the visible: off-screen groups first, then descending camera distance, then
 * coldest last-visible tick. Runs once per frame after all entries are
 * evaluated; eviction is rare (only under genuine VRAM pressure), preserving
 * the registry's no-churn retention property.
 *
 * Types are minimal structural shapes (the `lod-freshness.ts` pattern) so
 * this module never imports the registry back.
 *
 * @module scene/lod-eviction
 */

import * as THREE from 'three';

import type { BoundingBox } from './scene-manager/clipping/bounds-math';
import { isReady } from './lod-freshness';

/** Minimal structural shape of a registry child this module reads/releases. */
export interface EvictableChild {
  /** ``false`` ⇒ geometry not committed yet (never an eviction candidate). */
  ready?: boolean;
  /** ``true`` while a deferred (re)load is in flight — skip (see below). */
  loading?: boolean;
  /** Release thunk; absent on eager fallback levels (never demoted). */
  release?: () => void;
  /** LRU key; ``undefined`` ⇒ never shown ⇒ not an eviction candidate. */
  lastVisibleTick?: number;
}

/** Minimal structural shape of a registry entry the eviction pass reads. */
export interface EvictableEntry {
  children: readonly EvictableChild[];
  /** The level ACTUALLY on screen this frame (never evicted). */
  displayedChildIndex?: number;
  /** Fallback for ``displayedChildIndex`` before the first evaluation. */
  activeChildIndex: number;
}

/**
 * Module-scope scratch for the eviction ranking. ``evaluatePerFrame`` (the
 * sole caller's per-frame entry point) is single-threaded and non-re-entrant,
 * so sharing these across all entries within one frame is safe.
 */
const WORLD_BOX3_SCRATCH = new THREE.Box3();
const BOX_CENTER_SCRATCH = new THREE.Vector3();
const CAMERA_POS_SCRATCH = new THREE.Vector3();

/**
 * Bound resident LOD geometry to the GPU-pool byte budget. Runs once per
 * frame after all entries are evaluated. This is pure *policy*: it does not
 * track bytes itself — it asks the pool for the live resident total
 * (``getResidentBytes``, the single accounting truth) and, while over the
 * budget ceiling (``getResidentByteBudget``), demotes evictable levels
 * (loaded, has a ``release`` thunk, not the visible child, shown at least
 * once).
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
 *
 * ``computeWorldBox`` is the registry's per-entry world-box fold (shared with
 * the auto selector so both reason over identical geometry).
 */
export function enforceResidentByteBudget<E extends EvictableEntry>(opts: {
  entries: Iterable<E>;
  camera: THREE.Camera;
  frustum: THREE.Frustum;
  getResidentByteBudget?: () => number;
  getResidentBytes?: () => number;
  computeWorldBox: (entry: E) => BoundingBox | null;
}): void {
  const { entries, camera, frustum, getResidentBytes, computeWorldBox } = opts;
  const budget = opts.getResidentByteBudget?.();
  // No budget or no measurement wired ⇒ pure retention.
  if (budget == null || budget <= 0 || !getResidentBytes) return;
  if (getResidentBytes() <= budget) return;

  camera.getWorldPosition(CAMERA_POS_SCRATCH);

  // Collect evictable levels, tagging each with its group's off-screen flag
  // and camera distance for spatial-priority ranking.
  const evictable: { child: EvictableChild; offscreen: boolean; distance: number }[] = [];
  for (const entry of entries) {
    const worldBox = computeWorldBox(entry);
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
