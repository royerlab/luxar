/**
 * Resident-byte eviction policy for LOD-group levels.
 *
 * The VRAM-pressure half of `lod-group-registry.ts`, extracted as a pure
 * policy function: given the registry's entries and the GPU buffer pool's
 * live accounting (`getResidentBytes` — the single truth; the registry keeps
 * no per-level byte estimate), demote cold hidden levels while the pool is
 * over the shared budget ceiling. "Hidden" is EFFECTIVE (ancestor-aware)
 * visibility — a level under a hidden layer is cold no matter what its own
 * ``visible`` flag says. Ranking prioritises what is furthest from
 * the visible: levels under a hidden layer (undrawable, so coldest of all)
 * first, then off-screen groups, then descending camera distance, then
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
import { isEffectivelyVisible, type VisibilityNode } from '../utils/object-visibility';

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
  /**
   * The child's THREE node (structural — only ``visible`` and the ``parent``
   * chain are read). ``visible === true`` **and no hidden ancestor** ⇒ the
   * level renders THIS frame: the displayed level or the cross-fade blend
   * partner mid-dissolve. Never an eviction candidate — releasing an on-screen
   * level blanks (or half-blanks) the group mid-frame. The
   * ``displayedChildIndex`` guard alone misses the blend partner, which is on
   * screen but not the entry's displayed index. The registry's visibility pass
   * runs earlier in the same synchronous per-frame call, so the flag is current
   * here.
   *
   * The ancestor walk matters because ``visible`` is a LOCAL flag: a layer
   * authored ``visible=false`` (or toggled off in the layers panel) hides the
   * LAYER object while the level underneath keeps ``visible === true``. Without
   * the walk such a level looked permanently on-screen and was exempt from
   * eviction forever — hidden data that could not be drawn AND could not be
   * reclaimed.
   *
   * Demoting a level also clears this flag (``visible = false``): the released
   * geometry is no longer ready, so leaving it ``visible`` would let a
   * same-frame re-show of a hidden ancestor briefly expose stale buffers.
   */
  object?: VisibilityNode;
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
 * under a hidden layer (effectively invisible ⇒ undrawable, so the coldest
 * data of all) first, then levels whose group is entirely outside the camera
 * frustum, then by descending camera distance, then coldest-last-visible-tick
 * as the final tiebreak (preserving the previous time-LRU behaviour when
 * spatial keys tie). Making hidden the PRIMARY key keeps a hidden layer that
 * happens to sit in-frustum near the camera from ranking behind a visible
 * layer's off-screen fine levels — undrawable geometry is always reclaimed
 * before anything the viewer can still draw. This pairs with the off-screen
 * selector gate: groups the camera turned away from drop to coarsest *and* are
 * the first to give back VRAM, keeping the on-screen working set resident.
 *
 * Each ``release()`` moves that level's buffer active→pooled and
 * synchronously triggers the pool's byte-eviction pass, which disposes
 * pooled buffers (largest-first) until total resident is back under
 * budget — so the loop typically demotes one level then exits. The
 * EFFECTIVELY-visible level of each group (visible flag set *and* no hidden
 * ancestor) and eager fallback levels (no ``release``) are never demoted; a
 * level whose layer is hidden is an ordinary cold candidate, including the
 * group's nominal ``displayedChildIndex`` — nothing of that group is on screen
 * to protect. Eviction is rare (only under genuine VRAM pressure),
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

  // Collect evictable levels, tagging each with whether its group is hidden
  // (undrawable), its off-screen flag, and camera distance for ranking.
  const evictable: {
    child: EvictableChild;
    hidden: boolean;
    offscreen: boolean;
    distance: number;
  }[] = [];
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
      // Is anything in this group drawable at all? A hidden ANCESTOR (a layer
      // authored ``visible=false`` or toggled off in the layers panel) means no
      // — whatever the levels' own flags or the entry's displayed index say —
      // so every ready level of that group is an ordinary cold candidate. The
      // walk starts at the PARENT, not the child: the registry sets the
      // displayed level's own ``visible = true``, so including it would make
      // the two guards below circular.
      const ancestorsVisible = isEffectivelyVisible(child.object?.parent);
      // Never evict a level that is ON SCREEN this frame (``object.visible``):
      // during a coverage-band cross-fade TWO levels render — the displayed
      // primary and its blend partner — and only the primary is
      // ``displayedChildIndex``. Releasing the visible partner would drop half
      // the dissolve mid-fade (and leave a visible-but-not-ready level behind).
      // "Eviction must never release the level currently displayed" covers
      // everything actually rendering, not just the entry's displayed index.
      if (ancestorsVisible && child.object?.visible === true) continue;
      // The displayed level is likewise exempt only while the group can render
      // it; under a hidden ancestor there is nothing on screen to protect.
      if (ancestorsVisible && i === displayed) continue;
      // Skip a child mid-(re)load: ``release()`` resets ``loading=false`` and
      // ``ready=false``, so evicting one whose deferred reload is in flight
      // would let the registry kick a SECOND concurrent ``ensureLoaded`` for
      // the same loader. The not-ready guard above misses it because a stale
      // RELOAD keeps ``ready=true`` while ``loading=true``.
      if (!child.loading && child.release && child.lastVisibleTick != null) {
        evictable.push({ child, hidden: !ancestorsVisible, offscreen, distance });
      }
    }
  }
  // Hidden (undrawable) first, then off-screen, then furthest-first, then
  // coldest-first.
  evictable.sort((a, b) => {
    if (a.hidden !== b.hidden) return a.hidden ? -1 : 1;
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
    // Released geometry is no longer ready. Clear the (possibly stale) visible
    // flag so a same-frame re-show of a hidden ancestor cannot expose the stale
    // buffer before the registry's gated reload commits. Safe because eviction
    // never releases an EFFECTIVELY-visible level (the two guards above).
    if (child.object) child.object.visible = false;
  }
}
