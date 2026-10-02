/**
 * Offline-capture quiescence: whether every lod_group and partition part that contributes pixels to the
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
 *   ``LODGroupRegistry.kickDeferredLoadIfVisible`` refuses to START a deferred load while
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
 * - No child that can affect the frame may be ``loading`` — an in-flight
 *   commit can change what renders later. A hidden band preload is exempt
 *   until it becomes the selected level.
 * - No level dissolve may be in flight (``LODGroupRegistry.isAnimating``). The dissolve
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
 *
 * @module scene/capture-quiescence
 */

import { isEffectivelyVisible } from '../utils/object-visibility';
import { isReady } from './lod-freshness';
import { childFreshAndCount } from './lod-display-gate';
import type { LODGroupChild, LODGroupEntry } from './lod-group-registry';

/** What {@link isCaptureQuiescent} reads off the registry. */
export interface CaptureQuiescenceSource {
  /** A level dissolve is in flight. */
  isAnimating(): boolean;
  /** Any registered partition part contributes pixels to this frame. */
  anyVisiblePartitionPart(): boolean;
  /** The owning loader has a view pass in flight or queued. */
  isUpdateInProgress(): boolean;
  /** The owning loader has latched an archive fault. */
  hasArchiveFault(): boolean;
  /** Any lazy level has an ``ensureLoaded`` in flight. */
  anyChildLoading(): boolean;
  /** The current view version, or ``null`` when freshness is not tracked. */
  viewVersion(): number | null;
  /** Visible partition parts have no pending resync or incomplete commit. */
  partitionsCaptureQuiescent(version: number | null): boolean;
  /** The registered lod_groups. */
  entries(): Iterable<LODGroupEntry>;
  /** The level a group's band preload is making resident, if any. */
  preloadIdx(path: string): number | undefined;
}

/** See the module doc. */
export function isCaptureQuiescent(src: CaptureQuiescenceSource): boolean {
  if (src.isAnimating()) return false;
  // Wired to SceneLoader.isLoadPassInProgress: any pass can still change a
  // visible partition part before this fixed-pose capture frame is exported.
  if (src.anyVisiblePartitionPart() && src.isUpdateInProgress()) return false;
  if (src.hasArchiveFault()) return !src.anyChildLoading();
  const version = src.viewVersion();
  if (!src.partitionsCaptureQuiescent(version)) return false;
  for (const entry of src.entries()) {
    if (!entryCaptureQuiescent(entry, version, src.preloadIdx(entry.path))) return false;
  }
  return true;
}

/** One lod_group's share of {@link isCaptureQuiescent}. */
function entryCaptureQuiescent(
  entry: LODGroupEntry,
  version: number | null,
  preloadIdx: number | undefined
): boolean {
  // Deliberately excluded: an off-screen group is held coarse on purpose
  // and contributes no pixels to the frame being captured.
  if (entry.offScreen === true) return true;
  // Likewise excluded, and this one is load-bearing rather than merely an
  // optimisation: a hidden group draws nothing AND cannot start a deferred
  // load (``kickDeferredLoadIfVisible``), while the selector — whose
  // frustum test is pure geometry — happily records a fine
  // ``desiredChildIndex`` for it. Blocking on that combination never
  // resolves.
  if (!isEffectivelyVisible(entry.groupObject)) return true;

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
  if (!aspiration) return true;

  if (displayed !== active || awaitingDesired(entry, desired)) return false;
  return aspirationAtFinalQuality(aspiration, version) && !childLoading(entry, desired, preloadIdx);
}

/** The selector wants a level other than the aspiration that can still become ready. */
function awaitingDesired(entry: LODGroupEntry, desired: number): boolean {
  if (desired === entry.activeChildIndex) return false;
  const target = entry.children[desired];
  // A failed level never becomes ready, so waiting on it only times out.
  // An out-of-range ``desired`` (no child there at all) is the same trap
  // as the missing aspiration above and likewise must not block.
  return target !== undefined && target.failed !== true;
}

/** The aspiration is ready, fresh for ``version``, and its ladder is complete. */
function aspirationAtFinalQuality(aspiration: LODGroupChild, version: number | null): boolean {
  if (!isReady(aspiration)) return false;
  // One fold for both group-aware answers: per-slice freshness AND — for a
  // deferred GROUP child, which has no ``hasMoreLODs`` thunk — whether its
  // subtree's committed additive ladders are complete.
  const progress = childFreshAndCount(aspiration, version);
  if (version != null && !progress.fresh) return false;
  if (aspiration.hasMoreLODs?.() === true) return false;
  return progress.subtreeLadderComplete;
}

/** A child that can affect the frame is loading (a hidden band preload cannot). */
function childLoading(
  entry: LODGroupEntry,
  desired: number,
  preloadIdx: number | undefined
): boolean {
  const active = entry.activeChildIndex;
  for (let i = 0; i < entry.children.length; i++) {
    const child = entry.children[i];
    // The neighbour is loaded only to make a future threshold crossing
    // immediate; it cannot change this frame while it stays unselected.
    if (i === preloadIdx && i !== active && i !== desired && !child.object.visible) continue;
    if (child.loading) return true;
  }
  return false;
}
