/**
 * Pure, dependency-free LOD freshness + settle helpers used by the
 * `LODGroupRegistry` selector. Kept out of `lod-group-registry.ts` (which is
 * already large and THREE/camera-bound) so this logic is unit-testable in
 * isolation — every function here is pure over plain inputs (a child stub +
 * numbers), no THREE, no camera, no mocking.
 *
 * **Freshness** distinguishes a level whose committed geometry reflects the
 * CURRENT view (slice / displayDims) version from one that is merely "ready"
 * (geometry committed) but stale — a re-slice overwrites geometry in place
 * without flipping readiness, so the registry needs the per-mesh
 * `loadedViewVersion` stamp (written at commit by `stamp-view-version.ts`) to
 * know which level to actually display. Freshness is tracked for the three leaf
 * geometry types (gsplats / points / lines); nested groups carry no per-slice
 * staleness and are always "fresh".
 *
 * **Settle** is the debounce that drives deferred reloads: while the view
 * version changes every frame (active scrub) we want to show only the cheap
 * coarse level; once it has been stable for a few frames we reload the fine
 * level. `SettleTracker` answers "has the version been stable for N ticks?".
 *
 * The **never-downgrade display gate** built on these primitives lives in
 * the sibling module `lod-display-gate.ts` (`shouldHoldPreviousDisplay`,
 * `subtreeDisplayProgress`).
 *
 * @module scene/lod-freshness
 */

/** Minimal structural shape this module reads off a registry child. */
export interface FreshnessChild {
  /** ``false`` ⇒ geometry not committed yet. Absent/``true`` ⇒ committed. */
  ready?: boolean;
  /** The leaf THREE node; its ``userData`` carries the commit-time stamps. */
  object: {
    userData?: {
      nodeType?: string;
      loadedViewVersion?: number;
      visiblePointCount?: number;
      visibleSegmentCount?: number;
      visibleSplatCount?: number;
      /** Commit-time ladder stamp — see ``stamp-view-version.ts``. */
      committedLadderComplete?: boolean;
    };
  };
}

/** Leaf geometry types whose meshes are stamped with ``loadedViewVersion``. */
const FRESHNESS_TRACKED_TYPES: ReadonlySet<string> = new Set(['gsplats', 'points', 'lines']);

/** A child is renderable iff its geometry is committed. Absent flag ⇒ ready. */
export function isReady(child: { ready?: boolean }): boolean {
  return child.ready !== false;
}

/**
 * Whether `child` is ready AND fresh for view-version `version`. Freshness is
 * tracked only for the three stamped leaf types; a ready non-leaf child (nested
 * group / partition wrapper) has no per-slice staleness and is always fresh —
 * so the slice-aware fallback is a no-op for those. A ready leaf whose stamp is
 * absent/older than `version` is stale (its geometry reflects a previous slice).
 */
export function isFresh(child: FreshnessChild | undefined, version: number): boolean {
  if (!child || !isReady(child)) return false;
  const ud = child.object.userData;
  if (ud && FRESHNESS_TRACKED_TYPES.has(ud.nodeType ?? '')) {
    return ud.loadedViewVersion === version;
  }
  return true; // nested groups / unstamped leaves: no per-slice staleness
}

/**
 * Index of the coarsest child that is ready AND fresh for `version`, or `-1`
 * when none is fresh yet (the ≤1-frame window right after a re-slice, before
 * even the coarse level recommits). Children are stored coarsest→finest, so the
 * first match is the coarsest. The caller falls back to the coarsest READY
 * level on `-1` so the group shows stale-but-ready geometry rather than blank.
 */
export function coarsestFreshIndex(children: readonly FreshnessChild[], version: number): number {
  for (let i = 0; i < children.length; i++) {
    if (isFresh(children[i], version)) return i;
  }
  return -1;
}

/**
 * Committed visible-element count of a child's leaf mesh, or ``null`` when
 * untracked. Reads the per-type commit stamps (``visiblePointCount`` /
 * ``visibleSegmentCount`` / ``visibleSplatCount`` — written by the
 * commit-*-geometry helpers). Non-leaf children (nested groups) and
 * not-yet-committed leaves carry no stamp and return ``null`` — callers must
 * only act on a KNOWN-empty level (``0``), never on ``null``.
 */
export function visibleElementCount(child: FreshnessChild): number | null {
  return countFromUserData(child.object.userData);
}

/**
 * Committed visible-element count read straight off a leaf mesh's ``userData``,
 * or ``null`` when untracked. The allocation-free primitive behind
 * :func:`visibleElementCount` — the subtree fold in ``lod-display-gate.ts``
 * calls this per descendant on the per-frame hot path (during a hold), so it
 * must not wrap ``ud`` in a throwaway ``{ object: { userData } }`` object.
 */
export function countFromUserData(ud: FreshnessChild['object']['userData']): number | null {
  if (!ud || !FRESHNESS_TRACKED_TYPES.has(ud.nodeType ?? '')) return null;
  switch (ud.nodeType) {
    case 'points':
      return ud.visiblePointCount ?? null;
    case 'lines':
      return ud.visibleSegmentCount ?? null;
    case 'gsplats':
      return ud.visibleSplatCount ?? null;
    default:
      return null;
  }
}

/**
 * Index of the coarsest child that is ready, fresh for `version`, AND has a
 * non-zero committed element count — or `-1` when none qualifies. Companion
 * to :func:`coarsestFreshIndex` for the registry's empty-level display guard:
 * a fresh level that committed 0 elements while a coarser fresh level holds
 * visible geometry signals inconsistent data (with consistent LOD data a
 * finer level can never be empty where a coarser one is not — coarse levels
 * are derived from fine), and displaying the empty level would blank the
 * screen. Children with an UNTRACKED count (``null``) are accepted — the
 * guard only redirects away from known-empty levels.
 */
export function coarsestFreshNonEmptyIndex(
  children: readonly FreshnessChild[],
  version: number
): number {
  for (let i = 0; i < children.length; i++) {
    if (!isFresh(children[i], version)) continue;
    if (visibleElementCount(children[i]) === 0) continue;
    return i;
  }
  return -1;
}

/**
 * Tracks when the (global) view-update version last changed, in registry ticks,
 * so the selector can tell whether the view has "settled" (stopped scrubbing).
 * Single instance per registry — the version is global (one `getViewVersion`).
 */
export class SettleTracker {
  private lastVersion = Number.NaN;
  private lastChangeTick = 0;

  /** Record the current version at `tick`; resets the settle clock on change. */
  observe(version: number, tick: number): void {
    if (version !== this.lastVersion) {
      this.lastVersion = version;
      this.lastChangeTick = tick;
    }
  }

  /** True once the version has been unchanged for ≥ `settleTicks` frames. */
  isSettled(tick: number, settleTicks: number): boolean {
    return tick - this.lastChangeTick >= settleTicks;
  }
}
