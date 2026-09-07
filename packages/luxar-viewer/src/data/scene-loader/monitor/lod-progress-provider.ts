/**
 * Producer for the monitor's live LOD / progressive-refinement /
 * cache-residency snapshot.
 *
 * Closes over the per-geometry loader maps (for additive progressive
 * loaders) and the `LODGroupRegistry` (for substitutive `kind=lod`
 * groups), and on each poll returns a `path → LODProgressState` map the
 * monitor merges into its scene-graph tree. Lives in the data layer; the
 * monitor consumes it through `SceneLoaderMonitorPort.setLODProgressProvider`.
 *
 * @module data/scene-loader/monitor/lod-progress-provider
 */

import type {
  LODProgressProvider,
  LODProgressState,
  RefinementHoldReason,
} from '../../../types/data-monitor-types';
import type { LODGroupRegistry } from '../../../scene/lod-group-registry';

/**
 * Duck-typed subset of the progressive loaders' public surface
 * (`{points,lines,gsplats}-progressive-loader.ts`). Non-progressive
 * (single-LOD) loaders don't expose these getters, so the producer
 * treats them as plain loaders and skips them.
 */
interface ProgressiveLike {
  readonly totalLODCount?: number;
  readonly loadedLODCount?: number;
  readonly hasMoreLODs?: boolean;
  readonly lastAllResident?: boolean;
  readonly committedEnergyFraction?: number | null;
}

export interface LODProgressProviderDeps {
  /**
   * The four per-geometry loader maps (points / lines / gsplats / mesh),
   * keyed by scene-graph path. Additive nodes connect a single
   * progressive loader at the node path. Mesh joins them because a reveal
   * ladder is a progressive loader with the same three getters, even though
   * mesh takes part in none of the monitor's other per-type providers.
   */
  loaderMaps: ReadonlyArray<ReadonlyMap<string, unknown>>;
  /** The scene's LOD-group registry (substitutive levels), or null. */
  lodGroupRegistry: LODGroupRegistry | null;
  /**
   * Snapshot of partition groups (`{ path, partCount }`) captured at scene
   * build. A partition's frustum selector changes child visibility, not its
   * structural part count, so a one-time snapshot is sufficient — groups
   * surface in the monitor's scene-graph summary as `kind:'partition'` states.
   * Empty/omitted for scenes without partitions.
   */
  partitionGroups?: ReadonlyArray<{ path: string; partCount: number }>;
  /**
   * Rungs actually COMMITTED per node path, from the `committedLODCount`
   * stamp each commit writes to the mesh userData. Supplied by the caller
   * because it owns the live THREE root group (same arrangement as the
   * draw-order provider).
   *
   * Without it the panel reports the LOADER'S CURSOR, which advances as each
   * rung ARRIVES rather than when it is drawn. A pass whose commit fails or is
   * superseded therefore leaves the two disagreeing — and it disagrees loudest
   * in exactly the situation a user is trying to diagnose. On the hosted
   * Laniakea demo, basins showed "LOD 7/7 ~100%" while rendering a coarse
   * prefix and no longer refining; the monitor was the one surface that could
   * have revealed the stall and it actively concealed it (#2426).
   *
   * Optional so a caller without a root group (tests, headless wiring) keeps
   * the previous behaviour rather than losing the panel entirely.
   */
  committedLODCounts?: () => ReadonlyMap<string, number>;
  /**
   * Why this path's next rung is HELD back, if it is (`SceneLoader.
   * refinementHoldReason`: the density gate at the current framing, or the
   * residency ceiling). Lets the monitor tell a held rung from one still
   * streaming; both read as `refining` to the loader.
   */
  refinementHold?: (path: string) => RefinementHoldReason | null;
}

/** `{ held }` when a reason is known, `{}` otherwise — keeps unheld states free of the key. */
function heldEntry(reason: RefinementHoldReason | null | undefined): {
  held?: RefinementHoldReason;
} {
  return reason ? { held: reason } : {};
}

/**
 * Build a {@link LODProgressProvider} over the given loader maps and LOD
 * registry. The returned provider is cheap to call each tick — it reads
 * already-computed getters, never triggers loads.
 */
export function createLODProgressProvider(deps: LODProgressProviderDeps): LODProgressProvider {
  return {
    getLODStates(): Map<string, LODProgressState> {
      const out = new Map<string, LODProgressState>();

      const committed = deps.committedLODCounts?.();

      // Additive progressive loaders: one per node across all four maps.
      for (const map of deps.loaderMaps) {
        for (const [path, loader] of map) {
          const p = loader as ProgressiveLike;
          if (typeof p.totalLODCount === 'number' && p.totalLODCount > 1) {
            // Report what is ON SCREEN, not what the loader has fetched — the
            // same rule the substitutive branch below already follows. The
            // loader cursor advances on arrival, so a node stalled after a
            // failed or superseded commit would otherwise read as complete.
            const onScreen = committed?.get(path);
            out.set(path, {
              kind: 'additive',
              loaded: onScreen ?? p.loadedLODCount ?? 0,
              total: p.totalLODCount,
              refining: p.hasMoreLODs === true,
              ...heldEntry(deps.refinementHold?.(path)),
              lastAllResident: p.lastAllResident,
              energy:
                typeof p.committedEnergyFraction === 'number'
                  ? p.committedEnergyFraction
                  : undefined,
            });
          }
        }
      }

      // Substitutive LOD groups: active level + selector mode from the registry.
      const reg = deps.lodGroupRegistry;
      if (reg) {
        for (const entry of reg.list()) {
          const selector =
            entry.selectorMode === 'auto' ? 'auto' : `locked L${entry.selectorMode.lockLevel + 1}`;
          out.set(entry.path, {
            kind: 'lod',
            levelCount: entry.children.length,
            // Report the level actually ON SCREEN, not the selector's aspiration
            // — during a slice scrub the registry shows a coarser fresh level
            // (``displayedChildIndex``) while ``activeChildIndex`` points at the
            // stale fine level being reloaded. ``?? activeChildIndex`` preserves
            // pre-evaluation behaviour (the two are equal until the first frame).
            activeLevel: entry.displayedChildIndex ?? entry.activeChildIndex,
            selector,
          });
        }
      }

      // Partition groups: the structural part count is static even though
      // per-frame frustum selection may hide individual parts.
      for (const part of deps.partitionGroups ?? []) {
        out.set(part.path, { kind: 'partition', partCount: part.partCount });
      }

      return out;
    },
  };
}
