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

import type { LODProgressProvider, LODProgressState } from '../../../types/data-monitor-types';
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
   * build. Partition groups are static `THREE.Group`s with no per-frame
   * selector, so a one-time snapshot is sufficient — they surface in the
   * monitor's scene-graph summary as `kind:'partition'` states. Empty/omitted
   * for scenes without partitions.
   */
  partitionGroups?: ReadonlyArray<{ path: string; partCount: number }>;
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

      // Additive progressive loaders: one per node across all four maps.
      for (const map of deps.loaderMaps) {
        for (const [path, loader] of map) {
          const p = loader as ProgressiveLike;
          if (typeof p.totalLODCount === 'number' && p.totalLODCount > 1) {
            out.set(path, {
              kind: 'additive',
              loaded: p.loadedLODCount ?? 0,
              total: p.totalLODCount,
              refining: p.hasMoreLODs === true,
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

      // Partition groups: static spatial subdivisions (all parts render at
      // once). A one-time snapshot of part counts — no per-frame state.
      for (const part of deps.partitionGroups ?? []) {
        out.set(part.path, { kind: 'partition', partCount: part.partCount });
      }

      return out;
    },
  };
}
