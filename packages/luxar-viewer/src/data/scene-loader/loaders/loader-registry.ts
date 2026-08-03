/**
 * LoaderRegistry - Manages lifecycle of geometry loaders.
 *
 * Extracted from SceneLoader to reduce God Object complexity. Holds every
 * loader bucketed by {@link GeometryKind}, plus failure tracking, and provides
 * registration, lookup, disposal and retry-budget methods.
 *
 * Registration/lookup/disposal are written **once** against the keyed store
 * rather than repeated per geometry type; the `registerPointsLoader`-style
 * methods and the `loaders` / `linesLoaders` / `gsplatLoaders` accessors are
 * typed conveniences over it, preserved so call sites read naturally.
 *
 * @module data/loader-registry
 */

import type { DataLoader, GeometryKind } from '../../data-loader-types';
import { GEOMETRY_TYPES } from '../../../types/format-contract';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';

import { log, Modules } from '../../../utils/log';
import { classifyLoaderError, type LoaderErrorKind } from '../nodes/load-leaf-error-dispatch';

/**
 * Which loader interface belongs to which geometry kind.
 *
 * The three loader interfaces are structurally distinct, so this mapping is
 * what keeps the kind-keyed store honest: `register`/`loadersOf` are generic in
 * `K`, so `register('points', path, someLinesLoader)` is a compile error rather
 * than a silent mis-route through the wrong update path.
 *
 * Every {@link GeometryKind} must appear here. That is enforced — not merely
 * asked for — by {@link AnyDataLoader} indexing this type with the full
 * `GeometryKind` union: a kind added to `contract.yaml` without a loader entry
 * above fails to compile with `TS2339: Property '<kind>' does not exist on type
 * 'LoaderByKind'` at that indexed access, plus `TS2536` inside each generic
 * accessor below.
 *
 * Note the enforcement can NOT be written as `interface LoaderByKind extends
 * Record<GeometryKind, …>` — an interface *inherits* members it does not
 * redeclare, so a new kind would silently pick up the permissive base type
 * instead of erroring.
 */
export type LoaderByKind = {
  points: DataLoader;
  lines: LinesDataLoader;
  gsplats: GSplatsDataLoader;
};

/**
 * Any geometry loader — derived from {@link LoaderByKind} rather than listing
 * the three interfaces again, so the two cannot drift apart.
 */
export type AnyDataLoader = LoaderByKind[GeometryKind];

/**
 * Error information tracked for failed loaders.
 */
export interface FailedLoaderInfo {
  error: Error;
  timestamp: number;
  retryCount: number;
  /**
   * Automatic (connectivity-triggered) retry attempts charged against this
   * path. Distinct from `retryCount`, which logs EVERY failure (update
   * sweeps + manual retries included). The automatic budget must not be
   * consumed by ordinary interaction, or a scene that failed a few slices
   * offline would be past the cap before `online` ever fires — so only
   * `markAutoRetryAttempt` (called by the connectivity retry) bumps this.
   */
  autoRetryCount: number;
  /**
   * Classified cause. Persisted so retry policy can tell a transient failure
   * from a deterministic one — previously the kind was computed for logging and
   * then thrown away, so a WASM trap was retried on every reconnect exactly like
   * a 503.
   */
  kind: LoaderErrorKind;
}

/**
 * Automatic (connectivity-triggered) retries a single path gets before it is
 * left to the monitor banner and a manual Retry.
 *
 * Mirrors `MAX_CONSECUTIVE_REFINEMENT_FAILURES` in `progressive/refinement.ts`,
 * for the same reason: without a cap, a path that fails for a reason
 * connectivity cannot fix is re-fetched on every `online` transition forever. A
 * permanently-404 chunk classifies as `Network`, so the kind filter alone does
 * not bound it. `autoRetryCount` starts at 0 and is charged only by
 * `markAutoRetryAttempt` on each connectivity-triggered attempt, so this allows
 * exactly this many automatic attempts regardless of how many times ordinary
 * update sweeps or manual retries recorded the same failure.
 */
export const MAX_AUTO_RETRY_ATTEMPTS = 3;

/**
 * Registry that manages all geometry loaders (Points, Lines, GSplats)
 * and tracks loading failures for retry/recovery.
 */
export class LoaderRegistry {
  /**
   * Every loader, bucketed by geometry kind, path → loader.
   *
   * One bucket per {@link GEOMETRY_TYPES} entry, created up front so
   * {@link loadersOf} never has to handle a missing bucket. The kind-specific
   * accessors below are thin views onto these same `Map` objects — callers that
   * hold `registry.loaders` and mutate it directly are mutating this store, as
   * they always were.
   */
  private readonly byKind: ReadonlyMap<GeometryKind, Map<string, AnyDataLoader>> = new Map(
    GEOMETRY_TYPES.map((kind) => [kind as GeometryKind, new Map<string, AnyDataLoader>()])
  );

  /**
   * Loaders of one geometry kind, keyed by scene path, narrowed to that kind's
   * loader interface via {@link LoaderByKind}.
   *
   * The single cast in this class: the heterogeneous store cannot express
   * "bucket `K` holds `LoaderByKind[K]`" internally, so the invariant is
   * enforced at this boundary and every caller above it is fully typed.
   */
  loadersOf<K extends GeometryKind>(kind: K): Map<string, LoaderByKind[K]> {
    const bucket = this.byKind.get(kind);
    if (!bucket) throw new Error(`LoaderRegistry: unknown geometry kind '${kind}'`);
    return bucket as Map<string, LoaderByKind[K]>;
  }

  /** Points loaders indexed by scene path */
  get loaders(): Map<string, DataLoader> {
    return this.loadersOf('points');
  }

  /** Lines loaders indexed by scene path */
  get linesLoaders(): Map<string, LinesDataLoader> {
    return this.loadersOf('lines');
  }

  /** GSplats loaders indexed by scene path */
  get gsplatLoaders(): Map<string, GSplatsDataLoader> {
    return this.loadersOf('gsplats');
  }

  /** Error tracking for failed loaders */
  readonly failedLoaders = new Map<string, FailedLoaderInfo>();

  // ---------------------------------------------------------------------------
  // Registration
  // ---------------------------------------------------------------------------

  /**
   * Register a loader for a given path. The loader type must match the kind —
   * `register('points', p, someLinesLoader)` is a compile error.
   */
  register<K extends GeometryKind>(kind: K, path: string, loader: LoaderByKind[K]): void {
    this.loadersOf(kind).set(path, loader);
  }

  /**
   * Drop a single loader so it no longer participates in scene-wide
   * ``updateView`` sweeps. Defensive: lazy substitutive LOD levels are never
   * registered in the first place (they stay out of the sweep by design — see
   * ``load-lod-group-node.ts``; the registry drives their reloads), so on the
   * lazy-release path this is a no-op. It exists so a future path that DOES
   * register such a loader cannot leak it into the sweep after its geometry was
   * released. The loader object itself stays alive in the lod_group's
   * ``ensureLoaded`` closure for reload.
   */
  unregister<K extends GeometryKind>(kind: K, path: string): void {
    this.loadersOf(kind).delete(path);
  }

  /**
   * Register a points loader for a given path.
   */
  registerPointsLoader(path: string, loader: DataLoader): void {
    this.register('points', path, loader);
  }

  /**
   * Register a lines loader for a given path.
   */
  registerLinesLoader(path: string, loader: LinesDataLoader): void {
    this.register('lines', path, loader);
  }

  /**
   * Register a gsplats loader for a given path.
   */
  registerGSplatsLoader(path: string, loader: GSplatsDataLoader): void {
    this.register('gsplats', path, loader);
  }

  /** Peer of {@link unregister}, kept for call-site readability. */
  unregisterGSplatsLoader(path: string): void {
    this.unregister('gsplats', path);
  }

  /** Peer of {@link unregister}, kept for call-site readability. */
  unregisterPointsLoader(path: string): void {
    this.unregister('points', path);
  }

  /** Peer of {@link unregister}, kept for call-site readability. */
  unregisterLinesLoader(path: string): void {
    this.unregister('lines', path);
  }

  // ---------------------------------------------------------------------------
  // Lookup
  // ---------------------------------------------------------------------------

  /** Total number of loaders across all geometry types. */
  get totalLoaderCount(): number {
    let total = 0;
    for (const bucket of this.byKind.values()) total += bucket.size;
    return total;
  }

  /** Whether there are any registered loaders. */
  get hasLoaders(): boolean {
    for (const bucket of this.byKind.values()) if (bucket.size > 0) return true;
    return false;
  }

  /**
   * Find which geometry kind owns a given path, or null if none does.
   *
   * Nothing prevents the same path from being registered under two kinds, so
   * this has a documented precedence: points > lines > gsplats. That order is
   * now the **bucket insertion order**, which comes from `GEOMETRY_TYPES` —
   * i.e. from the order of `geometry_types` in `format-contract/contract.yaml`.
   * Reordering that list would silently reorder this precedence; the
   * "checks points first when a path collides" test is the guard.
   */
  getLoaderType(path: string): GeometryKind | null {
    for (const [kind, bucket] of this.byKind) {
      if (bucket.has(path)) return kind;
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Error tracking
  // ---------------------------------------------------------------------------

  /**
   * Record a loader failure. `kind` defaults to the heuristic classification of
   * `error`; pass it explicitly when the caller already computed one.
   *
   * The single writer for `failedLoaders` — the retry and update-sweep paths
   * route through here rather than calling `.set` inline, which also removes a
   * pre-existing skew where those two baselined `retryCount` at 1 and 0.
   *
   * Bumps the diagnostic `retryCount` but leaves `autoRetryCount` untouched —
   * only `markAutoRetryAttempt` charges the automatic-retry budget.
   */
  recordFailure(path: string, error: Error, kind?: LoaderErrorKind): void {
    const existing = this.failedLoaders.get(path);
    const retryCount = existing ? existing.retryCount + 1 : 0;
    this.failedLoaders.set(path, {
      error,
      timestamp: Date.now(),
      retryCount,
      autoRetryCount: existing ? existing.autoRetryCount : 0,
      kind: kind ?? classifyLoaderError(error),
    });
  }

  /**
   * Charge one automatic (connectivity-triggered) retry attempt against a
   * path. No-op if the path has no failure record. Kept separate from
   * `recordFailure` so update sweeps and manual retries — which also record
   * failures — cannot drain the automatic budget gated by
   * {@link autoRetryablePaths}.
   */
  markAutoRetryAttempt(path: string): void {
    const info = this.failedLoaders.get(path);
    if (info) info.autoRetryCount += 1;
  }

  /**
   * Clear failure tracking for a specific path.
   */
  clearFailure(path: string): void {
    this.failedLoaders.delete(path);
  }

  /**
   * Get information about failed loaders (read-only view).
   */
  getFailedLoaders(): ReadonlyMap<string, FailedLoaderInfo> {
    return this.failedLoaders;
  }

  /** Whether there are any failed loaders. */
  hasFailures(): boolean {
    return this.failedLoaders.size > 0;
  }

  /**
   * Paths an AUTOMATIC retry should attempt: a transient (`Network`) cause that
   * is still under {@link MAX_AUTO_RETRY_ATTEMPTS}.
   *
   * A manual Retry deliberately ignores both filters — the user pressing the
   * button is new information (they may have just fixed the server), and a
   * deterministic failure is still worth one more look on request.
   */
  autoRetryablePaths(): string[] {
    const paths: string[] = [];
    for (const [path, info] of this.failedLoaders) {
      if (info.kind === 'Network' && info.autoRetryCount < MAX_AUTO_RETRY_ATTEMPTS) {
        paths.push(path);
      }
    }
    return paths;
  }

  /** Whether any failed loader is worth an automatic retry. */
  hasAutoRetryableFailures(): boolean {
    return this.autoRetryablePaths().length > 0;
  }

  /**
   * Clear all failure tracking.
   */
  clearAllFailures(): void {
    const count = this.failedLoaders.size;
    this.failedLoaders.clear();
    if (count > 0) {
      log.info(Modules.SCENE_LOADER, `Cleared ${count} failed loader(s) from tracking`);
    }
  }

  // ---------------------------------------------------------------------------
  // Disposal
  // ---------------------------------------------------------------------------

  /**
   * Dispose all loaders and clear all maps.
   */
  disposeAll(): void {
    for (const bucket of this.byKind.values()) {
      for (const loader of bucket.values()) {
        loader.dispose();
      }
      bucket.clear();
    }
  }
}
