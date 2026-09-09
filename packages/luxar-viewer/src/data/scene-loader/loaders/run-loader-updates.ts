/**
 * Loader eligibility helpers and per-geometry update-loop scaffolding.
 *
 * Points / Lines / GSplats branches differ only in the type-specific
 * work (deriveNodeViewState, call loader.updateView, post-process,
 * setMetadata, return staged) — the surrounding try/catch with
 * failedLoaders bookkeeping + retryCount tracking, profiler dispatch,
 * archive-fault hoisting, and Promise.all are identical and live here.
 */

import { log, Modules } from '../../../utils/log';
import type { UpdateProfiler, UpdateSession } from '../../../profiling/update-profiler';
import type { ViewStateQueue } from '../view-state/view-state-queue';
import type { LoaderRegistry } from './loader-registry';
import { isAbortError } from '../../loaders/abort-error';
import { archiveFaultFrom, type ArchiveFaultError } from '../../../cache/chunk-source';
import { tryRollbackToPassStart } from '../../loaders/progressive/pass-rollback';
import type * as THREE from 'three';

const NOOP_SESSION: UpdateSession = {
  begin: () => NOOP_SESSION,
  end: () => {},
  setMetadata: () => {},
  markSkipped: () => {},
};

/** Whether an object is eligible for foreground or background loading. */
export function isObjectLoadEligible(object: THREE.Object3D | null | undefined): boolean {
  while (object) {
    if (object.userData.partitionFrustumVisible === false) return false;
    if (object.userData.layerVisible === false) return false;
    object = object.parent;
  }
  return true;
}

/** Resolve a loader path and test its foreground/background load eligibility. */
export function isLoaderPathEligible(root: THREE.Object3D | null, path: string): boolean {
  return isObjectLoadEligible(root?.getObjectByName(path));
}

/**
 * Whether ``path`` is one of ``targets`` or nested under one. Node paths are
 * ``/``-separated, so ``/a/b`` is under ``/a`` but ``/a/bc`` is not.
 */
export function isUnderAny(path: string, targets: ReadonlySet<string>): boolean {
  if (targets.has(path)) return true;
  for (const target of targets) {
    const prefix = target.endsWith('/') ? target : `${target}/`;
    if (path.startsWith(prefix)) return true;
  }
  return false;
}

/** Resolve loader objects once and copy paths eligible for background loading. */
export function resolveLoadEligibleLoaders<TLoader>(
  root: THREE.Object3D | null,
  loaders: Map<string, TLoader>
): {
  loaders: Map<string, TLoader>;
  objects: Map<string, THREE.Object3D | undefined>;
} {
  const eligibleLoaders = new Map<string, TLoader>();
  const objects = new Map<string, THREE.Object3D | undefined>();
  for (const [path, loader] of loaders) {
    const object = root?.getObjectByName(path);
    if (!isObjectLoadEligible(object)) continue;
    eligibleLoaders.set(path, loader);
    objects.set(path, object);
  }
  return { loaders: eligibleLoaders, objects };
}

/**
 * Run a per-loader update task for every entry in `loaders`, recording
 * failures into `failedLoaders` and forgetting the predictive-prefetch
 * baseline for failed, culled, or hidden paths. Loaders outside a targeted
 * partition resync (`isResyncTarget`) are skipped WITHOUT forgetting their
 * baseline — nothing about their view changed. Archive faults are excluded
 * from that bookkeeping and reported once after every task has settled.
 */
export async function runLoaderUpdates<TLoader, TStaged>(
  loaders: Map<string, TLoader>,
  loaderType: 'Points' | 'Lines' | 'GSplats' | 'Mesh',
  updateFn: (path: string, loader: TLoader, session: UpdateSession) => Promise<TStaged | null>,
  ctx: {
    profiler: UpdateProfiler | null;
    viewStateQueue: ViewStateQueue;
    registry: Pick<LoaderRegistry, 'failedLoaders' | 'recordFailure'>;
    shouldUpdatePath?: (path: string) => boolean;
    /**
     * Targeted partition resync: `false` ⇒ this loader is not under any
     * re-entering part, skip it without dropping its prefetch baseline —
     * checked BEFORE the culled check, so a culled non-target keeps its
     * baseline too.
     */
    isResyncTarget?: (path: string) => boolean;
    /** Called at most once per sweep, after Promise.all, with the first archive fault. */
    onArchiveFault: (fault: ArchiveFaultError) => void;
  }
): Promise<Array<{ staged: TStaged | null; session: UpdateSession }>> {
  let archiveFault: ArchiveFaultError | undefined;
  const tasks = Array.from(loaders.entries()).map(async ([path, loader]) => {
    // Open a top-level session per node and keep it alive across the
    // atomic commit stage so the per-node "Update Buffers" child entry
    // nests under this session. The caller is responsible for calling
    // session.end() once the commit has run.
    const session = ctx.profiler
      ? ctx.profiler.beginTopLevel(`${loaderType} (${path})`)
      : NOOP_SESSION;
    // Resync-target check FIRST: a targeted resync must leave every non-target
    // loader untouched, culled or not. Rising edges fire precisely when many
    // parts are culled, and running the culled check first would drop those
    // parts' prefetch baselines on every rising edge.
    if (ctx.isResyncTarget?.(path) === false) {
      session.markSkipped('outside partition resync targets');
      return { staged: null, session };
    }
    if (ctx.shouldUpdatePath?.(path) === false) {
      session.markSkipped('loader path culled or under a hidden layer');
      ctx.viewStateQueue.forgetPath(path);
      return { staged: null, session };
    }
    try {
      const staged = await updateFn(path, loader, session);
      return { staged, session };
    } catch (error) {
      // Superseded, not failed: a newer view-state aborted this in-flight
      // update (per-update AbortSignal). zarrita's chunk reads throw a
      // DOMException named 'AbortError'; the worker pool throws
      // 'WorkerAbortError'. Either way this path was abandoned on purpose —
      // do NOT record a failure or drop the prefetch baseline. The winning
      // update re-derives and reloads this path. Returning staged:null means
      // the atomic commit leaves this node's geometry untouched.
      if (isAbortError(error)) {
        return { staged: null, session };
      }

      const unwound = tryRollbackToPassStart(
        loader as TLoader & { rollbackToPassStart?: () => number }
      );

      const fault = archiveFaultFrom(error);
      if (fault) {
        archiveFault ??= fault;
        return { staged: null, session };
      }

      // Predictive prefetch is keyed by the previous successful
      // derived view-state for this path. If the demand update
      // fails, discard that baseline so the next success
      // re-baselines instead of extrapolating across a stale/error
      // gap and warming irrelevant chunks.
      ctx.viewStateQueue.forgetPath(path);

      // Routed through `recordFailure` so the classified `kind` is persisted
      // (retry policy consumes it) and the counter has a single owner.
      ctx.registry.recordFailure(path, error as Error);
      const retryCount = ctx.registry.failedLoaders.get(path)?.retryCount ?? 0;
      const lcType = loaderType === 'Points' ? '' : `${loaderType.toLowerCase()} `;
      const unwindSuffix = unwound > 0 ? ` (unwound ${unwound} level(s))` : '';
      log.error(
        Modules.SCENE_LOADER,
        `Failed to update ${lcType}${path} (attempt ${retryCount + 1}): ${(error as Error).message}` +
          unwindSuffix
      );
      return { staged: null, session };
    }
  });
  const results = await Promise.all(tasks);
  if (archiveFault) ctx.onArchiveFault(archiveFault);
  return results;
}
