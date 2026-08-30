/**
 * Shared scaffolding for the per-geometry update loops in `updateView`.
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
import type * as THREE from 'three';

const NOOP_SESSION: UpdateSession = {
  begin: () => NOOP_SESSION,
  end: () => {},
  setMetadata: () => {},
  markSkipped: () => {},
};

export function isPartitionPathVisible(root: THREE.Object3D | null, path: string): boolean {
  let object: THREE.Object3D | null | undefined = root?.getObjectByName(path);
  while (object) {
    if (object.userData.partitionFrustumVisible === false) return false;
    object = object.parent;
  }
  return true;
}

export function filterPartitionVisibleLoaders<TLoader>(
  root: THREE.Object3D | null,
  loaders: Map<string, TLoader>
): Map<string, TLoader> {
  return new Map([...loaders].filter(([path]) => isPartitionPathVisible(root, path)));
}

/**
 * Run a per-loader update task for every entry in `loaders`, recording
 * failures into `failedLoaders` and forgetting the predictive-prefetch
 * baseline for failed paths. Archive faults are excluded from that bookkeeping
 * and reported once after every task has settled.
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
    if (ctx.shouldUpdatePath?.(path) === false) {
      session.markSkipped('partition part outside camera frustum');
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
      log.error(
        Modules.SCENE_LOADER,
        `Failed to update ${lcType}${path} (attempt ${retryCount + 1}): ${(error as Error).message}`
      );
      return { staged: null, session };
    }
  });
  const results = await Promise.all(tasks);
  if (archiveFault) ctx.onArchiveFault(archiveFault);
  return results;
}
