/**
 * Leaf-loader error dispatch — the partial-scene-resilience layer.
 *
 * `loadPoints` / `loadLines` / `loadGSplats` may throw `LoaderError`
 * with a `kind` from { Network, Decode, Validation, Unexpected }.
 * `loadLeafNode` catches the LoaderError, logs according to the kind, and
 * returns `null` so the failing leaf doesn't take down the scene — its
 * siblings still render.
 *
 * User-facing notification is deliberately NOT done here: it belongs to the
 * end-of-load aggregate (`loaders/failure-report.ts`). Per-node toasts could not
 * work — leaves load sequentially and the toast surface holds one message at a
 * time, so N failures showed a single toast naming the LAST path, the least
 * useful one, while `Network`-kind failures never toasted at all. The aggregate
 * names every failed path once and covers every kind.
 *
 * Validation skips (missing attr, ndim mismatch, etc.) `return null`
 * from the loader without throwing; only genuine errors throw
 * LoaderError. Anything that is not a LoaderError is re-thrown so the
 * caller can decide on it.
 */

import type * as THREE from 'three';
import { log, Modules } from '../../../utils/log';

export type LoaderErrorKind = 'Network' | 'Decode' | 'Validation' | 'Unexpected';

/**
 * Thrown by `loadPoints` / `loadLines` / `loadGSplats` when a node
 * fails to load for an unexpected reason. The `kind` field tells
 * `loadSceneNodes` how to handle the failure.
 *
 * Note: validation skips (missing attr, ndim mismatch, etc.) still
 * `return null` from the loader — only genuine errors throw this.
 */
export class LoaderError extends Error {
  constructor(
    readonly kind: LoaderErrorKind,
    readonly path: string,
    cause: unknown
  ) {
    const causeMsg = cause instanceof Error ? cause.message : String(cause);
    super(`${kind} loading ${path}: ${causeMsg}`);
    this.name = 'LoaderError';
    this.cause = cause;
  }
}

/** Heuristic classifier for raw thrown errors. */
export function classifyLoaderError(error: unknown): LoaderErrorKind {
  if (!(error instanceof Error)) return 'Unexpected';
  if (error.name === 'AbortError') return 'Network';
  const msg = error.message.toLowerCase();
  if (
    msg.includes('fetch') ||
    msg.includes('network') ||
    msg.includes('timeout') ||
    msg.includes('http ')
  ) {
    return 'Network';
  }
  if (
    msg.includes('decode') ||
    msg.includes('parse') ||
    msg.includes('invalid') ||
    msg.includes('corrupt')
  ) {
    return 'Decode';
  }
  if (msg.includes('validation') || msg.includes('expected') || msg.includes('required')) {
    return 'Validation';
  }
  return 'Unexpected';
}

/**
 * Run a leaf-node loader, dispatching {@link LoaderError} by kind so a
 * single failing node doesn't take the whole scene down. Re-throws
 * anything that isn't a LoaderError.
 */
export async function loadLeafNode<T extends THREE.Object3D>(
  load: () => Promise<T | null>,
  path: string
): Promise<T | null> {
  try {
    return await load();
  } catch (error) {
    if (!(error instanceof LoaderError)) throw error;
    const causeStack = error.cause instanceof Error ? error.cause.stack : undefined;
    switch (error.kind) {
      case 'Network':
        log.warning(Modules.SCENE_LOADER, `Network error loading ${path}: ${error.message}`);
        break;
      case 'Decode':
      case 'Validation':
      case 'Unexpected':
        log.error(Modules.SCENE_LOADER, `Failed to load ${path}: ${error.message}`);
        if (causeStack) log.error(Modules.SCENE_LOADER, `Stack trace for ${path}`, causeStack);
        break;
    }
    return null;
  }
}
