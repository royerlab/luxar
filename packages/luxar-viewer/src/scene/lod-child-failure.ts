import type { LODGroupChild } from './lod-group-registry';
import { bumpFailedLoadsVersion } from '../utils/failed-loads-version';

/** Latch a lazy child failure and invalidate failure-status consumers. */
export function latchChildFailure(child: LODGroupChild, reason: string): void {
  child.permanentlyFailed = true;
  child.failureReason = reason;
  bumpFailedLoadsVersion();
}

/** Clear a lazy child failure on explicit retry. */
export function clearChildFailure(child: LODGroupChild): void {
  if (child.permanentlyFailed) bumpFailedLoadsVersion();
  child.permanentlyFailed = false;
  child.failureReason = undefined;
}
