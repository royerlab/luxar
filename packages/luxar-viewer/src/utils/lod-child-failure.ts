/**
 * Keep every lazy-child failure transition paired with status invalidation.
 * Shared by the data loader and scene registry without crossing runtime layers.
 *
 * @module utils/lod-child-failure
 */
import { bumpFailedLoadsVersion } from './failed-loads-version';

/** Failure-latch fields shared by lazy LOD children. */
export interface LazyChildFailureState {
  permanentlyFailed?: boolean;
  failureReason?: string;
}

/** Latch a lazy child failure and invalidate failure-status consumers. */
export function latchChildFailure(child: LazyChildFailureState, reason: string): void {
  child.permanentlyFailed = true;
  child.failureReason = reason;
  bumpFailedLoadsVersion();
}

/** Clear a lazy child failure on explicit retry. */
export function clearChildFailure(child: LazyChildFailureState): void {
  if (child.permanentlyFailed) bumpFailedLoadsVersion();
  child.permanentlyFailed = false;
  child.failureReason = undefined;
}
