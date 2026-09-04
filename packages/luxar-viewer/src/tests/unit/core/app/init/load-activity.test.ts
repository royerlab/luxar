/**
 * `isLoadActivity` — the adaptive-DPR controller's suppression predicate.
 * Every arm must suppress on its own: the audit's misattribution came from
 * the refinement drain, which the old update-lock-only predicate missed.
 */
import { describe, expect, it } from 'vitest';

import {
  isLoadActivity,
  refinementCompleteFromTimeline,
  type LoadActivityDeps,
} from '../../../../../core/app/init/load-activity';
import {
  markLoad,
  noteRefinementComplete,
  resetLoadTimeline,
} from '../../../../../profiling/load-timeline';

function deps(overrides: Partial<LoadActivityDeps> = {}): LoadActivityDeps {
  return {
    isUpdateInProgress: () => false,
    isAnyLoadPassInProgress: () => false,
    isAnyLodLevelLoading: () => false,
    isRefinementComplete: () => true,
    ...overrides,
  };
}

describe('isLoadActivity', () => {
  it('is false only when every source is quiet', () => {
    expect(isLoadActivity(deps())).toBe(false);
  });

  it.each<[string, Partial<LoadActivityDeps>]>([
    ['update lock held', { isUpdateInProgress: () => true }],
    ['a load pass in flight', { isAnyLoadPassInProgress: () => true }],
    ['a lazy LOD level loading', { isAnyLodLevelLoading: () => true }],
    ['the refinement drain still running', { isRefinementComplete: () => false }],
  ])('suppresses while %s', (_label, overrides) => {
    expect(isLoadActivity(deps(overrides))).toBe(true);
  });
});

describe('refinementCompleteFromTimeline', () => {
  it('follows the load timeline: false from loadStart until the drain completes', () => {
    resetLoadTimeline();
    markLoad('loadStart');
    expect(refinementCompleteFromTimeline()).toBe(false);
    noteRefinementComplete();
    expect(refinementCompleteFromTimeline()).toBe(true);
    // A new load resets it.
    markLoad('loadStart');
    expect(refinementCompleteFromTimeline()).toBe(false);
    resetLoadTimeline();
  });
});
