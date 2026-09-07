/**
 * `isLoadActivity` — the adaptive-DPR controller's suppression predicate.
 * Every arm must suppress on its own: the audit's misattribution came from
 * the refinement drain, which the old update-lock-only predicate missed.
 */
import { describe, expect, it } from 'vitest';

import {
  buildLoadActivityPredicate,
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

describe('buildLoadActivityPredicate', () => {
  it('reads the live loader each tick and treats a missing loader as quiet', () => {
    resetLoadTimeline();
    markLoad('loadStart');
    noteRefinementComplete();
    let loader: {
      isUpdateInProgress(): boolean;
      lodGroupRegistry?: { isAnyLevelLoading(): boolean } | null;
    } | null = null;
    let loadPass = false;
    const predicate = buildLoadActivityPredicate({
      getDefaultLoader: () => loader,
      isAnyLoadPassInProgress: () => loadPass,
    });
    expect(predicate()).toBe(false); // no loader yet, drain complete
    loader = { isUpdateInProgress: () => true };
    expect(predicate()).toBe(true);
    loader = {
      isUpdateInProgress: () => false,
      lodGroupRegistry: { isAnyLevelLoading: () => true },
    };
    expect(predicate()).toBe(true);
    loader = { isUpdateInProgress: () => false, lodGroupRegistry: null };
    expect(predicate()).toBe(false);
    loadPass = true;
    expect(predicate()).toBe(true);
    loadPass = false;
    markLoad('loadStart'); // a new load: refinement not complete again
    expect(predicate()).toBe(true);
    resetLoadTimeline();
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
