/**
 * Direct tests for the shared prefix-lineage helper (types/prefix-lineage.ts).
 *
 * The loader-integration behavior (forward-chaining, view-change reset,
 * memoized no-op identity) is covered per geometry in the progressive-loader
 * suites; the commit-side gate consumption in the commit suites. This file
 * pins the module's own contracts — in particular the RETENTION CONTRACT:
 * a WeakMap holds its value strongly while the key is reachable, so an
 * uncapped forward chain would pin every intermediate concat of a generation
 * (~(n−1)/2 × the final CPU arrays on an n-level ladder).
 */

import { describe, it, expect } from 'vitest';
import { getPrefixParent, setPrefixParent } from '../../../types/prefix-lineage';

describe('prefix-lineage', () => {
  it('records and reads a parent link', () => {
    const parent = { id: 'p' };
    const child = { id: 'c' };
    setPrefixParent(child, parent);
    expect(getPrefixParent(child)).toBe(parent);
    expect(getPrefixParent(parent)).toBeUndefined();
  });

  it('a null parent clears any stale entry (post-reset first concat)', () => {
    const parent = { id: 'p' };
    const child = { id: 'c' };
    setPrefixParent(child, parent);
    setPrefixParent(child, null);
    expect(getPrefixParent(child)).toBeUndefined();
  });

  it('caps the chain at depth 1: linking a child deletes the parent-own entry', () => {
    // Without the cap, committedData pinning c3 would transitively retain
    // c2 AND c1 (WeakMap values are strong while their key lives) — the
    // whole generation's ladder of concat results.
    const c1 = { id: 1 };
    const c2 = { id: 2 };
    const c3 = { id: 3 };
    setPrefixParent(c2, c1);
    setPrefixParent(c3, c2);
    expect(getPrefixParent(c3)).toBe(c2); // latest link intact
    expect(getPrefixParent(c2)).toBeUndefined(); // grandparent link released
  });

  it('self-link does not delete the fresh entry', () => {
    const a = { id: 'a' };
    setPrefixParent(a, a);
    expect(getPrefixParent(a)).toBe(a);
  });
});
