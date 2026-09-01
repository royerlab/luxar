/**
 * Regression tests for the lineage retention half of the prefix-lineage
 * contract (#2426).
 *
 * The commit layer's `setPrefixParent(data, null)` was unconditional and
 * correct, but a refinement pass does not always reach a commit — it skips one
 * when processing stages nothing, and when a superseding view-state aborts
 * mid-flight, which happens constantly during ordinary camera motion. On those
 * exits the entry survived, pinning the PREVIOUS CUMULATIVE for as long as the
 * newest concat stayed reachable (indefinitely, via `committedData` and the
 * loader's memo).
 *
 * The cost is depth-dependent, which is what made it hard to see: the pinned
 * parent is `(passes - 1) / passes` of the final payload, so it grows with
 * ladder depth and saturates at a whole redundant copy. Measured in-process on
 * an equal-count ladder climbed one rung per pass: 0.0 / 0.5 / 0.75 / 0.875 at
 * 2 / 4 / 8 / 16 rungs. A two-rung ladder retains nothing because it completes
 * in ONE pass — which is why 4-rung test arms made the fold look partly broken
 * and 2-rung arms made it look perfect, when neither was true.
 */

import { describe, it, expect } from 'vitest';
import {
  getPrefixParent,
  releaseLineageIfUncommitted,
  setPrefixParent,
} from '../../../types/prefix-lineage';

describe('releaseLineageIfUncommitted', () => {
  it('releases the pinned parent when the pass did not commit', () => {
    const parent = { id: 'cumulative-through-rung-3' };
    const child = { id: 'cumulative-through-rung-4' };
    setPrefixParent(child, parent);
    expect(getPrefixParent(child)).toBe(parent);

    releaseLineageIfUncommitted(child, false);

    expect(getPrefixParent(child)).toBeUndefined();
  });

  it('leaves a committed pass alone', () => {
    // The commit layer already cleared it; re-clearing would be harmless but
    // this pins the intent — the helper is for the NO-commit exits only.
    const parent = {};
    const child = {};
    setPrefixParent(child, parent);

    releaseLineageIfUncommitted(child, true);

    expect(getPrefixParent(child)).toBe(parent);
  });

  it('tolerates absent data', () => {
    expect(() => releaseLineageIfUncommitted(null, false)).not.toThrow();
    expect(() => releaseLineageIfUncommitted(undefined, false)).not.toThrow();
  });

  it('is idempotent', () => {
    const child = {};
    setPrefixParent(child, {});
    releaseLineageIfUncommitted(child, false);
    expect(() => releaseLineageIfUncommitted(child, false)).not.toThrow();
    expect(getPrefixParent(child)).toBeUndefined();
  });

  it('unpins a chain of passes so only the newest survives', () => {
    // The depth curve in one assertion: each pass links a new cumulative to the
    // previous one. Without the release, the last uncommitted pass leaves its
    // parent — the whole ladder minus one rung — reachable for as long as the
    // newest result is.
    const cumulatives = [{ n: 1 }, { n: 2 }, { n: 3 }, { n: 4 }];
    for (let i = 1; i < cumulatives.length; i++) {
      setPrefixParent(cumulatives[i], cumulatives[i - 1]);
    }
    const newest = cumulatives[cumulatives.length - 1];
    expect(getPrefixParent(newest)).toBe(cumulatives[2]);

    releaseLineageIfUncommitted(newest, false);

    for (const c of cumulatives) expect(getPrefixParent(c)).toBeUndefined();
  });
});
