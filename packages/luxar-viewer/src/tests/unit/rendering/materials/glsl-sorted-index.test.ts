/**
 * Selection semantics of the shared GLSL ordering-index snippet.
 *
 * `GLSL_SORTED_INDEX` is the WebGL half of the double-buffered
 * draw-slot → storage-slot indirection; `sortedIndexNode` in
 * `tsl-helpers.ts` is its WebGPU twin. They must agree:
 *
 *   slot 0 -> aSortedIndex      slot 1 -> aSortedIndexB
 *
 * COVERAGE NOTE — why this file exists. The TSL twin's expression is
 * pinned textually by all 24 `__codegen__/*.vertex.glsl.txt` snapshots.
 * NOTHING pinned the GLSL one, and no rendering test exercises either at
 * slot 1: the TSL parity harness drains a finished ordering back onto the
 * FRONT buffer (`drainOrderingOntoFrontBuffer`) precisely so the selector
 * can stay at its default 0. So swapping the two branches here — the
 * exact silent inversion that would make every WebGL node draw the stale
 * buffer after its first flip — passed the entire suite.
 */

import { describe, it, expect } from 'vitest';
import { GLSL_SORTED_INDEX } from '../../../../rendering/materials/_shared/glsl-lib';

/**
 * Evaluate the snippet's ternary the way GLSL would, by parsing the
 * branch identifiers out of the source. Matching the RETURNED NAMES (not
 * the whole string) keeps this robust to formatting while still failing
 * if the branches are swapped.
 */
function branches(): { cond: string; whenTrue: string; whenFalse: string } {
  const m = /return\s+(\w+)\s*==\s*1\s*\?\s*(\w+)\s*:\s*(\w+)\s*;/.exec(GLSL_SORTED_INDEX);
  if (!m) throw new Error(`luxarSortedIndex() is not the expected ternary:\n${GLSL_SORTED_INDEX}`);
  return { cond: m[1], whenTrue: m[2], whenFalse: m[3] };
}

describe('GLSL_SORTED_INDEX', () => {
  it('declares both ordering attributes and the runtime selector uniform', () => {
    expect(GLSL_SORTED_INDEX).toContain('in uint aSortedIndex;');
    expect(GLSL_SORTED_INDEX).toContain('in uint aSortedIndexB;');
    // A UNIFORM, never a #define: a slot flip must not recompile.
    expect(GLSL_SORTED_INDEX).toContain('uniform int uSortedIndexSlot;');
    expect(GLSL_SORTED_INDEX).not.toContain('#define');
  });

  it('selects the BACK buffer on slot 1 and the front buffer otherwise', () => {
    const { cond, whenTrue, whenFalse } = branches();
    expect(cond).toBe('uSortedIndexSlot');
    // The load-bearing assertion: branches in this order, not swapped.
    expect(whenTrue).toBe('aSortedIndexB');
    expect(whenFalse).toBe('aSortedIndex');
  });
});

// The TSL twin is NOT re-asserted here. Re-deriving `a*(1-slot) + b*slot`
// in JS would only test a hand-written mirror of it, not the shipped
// node graph — a tautology. The real expression is pinned verbatim in all
// 24 `__codegen__/*.vertex.glsl.txt` vertex snapshots, e.g.
//   ( ( int( aSortedIndex ) * ( 1 - int( nodeUniformN ) ) )
//     + ( int( aSortedIndexB ) * int( nodeUniformN ) ) )
// which is the same 0 -> front / 1 -> back mapping asserted above.
