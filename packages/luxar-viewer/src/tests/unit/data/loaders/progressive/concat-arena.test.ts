/**
 * Unit tests for data/loaders/progressive/concat-arena.ts — the growable
 * arena primitives behind the amortized progressive-ladder concat (perf
 * lever L2) and the per-result append-span registry.
 *
 * The per-geometry arena classes (GSplats/Points/Lines LadderArena) are
 * covered by concat-arena-equivalence.test.ts; this file pins the shared
 * building blocks: growth/trim capacity policy, prefix-stable views,
 * copy-work accounting, the all-or-nothing optional field, and the span
 * WeakMap.
 */

import { describe, it, expect } from 'vitest';
import {
  ARENA_GROWTH_FACTOR,
  ArenaField,
  OptionalLadderField,
  getAppendSpan,
  setAppendSpan,
  validateLadderFieldDtype,
} from '../../../../../data/loaders/progressive/concat-arena';

describe('ArenaField', () => {
  it('appends at a running offset with perItem striding', () => {
    const f = new ArenaField(Float32Array, 3, 0);
    f.append(new Float32Array([1, 2, 3, 4, 5, 6]), 2);
    f.append(new Float32Array([7, 8, 9]), 1);
    expect(f.lengthElements).toBe(3);
    expect(Array.from(f.view())).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('appendFill writes a constant range; appendWith uses the raw base entry', () => {
    const f = new ArenaField(Uint32Array, 2, 0);
    f.appendFill(7, 2);
    f.appendWith(1, (buf, base) => {
      buf[base] = 100;
      buf[base + 1] = 200;
    });
    expect(Array.from(f.view())).toEqual([7, 7, 7, 7, 100, 200]);
  });

  it('grows by ARENA_GROWTH_FACTOR (or to exact need when larger)', () => {
    const f = new ArenaField(Float32Array, 1, 4);
    f.append(new Float32Array(4).fill(1), 4);
    expect(f.capacityElements).toBe(4);
    // +1 element: 1.5× growth dominates the need.
    f.append(new Float32Array([2]), 1);
    expect(f.capacityElements).toBe(Math.ceil(4 * ARENA_GROWTH_FACTOR));
    // A huge append: exact need dominates the 1.5× step.
    f.append(new Float32Array(100).fill(3), 100);
    expect(f.capacityElements).toBe(105);
  });

  it('exact growth (final level) allocates exactly the need', () => {
    const f = new ArenaField(Float32Array, 1, 4);
    f.append(new Float32Array(4).fill(1), 4);
    f.ensureCapacity(5, true);
    expect(f.capacityElements).toBe(5);
  });

  it('prefix stability: earlier views keep their exact bytes across in-place appends AND growth reallocs', () => {
    const f = new ArenaField(Float32Array, 1, 2);
    f.append(new Float32Array([1, 2]), 2);
    const v1 = f.view();
    const v1Copy = v1.slice();
    // In-place append (capacity available after growth): beyond v1's length.
    f.ensureCapacity(10);
    f.append(new Float32Array([3]), 1);
    expect(Array.from(v1)).toEqual(Array.from(v1Copy));
    const v2 = f.view();
    // Growth realloc: v1/v2 keep the OLD buffer untouched.
    f.append(new Float32Array(100).fill(9), 100);
    expect(Array.from(v1)).toEqual(Array.from(v1Copy));
    expect(Array.from(v2.subarray(0, 2))).toEqual(Array.from(v1Copy));
    // The new view's prefix matches the old contents (values, same positions).
    expect(Array.from(f.view().subarray(0, 3))).toEqual([1, 2, 3]);
  });

  it('view() returns a NEW object each call, at byteOffset 0', () => {
    const f = new ArenaField(Float32Array, 1, 4);
    f.append(new Float32Array([1]), 1);
    const a = f.view();
    const b = f.view();
    expect(a).not.toBe(b);
    expect(a.byteOffset).toBe(0);
  });

  it('trimToFit releases slack above the threshold and keeps exact-fit buffers', () => {
    const f = new ArenaField(Float32Array, 1, 100);
    f.append(new Float32Array(10).fill(5), 10);
    expect(f.capacityElements).toBe(100);
    f.trimToFit();
    expect(f.capacityElements).toBe(10);
    expect(Array.from(f.view())).toEqual(new Array(10).fill(5));
    // Second trim is a no-op (no slack).
    const before = f.view().buffer;
    f.trimToFit();
    expect(f.view().buffer).toBe(before);
  });

  it('copy-work accounting: appends tally O(N_k) entries; reallocs tally amortized copies', () => {
    const f = new ArenaField(Float32Array, 1, 0);
    f.append(new Float32Array(4), 4);
    expect(f.stats.appendedEntries).toBe(4);
    const reallocsAfterFirst = f.stats.reallocCopiedEntries;
    f.append(new Float32Array(2), 2);
    // The second append tallies ONLY its own entries...
    expect(f.stats.appendedEntries).toBe(6);
    // ...while any growth copy is accounted separately (4 entries moved).
    expect(f.stats.reallocCopiedEntries - reallocsAfterFirst).toBe(4);
  });
});

describe('validateLadderFieldDtype', () => {
  it('throws the concatRequiredField-style message naming the level and label', () => {
    const parts = [{ v: new Float32Array(1) }, { v: new Uint8Array(1) as unknown as Float32Array }];
    expect(() =>
      validateLadderFieldDtype(parts, 0, (p) => p.v, Float32Array, 'radii')
    ).toThrowError(/concatRequiredField: LOD level 1 carries 'radii' as Uint8Array/);
  });

  it('passes matching ladders and skips levels before `from`', () => {
    const parts = [{ v: new Uint8Array(1) as unknown as Float32Array }, { v: new Float32Array(1) }];
    // from=2: nothing to check even though level 0 mismatches.
    expect(() => validateLadderFieldDtype(parts, 2, (p) => p.v, Float32Array, 'x')).not.toThrow();
  });
});

describe('OptionalLadderField (all-or-nothing)', () => {
  interface Part {
    n: number;
    attr?: Float32Array;
  }
  const get = (p: Part) => p.attr;

  it('appends while every level carries the attribute', () => {
    const opt = new OptionalLadderField<Float32Array>();
    const parts: Part[] = [
      { n: 2, attr: new Float32Array([1, 2]) },
      { n: 1, attr: new Float32Array([3]) },
    ];
    const plan = opt.plan(parts, 0, get, 'attr');
    expect(plan).not.toBe('skip');
    expect(plan).not.toBe('drop');
    opt.apply(plan, 1, 3, false);
    expect(opt.field).not.toBeNull();
  });

  it('drops the field for good when a level lacks it (skip thereafter)', () => {
    const opt = new OptionalLadderField<Float32Array>();
    const parts: Part[] = [{ n: 2, attr: new Float32Array([1, 2]) }, { n: 1 }];
    const plan = opt.plan(parts, 0, get, 'attr');
    expect(plan).toBe('drop');
    opt.apply(plan, 1, 3, false);
    expect(opt.field).toBeNull();
    expect(opt.dropped).toBe(true);
    // A later carrier level cannot resurrect it (the lacking level is still
    // part of the ladder — matches concatOptionalField's `every` semantics).
    parts.push({ n: 1, attr: new Float32Array([9]) });
    expect(opt.plan(parts, 2, get, 'attr')).toBe('skip');
  });

  it('fails fast on a dtype flip with the reference message', () => {
    const opt = new OptionalLadderField<Float32Array>();
    const parts: Part[] = [
      { n: 1, attr: new Float32Array([1]) },
      { n: 1, attr: new Uint8Array([2]) as unknown as Float32Array },
    ];
    expect(() => opt.plan(parts, 0, get, 'attr')).toThrowError(/mixed|carries 'attr'/);
  });
});

describe('append-span registry', () => {
  it('round-trips a span keyed on result identity; unknown objects read undefined', () => {
    const result = { some: 'result' };
    setAppendSpan(result, { fromElement: 100, elementCount: 25 });
    expect(getAppendSpan(result)).toEqual({ fromElement: 100, elementCount: 25 });
    expect(getAppendSpan({ some: 'result' })).toBeUndefined();
  });

  it('carries the lines vertex-space extension', () => {
    const result = {};
    setAppendSpan(result, { fromElement: 10, elementCount: 5, fromVertex: 20, vertexCount: 10 });
    expect(getAppendSpan(result)?.fromVertex).toBe(20);
    expect(getAppendSpan(result)?.vertexCount).toBe(10);
  });
});
