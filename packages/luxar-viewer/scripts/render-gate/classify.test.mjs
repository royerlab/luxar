import { describe, expect, it } from 'vitest';

import { classifyRow, scoreControlPick } from './classify.mjs';
import { differingBBox, scoreBlocks, scoreFloatBuffers, scorePickBuffers } from './exactness.mjs';

const W = 4;
const H = 3;

/** A W x H RGBA frame at `value`, with `pixels` (indices) set to `value + 1`. */
function frame(value, pixels = []) {
  const f = new Float32Array(W * H * 4).fill(value);
  for (const p of pixels) f.fill(value + 1, p * 4, p * 4 + 4);
  return f;
}

/** A pick buffer: every pixel carries node 1 / element `element`. */
function pickBuffer(pixels, element = 1) {
  const f = new Float32Array(pixels * 4);
  for (let p = 0; p < pixels; p++) f.set([1, element, 0, 1], p * 4);
  return f;
}

function shot(overrides = {}) {
  return {
    settled: true,
    stable: true,
    bufferStability: { hdr: true, ldr: true },
    counts: { points: 'points:10' },
    camera: { world: [1] },
    width: W,
    height: H,
    hdr: frame(1),
    ldr: frame(0.5),
    pick: null,
    ...overrides,
  };
}

/** Score three shots the way `runExact` does and classify the view. */
function classify({
  a = shot(),
  a2 = shot(),
  b = shot(),
  cls = 'IDENTICAL',
  requiresPick = false,
}) {
  const pickOf = (x, y) =>
    x.pick && y.pick && x.pick.length === y.pick.length ? scorePickBuffers(x.pick, y.pick) : null;
  return classifyRow({
    a,
    a2,
    b,
    aa: scoreFloatBuffers(a.hdr, a2.hdr),
    aaLdr: scoreFloatBuffers(a.ldr, a2.ldr),
    aaPick: scoreControlPick(a.pick, a2.pick),
    hdr: scoreFloatBuffers(a.hdr, b.hdr),
    ldr: scoreFloatBuffers(a.ldr, b.ldr),
    pick: pickOf(a, b),
    blocks: scoreBlocks(a.hdr, b.hdr, W, H),
    cls,
    requiresPick,
  });
}

describe('differingBBox', () => {
  it('is null when nothing differs', () => {
    expect(differingBBox(new Float64Array(W * H), W, H)).toBeNull();
  });

  it('bounds the differing pixels inclusively and counts them', () => {
    const ulp = new Float64Array(W * H);
    ulp[1 * W + 1] = 3;
    ulp[2 * W + 3] = 1;
    expect(differingBBox(ulp, W, H)).toEqual({ count: 2, minX: 1, minY: 1, maxX: 3, maxY: 2 });
  });
});

describe('scoreControlPick', () => {
  it('is null when neither capture has a pick buffer', () => {
    expect(scoreControlPick(null, null)).toBeNull();
  });

  it('records a length mismatch instead of dropping it', () => {
    expect(scoreControlPick(pickBuffer(12), pickBuffer(6))).toEqual({
      lengthMismatch: true,
      first: 48,
      second: 24,
    });
    expect(scoreControlPick(pickBuffer(12), null)).toEqual({
      lengthMismatch: true,
      first: 48,
      second: null,
    });
  });
});

describe('classifyRow', () => {
  const pick = pickBuffer(W * H);
  const table = [
    { name: 'clean A/A, identical candidate', input: {}, status: 'pass', control: false },
    {
      name: 'clean A/A, intended case unchanged',
      input: { cls: 'INTENDED' },
      status: 'unchanged',
      control: false,
    },
    {
      name: 'clean A/A with matching pick buffers',
      input: { a: shot({ pick }), a2: shot({ pick }), b: shot({ pick }), requiresPick: true },
      status: 'pass',
      control: false,
    },
    {
      name: 'unstable within a page',
      input: { a2: shot({ stable: false }) },
      status: 'excluded',
      control: true,
    },
    {
      name: 'A/A HDR differs',
      input: { a2: shot({ hdr: frame(1, [5, 6]) }) },
      status: 'excluded',
      control: true,
    },
    {
      name: 'A/A LDR differs',
      input: { a2: shot({ ldr: frame(0.5, [0]) }) },
      status: 'excluded',
      control: true,
    },
    {
      name: 'A/A pick ids differ',
      input: {
        a: shot({ pick }),
        a2: shot({ pick: pickBuffer(W * H, 2) }),
        b: shot({ pick }),
        requiresPick: true,
      },
      status: 'excluded',
      control: true,
    },
    {
      name: 'A/A pick buffer sizes differ',
      input: {
        a: shot({ pick }),
        a2: shot({ pick: pickBuffer(W * H - 1) }),
        b: shot({ pick }),
        requiresPick: true,
      },
      status: 'excluded',
      control: true,
    },
  ];

  it.each(table)('$name -> $status', ({ input, status, control }) => {
    const row = classify(input);
    expect(row.status).toBe(status);
    expect('control' in row).toBe(control);
  });

  it('records the A/A scores, bounding boxes and scene state for an HDR difference', () => {
    const row = classify({ a2: shot({ hdr: frame(1, [5, 6]), counts: { points: 'points:9' } }) });
    expect(row.failures).toEqual(['nondeterministic baseline (A/A differs on 2 HDR px, 0 LDR px)']);
    expect(row.control.hdr.differing).toBe(2);
    expect(row.control.hdr.perPixelUlp).toBeUndefined();
    expect(row.control.hdr.bbox).toEqual({ count: 2, minX: 1, minY: 1, maxX: 2, maxY: 1 });
    expect(row.control.ldr.bbox).toBeNull();
    expect(row.control.pick).toBeNull();
    expect(row.control.counts).toEqual({
      first: { points: 'points:10' },
      second: { points: 'points:9' },
    });
    expect(row.control.stable).toEqual({ first: true, second: true });
  });

  it('records the A/A pick mismatch count', () => {
    const row = classify({
      a: shot({ pick }),
      a2: shot({ pick: pickBuffer(W * H, 2) }),
      b: shot({ pick }),
    });
    expect(row.control.pick.mismatches).toBe(W * H);
    expect(row.failures[0]).toContain(`${W * H} pick px`);
  });

  it('records an A/A pick size mismatch', () => {
    const row = classify({
      a: shot({ pick }),
      a2: shot({ pick: pickBuffer(W * H - 1) }),
      b: shot({ pick }),
    });
    expect(row.control.pick).toEqual({
      lengthMismatch: true,
      first: W * H * 4,
      second: (W * H - 1) * 4,
    });
    expect(row.failures[0]).toContain('pick buffer sizes differ');
  });

  it('checks settling, empty frames and missing pick ids before the control arm', () => {
    const differ = { a2: shot({ hdr: frame(1, [0]) }) };
    expect(classify({ ...differ, b: shot({ settled: false }) }).failures).toEqual([
      'did not settle',
    ]);
    expect(classify({ a: shot({ hdr: frame(0) }), a2: shot({ hdr: frame(0) }) }).status).toBe(
      'error'
    );
    expect(classify({ ...differ, requiresPick: true }).failures).toEqual([
      'pick buffer not captured',
    ]);
  });

  it('fails a candidate whose drawn counts differ', () => {
    const row = classify({ b: shot({ counts: { points: 'points:9' } }) });
    expect(row.status).toBe('fail');
    expect(row.counts).toEqual({ base: { points: 'points:10' }, cand: { points: 'points:9' } });
  });
});
