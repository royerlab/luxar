import { afterEach, describe, expect, it, vi } from 'vitest';

import * as ops from './page-ops.mjs';
import {
  countTimingJudgements,
  dragWorkload,
  judgeMetric,
  meetsExpectation,
  METRIC_DIRECTIONS,
  runSuite,
  serverMetrics,
  validateExpectations,
  validateMetricDirections,
} from './suites.mjs';

afterEach(() => vi.unstubAllGlobals());

describe('waitForQuiet', () => {
  it('requires 30 consecutive render-free frames after the minimum tail', async () => {
    let now = 0;
    let frame = 0;
    const pp = { render() {} };
    const original = pp.render;
    vi.stubGlobal('window', { __luxarDebug: { app: { sceneManager: { postProcessing: pp } } } });
    vi.stubGlobal('performance', { now: () => now });
    vi.stubGlobal('requestAnimationFrame', (callback) => {
      queueMicrotask(() => {
        frame++;
        now += 100;
        if (frame === 3 || frame === 12) pp.render();
        callback(now);
      });
    });
    expect(await ops.waitForQuiet({ minMs: 3000, timeoutMs: 5000 })).toEqual({
      renders: 2,
      frames: 42,
      settled: true,
    });
    expect(pp.render).toBe(original);
  });

  it('times out while rendering continues and restores the render method', async () => {
    let now = 0;
    const pp = { render() {} };
    const original = pp.render;
    vi.stubGlobal('window', { __luxarDebug: { app: { sceneManager: { postProcessing: pp } } } });
    vi.stubGlobal('performance', { now: () => now });
    vi.stubGlobal('requestAnimationFrame', (callback) => {
      queueMicrotask(() => {
        now += 100;
        pp.render();
        callback(now);
      });
    });
    expect(await ops.waitForQuiet({ minMs: 0, timeoutMs: 1000 })).toEqual({
      renders: 10,
      frames: 10,
      settled: false,
    });
    expect(pp.render).toBe(original);
  });

  it('times out when animation frames stop', async () => {
    const pp = { render() {} };
    const original = pp.render;
    vi.stubGlobal('window', { __luxarDebug: { app: { sceneManager: { postProcessing: pp } } } });
    vi.stubGlobal('requestAnimationFrame', () => {});
    expect(await ops.waitForQuiet({ minMs: 0, timeoutMs: 10 })).toEqual({
      renders: 0,
      frames: 0,
      settled: false,
    });
    expect(pp.render).toBe(original);
  });
});

describe('dragWorkload', () => {
  it('reports only uploads after mouse release as tail work', async () => {
    const calls = [];
    let uploads = { gpuUploadBytesBuffer: 0, gpuUploadBytesTexture: 0 };
    let idleWindows = 0;
    const page = {
      evaluate: async (fn) => {
        if (fn === ops.beginRenderCount) return true;
        if (fn === ops.readExtCounters) return { ...uploads };
        if (fn === ops.waitForQuiet || fn === ops.idle) {
          idleWindows++;
          uploads =
            idleWindows === 1
              ? { gpuUploadBytesBuffer: 116, gpuUploadBytesTexture: 221 }
              : { gpuUploadBytesBuffer: 119, gpuUploadBytesTexture: 223 };
          return {
            renders: idleWindows === 1 ? 2 : 1,
            frames: idleWindows === 1 ? 180 : 60,
            settled: idleWindows === 1,
          };
        }
        if (fn === ops.endRenderCount) return 9;
        return { x: 10, y: 20 };
      },
      mouse: {
        move: async () => calls.push('move'),
        down: async () => calls.push('down'),
        up: async () => {
          calls.push('up');
          uploads = { gpuUploadBytesBuffer: 100, gpuUploadBytesTexture: 200 };
        },
      },
    };
    expect(await dragWorkload(page, { tailMs: 3000 })).toEqual({
      renders: 9,
      tailRenders: 2,
      tailFrames: 180,
      tailSettled: 1,
      tailUploadBytesBuffer: 16,
      tailUploadBytesTexture: 21,
      idleRenders: 1,
      idleUploadBytesBuffer: 3,
      idleUploadBytesTexture: 2,
    });
    expect(idleWindows).toBe(2);
    expect(calls).toEqual(['move', 'down', 'move', 'up']);
  });

  it('reports an unsettled tail as a harness error before measuring idle', async () => {
    const uploads = { gpuUploadBytesBuffer: 0, gpuUploadBytesTexture: 0 };
    let measuredIdle = false;
    const page = {
      evaluate: async (fn) => {
        if (fn === ops.waitForQuiet) return { renders: 4, frames: 60, settled: false };
        if (fn === ops.idle) measuredIdle = true;
        if (fn === ops.readExtCounters) return uploads;
        if (fn === ops.endRenderCount) return 4;
        return { x: 10, y: 20 };
      },
      mouse: { move: async () => {}, down: async () => {}, up: async () => {} },
    };
    await expect(dragWorkload(page, {})).rejects.toThrow(/drag tail did not settle/);
    expect(measuredIdle).toBe(false);
  });
});

describe('manifest expectations', () => {
  const run = async (renders, expectations) => {
    const page = {
      evaluate: async (fn) => {
        if (fn === ops.settle) return { settled: true };
        if (fn === ops.idle) return { renders, frames: 60 };
        return {};
      },
    };
    const server = {
      origin: 'http://localhost',
      reset() {},
      now: () => 0,
      log: [],
      stats: () => ({ maxInflight: 0 }),
    };
    return runSuite({
      name: 'counters',
      suite: {
        cases: [
          {
            id: 'idle',
            workload: { kind: 'idle' },
            backends: ['webgl', 'webgpu'],
            metrics: [
              { name: 'renders', kind: 'counter' },
              { name: 'frames', kind: 'counter' },
            ],
            expectations: { renders: 'zero' },
          },
        ],
      },
      profile: { name: 'local' },
      session: { run: async (fn) => fn({}) },
      openCase: async () => ({ page, context: { close: async () => {} } }),
      servers: { base: server, cand: server },
      dataRoot: '.',
      opts: { rounds: 1 },
      expectations,
      log() {},
    });
  };

  it('enforces the shipped zero expectation on each backend even when both builds agree', async () => {
    const busy = await run(1);
    expect(busy.map((row) => row.backend)).toEqual(['webgl', 'webgpu']);
    for (const row of busy) {
      expect(row.status).toBe('fail');
      expect(row.metrics.renders).toMatchObject({ verdict: 'pass', expected: 'zero', met: false });
    }
    const quiet = await run(0);
    for (const row of quiet) {
      expect(row.status).toBe('pass');
      expect(row.metrics.renders).toMatchObject({ expected: 'zero', met: true });
    }
  });

  it('merges a nonconflicting --expect metric with a shipped expectation', async () => {
    const rows = await run(0, { idle: { frames: 'pass' } });
    for (const row of rows) {
      expect(row.status).toBe('pass');
      expect(row.metrics.renders).toMatchObject({ expected: 'zero', met: true });
      expect(row.metrics.frames).toMatchObject({ expected: 'pass', met: true });
    }
  });
});

describe('unsettled drag row', () => {
  it('reports a harness error on both backends, not an idle-zero failure', async () => {
    const page = {
      evaluate: async (fn) => {
        if (fn === ops.settle) return { settled: true };
        if (fn === ops.waitForQuiet) return { renders: 1, frames: 10, settled: false };
        if (fn === ops.readExtCounters) {
          return { gpuUploadBytesBuffer: 0, gpuUploadBytesTexture: 0 };
        }
        if (fn === ops.endRenderCount) return 1;
        return { x: 10, y: 20 };
      },
      mouse: { move: async () => {}, down: async () => {}, up: async () => {} },
    };
    const server = {
      origin: 'http://localhost',
      reset() {},
      now: () => 0,
      log: [],
      stats: () => ({ maxInflight: 0 }),
    };
    const rows = await runSuite({
      name: 'counters',
      suite: {
        cases: [
          {
            id: 'drag',
            workload: { kind: 'drag' },
            backends: ['webgl', 'webgpu'],
            metrics: [{ name: 'idleRenders' }],
            expectations: { idleRenders: 'zero' },
          },
        ],
      },
      profile: { name: 'local' },
      session: { run: async (fn) => fn({}) },
      openCase: async () => ({ page, context: { close: async () => {} } }),
      servers: { base: server, cand: server },
      dataRoot: '.',
      opts: { rounds: 1 },
      log() {},
    });
    expect(rows.map((row) => row.backend)).toEqual(['webgl', 'webgpu']);
    for (const row of rows) {
      expect(row).toMatchObject({
        status: 'error',
        metrics: {},
        failures: ['drag tail did not settle (tailSettled=0)'],
      });
    }
  });
});

const manifestSuites = {
  counters: {
    cases: [
      {
        id: 'a',
        metrics: [
          { name: 'render.count', kind: 'counter', better: 'lower' },
          { name: 'fps', kind: 'timing', better: 'higher' },
        ],
      },
    ],
  },
};

describe('validateExpectations', () => {
  it('accepts declared cases and metrics', () => {
    const e = { a: { 'render.count': 'zero', fps: 'win' } };
    expect(validateExpectations(e, manifestSuites)).toEqual(e);
  });

  it('rejects an unknown case, an undeclared metric and a bad value', () => {
    expect(() => validateExpectations({ b: {} }, manifestSuites)).toThrow(/unknown case 'b'/);
    expect(() => validateExpectations({ a: { nope: 'win' } }, manifestSuites)).toThrow(
      /no metric 'nope'/
    );
    expect(() => validateExpectations({ a: { fps: 'better' } }, manifestSuites)).toThrow(
      /not one of/
    );
  });

  it('rejects a conflicting shipped expectation', () => {
    const suites = {
      counters: {
        cases: [{ id: 'idle', metrics: [{ name: 'renders' }], expectations: { renders: 'zero' } }],
      },
    };
    expect(() => validateExpectations({ idle: { renders: 'pass' } }, suites)).toThrow(
      /idle\.renders.*conflicts with shipped expectation 'zero'/
    );
    expect(validateExpectations({ idle: { renders: 'zero' } }, suites)).toEqual({
      idle: { renders: 'zero' },
    });
  });
});

describe('meetsExpectation', () => {
  it('never counts n/a as met', () => {
    for (const want of ['win', 'zero', 'same', 'pass']) {
      expect(meetsExpectation(want, { verdict: 'n/a' }, 'counter')).toBe(false);
    }
  });

  it('applies each expectation', () => {
    expect(meetsExpectation('win', { verdict: 'win' }, 'counter')).toBe(true);
    expect(meetsExpectation('win', { verdict: 'pass' }, 'counter')).toBe(false);
    expect(meetsExpectation('zero', { verdict: 'win', candMedian: 0 }, 'counter')).toBe(true);
    expect(meetsExpectation('zero', { verdict: 'win', candMedian: 1 }, 'counter')).toBe(false);
    expect(
      meetsExpectation('same', { verdict: 'pass', baseMedian: 3, candMedian: 3 }, 'counter')
    ).toBe(true);
    expect(
      meetsExpectation('same', { verdict: 'pass', baseMedian: 3, candMedian: 3.2 }, 'counter')
    ).toBe(false);
    expect(
      meetsExpectation('same', { verdict: 'pass', baseMedian: 3, candMedian: 3.2 }, 'timing')
    ).toBe(true);
    expect(meetsExpectation('pass', { verdict: 'win' }, 'timing')).toBe(true);
    expect(meetsExpectation('pass', { verdict: 'fail' }, 'timing')).toBe(false);
  });
});

describe('judgeMetric', () => {
  const arm = (vals, key) => vals.map((v) => ({ [key]: v }));

  it('judges a per-tick ratio, not the timed total', () => {
    // A faster build plays 10% more ticks in the window and uploads 10% more
    // bytes in total: the same work per tick.
    const s = (bytes, ticks) => ({ 'gpu.uploadBytes': bytes, ticks });
    const m = { name: 'gpu.uploadBytes/tick', of: 'gpu.uploadBytes', per: 'ticks', relTol: 0.05 };
    const samples = {
      base: [s(1000, 50), s(1000, 50), s(1000, 50)],
      base2: [s(1000, 50), s(1000, 50), s(1000, 50)],
      cand: [s(1100, 55), s(1100, 55), s(1100, 55)],
    };
    expect(judgeMetric(m, samples)).toMatchObject({
      verdict: 'pass',
      baseMedian: 20,
      candMedian: 20,
    });
    expect(judgeMetric({ name: 'gpu.uploadBytes' }, samples).verdict).toBe('fail');
    const noTicks = { ...samples, cand: [s(1100, 0), s(1100, 0), s(1100, 0)] };
    expect(judgeMetric(m, noTicks).verdict).toBe('n/a');
  });

  it('judges counters exactly and timings with direction', () => {
    const counter = judgeMetric(
      { name: 'render.count', kind: 'counter' },
      {
        base: arm([5, 5, 5], 'render.count'),
        base2: arm([5, 5, 5], 'render.count'),
        cand: arm([0, 0, 0], 'render.count'),
      }
    );
    expect(counter).toMatchObject({
      verdict: 'win',
      baseMedian: 5,
      candMedian: 0,
      kind: 'counter',
    });
    const fps = [10, 10.1, 9.9, 10, 10.05];
    const timing = judgeMetric(
      { name: 'fps', kind: 'timing', better: 'higher' },
      {
        base: arm(fps, 'fps'),
        base2: arm(fps, 'fps'),
        cand: arm(
          fps.map((v) => v * 1.5),
          'fps'
        ),
      }
    );
    expect(timing.verdict).toBe('win');
  });

  it('reads n/a when a build does not report the metric', () => {
    const missing = judgeMetric(
      { name: 'decode.duplicates', kind: 'counter' },
      { base: [{}, {}], base2: [{}, {}], cand: arm([0, 0], 'decode.duplicates') }
    );
    expect(missing.verdict).toBe('n/a');
    const timing = judgeMetric(
      { name: 'x', kind: 'timing' },
      { base: [{}], base2: [{}], cand: [{ x: 1 }] }
    );
    expect(timing.verdict).toBe('n/a');
  });

  it('passes an A/A fluctuation at the run-wide floor and preserves the raw floor', () => {
    const samples = {
      base: arm([100, 100, 100, 100, 100], 'firstFrameMs'),
      base2: arm([100, 100, 100, 100, 100], 'firstFrameMs'),
      cand: arm([101.2, 101.2, 101.2, 101.2, 101.2], 'firstFrameMs'),
    };
    const timing = { name: 'firstFrameMs', kind: 'timing' };
    expect(judgeMetric(timing, samples, 1).verdict).toBe('fail');
    expect(judgeMetric(timing, samples, 4)).toMatchObject({ verdict: 'pass', aaFloor: 0.01 });
    expect(judgeMetric(timing, samples, 4).floor).toBeGreaterThan(0.01);
    expect(judgeMetric({ name: 'firstFrameMs', kind: 'counter' }, samples, 4).verdict).toBe('fail');
  });
});

describe('countTimingJudgements', () => {
  it('counts only selected gating timings across eligible backends and suites', () => {
    const suites = {
      hosted: {
        defaults: { backends: ['webgl'] },
        cases: [
          {
            id: 'load',
            metrics: [
              { name: 'firstFrameMs', kind: 'timing' },
              { name: 'settledMs', kind: 'timing' },
              { name: 'bytes', kind: 'counter' },
              { name: 'totalMs', kind: 'timing', report: true },
            ],
          },
          { id: 'heavy', heavy: true, metrics: [{ name: 'stepMs', kind: 'timing' }] },
        ],
      },
      playback: {
        cases: [
          { id: 'play', backends: ['webgpu'], metrics: [{ name: 'achievedFps', kind: 'timing' }] },
        ],
      },
    };
    expect(
      countTimingJudgements(suites, ['hosted', 'playback'], { backends: ['webgl', 'webgpu'] })
    ).toBe(3);
    expect(
      countTimingJudgements(suites, ['hosted'], { backends: ['webgl', 'webgpu'], only: ['load'] })
    ).toBe(2);
    expect(
      countTimingJudgements(suites, ['hosted', 'playback'], {
        backends: ['webgl', 'webgpu'],
        heavy: true,
      })
    ).toBe(4);
  });
});

describe('validateMetricDirections', () => {
  const suites = (metrics) => ({ playback: { cases: [{ id: 'scrub', metrics }] } });

  it('pins dragFrames as higher-is-better', () => {
    expect(METRIC_DIRECTIONS.dragFrames).toBe('higher');
  });

  it('rejects a manifest that judges dragFrames lower-is-better', () => {
    // The shape an audit's derived manifest declared: every gain read as a FAIL.
    const bad = suites([{ name: 'dragFrames', better: 'lower', kind: 'counter' }]);
    expect(() => validateMetricDirections(bad)).toThrow(
      /scrub\.dragFrames: declared better: 'lower', but dragFrames is higher-is-better/
    );
  });

  it('treats a missing better as lower and checks a ratio by its numerator', () => {
    expect(() => validateMetricDirections(suites([{ name: 'achievedFps' }]))).toThrow(
      /achievedFps is higher-is-better/
    );
    const ratio = { name: 'gpu.uploadBytes/tick', of: 'gpu.uploadBytes', per: 'ticks' };
    expect(() => validateMetricDirections(suites([{ ...ratio, better: 'lower' }]))).not.toThrow();
    expect(() => validateMetricDirections(suites([{ ...ratio, better: 'higher' }]))).toThrow(
      /gpu\.uploadBytes is lower-is-better/
    );
  });

  it('leaves a metric it does not know alone', () => {
    expect(() =>
      validateMetricDirections(suites([{ name: 'newCounter', better: 'higher' }]))
    ).not.toThrow();
  });

  it('rejects misspelled or invalid manifest expectations', () => {
    const c = { id: 'idle', metrics: [{ name: 'renders' }], expectations: { misspelled: 'zero' } };
    expect(() => validateMetricDirections({ counters: { cases: [c] } })).toThrow(
      /expectation names undeclared metric 'misspelled'/
    );
    c.expectations = { renders: 'quiet' };
    expect(() => validateMetricDirections({ counters: { cases: [c] } })).toThrow(
      /unknown expectation 'quiet'/
    );
  });
});

describe('serverMetrics', () => {
  const e = (url, startMs, bytes) => ({ url, startMs, endMs: startMs + 50, bytes });

  it('counts dataset requests before the first frame and their start waves', () => {
    const log = [
      e('/index.html', 0, 1000),
      e('/datasets/s/zarr.json', 10, 100),
      e('/datasets/s/a/zarr.json', 120, 50),
      e('/datasets/s/b/zarr.json', 125, 50),
      e('/datasets/s/a/c/0', 240, 4000),
      e('/datasets/s/a/c/1', 900, 4000),
    ];
    const m = serverMetrics(log, { navT: 0, firstFrameMs: 500, latencyMs: 100, maxInflight: 3 });
    expect(m).toEqual({
      serverRequests: 5,
      serverBytes: 8200,
      maxInflight: 3,
      requestsToFirstFrame: 4,
      bytesToFirstFrame: 4200,
      serialDepth: 3,
    });
  });

  it('omits the first-frame figures when the workload reported no first frame', () => {
    const m = serverMetrics([], { navT: 0, firstFrameMs: undefined, maxInflight: 0 });
    expect(m).toEqual({ serverRequests: 0, serverBytes: 0, maxInflight: 0 });
  });
});
