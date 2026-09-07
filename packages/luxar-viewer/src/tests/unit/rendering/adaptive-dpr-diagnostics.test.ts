// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../config', () => ({
  config: {
    adaptiveDPR: {
      enabled: true,
      scaleDownFactor: 0.7,
      scaleUpFactor: 1.2,
      minDPR: 0.5,
      hysteresisSeconds: 2,
      evaluationIntervalMs: 500,
    },
  },
}));

import { AdaptiveDPRManager, type DPRRenderer } from '../../../rendering/adaptive-dpr-manager';
import { allowHighDPR, setNativeDPR } from '../../helpers/device-pixel-ratio';

let restoreCap: () => void;
let restoreDPR: () => void;
beforeEach(() => {
  restoreCap = allowHighDPR();
  restoreDPR = setNativeDPR(1.0);
});
afterEach(() => {
  restoreDPR();
  restoreCap();
});

function makeRenderer(): DPRRenderer {
  return { setAdaptivePixelRatio: vi.fn() } as unknown as DPRRenderer;
}

/** Push `frameCount` frames evenly spread over `durationMs` starting at `start`. */
function pushFrames(m: AdaptiveDPRManager, start: number, frameCount: number, durationMs: number) {
  for (let i = 0; i < frameCount; i++) {
    m.recordFrame(start + (i * durationMs) / Math.max(1, frameCount - 1));
  }
  return start + durationMs;
}

describe('AdaptiveDPRManager.getDiagnostics', () => {
  it('starts as getState() plus empty history', () => {
    const m = new AdaptiveDPRManager();
    const d = m.getDiagnostics();
    expect(d).toMatchObject({
      ...m.getState(),
      suppressedBy: null,
      backoffLevel: expect.any(Number),
      scaleDowns: 0,
      scaleUps: 0,
      verdicts: [],
    });
  });

  it('counts scale-downs and records the probe verdict that follows', () => {
    const m = new AdaptiveDPRManager({ gapResetMs: 60_000 });
    m.setRenderer(makeRenderer());
    // 20 fps for 14 frames: below the down-threshold with a trusted cadence,
    // so the manager scales down AND arms a probe (same drive as the manager
    // suite's `reduceWithProbe`).
    let t = pushFrames(m, 0, 14, 650);
    expect(m.getCurrentDPR()).toBeCloseTo(0.7, 5);
    expect(m.getState().probing).toBe(true);
    let d = m.getDiagnostics();
    expect(d.scaleDowns).toBe(1);
    expect(d.verdicts).toEqual([]);

    // The same 20 fps through the probe window: the step did not help, so the
    // probe is rejected and the record carries the before/after numbers.
    t = pushFrames(m, t + 50, 44, 2150);
    d = m.getDiagnostics();
    expect(d.verdicts).toHaveLength(1);
    const v = d.verdicts[0];
    expect(v.kind).toBe('rejected');
    expect(v.previousDPR).toBeCloseTo(1.0, 5);
    expect(v.probedDPR).toBeCloseTo(0.7, 5);
    expect(v.previousFPS).toBeGreaterThan(0);
    expect(v.currentFPS).toBeGreaterThan(0);
    expect(v.fpsRatio).toBeCloseTo((v.currentFPS ?? 0) / v.previousFPS, 5);
    expect(v.timestamp).toBeLessThanOrEqual(t);
    expect(m.getCurrentDPR()).toBeCloseTo(1.0, 5); // reverted
    // The returned history is a copy.
    d.verdicts.length = 0;
    expect(m.getDiagnostics().verdicts.length).toBeGreaterThanOrEqual(1);
  });

  it('reports the suppression cause of the last evaluation', () => {
    const m = new AdaptiveDPRManager();
    m.setRenderer(makeRenderer());
    m.setLoadActivityPredicate(() => true);
    m.recordFrame(0);
    m.recordFrame(1000);
    m.recordFrame(1600);
    expect(m.getDiagnostics().suppressedBy).toBe('load');
  });
});
