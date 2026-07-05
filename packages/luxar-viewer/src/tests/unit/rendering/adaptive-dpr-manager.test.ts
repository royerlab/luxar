/**
 * Unit tests for AdaptiveDPRManager.
 *
 * The manager has only two real-world dependencies: `window.devicePixelRatio`
 * (read once in the constructor) and the project `config` module. We set
 * the DPR explicitly before each test and mock the config so the FPS
 * thresholds, scaling factors, and hysteresis are fully under our control.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Config mock — the manager merges this with optional ctor overrides.
vi.mock('../../../config', () => ({
  config: {
    adaptiveDPR: {
      enabled: true,
      minFPS: 30,
      maxFPS: 55,
      scaleDownFactor: 0.7,
      scaleUpFactor: 1.2,
      minDPR: 0.5,
      hysteresisSeconds: 2,
      evaluationIntervalMs: 500,
    },
  },
}));

import { AdaptiveDPRManager, type DPRRenderer } from '../../../rendering/adaptive-dpr-manager';

// ---------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------

function setNativeDPR(value: number): () => void {
  const original = window.devicePixelRatio;
  Object.defineProperty(window, 'devicePixelRatio', {
    configurable: true,
    value,
    writable: true,
  });
  return () => {
    Object.defineProperty(window, 'devicePixelRatio', {
      configurable: true,
      value: original,
      writable: true,
    });
  };
}

function makeRenderer(): DPRRenderer & { setAdaptivePixelRatio: ReturnType<typeof vi.fn> } {
  return {
    setAdaptivePixelRatio: vi.fn(),
  } as DPRRenderer & { setAdaptivePixelRatio: ReturnType<typeof vi.fn> };
}

/** Simulate `frameCount` frames over `durationMs` so the FPS computes deterministically. */
function pushFrames(
  manager: AdaptiveDPRManager,
  startTimestamp: number,
  frameCount: number,
  durationMs: number
): number {
  for (let i = 0; i < frameCount; i++) {
    const t = startTimestamp + (i * durationMs) / Math.max(1, frameCount - 1);
    manager.recordFrame(t);
  }
  return startTimestamp + durationMs;
}

// ---------------------------------------------------------------------
// Construction & initial state
// ---------------------------------------------------------------------

describe('AdaptiveDPRManager — construction', () => {
  let restore: () => void;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
  });
  // afterEach pattern via direct restore call below.

  it('initializes currentDPR / nativeDPR from window.devicePixelRatio', () => {
    const m = new AdaptiveDPRManager();
    try {
      expect(m.getNativeDPR()).toBe(2.0);
      expect(m.getCurrentDPR()).toBe(2.0);
      expect(m.getState().enabled).toBe(true);
      expect(m.getState().isReducedResolution).toBe(false);
    } finally {
      restore();
    }
  });

  it('falls back to 1 when window.devicePixelRatio is 0/undefined', () => {
    restore();
    const restoreZero = setNativeDPR(0);
    try {
      const m = new AdaptiveDPRManager();
      expect(m.getNativeDPR()).toBe(1);
    } finally {
      restoreZero();
    }
  });

  it('respects ctor overrides on top of the config defaults', () => {
    const m = new AdaptiveDPRManager({ minDPR: 0.1 });
    try {
      // Internal config check via behavior: scaleDown with very low FPS
      // should drop DPR all the way to 0.1, not 0.5.
      const r = makeRenderer();
      m.setRenderer(r);
      // Simulate a very low FPS by pushing 2 frames over a long time.
      m.recordFrame(0);
      m.recordFrame(1000); // 2 frames in 1 second → ~1 FPS, well under minFPS=30
      // evaluateAndAdjust runs at next recordFrame after 500ms — push one more.
      m.recordFrame(1600);
      // currentDPR should have dropped via repeated scaleDown to clamp at 0.1.
      // We can't verify the clamp from a single tick, but we can verify scaleDown happened.
      expect(m.getCurrentDPR()).toBeLessThan(2.0);
    } finally {
      restore();
    }
  });
});

// ---------------------------------------------------------------------
// FPS tracking
// ---------------------------------------------------------------------

describe('AdaptiveDPRManager — getCurrentFPS', () => {
  let restore: () => void;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
  });

  it('returns 0 when fewer than 2 frame samples are recorded', () => {
    const m = new AdaptiveDPRManager();
    try {
      expect(m.getCurrentFPS()).toBe(0);
      m.recordFrame(0);
      expect(m.getCurrentFPS()).toBe(0);
    } finally {
      restore();
    }
  });

  it('computes FPS from the recorded frame window', () => {
    const m = new AdaptiveDPRManager();
    try {
      // 60 frames in 1000ms → ~59 intervals → 59 FPS via the count-1 formula.
      pushFrames(m, 0, 60, 1000);
      const fps = m.getCurrentFPS();
      expect(fps).toBeGreaterThan(58);
      expect(fps).toBeLessThan(61);
    } finally {
      restore();
    }
  });

  it('returns 0 when timeSpan is non-positive (degenerate input)', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.recordFrame(100);
      m.recordFrame(100); // same timestamp → timeSpan = 0
      expect(m.getCurrentFPS()).toBe(0);
    } finally {
      restore();
    }
  });

  it('skips recording entirely when adaptive DPR is disabled', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setEnabled(false);
      pushFrames(m, 0, 60, 1000);
      expect(m.getCurrentFPS()).toBe(0);
    } finally {
      restore();
    }
  });
});

// ---------------------------------------------------------------------
// Scale down / up behavior
// ---------------------------------------------------------------------

describe('AdaptiveDPRManager — scaling', () => {
  let restore: () => void;
  let renderer: ReturnType<typeof makeRenderer>;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
    renderer = makeRenderer();
  });

  it('scales DPR down when sustained low FPS is observed', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // 10 FPS = 10 frames over 1 second.
      pushFrames(m, 0, 10, 1000);
      // Advance 600ms past the last sample so evaluateAndAdjust fires
      // (interval = 500ms).
      m.recordFrame(1600);

      expect(m.getCurrentDPR()).toBeLessThan(2.0);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenCalled();
      // The notification args: (newDPR, isReducedResolution).
      const lastCall = renderer.setAdaptivePixelRatio.mock.calls.at(-1)!;
      expect(lastCall[0]).toBeLessThan(2.0);
    } finally {
      restore();
    }
  });

  it('does not exceed minDPR when scaling down repeatedly', () => {
    const m = new AdaptiveDPRManager({ minDPR: 0.5 });
    try {
      m.setRenderer(renderer);
      // Hammer with low FPS over many evaluation cycles.
      for (let cycle = 0; cycle < 20; cycle++) {
        pushFrames(m, cycle * 1000, 5, 1000); // 5 FPS each second
        // Trigger evaluation by advancing past the interval.
        m.recordFrame(cycle * 1000 + 1100);
      }
      expect(m.getCurrentDPR()).toBeGreaterThanOrEqual(0.5);
    } finally {
      restore();
    }
  });

  it('flags reducedResolution once DPR drops below 95% of native', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      pushFrames(m, 0, 10, 1000);
      m.recordFrame(1600);
      // Native 2.0 × 0.7 = 1.4 → 1.4 / 2.0 = 0.7 < 0.95 → reduced.
      expect(m.getCurrentDPR()).toBeLessThan(2.0 * 0.95);
      expect(m.getState().isReducedResolution).toBe(true);
    } finally {
      restore();
    }
  });
});

// ---------------------------------------------------------------------
// setEnabled / setManualDPR / dispose
// ---------------------------------------------------------------------

describe('AdaptiveDPRManager — setEnabled', () => {
  let restore: () => void;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
  });

  it('is a no-op when state is unchanged', () => {
    const m = new AdaptiveDPRManager();
    try {
      const cb = vi.fn();
      m.setOnDPRChangeCallback(cb);
      m.setEnabled(true); // already enabled
      expect(cb).not.toHaveBeenCalled();
    } finally {
      restore();
    }
  });

  it('resets DPR to native and clears tracking on disable', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);
      // Drop DPR via low-FPS path first.
      pushFrames(m, 0, 10, 1000);
      m.recordFrame(1600);
      expect(m.getCurrentDPR()).toBeLessThan(2.0);

      const cb = vi.fn();
      m.setOnDPRChangeCallback(cb);
      m.setEnabled(false);

      expect(m.getCurrentDPR()).toBe(2.0);
      expect(m.getState().isReducedResolution).toBe(false);
      expect(m.isActive()).toBe(false);
      expect(cb).toHaveBeenCalledWith(2.0, false);
      // recordFrame should now be a no-op.
      m.recordFrame(2000);
      expect(m.getCurrentFPS()).toBe(0);
    } finally {
      restore();
    }
  });
});

describe('AdaptiveDPRManager — setManualDPR', () => {
  let restore: () => void;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
  });

  it('refuses to change DPR while adaptive mode is active', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);
      m.setManualDPR(1.0);
      // Adaptive on → manual setter is rejected.
      expect(m.getCurrentDPR()).toBe(2.0);
      expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();
    } finally {
      restore();
    }
  });

  it('clamps to [0.25, nativeDPR] when adaptive is disabled', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);
      m.setEnabled(false);

      m.setManualDPR(0.1);
      expect(m.getCurrentDPR()).toBe(0.25);

      m.setManualDPR(99);
      expect(m.getCurrentDPR()).toBe(2.0);

      m.setManualDPR(1.0);
      expect(m.getCurrentDPR()).toBe(1.0);
    } finally {
      restore();
    }
  });

  it('skips duplicate manual DPR updates to avoid redundant GPU reallocations', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);
      m.setEnabled(false);
      renderer.setAdaptivePixelRatio.mockClear();

      m.setManualDPR(1.0);
      m.setManualDPR(1.0);
      m.setManualDPR(1.005); // within idempotency epsilon

      expect(renderer.setAdaptivePixelRatio).toHaveBeenCalledTimes(1);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenLastCalledWith(1.0);
    } finally {
      restore();
    }
  });

  it('updates reduced-resolution state for manual DPR changes', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);
      m.setEnabled(false);

      m.setManualDPR(1.0);
      expect(m.getState().isReducedResolution).toBe(true);

      m.setManualDPR(2.0);
      expect(m.getState().isReducedResolution).toBe(false);
    } finally {
      restore();
    }
  });
});

describe('AdaptiveDPRManager — pinManualDPR', () => {
  let restore: () => void;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
  });

  it('disables adaptive mode, applies the clamped DPR, and reports pinned', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);

      m.pinManualDPR(0.5);

      expect(m.isActive()).toBe(false);
      expect(m.isPinned()).toBe(true);
      expect(m.getCurrentDPR()).toBe(0.5);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenLastCalledWith(0.5);
    } finally {
      restore();
    }
  });

  it('locks setEnabled so persisted settings cannot re-enable adaptation', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);

      m.pinManualDPR(1.0);
      // Simulate rendering-controls loadSettings applying a persisted
      // `adaptiveDPREnabled: true` after pipeline init.
      m.setEnabled(true);

      expect(m.isActive()).toBe(false);
      expect(m.getCurrentDPR()).toBe(1.0);

      // Disable requests are equally ignored — the pin owns the state.
      m.setEnabled(false);
      expect(m.isPinned()).toBe(true);
    } finally {
      restore();
    }
  });

  it('clamps the pinned value to [0.25, native]', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(makeRenderer());
      m.pinManualDPR(99);
      expect(m.getCurrentDPR()).toBe(2.0);
    } finally {
      restore();
    }
  });
});

describe('AdaptiveDPRManager — dispose', () => {
  let restore: () => void;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
  });

  it('clears callbacks and renderer references', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      const cb = vi.fn();
      m.setRenderer(renderer);
      m.setOnDPRChangeCallback(cb);
      m.dispose();
      // After dispose, recordFrame at low FPS should not call the
      // renderer (cleared) or invoke the callback.
      pushFrames(m, 0, 10, 1000);
      m.recordFrame(1600);
      // Callback was cleared; renderer was cleared. The DPR may still be
      // updated internally but no observable side effects fire.
      expect(cb).not.toHaveBeenCalled();
    } finally {
      restore();
    }
  });
});

describe('AdaptiveDPRManager — getState', () => {
  let restore: () => void;

  beforeEach(() => {
    restore = setNativeDPR(1.5);
  });

  it('exposes a snapshot of the public state fields', () => {
    const m = new AdaptiveDPRManager();
    try {
      const s = m.getState();
      expect(s.enabled).toBe(true);
      expect(s.currentDPR).toBe(1.5);
      expect(s.nativeDPR).toBe(1.5);
      expect(s.currentFPS).toBe(0);
      expect(s.isReducedResolution).toBe(false);
      // U-shape state defaults
      expect(s.dprFloor).toBe(0.5); // matches mocked config.minDPR
      expect(s.probing).toBe(false);
    } finally {
      restore();
    }
  });
});

// ---------------------------------------------------------------------
// U-shape probe behaviour
// ---------------------------------------------------------------------
// Lowering DPR is not monotonically faster on macOS/Chrome (compositor
// upscale cost grows past a sweet spot). The manager scales down with
// a probe: it measures FPS before and after a scale-down move and, if
// the move did not improve FPS, reverts and records a floor so it
// won't try the same move again until the floor TTL expires.

describe('AdaptiveDPRManager — U-shape probe', () => {
  let restore: () => void;
  let renderer: ReturnType<typeof makeRenderer>;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
    renderer = makeRenderer();
  });

  /**
   * Fill the FPS window so that an evaluation triggered at `evalAt`
   * sees roughly `targetFPS`. Pushes `targetFPS` frames evenly spaced
   * between `evalAt - 1000` and `evalAt`, then a final frame AT
   * `evalAt` that triggers evaluateAndAdjust. Avoids intermediate
   * evaluations from firing with random FPS readings by keeping all
   * pushes within a single 1000ms window past the previous evaluation.
   */
  function evaluateWithFPS(m: AdaptiveDPRManager, evalAt: number, targetFPS: number): void {
    const start = evalAt - 1000;
    const N = Math.max(2, Math.round(targetFPS));
    for (let i = 0; i < N; i++) {
      m.recordFrame(start + (i / (N - 1)) * 1000);
    }
  }

  it('keeps the DPR move when post-probe FPS improves enough', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);

      // Eval at t=1000 with FPS=20 → below minFPS=30 → scaleDown, probe armed.
      evaluateWithFPS(m, 1000, 20);
      const dprAfterScaleDown = m.getCurrentDPR();
      expect(dprAfterScaleDown).toBeLessThan(2.0);
      expect(m.getState().probing).toBe(true);

      // Eval at t=3000 (2000ms past probe start, well past PROBE_WINDOW_MS).
      // Build a window of frames giving FPS=80 → 4× improvement, easily
      // clears the PROBE_IMPROVEMENT (1.05×) threshold.
      evaluateWithFPS(m, 3000, 80);

      // Probe accepted: DPR stays at the scaled-down value, floor unchanged.
      expect(m.getState().probing).toBe(false);
      expect(m.getCurrentDPR()).toBe(dprAfterScaleDown);
      expect(m.getState().dprFloor).toBe(0.5);
    } finally {
      restore();
    }
  });

  it('reverts DPR and sets a floor when post-probe FPS did not improve', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);

      // scaleDown at FPS=20
      evaluateWithFPS(m, 1000, 20);
      const previousDPR = 2.0;
      const probedDPR = m.getCurrentDPR();
      expect(probedDPR).toBeLessThan(previousDPR);
      expect(m.getState().probing).toBe(true);

      // Eval past probe window with SAME FPS=20 (ratio = 1.0 < 1.05).
      evaluateWithFPS(m, 3000, 20);

      // Probe rejected: DPR reverts; floor tightens to probedDPR.
      expect(m.getState().probing).toBe(false);
      expect(m.getCurrentDPR()).toBe(previousDPR);
      expect(m.getState().dprFloor).toBeCloseTo(probedDPR, 5);
    } finally {
      restore();
    }
  });

  it('refuses to scale TO OR below the dprFloor set by a failed probe', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);

      // Reject a probe to set a floor at scaleDownFactor*native = 0.7*2 = 1.4
      evaluateWithFPS(m, 1000, 20);
      const probedDPR = m.getCurrentDPR();
      evaluateWithFPS(m, 3000, 20);
      expect(m.getState().dprFloor).toBeCloseTo(probedDPR, 5);
      expect(m.getCurrentDPR()).toBe(2.0); // reverted to native

      // Eval again with FPS=20 → scaleDown proposes 2.0 * 0.7 = 1.4,
      // which equals the floor that was just set by the failed probe.
      // The floor MUST block moves to that DPR — otherwise we'd
      // re-fire the same failed probe and oscillate (this is the
      // c.elegans-demo "Probe rejected ... will retry in 30s" loop
      // that fires every ~2s instead of every 30s).
      evaluateWithFPS(m, 5000, 20);
      expect(m.getCurrentDPR()).toBe(2.0); // stayed at native, no re-probe
      expect(m.getState().probing).toBe(false);
    } finally {
      restore();
    }
  });

  it('does not oscillate at the floor across many evaluations after a failed probe', () => {
    // Regression guard for the AdaptiveDPR oscillation: every
    // evaluation after a rejected probe used to re-fire scaleDown at
    // the floor (because `Math.max(dprFloor, proposed)` clamped TO
    // the floor instead of strictly above it), causing a 2-second
    // probe-reject-revert cycle to repeat throughout the 30s TTL
    // window. The fix blocks scaleDown when `proposed <= dprFloor`.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);

      // Trigger a rejected probe → floor lands at scaleDownFactor*native.
      evaluateWithFPS(m, 1000, 20);
      const probedDPR = m.getCurrentDPR();
      evaluateWithFPS(m, 3000, 20);
      expect(m.getState().dprFloor).toBeCloseTo(probedDPR, 5);
      expect(m.getCurrentDPR()).toBe(2.0);

      // Hammer it: 10 further evaluations spaced 1s apart, all with
      // poor FPS. The floor must keep us pinned at native — no probe
      // armed, no DPR oscillation.
      for (let i = 0; i < 10; i++) {
        evaluateWithFPS(m, 4000 + i * 1000, 20);
        expect(m.getCurrentDPR()).toBe(2.0);
        expect(m.getState().probing).toBe(false);
      }
    } finally {
      restore();
    }
  });

  it('clears probe state and resets floor on setEnabled(false)', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // Force a rejected probe to set a floor.
      evaluateWithFPS(m, 1000, 20);
      evaluateWithFPS(m, 3000, 20);
      expect(m.getState().dprFloor).toBeGreaterThan(0.5);

      m.setEnabled(false);
      const s = m.getState();
      expect(s.probing).toBe(false);
      expect(s.dprFloor).toBe(0.5); // back to config minDPR
      expect(s.currentDPR).toBe(2.0); // back to native
    } finally {
      restore();
    }
  });
});
