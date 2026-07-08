/**
 * Unit tests for AdaptiveDPRManager.
 *
 * The manager has only two real-world dependencies: `window.devicePixelRatio`
 * (read once in the constructor) and the project `config` module. We set
 * the DPR explicitly before each test and mock the config so the FPS
 * thresholds, scaling factors, and hysteresis are fully under our control.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Config mock — the manager layers this OVER the real data.ts defaults
// (structural per-key fallback), so only the keys under test need to be
// listed; missing knobs (probe/backoff/ceiling timings, FPS ratios)
// resolve to production defaults instead of `undefined`.
//
// Effective thresholds with the default ratios and a warmup cap of 60:
// scale down below 45 fps, count toward scale-up above 54 fps.
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
    // gapResetMs is raised because this test deliberately simulates
    // ~1fps with widely spaced frames — production gap detection would
    // (correctly) treat those as stalls and never evaluate. The halved
    // scaleDownFactor makes the override observable EXACTLY: one
    // scale-down lands at 2.0 × 0.5 = 1.0, not the default 2.0 × 0.9.
    const m = new AdaptiveDPRManager({ scaleDownFactor: 0.5, gapResetMs: 10_000 });
    try {
      const r = makeRenderer();
      m.setRenderer(r);
      // Simulate a very low FPS by pushing 2 frames over a long time.
      m.recordFrame(0);
      m.recordFrame(1000); // 2 frames in 1 second → ~1 FPS, well under the 45fps down-threshold
      // evaluateAndAdjust runs at next recordFrame after 500ms — push one more.
      m.recordFrame(1600);
      expect(m.getCurrentDPR()).toBeCloseTo(1.0, 5);
      expect(r.setAdaptivePixelRatio).toHaveBeenLastCalledWith(1.0);
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

  it('scales DPR down by exactly scaleDownFactor when sustained low FPS is observed', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // 10 FPS = 10 frames over 1 second. The scale-down fires at the
      // first evaluation tick DURING these pushes (t ≥ 500); the trailing
      // frame at 1600 is a >gapResetMs gap and only resets the window.
      pushFrames(m, 0, 10, 1000);
      m.recordFrame(1600);

      // Exactly one multiplicative step: 2.0 × 0.7 (mock scaleDownFactor).
      expect(m.getCurrentDPR()).toBeCloseTo(1.4, 5);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenCalledTimes(1);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenLastCalledWith(1.4);
    } finally {
      restore();
    }
  });

  it('walks accepted probes down to the exact minDPR floor block and never below', () => {
    const m = new AdaptiveDPRManager({ minDPR: 0.5, gapResetMs: 60_000 });
    try {
      m.setRenderer(renderer);
      // Each phase renders ~20% faster than the last, so every probe's
      // settle sample clears the +5% improvement bar and is ACCEPTED —
      // the walk continues: 2.0 → 1.4 → 0.98 → 0.686, then the proposed
      // 0.48 is blocked by the 0.5 floor (to-or-below rule).
      let t = 0;
      const phase = (fps: number, durationMs: number): void => {
        const step = 1000 / fps;
        const end = t + durationMs;
        while (t < end) {
          m.recordFrame(t);
          t += step;
        }
      };
      // Cadence: scale-down+arm fires ~500ms into a phase's predecessor;
      // the settle (arm+1500ms) then reads a window filled entirely with
      // the NEXT phase's faster frames → ratio ≈ 1.2 → accepted.
      phase(12, 1000); // scale-down #1 arms probe (baseline ~12)
      phase(14.4, 2000); // settle #1 accepted; scale-down #2 arms
      phase(17.3, 2000); // settle #2 accepted; scale-down #3 arms
      phase(20.7, 2000); // settle #3 accepted; proposal 0.48 blocked
      phase(24.8, 2000); // stays blocked — no further reduction

      expect(m.getCurrentDPR()).toBeCloseTo(2.0 * 0.7 ** 3, 5);
      // All probes accepted → no rejection floor was ever learned.
      expect(m.getState().dprFloor).toBe(0.5);
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

describe('AdaptiveDPRManager — evidence-based DPR ceiling', () => {
  let restore: () => void;
  let renderer: ReturnType<typeof makeRenderer>;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
    renderer = makeRenderer();
  });

  /** Push frames at `fps` from `t` for `durationMs`; returns the new cursor. */
  function drive(m: AdaptiveDPRManager, t: number, fps: number, durationMs: number): number {
    const step = 1000 / fps;
    const end = t + durationMs;
    while (t < end) {
      m.recordFrame(t);
      t += step;
    }
    return t;
  }

  it('demotes the ceiling to 1.0 after a punished ascent and clamps in one step', () => {
    // threshold 1 keeps the scenario tractable; hysteresis 1s speeds the
    // scale-up; gap detection off (deliberate fps regime switches).
    const m = new AdaptiveDPRManager({
      punishedAscentThreshold: 1,
      hysteresisSeconds: 1,
      gapResetMs: 60_000,
    });
    try {
      m.setRenderer(renderer);

      // Phase 1 — heavy: scale down off native (2.0 → 1.4), probe armed.
      let t = drive(m, 0, 20, 700);
      expect(m.getCurrentDPR()).toBeCloseTo(1.4, 5);

      // Phase 2 — light: probe ACCEPTED (fps ×4), then the sustained
      // high streak scales up ABOVE 1.0 (1.4 → 1.68): an ascent on
      // probation.
      t = drive(m, t, 80, 3200);
      expect(m.getState().probing).toBe(false);
      expect(m.getCurrentDPR()).toBeCloseTo(1.68, 2);

      // Phase 3 — collapse right after the ascent: the slow sample
      // punishes it; at threshold 1 the ceiling demotes to exactly 1.0
      // and the operating DPR clamps there in ONE step (1.68 → 1.0, not
      // a 0.7× walk). The scene is still slow at 1.0, so later ticks
      // may legitimately probe below it — the clamp itself must appear
      // verbatim in the renderer call history.
      renderer.setAdaptivePixelRatio.mockClear();
      t = drive(m, t, 20, 1300);
      expect(m.getState().dprCeiling).toBe(1.0);
      expect(renderer.setAdaptivePixelRatio.mock.calls.map((c) => c[0])).toContain(1.0);
      expect(m.getCurrentDPR()).toBeLessThanOrEqual(1.0);

      // Scale-up can no longer cross 1.0 while demoted.
      t = drive(m, t, 80, 3200);
      expect(m.getCurrentDPR()).toBeLessThanOrEqual(1.0);
      void t;
    } finally {
      restore();
    }
  });

  it('resume-from-idle snaps to min(ceiling, remembered operating DPR)', () => {
    const m = new AdaptiveDPRManager({
      punishedAscentThreshold: 1,
      hysteresisSeconds: 1,
      gapResetMs: 60_000,
    });
    try {
      m.setRenderer(renderer);
      // Earn the demotion (same choreography as above).
      let t = drive(m, 0, 20, 700);
      t = drive(m, t, 80, 3200);
      t = drive(m, t, 20, 1300);
      expect(m.getState().dprCeiling).toBe(1.0);

      // Idle: the RESTING frame still renders at full native (the
      // ceiling governs interactive rendering, not the still image)...
      const operatingDPR = m.getCurrentDPR(); // ≤ 1.0 post-demotion
      m.prepareIdleFrame();
      expect(m.getCurrentDPR()).toBe(2.0);

      // ...and resume snaps back EXACTLY to the remembered operating
      // DPR (which post-demotion is already at or below the ceiling —
      // the ceiling term in the resume min() is a defensive invariant,
      // structurally non-binding today).
      m.notifyResumed();
      expect(m.getCurrentDPR()).toBe(operatingDPR);
      expect(m.getCurrentDPR()).toBeLessThanOrEqual(1.0);
      void t;
    } finally {
      restore();
    }
  });

  it('exposes the effective ceiling as native when not demoted', () => {
    const m = new AdaptiveDPRManager();
    try {
      expect(m.getState().dprCeiling).toBe(2.0);
    } finally {
      restore();
    }
  });
});

describe('AdaptiveDPRManager — pause / idle-restore / resume', () => {
  let restore: () => void;
  let renderer: ReturnType<typeof makeRenderer>;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
    renderer = makeRenderer();
  });

  /** Drive the manager to a reduced DPR with an armed probe (20fps). */
  function reduceWithProbe(m: AdaptiveDPRManager): number {
    let t = 0;
    for (let i = 0; i < 14; i++) {
      m.recordFrame(t);
      t += 50;
    }
    expect(m.getState().probing).toBe(true);
    return t;
  }

  it('notifyPaused clears session state but PRESERVES the learned floor and backoff', () => {
    const m = new AdaptiveDPRManager({ gapResetMs: 60_000 });
    try {
      m.setRenderer(renderer);
      // Build a REJECTED probe so a floor + backoff streak exist.
      let t = reduceWithProbe(m);
      for (let i = 0; i < 44; i++) {
        m.recordFrame(t);
        t += 50;
      }
      const floorBefore = m.getState().dprFloor;
      expect(floorBefore).toBeGreaterThan(0.5);

      m.notifyPaused();

      const s = m.getState();
      // Session state gone...
      expect(s.currentFPS).toBe(0);
      expect(s.probing).toBe(false);
      // ...learned evidence intact. (Copying the setEnabled(false) wipe
      // here would replay a rejected-probe blur episode on EVERY
      // interaction burst — strictly worse than the old 30s cycle.)
      expect(s.dprFloor).toBe(floorBefore);
    } finally {
      restore();
    }
  });

  it('notifyPaused voids an in-flight probe without judging it', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      const dpr = (reduceWithProbe(m), m.getCurrentDPR());

      m.notifyPaused();
      expect(m.getState().probing).toBe(false);
      expect(m.getCurrentDPR()).toBe(dpr); // kept, not reverted
      expect(m.getState().dprFloor).toBe(0.5); // nothing learned
      // Idempotent + safe to repeat (dispose path calls stopAnimation).
      m.notifyPaused();
    } finally {
      restore();
    }
  });

  it('prepareIdleFrame restores native for the resting frame and remembers the operating DPR', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      reduceWithProbe(m);
      const operatingDPR = m.getCurrentDPR();
      expect(operatingDPR).toBeLessThan(2.0);

      const changed = m.prepareIdleFrame();
      expect(changed).toBe(true); // caller must render one frame
      expect(m.getCurrentDPR()).toBe(2.0);
      expect(m.getIsReducedResolution()).toBe(false);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenLastCalledWith(2.0);

      // Resume snaps straight back in ONE step — no reactive re-walk.
      renderer.setAdaptivePixelRatio.mockClear();
      m.notifyResumed();
      expect(m.getCurrentDPR()).toBe(operatingDPR);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenCalledTimes(1);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenLastCalledWith(operatingDPR);
    } finally {
      restore();
    }
  });

  it('prepareIdleFrame preserves the floor/backoff across the idle cycle', () => {
    const m = new AdaptiveDPRManager({ gapResetMs: 60_000 });
    try {
      m.setRenderer(renderer);
      let t = reduceWithProbe(m);
      for (let i = 0; i < 44; i++) {
        m.recordFrame(t);
        t += 50;
      }
      const floorBefore = m.getState().dprFloor;
      expect(floorBefore).toBeGreaterThan(0.5);

      m.notifyPaused();
      m.prepareIdleFrame();
      m.notifyResumed();

      expect(m.getState().dprFloor).toBe(floorBefore);
    } finally {
      restore();
    }
  });

  it('prepareIdleFrame is a no-op at native DPR or when disabled', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // Already at native: nothing to restore, caller must NOT render.
      expect(m.prepareIdleFrame()).toBe(false);
      expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();

      reduceWithProbe(m);
      m.setEnabled(false); // resets to native and disables
      renderer.setAdaptivePixelRatio.mockClear();
      expect(m.prepareIdleFrame()).toBe(false);
      expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();
    } finally {
      restore();
    }
  });

  it('notifyResumed without a preceding idle restore is a no-op', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      reduceWithProbe(m);
      const dpr = m.getCurrentDPR();
      renderer.setAdaptivePixelRatio.mockClear();

      m.notifyResumed(); // no restingAtNative flag set
      expect(m.getCurrentDPR()).toBe(dpr);
      expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();
    } finally {
      restore();
    }
  });

  it('resume clamps the remembered operating DPR to the live native (monitor change while idle)', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      reduceWithProbe(m); // operating DPR 1.4 on native 2.0
      expect(m.getCurrentDPR()).toBeCloseTo(1.4, 5);
      m.prepareIdleFrame();

      setNativeDPR(1.0); // display changed while resting
      m.notifyResumed();
      // min(remembered 1.4, live 1.0) — never supersample on resume.
      expect(m.getCurrentDPR()).toBeLessThanOrEqual(1.0);
    } finally {
      restore();
    }
  });
});

describe('AdaptiveDPRManager — gap detection & load suppression', () => {
  let restore: () => void;
  let renderer: ReturnType<typeof makeRenderer>;

  beforeEach(() => {
    restore = setNativeDPR(1.0);
    renderer = makeRenderer();
  });

  it('a frame gap resets the FPS window instead of ratcheting a spurious scale-down', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // 500ms of healthy 60fps...
      let t = 0;
      for (let i = 0; i < 31; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      // ...then a 400ms stall (GC pause / sync decode), then healthy
      // 60fps again. Without gap detection the window would blend the
      // dead time into the estimate (~36fps < 45) and scale down a
      // scene that renders perfectly fine.
      t += 400;
      for (let i = 0; i < 60; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      expect(m.getCurrentDPR()).toBe(1.0);
      expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();
    } finally {
      restore();
    }
  });

  it('a gap voids a pending probe without judging it', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // Sustained 20fps arms a scale-down probe.
      let t = 0;
      for (let i = 0; i < 14; i++) {
        m.recordFrame(t);
        t += 50;
      }
      expect(m.getState().probing).toBe(true);
      const probedDPR = m.getCurrentDPR();

      // Stall across the probe window, then resume: the probe's data is
      // contaminated — it must be voided (DPR kept, no floor learned),
      // not settled against post-stall jank.
      m.recordFrame(t + 5000);
      expect(m.getState().probing).toBe(false);
      expect(m.getCurrentDPR()).toBe(probedDPR);
      expect(m.getState().dprFloor).toBe(0.5);
    } finally {
      restore();
    }
  });

  it('load-suppressed low FPS scales down WITHOUT arming a probe', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      m.setLoadActivityPredicate(() => true);

      let t = 0;
      for (let i = 0; i < 14; i++) {
        m.recordFrame(t);
        t += 50; // 20fps < 45 down-threshold
      }
      // The reduction applies (fewer pixels help a janky load too)...
      expect(m.getCurrentDPR()).toBeLessThan(1.0);
      // ...but jank samples never become probe/floor evidence.
      expect(m.getState().probing).toBe(false);
      expect(m.getState().dprFloor).toBe(0.5);
    } finally {
      restore();
    }
  });

  it('notifyContentChanged pulls a learned floor forward (burst calls stay observably inert)', () => {
    // NOTE on coalescing: softenForContentChange only ever MIN()s the
    // expiry, so an uncoalesced second call could not extend it either —
    // the coalescing guard saves ledger/log churn, not correctness, and
    // is therefore not separately observable through the floor clock.
    const m = new AdaptiveDPRManager({ gapResetMs: 60_000 });
    try {
      m.setRenderer(renderer);
      // Build a rejected probe: 20fps scale-down, then settle at the
      // same 20fps past the probe window → revert + floor.
      let t = 0;
      for (let i = 0; i < 14; i++) {
        m.recordFrame(t);
        t += 50;
      }
      expect(m.getState().probing).toBe(true);
      for (let i = 0; i < 44; i++) {
        m.recordFrame(t);
        t += 50; // continues 20fps through settle (~2.9s total)
      }
      expect(m.getState().probing).toBe(false);
      expect(m.getState().dprFloor).toBeGreaterThan(0.5); // floor learned

      // Content changed: floor expiry is pulled to recheckMs (5s) from
      // now instead of the 30s TTL.
      m.notifyContentChanged(t);
      // Coalesced: an immediate second call within recheckMs must be a
      // no-op (no clock resets/extensions).
      m.notifyContentChanged(t + 100);

      // Advance past recheckMs (plus one full evaluation interval so a
      // post-expiry evaluation actually fires) — floor decays.
      const decayAt = t + 6500;
      while (t < decayAt) {
        m.recordFrame(t);
        t += 50;
      }
      expect(m.getState().dprFloor).toBe(0.5);
    } finally {
      restore();
    }
  });
});

describe('AdaptiveDPRManager — refresh-rate-relative thresholds', () => {
  let restore: () => void;

  beforeEach(() => {
    restore = setNativeDPR(1.0);
  });

  it('scales down at 80fps on a 120Hz display (fixed 50fps thresholds never would)', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);

      // A light 1.5s teaches the estimator the display can do 120Hz.
      let t = 0;
      for (let i = 0; i < 180; i++) {
        m.recordFrame(t);
        t += 1000 / 120;
      }
      expect(m.getState().refreshRateCap).toBeGreaterThan(115);
      expect(m.getCurrentDPR()).toBe(1.0); // 120 > upThreshold but already at native

      // Sustained 80fps: healthy under the old fixed 50fps rule, but
      // below 0.75 × 120 = 90 → the relative rule scales down.
      for (let i = 0; i < 160; i++) {
        m.recordFrame(t);
        t += 1000 / 80;
      }
      expect(m.getCurrentDPR()).toBeLessThan(1.0);
    } finally {
      restore();
    }
  });

  it('notifyContentChanged resets a LATCHED throttle cap, not only the downshift permission', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(makeRenderer());
      // Prove 120, then a genuine throttle signature latches the low cap.
      let t = 0;
      for (let i = 0; i < 180; i++) {
        m.recordFrame(t);
        t += 1000 / 120;
      }
      for (let i = 0; i < 480; i++) {
        m.recordFrame(t);
        t += 1000 / 30; // uniform 30fps for 16s (mixed-window ramp +
        // 10s persistence) → downshift latches
      }
      expect(m.getState().refreshRateCap).toBeCloseTo(30, 6);

      // The throttle verdict is content-relative: after a content change
      // the cap must return to the fallback-floored bound so heavy new
      // content at ~35fps is correctly below the down-threshold instead
      // of above an inverted up-threshold (31.5).
      m.notifyContentChanged(t);
      expect(m.getState().refreshRateCap).toBeGreaterThanOrEqual(60);
    } finally {
      restore();
    }
  });

  it('notifyContentChanged disarms the throttle downshift for the new content', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(makeRenderer());
      // Light loading phase proves a 120Hz cap...
      let t = 0;
      for (let i = 0; i < 180; i++) {
        m.recordFrame(t);
        t += 1000 / 120;
      }
      expect(m.getState().refreshRateCap).toBeGreaterThan(115);

      // ...then the content changes and the new scene parks at a steady
      // uniform 20fps. The old proof must not license a throttle
      // downshift: the cap must NOT collapse onto the loaded FPS (which
      // would invert the thresholds and arm scale-up on a GPU-bound
      // scene).
      m.notifyContentChanged(t);
      for (let i = 0; i < 300; i++) {
        m.recordFrame(t);
        t += 50; // 20fps for 15s
      }
      expect(m.getState().refreshRateCap).toBeGreaterThan(115);
    } finally {
      restore();
    }
  });

  it('setEnabled(false) wipes the learned refresh-cap estimate along with the other learned state', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(makeRenderer());
      // Teach the estimator a 120Hz cap.
      let t = 0;
      for (let i = 0; i < 180; i++) {
        m.recordFrame(t);
        t += 1000 / 120;
      }
      expect(m.getState().refreshRateCap).toBeGreaterThan(115);

      // Disable/re-enable on the SAME display must reset the estimate
      // to the warmup fallback — a learned (possibly throttled) cap
      // surviving the toggle would mis-arm the relative thresholds.
      m.setEnabled(false);
      m.setEnabled(true);
      expect(m.getState().refreshRateCap).toBe(60);
    } finally {
      restore();
    }
  });
});

describe('AdaptiveDPRManager — live native DPR (monitor / zoom changes)', () => {
  let restore: () => void;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
  });

  it('getNativeDPR() reflects a devicePixelRatio change after construction', () => {
    const m = new AdaptiveDPRManager();
    try {
      expect(m.getNativeDPR()).toBe(2.0);
      setNativeDPR(1.0);
      expect(m.getNativeDPR()).toBe(1.0);
      expect(m.getState().nativeDPR).toBe(1.0);
    } finally {
      restore();
    }
  });

  it('follows the new native silently when tracking native (no redundant re-apply)', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);

      setNativeDPR(1.0);
      // Live-consistent getCurrentDPR: never reports the stale native.
      expect(m.getCurrentDPR()).toBe(1.0);
      // Tracking-native path must NOT reallocate render targets — the
      // renderer follows live DPR by itself via the null override.
      expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();
    } finally {
      restore();
    }
  });

  it('clamps an engaged manual DPR above the new native and re-applies (no supersampling)', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);
      m.setEnabled(false);
      m.setManualDPR(1.5); // engaged below native 2.0
      renderer.setAdaptivePixelRatio.mockClear();

      setNativeDPR(1.0); // move to a 1x monitor
      expect(m.getCurrentDPR()).toBe(1.0);
      // The 1.5 override would supersample on the 1x display — must be
      // clamped down and re-applied exactly once.
      expect(renderer.setAdaptivePixelRatio).toHaveBeenCalledTimes(1);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenLastCalledWith(1.0);
    } finally {
      restore();
    }
  });

  it('keeps an engaged reduced DPR that is still below the new native', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);
      m.setEnabled(false);
      m.setManualDPR(0.5);
      renderer.setAdaptivePixelRatio.mockClear();

      setNativeDPR(1.0);
      // 0.5 is still valid below native 1.0 — no clamp, no re-apply.
      expect(m.getCurrentDPR()).toBe(0.5);
      expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();
    } finally {
      restore();
    }
  });

  it('setEnabled(false) resets to the LIVE native, not the construction-time snapshot', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);

      setNativeDPR(1.0);
      m.setEnabled(false);

      expect(m.getCurrentDPR()).toBe(1.0);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenLastCalledWith(1.0);
    } finally {
      restore();
    }
  });

  it('setManualDPR clamps against the LIVE native', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(makeRenderer());
      m.setEnabled(false);

      setNativeDPR(1.0);
      m.setManualDPR(2.0); // old native — must clamp to live 1.0
      expect(m.getCurrentDPR()).toBe(1.0);
    } finally {
      restore();
    }
  });

  it('a native change clears the U-shape floor and any pending probe', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);

      // Drive a scale-down (arms a probe) with sustained low FPS.
      let t = pushFrames(m, 0, 12, 1100); // ~10 fps < 45fps down-threshold
      expect(m.getState().probing).toBe(true);

      setNativeDPR(1.0);
      const state = m.getState();
      // Probe voided and floor back at config minDPR (0.5) — the old
      // display's absolute-DPR calibrations are meaningless now.
      expect(state.probing).toBe(false);
      expect(state.dprFloor).toBe(0.5);
      // FPS window cleared: fresh samples required before any decision.
      expect(state.currentFPS).toBe(0);
      void t;
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
  /**
   * evaluateWithFPS timelines have >350ms dead zones between windows.
   * Production gap detection would (correctly) void the armed probes
   * across those zones — gap behavior has its own dedicated suite — so
   * the U-shape tests disable it to keep single-window semantics.
   */
  function makeProbeManager(): AdaptiveDPRManager {
    return new AdaptiveDPRManager({ gapResetMs: 60_000 });
  }

  function evaluateWithFPS(m: AdaptiveDPRManager, evalAt: number, targetFPS: number): void {
    const start = evalAt - 1000;
    const N = Math.max(2, Math.round(targetFPS));
    for (let i = 0; i < N; i++) {
      m.recordFrame(start + (i / (N - 1)) * 1000);
    }
  }

  it('keeps the DPR move when post-probe FPS improves enough', () => {
    const m = makeProbeManager();
    try {
      m.setRenderer(renderer);

      // Eval at t=1000 with FPS=20 → below the 45fps down-threshold → scaleDown, probe armed.
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
    const m = makeProbeManager();
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
    const m = makeProbeManager();
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
    const m = makeProbeManager();
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
    const m = makeProbeManager();
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

// ---------------------------------------------------------------------
// Sub-throttle distress → ceiling demotion (the 10fps-at-native latch)
// ---------------------------------------------------------------------

describe('AdaptiveDPRManager — sustained distress demotes the ceiling to 1.0', () => {
  it('a proven-fast display + GPU-bound ~10fps scene ends at DPR 1.0, not parked at native', () => {
    // Regression for the catastrophic estimator latch: the display
    // proves ~120Hz, then heavy content pins FPS at a uniform ~10.
    // The old throttle detector reseeded the cap onto 10fps, the
    // relative thresholds then read 10fps as "at the display cap =
    // healthy", and scale-up walked the DPR back to native 2.0 and
    // parked it there for the content's lifetime. Now the estimator
    // reports sub-throttle DISTRESS instead and the manager demotes
    // the ceiling: the session must end operating at exactly 1.0.
    const restore = setNativeDPR(2.0);
    const m = new AdaptiveDPRManager();
    const r = makeRenderer();
    m.setRenderer(r);
    try {
      // Phase A: light content proves the display can do ~120fps.
      let t = pushFrames(m, 0, 241, 2000); // 8.3ms cadence ≈ 120fps
      // Phase B: heavy content — a uniform ~10fps for 25s.
      pushFrames(m, t + 100, 251, 25_000); // 100ms cadence = 10fps

      const s = m.getState();
      expect(s.dprCeiling).toBe(1.0); // demoted
      expect(s.currentDPR).toBe(1.0); // clamped in one step
      expect(s.isReducedResolution).toBe(true);
      // The cap must NOT have collapsed onto the loaded FPS — that was
      // the latch. Scale-down thresholds stay armed against it.
      expect(s.refreshRateCap).toBeGreaterThanOrEqual(60);
      expect(r.setAdaptivePixelRatio).toHaveBeenCalledWith(1.0);
    } finally {
      restore();
    }
  });

  it('a scene heavy from the very first frame also gets the 1.0 clamp (no proven rate needed)', () => {
    const restore = setNativeDPR(2.0);
    const m = new AdaptiveDPRManager();
    const r = makeRenderer();
    m.setRenderer(r);
    try {
      pushFrames(m, 0, 251, 25_000); // ~10fps from the start
      const s = m.getState();
      expect(s.dprCeiling).toBe(1.0);
      expect(s.currentDPR).toBe(1.0);
    } finally {
      restore();
    }
  });
});
