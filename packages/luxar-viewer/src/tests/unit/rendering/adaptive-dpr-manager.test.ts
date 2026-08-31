// @vitest-environment jsdom
/**
 * Unit tests for AdaptiveDPRManager.
 *
 * The manager has three real-world dependencies: `window.devicePixelRatio`,
 * the shared pixel-ratio cap (`rendering/pixel-ratio-cap`), and the
 * project `config` module. We set the DPR and the cap explicitly before
 * each test and mock the config so the FPS thresholds, scaling factors,
 * and hysteresis are fully under our control.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

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
import { BoundsLedger } from '../../../rendering/adaptive-dpr/bounds-ledger';
import { DEFAULT_MAX_PIXEL_RATIO, setMaxPixelRatioCap } from '../../../rendering/pixel-ratio-cap';
import { allowHighDPR, setNativeDPR } from '../../helpers/device-pixel-ratio';

// ---------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------

// Every block below predates the pixel-ratio cap and asserts against the
// DISPLAY's DPR — "resets to native when disabled", "scales up to
// native", the 2.0 -> 1.0 learned-ceiling choreography — so the whole
// file runs with high DPR ALLOWED. That keeps the coverage those tests
// were written for: under the shipped 1.0 cap the ceiling-demotion and
// distress blocks would still PASS while testing nothing, because their
// starting DPR is already the value they assert the manager arrives at.
//
// The capped default has its own block at the end of this file.
let restoreCapForFile: () => void;
beforeEach(() => {
  restoreCapForFile = allowHighDPR();
});
afterEach(() => restoreCapForFile());

function makeRenderer(): DPRRenderer & { setAdaptivePixelRatio: ReturnType<typeof vi.fn> } {
  return {
    setAdaptivePixelRatio: vi.fn(),
  } as DPRRenderer & { setAdaptivePixelRatio: ReturnType<typeof vi.fn> };
}

/** Every DPR the manager pushed to the renderer, in order. */
function appliedDPRs(renderer: ReturnType<typeof makeRenderer>): number[] {
  return renderer.setAdaptivePixelRatio.mock.calls.map((c) => c[0] as number);
}

/**
 * Evaluations WITH a frame rate to judge that 40 cycles of
 * [400ms dead time, two 60fps frames] produce. Both the cadence and the
 * 500ms evaluation interval are fixed literals, so this is exact, not a
 * lower bound — pre-fix it was ZERO for as long as the pattern lasted.
 */
const EXPECTED_EVALUATIONS_ON_RECURRING_STALL = 20;
/**
 * DPR that 30 cycles of [400ms dead time, two 33ms frames] end at: ONE
 * multiplicative step off native (1.0 × the 0.7 mock scaleDownFactor).
 * The reduction lands and then holds — a probe is armed on the first
 * trusted window and every following gap voids it, so the walk does not
 * ratchet on this cadence.
 */
const EXPECTED_DPR_ON_SLOW_RECURRING_STALL = 0.7;

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

  it('adapts to a SUSTAINED frame rate slower than gapResetMs (software-rasterizer regime)', () => {
    // Regression for the structurally-inert adaptive DPR measured under
    // ANGLE/SwiftShader: below ~1000/gapResetMs fps EVERY interval looks
    // like a stall, so the pre-fix code cleared the window and pushed
    // lastEvaluationTime to `timestamp` on every single frame —
    // evaluateAndAdjust never ran and the DPR stayed parked at native
    // for the whole episode, exactly where shedding pixels helps most.
    const m = new AdaptiveDPRManager(); // production gapResetMs (350)
    try {
      m.setRenderer(renderer);
      // ~1.4fps: every 700ms interval is longer than gapResetMs. The
      // first one has no cadence to be an outlier against (cold memory →
      // the absolute floor decides alone) so it still resets the window;
      // from the second on the median IS 700ms, the interval is no
      // longer an outlier, and the samples are kept — so the third frame
      // completes a 700ms window at ~1.4fps and the evaluation fires.
      let t = 0;
      for (let i = 0; i < 3; i++) {
        m.recordFrame(t);
        t += 700;
      }
      expect(m.getCurrentFPS()).toBeCloseTo(1000 / 700, 2);
      // One multiplicative step off native: 1.0 × 0.7 (mock factor).
      expect(m.getCurrentDPR()).toBeCloseTo(0.7, 5);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenLastCalledWith(0.7);
    } finally {
      restore();
    }
  });

  it('a ONE-OFF long stall still resets the window and causes no scale-down', () => {
    // The behaviour the gap reset exists for must survive the
    // sustained-slow fix: an isolated 5s stall between healthy 60fps
    // stretches is dead time, not a frame rate. Without the reset the
    // window would blend the stall into the estimate (~0.2fps) and
    // ratchet a scale-down on a scene that renders perfectly fine.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let t = 0;
      for (let i = 0; i < 61; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      // One 5s stall (tab switch / synchronous decode).
      t += 5000;
      m.recordFrame(t);
      expect(m.getCurrentFPS()).toBe(0); // window cleared, sampling fresh
      t += 1000 / 60;

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

  it('catches a later isolated stall once the cadence is fast again', () => {
    // A sustained slow cadence stops being read as dead time (it IS the
    // frame rate); once the scene lightens, the cadence memory follows
    // it back down, so a later isolated stall is again a large outlier
    // and is discarded rather than folded into the estimate.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // Phase 1 — sustained ~1.4fps: the manager adapts (see above).
      let t = 0;
      for (let i = 0; i < 3; i++) {
        m.recordFrame(t);
        t += 700;
      }
      expect(m.getCurrentDPR()).toBeCloseTo(0.7, 5);

      // Phase 2 — the scene lightens to a healthy 60fps for 2s.
      for (let i = 0; i < 120; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      const beforeStall = m.getCurrentDPR();
      renderer.setAdaptivePixelRatio.mockClear();

      // Phase 3 — one isolated 5s stall. With the reset re-armed the
      // window is cleared; without it the stall would read as 0.2fps and
      // ratchet another scale-down.
      t += 5000;
      m.recordFrame(t);
      expect(m.getCurrentFPS()).toBe(0);
      t += 1000 / 60;
      for (let i = 0; i < 30; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      expect(m.getCurrentDPR()).toBe(beforeStall);
      expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();
    } finally {
      restore();
    }
  });

  it('TWO consecutive hitches inside a healthy session change nothing (F2 regression)', () => {
    // Regression for the discarded "only the first over-threshold
    // interval of a run is a stall" rule: with that rule the SECOND
    // 400ms hitch was declared the frame rate, the window kept the dead
    // time, and a clean 60fps session scaled DPR down (2.0 → 1.8 as
    // measured, with the probe then RATIFYING it against a 0.2fps
    // baseline). Under the outlier rule both hitches are ~24× the 16.7ms
    // median, so both are dead time and nothing moves.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let t = 0;
      for (let i = 0; i < 120; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      t += 400; // hitch #1
      m.recordFrame(t);
      t += 400; // hitch #2, back to back
      m.recordFrame(t);
      t += 1000 / 60;
      for (let i = 0; i < 240; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }

      expect(m.getCurrentDPR()).toBe(1.0);
      expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();
      expect(m.getState().dprFloor).toBe(0.5); // nothing learned either
    } finally {
      restore();
    }
  });

  it.each([
    ['400/300ms (~2.9fps)', 400, 300],
    ['500/100ms (~3.3fps)', 500, 100],
    ['2000/100ms (~0.95fps)', 2000, 100],
  ])('adapts to an ALTERNATING slow cadence — %s (F1 regression)', (_label, slowMs, fastMs) => {
    // The cadence-fragility class: with the discarded "first of a run"
    // rule, every sub-threshold interval re-armed the reset AND pushed
    // lastEvaluationTime forward, so these three cadences produced
    // ZERO DPR changes over minutes — the 2000/100 one being the same
    // class as the originally reported 0.48fps episode. The outlier
    // rule converges after one or two intervals and then adapts.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // Warm the cadence memory with a healthy 60fps stretch first —
      // the adversarial start, since a fast median makes the first
      // slow intervals look like stalls.
      let t = 0;
      for (let i = 0; i < 120; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      for (let i = 0; i < 20; i++) {
        t += i % 2 === 0 ? slowMs : fastMs;
        m.recordFrame(t);
      }

      // The loop ran and shed pixels. (Whether the reduction STANDS is
      // the U-shape probe's business — these synthetic streams keep the
      // exact same cadence after the scale-down, so some of them are
      // legitimately probe-rejected and reverted. What must never
      // happen again is the pre-fix outcome: zero applies, ever.)
      expect(appliedDPRs(renderer).length).toBeGreaterThan(0);
      expect(Math.min(...appliedDPRs(renderer))).toBeLessThan(1.0);
    } finally {
      restore();
    }
  });

  it('adapts to a uniform 0.5fps stream — slower than the FPS window itself (F3)', () => {
    // Pins the FPS tracker's retention floor at the MANAGER level. At a
    // 2000ms cadence the previous frame is twice the window's age when
    // the next arrives, so age-only trimming leaves ONE sample and
    // getFPS() returns 0 — evaluateAndAdjust bails on "not enough data"
    // and the DPR never moves, which is exactly the measured 0.48fps
    // episode. Reverting FPSTracker.minRetainedSamples must fail HERE,
    // not only in the tracker's own unit test.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let t = 0;
      for (let i = 0; i < 12; i++) {
        m.recordFrame(t);
        t += 2000; // 0.5fps
      }

      expect(m.getCurrentFPS()).toBeCloseTo(0.5, 3);
      expect(appliedDPRs(renderer)).toContain(0.7); // 1.0 × scaleDownFactor
      expect(Math.min(...appliedDPRs(renderer))).toBeLessThan(1.0);
    } finally {
      restore();
    }
  });

  it('recovers: DPR walks back up when the scene lightens, and a later stall is still caught', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // Phase 1 — 0.5fps: three frames are enough to scale down and arm
      // a probe (the third completes a 2s window at 0.5fps).
      let t = 0;
      m.recordFrame(t);
      m.recordFrame((t += 2000));
      m.recordFrame((t += 2000));
      const reduced = m.getCurrentDPR();
      expect(reduced).toBeCloseTo(0.7, 5);

      // Phase 2 — the scene lightens to 60fps for 6s: the pending probe
      // settles (hugely improved → accepted), then the sustained-high
      // streak clears hysteresisSeconds and the DPR walks back up.
      for (let i = 0; i < 360; i++) {
        t += 1000 / 60;
        m.recordFrame(t);
      }
      expect(m.getCurrentDPR()).toBeGreaterThan(reduced);

      // Phase 3 — one isolated 5s stall is again a large outlier against
      // the now-fast median: window cleared, no reduction ratcheted.
      const beforeStall = m.getCurrentDPR();
      renderer.setAdaptivePixelRatio.mockClear();
      t += 5000;
      m.recordFrame(t);
      expect(m.getCurrentFPS()).toBe(0); // window cleared → dead time dropped
      t += 1000 / 60;
      for (let i = 0; i < 30; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      expect(m.getCurrentDPR()).toBe(beforeStall);
      expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();
    } finally {
      restore();
    }
  });

  it('a session boundary forgets the cadence, so the first long interval after it is dead time again', () => {
    // The cadence memory survives the gap reset it drives (that is what
    // makes it converge) but NOT a real session boundary: after a pause
    // the pre-pause cadence describes nothing, and the first long
    // interval may well be resume dead time the pause hook mis-timed.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // Converge on 0.5fps: 2000ms intervals are "normal" for this scene.
      let t = 0;
      for (let i = 0; i < 6; i++) {
        m.recordFrame(t);
        t += 2000;
      }
      // Negative control: with the memory intact, the next 2000ms
      // interval is the frame rate and the window keeps both samples.
      m.recordFrame(t);
      m.recordFrame(t + 2000);
      expect(m.getCurrentFPS()).toBeCloseTo(0.5, 3);

      // Same two frames, but across a pause: cadence forgotten → the
      // cold-memory rule falls back to the absolute floor → dead time →
      // the window is thrown away.
      m.notifyPaused();
      m.recordFrame(t + 4000);
      m.recordFrame(t + 6000);
      expect(m.getCurrentFPS()).toBe(0);
    } finally {
      restore();
    }
  });

  it.each([
    [
      'a native-DPR change',
      (m: AdaptiveDPRManager) => {
        setNativeDPR(1.5);
        m.getState(); // any public read syncs the display
      },
    ],
    [
      'setEnabled(false)',
      (m: AdaptiveDPRManager) => {
        m.setEnabled(false);
        m.setEnabled(true);
      },
    ],
    ['dispose()', (m: AdaptiveDPRManager) => m.dispose()],
  ])('%s forgets the cadence too, not only notifyPaused', (_label, boundary) => {
    // Four documents state the contract "the cadence memory is cleared
    // wherever the frame STREAM breaks — pause, display change, disable,
    // dispose", but only the pause door was pinned: removing the
    // stallDetector.clear() call from any of the other three left every
    // test green. Each is observable the same way — a converged slow
    // cadence must read as DEAD TIME again once the memory is cold.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let t = 0;
      for (let i = 0; i < 6; i++) {
        m.recordFrame(t);
        t += 2000; // converge on 0.5fps: 2000ms is "normal" here
      }
      // Negative control: with the memory intact the pair below survives.
      m.recordFrame(t);
      m.recordFrame(t + 2000);
      expect(m.getCurrentFPS()).toBeCloseTo(0.5, 3);

      boundary(m);

      m.recordFrame(t + 4000);
      m.recordFrame(t + 6000);
      expect(m.getCurrentFPS()).toBe(0);
    } finally {
      restore();
    }
  });

  it('a stall burst absorbed as the frame rate is UNTRUSTED for learning', () => {
    // The detector compares each interval against the median of its four
    // PRECEDING neighbours, so by the THIRD consecutive dead interval
    // half that memory is dead time and the interval is reclassified as
    // the frame rate — on a window holding nothing but dead time.
    // Trusting that window armed a probe, settled it against a baseline
    // measured on the SAME dead time, and pinned a U-shape floor (with
    // exponential backoff) onto a session rendering at a healthy 60fps
    // either side of the burst.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let t = 0;
      for (let i = 0; i < 61; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      expect(m.getCurrentDPR()).toBe(1.0);

      // Three 5s dead intervals in a row.
      for (let i = 0; i < 3; i++) {
        t += 5000;
        m.recordFrame(t);
      }
      // A reduction applies — fewer pixels never hurt a stuttering
      // loop, and it is one multiplicative step, not a walk...
      expect(m.getCurrentDPR()).toBeCloseTo(0.7, 5);
      // ...but nothing is LEARNED from a window made only of dead time.
      expect(m.getState().probing).toBe(false);
      expect(m.getState().dprFloor).toBe(0.5);

      // Healthy 60fps again for 10s: scale-up restores the DPR and the
      // floor is still untouched, so no backoff ladder was started.
      for (let i = 0; i < 600; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      expect(m.getCurrentDPR()).toBeCloseTo(1.0, 2);
      expect(m.getState().dprFloor).toBe(0.5);
    } finally {
      restore();
    }
  });

  it('a burst of SHORT (400ms) hitches teaches nothing either — the bound in wall clock', () => {
    // The cadence-trust bound is counted in INTERVALS, so at 400ms a
    // burst of four is only 1.6s of wall clock — far less protection than
    // the 20s that four 5s gaps buy. What keeps a short burst harmless is
    // that a probe ALSO needs its own window and a representative span
    // before it can pin anything. Measured with the production 0.9 step,
    // the turnover sits at NINE: eight consecutive 400ms hitches (3.2s)
    // leave the floor untouched and the DPR lifts again, nine (3.6s) pin
    // a floor — at which point the "burst" is a slow regime worth
    // learning from. Both sides are pinned below, because the wall-clock
    // figure is the cadence-dependent one (the guarantee itself is
    // counted in intervals) and a claim about it has to be measured.
    // All the other stall tests use 5000ms gaps, so this length is its
    // own case.
    const m = new AdaptiveDPRManager({ scaleDownFactor: 0.9 });
    try {
      m.setRenderer(renderer);
      let t = 0;
      for (let i = 0; i < 1200; i++) {
        m.recordFrame(t);
        t += 1000 / 60; // 20s of healthy 60fps
      }
      for (let i = 0; i < 8; i++) {
        t += 400;
        m.recordFrame(t);
      }
      expect(m.getState().dprFloor).toBe(0.5); // nothing learned...
      expect(m.getCurrentDPR()).toBeLessThan(1.0); // ...but pixels were shed

      // Healthy again: the transient reduction lifts, floor still clean.
      const end = t + 20_000;
      while (t < end) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      expect(m.getCurrentDPR()).toBeCloseTo(1.0, 2);
      expect(m.getState().dprFloor).toBe(0.5);

      // The other side of the boundary: ONE more hitch is enough to pin a
      // floor, so the wall-clock bound is 3.6s here and not "roughly five
      // seconds".
      const m9 = new AdaptiveDPRManager({ scaleDownFactor: 0.9 });
      m9.setRenderer(makeRenderer());
      let t9 = 0;
      for (let i = 0; i < 1200; i++) {
        m9.recordFrame(t9);
        t9 += 1000 / 60;
      }
      for (let i = 0; i < 9; i++) {
        t9 += 400;
        m9.recordFrame(t9);
      }
      expect(m9.getState().dprFloor).toBeCloseTo(0.81, 5);
    } finally {
      restore();
    }
  });

  it('a genuine slowdown loses only the FIRST scale-down to the untrusted window', () => {
    // The other side of the cadence-trust gate: it must cost a
    // genuinely slow scene at most one unratified step. The reduction
    // still applies immediately either way — only the probe waits.
    // The production 0.9 step is used so the SECOND reduction (0.81) is
    // still clear of minDPR and the probe is not blocked by the floor.
    const m = new AdaptiveDPRManager({ scaleDownFactor: 0.9 });
    try {
      m.setRenderer(renderer);
      // 0.5fps from cold. Interval 1 is a cold-memory stall, interval 2
      // is absorbed as the frame rate but untrusted, interval 3 is
      // trusted and the normal probe machinery engages.
      let t = 0;
      m.recordFrame(t);
      m.recordFrame((t += 2000));
      m.recordFrame((t += 2000));
      expect(m.getCurrentDPR()).toBeCloseTo(0.9, 5); // applied...
      expect(m.getState().probing).toBe(false); // ...unratified

      m.recordFrame((t += 2000));
      expect(m.getState().probing).toBe(true); // normal machinery, next interval
    } finally {
      restore();
    }
  });

  it('a stall RECURRING faster than evaluationIntervalMs still lets the loop evaluate', () => {
    // A gap reset pushes the evaluation clock forward so dead time can't
    // count as progress toward the next evaluation. Pinned to the frame's
    // own timestamp, an outlier hitch recurring more often than
    // evaluationIntervalMs (500ms) re-pinned the clock before it could
    // ever mature, so no evaluation with data ever ran — no scale-down,
    // no scale-up, no probe settle — for as long as the pattern lasted.
    // Measured on the cadence below (16.7/16.7/400ms, ~6.9 perceived
    // fps): ZERO evaluations before, and one per cycle-pair after (the
    // 433ms cycle against the 500ms interval).
    //
    // The load-activity predicate is the observable: the manager polls it
    // once per evaluation that has a frame rate to judge.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let evaluations = 0;
      m.setLoadActivityPredicate(() => {
        evaluations++;
        return false;
      });
      // Warm a fast median so every 400ms gap stays an outlier...
      let t = 0;
      for (let i = 0; i < 600; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      evaluations = 0;
      // ...then 40 cycles of [400ms dead time, two 60fps frames]: a
      // 433ms period, inside the 500ms evaluation interval.
      for (let i = 0; i < 40; i++) {
        t += 400;
        m.recordFrame(t);
        t += 1000 / 60;
        m.recordFrame(t);
        t += 1000 / 60;
        m.recordFrame(t);
      }

      // Exact, not "> 0": the cadence and the evaluation interval are
      // fixed literals, so a regression from 20 evaluations to 1 must
      // fail here. (One per two 433ms cycles, the 500ms interval.)
      expect(evaluations).toBe(EXPECTED_EVALUATIONS_ON_RECURRING_STALL);
    } finally {
      restore();
    }
  });

  it('a slow window between recurring stalls still sheds pixels', () => {
    // The consequence of the clock fix: with the frames BETWEEN the gaps
    // slow enough to matter (~30fps, below the 45fps down threshold), the
    // loop must actually adapt and not just tick. Pre-fix the DPR sat at
    // native forever on this cadence.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let t = 0;
      for (let i = 0; i < 60; i++) {
        m.recordFrame(t);
        t += 1000 / 60;
      }
      // 30 cycles of [400ms dead time, two 33ms frames]: a 466ms period,
      // again inside the evaluation interval.
      for (let i = 0; i < 30; i++) {
        t += 400;
        m.recordFrame(t);
        t += 33;
        m.recordFrame(t);
        t += 33;
        m.recordFrame(t);
      }

      // Exact, not "< 1.0": the cadence is a fixed literal, so the
      // number of reductions it produces is knowable — one — and a
      // regression to zero OR to a runaway walk must both fail here.
      expect(m.getCurrentDPR()).toBeCloseTo(EXPECTED_DPR_ON_SLOW_RECURRING_STALL, 4);
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

  it('a content change keeps the cadence memory, so the next ordinary interval is not dead time', () => {
    // A content change is NOT a frame-stream boundary: frames keep
    // arriving at whatever rate they were arriving at. Wiping the
    // cadence would drop the detector onto its cold-memory fallback —
    // the absolute gapResetMs floor alone — so for any loop slower than
    // 1000/gapResetMs fps the very next ORDINARY interval reads as dead
    // time, throwing the freshly-restarted window away a second time.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      // Converge on 0.5fps: 2000ms intervals are "normal" for this scene.
      let t = 0;
      for (let i = 0; i < 6; i++) {
        m.recordFrame(t);
        t += 2000;
      }
      m.notifyContentChanged(t);
      // Two more frames at the SAME cadence. The window was cleared by
      // the content change, so this pair is all it has — and it must
      // survive: 2000ms is the frame rate here, not a stall.
      m.recordFrame(t);
      m.recordFrame(t + 2000);
      expect(m.getCurrentFPS()).toBeCloseTo(0.5, 3);

      // Positive control — a PAUSE is a real stream boundary and does
      // still forget the cadence, so the same pair is thrown away.
      m.notifyPaused();
      m.recordFrame(t + 4000);
      m.recordFrame(t + 6000);
      expect(m.getCurrentFPS()).toBe(0);
    } finally {
      restore();
    }
  });

  it('per-frame content churn keeps the reduction walk MONOTONE and teaches nothing', () => {
    // LOD-level swaps and `luxar-layers-changed` fire a content change
    // per frame, coalesced to one per contentChangeRecheckMs (5s) — only
    // ~2.5 frames at 0.5fps. Every armed probe used to be voided before
    // it could settle, so no verdict of any kind was ever produced. Two
    // things fix that and both are load-bearing: the content change must
    // not void a probe the loop is too slow to re-run, and it must not
    // wipe the cadence memory (which would fire a gap reset two frames
    // later and void the probe anyway — measured, that is what made the
    // gate inert).
    //
    // What the resulting CONFOUNDED verdict buys is NOT a stop to the
    // walk: under sustained churn no clean experiment exists, so the
    // pixel ratio keeps descending unratified, which is the correct
    // distress response (fewer pixels never hurt a stuttering loop) and
    // is bounded by the floor. What it buys is the absence of THRASH —
    // no revert upward onto a scene that just got heavier, and no 30s
    // floor pinned on a comparison that cannot mean anything.
    const m = new AdaptiveDPRManager({ scaleDownFactor: 0.9 });
    try {
      m.setRenderer(renderer);
      let t = 0;
      while (t < 120_000) {
        m.recordFrame(t);
        m.notifyContentChanged(t);
        t += 2000; // 0.5fps
      }
      // Every DPR the renderer ever saw was a REDUCTION on the one
      // before it: a confounded verdict must never revert upward (a swap
      // to a heavier LOD level reads as 'rejected', and acting on that
      // used to add pixels back to a scene that had just got heavier).
      // Deleting the confounded branch makes this fail — measured, that
      // costs 98 reverts over 20 minutes of churn.
      const applied = appliedDPRs(renderer);
      expect(applied).toEqual([...applied].sort((a, b) => b - a));
      // The walk descends to the static floor and stops there (0.9^6 of
      // native 1.0 — the next step would cross minDPR 0.5).
      expect(m.getCurrentDPR()).toBeCloseTo(0.5314, 4);
      // And NOTHING was learned from a confounded window: no probed DPR
      // became a floor and no backoff ladder was started, so the ledger
      // is still at minDPR after 120s of churn.
      expect(m.getState().dprFloor).toBe(0.5);
    } finally {
      restore();
    }
  });

  it('a probe settled across a content change keeps the DPR and learns nothing', () => {
    // The confounded-verdict contract, on one probe rather than a 120s
    // churn episode: keep the reduction (fewer pixels never hurt a
    // stuttering loop), never revert upward, pin no floor, and leave the
    // walk free to continue — a held walk was tried and measured to
    // invert its own goal (it pinned a 0.33fps scene at 1.62 forever,
    // and at churn periods just over its 5s window it walked FURTHER
    // than not having it).
    const restore2 = setNativeDPR(2.0);
    const m = new AdaptiveDPRManager({ scaleDownFactor: 0.9 });
    try {
      m.setRenderer(renderer);
      // 0.5fps: interval 1 is a cold-memory stall, 2 is absorbed but
      // untrusted (2.0 → 1.8 unprobed), 3 is trusted and both steps down
      // again and arms the probe at DPR 1.62.
      let t = 0;
      for (let i = 0; i < 4; i++) {
        m.recordFrame(t);
        t += 2000;
      }
      expect(m.getCurrentDPR()).toBeCloseTo(1.62, 5);
      expect(m.getState().probing).toBe(true);

      // Content changes mid-probe. At 0.5fps the loop cannot re-run the
      // experiment (fewer than two frames per probe window), so the probe
      // is KEPT — and marked confounded.
      m.notifyContentChanged(t);
      m.recordFrame(t);
      t += 2000;
      m.recordFrame(t); // settles the probe (age > probeWindowMs)

      expect(m.getState().probing).toBe(false);
      // Kept, NOT reverted to the pre-probe 1.8: every DPR the renderer
      // saw is a reduction on the one before it.
      expect(m.getCurrentDPR()).toBeCloseTo(1.62, 5);
      const applied = appliedDPRs(renderer);
      expect(applied).toEqual([...applied].sort((a, b) => b - a));
      // Nothing durable: no floor at the probed value, so the next step
      // is not blocked by one.
      expect(m.getState().dprFloor).toBe(0.5);

      // Content settles. The walk continues from where it was and the
      // NEXT probe is judged on its own evidence — rejected here, which
      // reverts and pins a real 30s floor: the normal machinery, intact.
      const end = t + 12_000;
      while (t < end) {
        m.recordFrame(t);
        t += 2000;
      }
      expect(Math.min(...appliedDPRs(renderer))).toBeLessThan(1.62);
      expect(m.getState().dprFloor).toBeGreaterThan(0.5);
    } finally {
      restore2();
      restore();
    }
  });

  it('a confounded settle leaves the rejection-backoff streak untouched', () => {
    // "The backoff streak is untouched" is the third clause of the
    // confounded contract and the only one with no visible consequence:
    // AdaptiveDPRState exposes no backoff field, so inserting
    // `recordAcceptance()` into the confounded path — which zeroes
    // floorBackoffLevel and lastRejectedProbeDPR, precisely the harm its
    // own JSDoc warns about — left every other test in this file green.
    // Assert it against the ledger the manager actually talks to.
    const acceptance = vi.spyOn(BoundsLedger.prototype, 'recordAcceptance');
    const restore2 = setNativeDPR(2.0);
    const m = new AdaptiveDPRManager({ scaleDownFactor: 0.9 });
    try {
      m.setRenderer(renderer);
      let t = 0;
      for (let i = 0; i < 4; i++) {
        m.recordFrame(t);
        t += 2000; // 0.5fps — probe armed at DPR 1.62
      }
      expect(m.getState().probing).toBe(true);
      m.notifyContentChanged(t);
      m.recordFrame(t);
      m.recordFrame(t + 2000); // settles the probe, confounded

      expect(m.getState().probing).toBe(false);
      expect(acceptance).not.toHaveBeenCalled();
    } finally {
      acceptance.mockRestore();
      restore2();
      restore();
    }
  });

  it('a content change at a HEALTHY frame rate still voids the probe outright', () => {
    // The gate is a slow-loop concession, not a new default: when the
    // loop can fit MIN_FRAMES_TO_RERUN_PROBE frames into a probe window
    // it can simply run the experiment again, so the contaminated one is
    // discarded unjudged (the historical behaviour). Nothing is learned
    // and the DPR stays where the scale-down put it.
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let t = 0;
      for (let i = 0; i < 14; i++) {
        m.recordFrame(t);
        t += 50; // 20fps < the 45fps down threshold, ~30 frames/probe window
      }
      expect(m.getState().probing).toBe(true);
      const reduced = m.getCurrentDPR();

      m.notifyContentChanged(t);
      expect(m.getState().probing).toBe(false); // voided, not kept
      expect(m.getCurrentDPR()).toBe(reduced);
      expect(m.getState().dprFloor).toBe(0.5);
    } finally {
      restore();
    }
  });

  it.each([
    ['1fps — 1.5 frames per probe window, too few to re-run', 1000, true],
    ['2fps — 3 frames per probe window, enough to re-run', 500, false],
  ])('the probe-rerun gate turns over between %s', (_label, frameMs, keptAcrossChange) => {
    // The gate is a threshold on FRAMES PER PROBE WINDOW
    // (MIN_FRAMES_TO_RERUN_PROBE = 2, the ProbeController's own settle
    // minimum), and it turns over between these two cadences: 1.5 frames
    // per 1500ms window is too few to judge a replacement experiment,
    // 3 is plenty. Straddling the threshold pins it from both sides.
    const m = new AdaptiveDPRManager({ scaleDownFactor: 0.9 });
    try {
      m.setRenderer(renderer);
      let t = 0;
      for (let i = 0; i < 4; i++) {
        m.recordFrame(t);
        t += frameMs;
      }
      expect(m.getState().probing).toBe(true);

      m.notifyContentChanged(t);
      expect(m.getState().probing).toBe(keptAcrossChange);
    } finally {
      restore();
    }
  });

  it('a content change with an UNKNOWN frame rate voids the probe (historical behaviour)', () => {
    // `getFPS()` is 0 with fewer than two samples in the window: nothing
    // says the loop is too slow to re-run the experiment, so the void
    // stands. Pins the `fps <= 0` half of the gate, which the
    // frames-per-window arithmetic alone cannot express — zero frames per
    // probe window would otherwise KEEP the probe.
    const m = new AdaptiveDPRManager({ scaleDownFactor: 0.9 });
    try {
      m.setRenderer(renderer);
      // 0.5fps until a probe is armed.
      m.recordFrame(0);
      m.recordFrame(2000);
      m.recordFrame(4000);
      m.recordFrame(6000);
      expect(m.getState().probing).toBe(true);

      // First change: the loop is measurably slow (0.5fps), so the probe
      // is kept — and the FPS window is cleared.
      m.notifyContentChanged(6000);
      expect(m.getState().probing).toBe(true);

      // One frame is not a frame rate. The next change (past the 5s
      // coalescing window) therefore knows nothing and voids the probe.
      m.recordFrame(12_000);
      expect(m.getCurrentFPS()).toBe(0);
      m.notifyContentChanged(12_000);
      expect(m.getState().probing).toBe(false);
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

  it('follows the new native when tracking it, and re-applies', () => {
    const m = new AdaptiveDPRManager();
    try {
      const renderer = makeRenderer();
      m.setRenderer(renderer);

      setNativeDPR(1.0);
      // Live-consistent getCurrentDPR: never reports the stale native.
      expect(m.getCurrentDPR()).toBe(1.0);
      // This used to assert the opposite — that the tracking path does
      // NOT re-apply, because a null override in dpr-policy follows the
      // live ceiling for free. That reasoning was unsound: the manager
      // cannot see the renderer's override, so "currentDPR equals the
      // ceiling" does not imply the renderer is tracking. See the
      // syncNativeDPR doc for the fuzzer-found sequence where the
      // manager reported 3.0 while the renderer drew at 1.0. The
      // optimization saved one reallocation during a monitor drag, which
      // fires a resize anyway.
      expect(renderer.setAdaptivePixelRatio).toHaveBeenCalledWith(1.0);
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
      // 0.5 is still valid below native 1.0, so the VALUE is untouched.
      // It is re-applied all the same: the rebase re-asserts the DPR
      // unconditionally now (see syncNativeDPR), because the manager
      // cannot tell whether the renderer is holding a stale override.
      expect(m.getCurrentDPR()).toBe(0.5);
      expect(renderer.setAdaptivePixelRatio).toHaveBeenCalledWith(0.5);
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

  it('a SUB-2fps scene reaches the distress verdict too (the regime the gap reset used to hide)', () => {
    // The whole point of the sustained-slow gap-reset fix: below
    // ~1000/gapResetMs fps the manager used to evaluate exactly never,
    // so this path — the one built for scenes too slow for any real
    // display mode — was structurally unreachable at the frame rates
    // that need it MOST. A software-rasterized 1.4fps scene must now
    // walk the same route as the 10fps one above: ceiling demoted to
    // 1.0, DPR clamped there, cap left at the fallback so scale-down
    // stays armed.
    const restore = setNativeDPR(2.0);
    const m = new AdaptiveDPRManager(); // production gapResetMs (350)
    const r = makeRenderer();
    m.setRenderer(r);
    try {
      let t = 0;
      for (let i = 0; i < 60; i++) {
        m.recordFrame(t);
        t += 700; // ~1.4fps — EVERY interval exceeds gapResetMs
      }
      const s = m.getState();
      expect(s.currentFPS).toBeCloseTo(1000 / 700, 2);
      expect(s.dprCeiling).toBe(1.0);
      expect(s.currentDPR).toBe(1.0);
      expect(s.refreshRateCap).toBeGreaterThanOrEqual(60);
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

  it('frame-gap dead-time never counts toward the sustained-distress bar (gap-reset leg)', () => {
    // Same staleness class as the pause leg, reached via the recordFrame
    // gapResetMs branch instead of notifyPaused(): a long single-frame
    // stall (tab hide the pause hook didn't see, synchronous decode)
    // must clear the estimator's plateau clock too. Pinned separately —
    // deleting the gap-reset's noteSessionInterrupted() call must fail
    // THIS test even while the pause test stays green.
    const restore = setNativeDPR(2.0);
    const m = new AdaptiveDPRManager();
    const r = makeRenderer();
    m.setRenderer(r);
    try {
      pushFrames(m, 0, 81, 8000); // ~10fps for 8s — under the 10s bar
      // One frame arrives 2 minutes later: recordFrame's gap detection
      // fires (no notifyPaused was ever called).
      pushFrames(m, 128_000, 41, 4000); // 4s at ~10fps after the gap
      expect(m.getState().dprCeiling).toBe(2.0); // NOT demoted yet

      pushFrames(m, 132_100, 201, 20_000); // sustained → demotes for real
      expect(m.getState().dprCeiling).toBe(1.0);
    } finally {
      restore();
    }
  });

  it('pause dead-time never counts toward the sustained-distress bar (stale-state regression)', () => {
    // Regression: the estimator's plateau clock and recent window used
    // to survive notifyPaused(), so ~8s of pre-pause lows + 2 minutes
    // of idle dead time + a couple of janky post-resume windows fired
    // an immediate wrongful demotion. After the fix, the sustained
    // requirement must be re-earned from FRESH post-resume samples.
    const restore = setNativeDPR(2.0);
    const m = new AdaptiveDPRManager();
    const r = makeRenderer();
    m.setRenderer(r);
    try {
      pushFrames(m, 0, 81, 8000); // ~10fps for 8s — under the 10s bar
      m.notifyPaused(); // idle pause; loop stops

      // Resume 2 minutes later, still slow for a few seconds.
      pushFrames(m, 128_000, 41, 4000); // 4s at ~10fps post-resume
      expect(m.getState().dprCeiling).toBe(2.0); // NOT demoted yet

      // ...but genuinely sustained post-resume distress still demotes.
      pushFrames(m, 132_100, 201, 20_000); // 20 more seconds at ~10fps
      expect(m.getState().dprCeiling).toBe(1.0);
      expect(m.getState().currentDPR).toBe(1.0);
    } finally {
      restore();
    }
  });
});

// ---------------------------------------------------------------------
// The pixel-ratio cap — the SHIPPED DEFAULT (high DPR disallowed)
//
// Everything above runs with the cap lifted so its pre-cap coverage
// survives. This block is the inverse: a 2x display where the viewer is
// only allowed to reach 1.0, which is what users actually get.
// ---------------------------------------------------------------------

describe('AdaptiveDPRManager — pixel-ratio cap (high DPR disallowed)', () => {
  let restore: () => void;

  beforeEach(() => {
    restore = setNativeDPR(2.0);
    // Overrides the file-wide allowHighDPR() in the outer beforeEach —
    // outer hooks run first, so this wins.
    setMaxPixelRatioCap(DEFAULT_MAX_PIXEL_RATIO);
  });
  afterEach(() => restore());

  it('starts at the ceiling, not the display DPR, and does not call that "reduced"', () => {
    const m = new AdaptiveDPRManager();
    try {
      // The display is still reported honestly — the cap governs what we
      // RENDER at, it does not lie about the hardware.
      expect(m.getNativeDPR()).toBe(2.0);
      expect(m.getCurrentDPR()).toBe(1.0);
      // Load-bearing: `isReducedResolution` drives the resolution-indicator
      // toast. Measured against native it would be permanently true, and
      // every scene would open with a "50%" toast.
      expect(m.getState().isReducedResolution).toBe(false);
      expect(m.getState().dprCeiling).toBe(1.0);
      expect(m.getState().allowHighDPR).toBe(false);
    } finally {
      m.dispose();
    }
  });

  it('never scales up past the ceiling however long FPS stays high', () => {
    const renderer = makeRenderer();
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let t = 1000;
      for (let i = 0; i < 40; i++) t = pushFrames(m, t, 30, 500);
      expect(m.getCurrentDPR()).toBeLessThanOrEqual(1.0);
      for (const dpr of appliedDPRs(renderer)) expect(dpr).toBeLessThanOrEqual(1.0);
    } finally {
      m.dispose();
    }
  });

  it('still scales DOWN normally beneath the ceiling', () => {
    const renderer = makeRenderer();
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let t = 1000;
      for (let i = 0; i < 6; i++) t = pushFrames(m, t, 6, 600);
      expect(m.getCurrentDPR()).toBeLessThan(1.0);
      expect(m.getState().isReducedResolution).toBe(true);
    } finally {
      m.dispose();
    }
  });

  it('disabling adaptation resets to the ceiling, not the display DPR', () => {
    const renderer = makeRenderer();
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      m.setEnabled(false);
      expect(m.getCurrentDPR()).toBe(1.0);
      expect(m.getState().isReducedResolution).toBe(false);
      for (const dpr of appliedDPRs(renderer)) expect(dpr).toBeLessThanOrEqual(1.0);
    } finally {
      m.dispose();
    }
  });

  it('clamps manual DPR to the ceiling', () => {
    const m = new AdaptiveDPRManager();
    try {
      m.setEnabled(false);
      m.setManualDPR(99);
      expect(m.getCurrentDPR()).toBe(1.0);
      m.setManualDPR(0.5);
      expect(m.getCurrentDPR()).toBe(0.5);
      expect(m.getState().isReducedResolution).toBe(true);
    } finally {
      m.dispose();
    }
  });

  it('rests the idle frame at the ceiling — no sharpen pop back to native', () => {
    const renderer = makeRenderer();
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      let t = 1000;
      for (let i = 0; i < 6; i++) t = pushFrames(m, t, 6, 600);
      const operating = m.getCurrentDPR();
      expect(operating).toBeLessThan(1.0);

      expect(m.prepareIdleFrame()).toBe(true);
      expect(m.getCurrentDPR()).toBe(1.0);

      m.notifyResumed();
      expect(m.getCurrentDPR()).toBeCloseTo(operating, 5);
    } finally {
      m.dispose();
    }
  });

  it('prepareIdleFrame is a no-op when already resting at the ceiling', () => {
    const renderer = makeRenderer();
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      expect(m.prepareIdleFrame()).toBe(false);
    } finally {
      m.dispose();
    }
  });

  describe('runtime toggle', () => {
    it('turning it ON while adaptive jumps straight to the display DPR', () => {
      const renderer = makeRenderer();
      const m = new AdaptiveDPRManager();
      try {
        m.setRenderer(renderer);
        expect(m.getCurrentDPR()).toBe(1.0);

        m.setHighDPRAllowed(true);

        expect(m.isHighDPRAllowed()).toBe(true);
        expect(m.getCurrentDPR()).toBe(2.0);
        expect(appliedDPRs(renderer)).toContain(2.0);
      } finally {
        m.dispose();
      }
    });

    it('turning it ON while MANUAL widens the range but keeps the chosen DPR', () => {
      const m = new AdaptiveDPRManager();
      try {
        m.setEnabled(false);
        m.setManualDPR(0.5);

        m.setHighDPRAllowed(true);

        // The user's number is authoritative in manual mode — it must not
        // be moved out from under them just because the ceiling rose.
        expect(m.getCurrentDPR()).toBe(0.5);
        expect(m.getState().dprCeiling).toBe(2.0);
        m.setManualDPR(2.0);
        expect(m.getCurrentDPR()).toBe(2.0);
      } finally {
        m.dispose();
      }
    });

    it('turning it OFF clamps down immediately', () => {
      const renderer = makeRenderer();
      const m = new AdaptiveDPRManager();
      try {
        m.setRenderer(renderer);
        m.setHighDPRAllowed(true);
        expect(m.getCurrentDPR()).toBe(2.0);

        m.setHighDPRAllowed(false);

        expect(m.getCurrentDPR()).toBe(1.0);
        expect(m.getState().isReducedResolution).toBe(false);
        expect(appliedDPRs(renderer).at(-1)).toBe(1.0);
      } finally {
        m.dispose();
      }
    });

    it('is a no-op when the value is unchanged', () => {
      const renderer = makeRenderer();
      const m = new AdaptiveDPRManager();
      try {
        m.setRenderer(renderer);
        m.setHighDPRAllowed(false);
        expect(renderer.setAdaptivePixelRatio).not.toHaveBeenCalled();
      } finally {
        m.dispose();
      }
    });
  });

  describe('URL-pinned DPR', () => {
    it('raises the cap so an above-ceiling pin is honoured', () => {
      const m = new AdaptiveDPRManager();
      try {
        m.pinManualDPR(2.0);
        expect(m.getCurrentDPR()).toBe(2.0);
        expect(m.isPinned()).toBe(true);
      } finally {
        m.dispose();
      }
    });

    it('is still bounded by the display', () => {
      const m = new AdaptiveDPRManager();
      try {
        m.pinManualDPR(4.0);
        expect(m.getCurrentDPR()).toBe(2.0);
      } finally {
        m.dispose();
      }
    });

    it('a below-ceiling pin leaves the cap alone', () => {
      const m = new AdaptiveDPRManager();
      try {
        m.pinManualDPR(0.5);
        expect(m.getCurrentDPR()).toBe(0.5);
        expect(m.isHighDPRAllowed()).toBe(false);
      } finally {
        m.dispose();
      }
    });

    it('ignores a later setHighDPRAllowed, like it ignores setEnabled', () => {
      const m = new AdaptiveDPRManager();
      try {
        m.pinManualDPR(0.5);
        m.setHighDPRAllowed(true);
        expect(m.getCurrentDPR()).toBe(0.5);
        expect(m.isHighDPRAllowed()).toBe(false);
      } finally {
        m.dispose();
      }
    });
  });

  it('a monitor change rebases against the new CEILING, not the new native', () => {
    const renderer = makeRenderer();
    const m = new AdaptiveDPRManager();
    try {
      m.setRenderer(renderer);
      expect(m.getCurrentDPR()).toBe(1.0);

      // Drag to a 3x display: the ceiling is still 1.0, so the operating
      // DPR does not move. What must NOT happen is the session being
      // treated as "explicitly reduced" and clamped to something else,
      // or the ceiling following the display past the cap.
      const restore3x = setNativeDPR(3.0);
      try {
        expect(m.getNativeDPR()).toBe(3.0);
        expect(m.getCurrentDPR()).toBe(1.0);
        expect(m.getState().isReducedResolution).toBe(false);
        // Re-asserted rather than left to the null override: see
        // syncNativeDPR for why the manager can no longer assume the
        // renderer is tracking just because the numbers agree.
        expect(appliedDPRs(renderer)).toEqual([1.0]);
      } finally {
        restore3x();
      }
    } finally {
      m.dispose();
    }
  });

  it('a move to a SUB-1 display lowers the ceiling with it', () => {
    const m = new AdaptiveDPRManager();
    try {
      const restoreSmall = setNativeDPR(0.75);
      try {
        expect(m.getCurrentDPR()).toBeCloseTo(0.75, 5);
        expect(m.getState().dprCeiling).toBeCloseTo(0.75, 5);
      } finally {
        restoreSmall();
      }
    } finally {
      m.dispose();
    }
  });
});
