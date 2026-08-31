// @vitest-environment jsdom
/**
 * Property-based invariant fuzzing for AdaptiveDPRManager.
 *
 * The scenario tests pin specific sequences; this harness instead drives
 * the REAL manager (with a real renderer stand-in) through thousands of
 * seeded, adversarially-interleaved operations — random frame cadences,
 * pause/idle/resume, content changes, enable/disable, manual DPR, and
 * live devicePixelRatio (monitor/zoom) changes — and asserts the hard
 * invariants that must hold no matter what:
 *
 *   I1  0.25 - eps <= currentDPR <= liveNative + eps          (bounds)
 *   I2  getState().refreshRateCap is finite and > 0           (no NaN cap)
 *   I3  0 < dprCeiling <= liveNative + eps, finite            (ceiling sane)
 *   I4  config.minDPR - eps <= dprFloor <= liveNative + eps    (floor sane)
 *   I5  the last DPR handed to the renderer == manager.currentDPR
 *       whenever a change was applied                         (no divergence)
 *   I6  no call ever throws
 *
 * Plus a fixed-point property: held at a STEADY fps the DPR must stop
 * changing within a bounded number of evaluations (no infinite
 * oscillation).
 *
 * Deterministic: a seeded PRNG, no Math.random / Date. A failure prints
 * the seed + step so it reproduces exactly.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Production-like config so the asserted bounds match real behavior
// (structural per-key defaults from data.ts fill anything omitted).
vi.mock('../../../config', () => ({
  config: {
    adaptiveDPR: {
      enabled: true,
      minDPR: 0.5,
      scaleDownFactor: 0.9,
      scaleUpFactor: 1.05,
      hysteresisSeconds: 3,
      evaluationIntervalMs: 500,
      scaleDownFpsRatio: 0.75,
      scaleUpFpsRatio: 0.9,
      refreshRateFallback: 60,
      midbandGraceSamples: 1,
      probeWindowMs: 1500,
      probeImprovement: 1.05,
      probeMinSamples: 8,
      floorTtlMs: 30_000,
      backoffMultiplier: 2,
      backoffMaxTtlMs: 300_000,
      ceilingTtlMs: 60_000,
      punishedAscentWindowMs: 3000,
      punishedAscentThreshold: 2,
      contentChangeRecheckMs: 5000,
      gapResetMs: 350,
    },
  },
}));

import { AdaptiveDPRManager, type DPRRenderer } from '../../../rendering/adaptive-dpr-manager';
import {
  computePixelRatioOverride,
  getActivePixelRatio,
} from '../../../scene/scene-manager/viewport/dpr-policy';
import {
  DEFAULT_MAX_PIXEL_RATIO,
  getMaxPixelRatio,
  getMaxPixelRatioCap,
  setMaxPixelRatioCap,
} from '../../../rendering/pixel-ratio-cap';

/** Tiny deterministic PRNG (mulberry32) — reproducible from a seed. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function setNativeDPR(value: number): void {
  Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value, writable: true });
}

/**
 * Renderer stand-in that runs the REAL SceneManager DPR path rather than
 * re-implementing it: `setAdaptivePixelRatio` stores whatever
 * dpr-policy.computePixelRatioOverride decides, and `effective()` is
 * dpr-policy.getActivePixelRatio of that — i.e. literally what the
 * screen renders at, which is the value I5 compares against
 * manager.currentDPR.
 *
 * Calling the policy instead of copying it matters twice over. The
 * manager deliberately does NOT re-apply while it is "tracking the
 * ceiling" (a null override follows a monitor-DPI change for free), so a
 * naive last-value mock reports a false divergence; and a hand-copied
 * threshold would silently stop modelling the real seam the first time
 * the policy changed — which is exactly what the pixel-ratio cap did to
 * it.
 */
function makeRenderer(): DPRRenderer & {
  override: number | null;
  touched: boolean;
  effective(): number;
} {
  const r = {
    override: null as number | null,
    touched: false,
    setAdaptivePixelRatio(dpr: number) {
      r.touched = true;
      r.override = computePixelRatioOverride(dpr).override;
    },
    effective() {
      return getActivePixelRatio(r.override);
    },
  };
  return r;
}

const EPS = 1e-6;
const NATIVE_CHOICES = [1, 1.5, 2, 3];
// The shipped default, a couple of partial lifts (a `?dpr=` pin or a
// capture override), and fully unrestricted.
const CAP_CHOICES = [DEFAULT_MAX_PIXEL_RATIO, 1.5, 2, Infinity];
const FPS_CHOICES = [2.5, 8, 20, 30, 45, 55, 61, 90, 120, 144];

describe('AdaptiveDPRManager — property/invariant fuzzing', () => {
  let restoreNative: number;
  let restoreCap: number;
  beforeEach(() => {
    restoreNative = window.devicePixelRatio;
    restoreCap = getMaxPixelRatioCap();
  });
  afterEach(() => {
    setNativeDPR(restoreNative);
    setMaxPixelRatioCap(restoreCap);
  });

  /** Assert the manager's cross-cutting invariants at the current state. */
  function assertInvariants(
    m: AdaptiveDPRManager,
    r: ReturnType<typeof makeRenderer>,
    ctx: string
  ) {
    const s = m.getState();
    const native = window.devicePixelRatio || 1;
    // Every bound below is stated against the CEILING — min(native, cap)
    // — not the display's raw DPR. Under the shipped cap the two differ,
    // and a bound of `<= native` would pass while the viewer rendered at
    // twice what the setting allows.
    const ceiling = getMaxPixelRatio();

    // I1 bounds
    expect(s.currentDPR, `${ctx} I1-lo`).toBeGreaterThanOrEqual(0.25 - EPS);
    expect(s.currentDPR, `${ctx} I1-hi`).toBeLessThanOrEqual(ceiling + EPS);
    expect(Number.isFinite(s.currentDPR), `${ctx} I1-finite`).toBe(true);

    // I2 cap finite & positive
    expect(Number.isFinite(s.refreshRateCap), `${ctx} I2-finite`).toBe(true);
    expect(s.refreshRateCap, `${ctx} I2-pos`).toBeGreaterThan(0);

    // I3 ceiling sane
    expect(Number.isFinite(s.dprCeiling), `${ctx} I3-finite`).toBe(true);
    expect(s.dprCeiling, `${ctx} I3-pos`).toBeGreaterThan(0);
    expect(s.dprCeiling, `${ctx} I3-hi`).toBeLessThanOrEqual(ceiling + EPS);
    // I3b the only LEARNED demotion is to exactly 1.0 (a Schelling point,
    // not a hunted estimate), so the reported ceiling is either the
    // session ceiling or 1.0 — whether the demotion came from punished
    // ascents or sustained sub-throttle distress. (When the cap is 1.0
    // the two coincide, which is the setting being the same demotion
    // applied up front.)
    expect(
      Math.abs(s.dprCeiling - 1.0) < EPS || Math.abs(s.dprCeiling - ceiling) < EPS,
      `${ctx} I3b-schelling (ceiling ${s.dprCeiling}, session ${ceiling}, native ${native})`
    ).toBe(true);

    // I4 floor sane (config.minDPR = 0.5)
    expect(s.dprFloor, `${ctx} I4-lo`).toBeGreaterThanOrEqual(0.5 - EPS);
    expect(s.dprFloor, `${ctx} I4-hi`).toBeLessThanOrEqual(native + EPS);

    // I5 no manager/renderer divergence: the EFFECTIVE rendered DPR
    // (after the null-override snap) must track manager.currentDPR once
    // the manager has driven the renderer at least once.
    if (r.touched) {
      expect(
        Math.abs(r.effective() - s.currentDPR),
        `${ctx} I5-diverge (eff ${r.effective()})`
      ).toBeLessThan(0.02);
    }
  }

  it('holds all invariants across randomized adversarial operation sequences (I1-I6)', () => {
    for (let seed = 1; seed <= 40; seed++) {
      const rand = rng(seed);
      setNativeDPR(NATIVE_CHOICES[Math.floor(rand() * NATIVE_CHOICES.length)]);
      const m = new AdaptiveDPRManager();
      const r = makeRenderer();
      m.setRenderer(r);
      // The manager fires the renderer once things move; before that
      // r.applied is null and I5 is vacuous.

      let t = 0;
      for (let step = 0; step < 400; step++) {
        const roll = rand();
        const ctx = `seed=${seed} step=${step}`;
        expect(() => {
          if (roll < 0.6) {
            // Feed a burst of frames at a chosen fps for a chosen span.
            const fps = FPS_CHOICES[Math.floor(rand() * FPS_CHOICES.length)];
            const spanMs = 200 + Math.floor(rand() * 2600);
            const step_ms = 1000 / fps;
            const end = t + spanMs;
            while (t < end) {
              m.recordFrame(t);
              t += step_ms;
            }
          } else if (roll < 0.68) {
            t += Math.floor(rand() * 5000); // idle gap of random length
            m.notifyPaused();
          } else if (roll < 0.74) {
            m.prepareIdleFrame();
          } else if (roll < 0.8) {
            m.notifyResumed();
          } else if (roll < 0.86) {
            m.notifyContentChanged(t);
          } else if (roll < 0.9) {
            m.setEnabled(rand() < 0.5);
          } else if (roll < 0.93) {
            // Only meaningful when adaptive is off; harmless otherwise.
            m.setManualDPR(0.2 + rand() * 3);
          } else if (roll < 0.96) {
            // Monitor drag / browser zoom: live devicePixelRatio changes.
            setNativeDPR(NATIVE_CHOICES[Math.floor(rand() * NATIVE_CHOICES.length)]);
          } else if (roll < 0.98) {
            // The Allow High DPR toggle, and the out-of-band cap writes a
            // `?dpr=` pin or a capture makes. Fuzzed alongside everything
            // else because the interesting failures are INTERACTIONS —
            // a cap lowered while resting at the old ceiling, or raised
            // mid-probe — not the toggle in isolation.
            if (rand() < 0.7) {
              m.setHighDPRAllowed(rand() < 0.5);
            } else {
              setMaxPixelRatioCap(
                CAP_CHOICES[Math.floor(rand() * CAP_CHOICES.length)] ?? DEFAULT_MAX_PIXEL_RATIO
              );
            }
          } else {
            m.getState(); // pure-ish poll (also triggers a sync)
          }
        }, `${ctx} threw`).not.toThrow();

        assertInvariants(m, r, ctx);
      }
      m.dispose();
    }
    // 40 seeds × 400 adversarial steps, each replaying frame bursts, is
    // CPU-bound; it finishes in a few seconds locally but can brush past the
    // default 15s cap on a heavily loaded CI runner. Give it explicit headroom.
  }, 60_000);

  it('reaches a fixed point (no infinite oscillation) under a steady fps regime', () => {
    // For a matrix of (native, steady fps), after a long warmup the DPR
    // must stop changing: the last K evaluations produce no renderer
    // call. Catches scale-up/scale-down and ceiling/floor oscillation.
    //
    // The sub-2.9fps regimes are the ones the gap-reset outlier rule
    // ACTIVATED (below 1000/gapResetMs fps every interval clears the
    // absolute floor, so the manager used to be structurally inert
    // there and could not oscillate for lack of running at all). They
    // are run at native 1 and 2 only — a 0.4fps regime replays so few
    // frames that a third native adds no coverage, only runtime.
    const regimes: Array<[number, number]> = [];
    for (const native of [1, 2, 3]) {
      for (const fps of [8, 20, 30, 45, 55, 90]) regimes.push([native, fps]);
    }
    for (const native of [1, 2]) {
      for (const fps of [0.4, 0.5, 1, 2, 2.5, 2.85]) regimes.push([native, fps]);
    }
    for (const [native, fps] of regimes) {
      setNativeDPR(native);
      const m = new AdaptiveDPRManager();
      const r = makeRenderer();
      m.setRenderer(r);

      const stepMs = 1000 / fps;
      let t = 0;
      // Warm up for 90s of steady frames — well past every TTL that
      // matters at this cadence except the 30s/60s learned-bound TTLs,
      // which at a steady fps re-settle to the same fixed point.
      const warmupEnd = 90_000;
      while (t < warmupEnd) {
        m.recordFrame(t);
        t += stepMs;
      }
      // Now observe 20s more and count DPR changes (by the effective
      // rendered value, so a null-override native track counts as no
      // change).
      let changes = 0;
      let last = r.effective();
      const observeEnd = t + 20_000;
      while (t < observeEnd) {
        m.recordFrame(t);
        t += stepMs;
        const eff = r.effective();
        if (Math.abs(eff - last) > 0.001) {
          changes++;
          last = eff;
        }
      }
      // A steady regime may still cross ONE learned-bound TTL boundary
      // in a 20s observation (floor re-probe), so allow a small bound,
      // not literally zero — but it must not thrash.
      expect(changes, `native=${native} fps=${fps} oscillation`).toBeLessThanOrEqual(2);
      m.dispose();
    }
    // Same CPU-bound profile as the sequence fuzzer above (30 regimes ×
    // ~110s of replayed frames each): fast locally, but give it the same
    // headroom so a loaded runner can't trip the default 15s cap.
  }, 60_000);

  it('a stall burst inside a healthy session never moves the learned floor', () => {
    // The invariant whose absence let a real regression through: once a
    // run of dead intervals is long enough that the stall detector's
    // median absorbs it as "the frame rate", the FPS window it lands in
    // holds NOTHING but dead time — and a window like that must never
    // become learned evidence. Left trusted it armed a probe, settled it
    // against a baseline measured on the same dead time, rejected it,
    // and pinned a U-shape floor (measured: 0.90) with exponential
    // backoff onto a session that was rendering at a healthy 60fps
    // before and after. A transient reduction is fine (fewer pixels
    // never hurt a stuttering loop); a learned floor is not.
    //
    // The guarantee is BOUNDED, and deliberately so: it covers a burst
    // up to CADENCE_TRUST_INTERVALS + 2 = 4 dead intervals (measured —
    // pre-fix a floor of 0.90 was pinned from FOUR). Past that, 25s of
    // nothing but 5s intervals is not a burst any more, it is a 0.2fps
    // regime the manager must be allowed to learn from; the boundary is
    // asserted below rather than left to drift silently.
    const untouched = [1, 2, 3, 4];
    for (const native of [1, 2, 3]) {
      for (const stalls of [...untouched, 5]) {
        setNativeDPR(native);
        const m = new AdaptiveDPRManager();
        const r = makeRenderer();
        m.setRenderer(r);

        const stepMs = 1000 / 60;
        let t = 0;
        while (t < 20_000) {
          m.recordFrame(t);
          t += stepMs;
        }
        const ctx = `native=${native} stalls=${stalls}`;
        expect(m.getState().dprFloor, `${ctx} floor before`).toBe(0.5);

        for (let i = 0; i < stalls; i++) {
          t += 5000;
          m.recordFrame(t);
        }
        const floorAfterBurst = m.getState().dprFloor;

        // Healthy again for another 20s — the reduction must lift.
        const end = t + 20_000;
        while (t < end) {
          m.recordFrame(t);
          t += stepMs;
        }

        const s = m.getState();
        if (untouched.includes(stalls)) {
          expect(floorAfterBurst, `${ctx} floor right after the burst`).toBe(0.5);
          expect(s.dprFloor, `${ctx} floor after recovery`).toBe(0.5);
        } else {
          // Boundary marker, not a wish: five in a row IS a frame rate.
          expect(floorAfterBurst, `${ctx} boundary — a floor is learned`).toBeGreaterThan(0.5);
        }
        expect(s.currentDPR, `${ctx} DPR restored`).toBeCloseTo(getMaxPixelRatio(), 1);
        m.dispose();
      }
    }
  }, 60_000);
});
