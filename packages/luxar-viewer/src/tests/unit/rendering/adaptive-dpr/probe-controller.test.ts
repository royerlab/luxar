import { describe, it, expect } from 'vitest';
import { ProbeController } from '../../../../rendering/adaptive-dpr/probe-controller';

const CONFIG = { windowMs: 1500, improvement: 1.05, minSamples: 8, minSpanMs: 500 };
/** A comfortably clean sample-quality snapshot. */
const CLEAN = { sampleCount: 30, spanMs: 900, suppressed: false };

function armed(): ProbeController {
  const probe = new ProbeController(CONFIG);
  probe.arm({ previousDPR: 2.0, previousFPS: 40, probedDPR: 1.8, startTime: 1000 });
  return probe;
}

describe('ProbeController', () => {
  it('evaluates to null when no probe is in flight', () => {
    const probe = new ProbeController(CONFIG);
    expect(probe.isPending).toBe(false);
    expect(probe.evaluate(5000, 60, CLEAN)).toBeNull();
  });

  it('stays pending inside the probe window and keeps the probe armed', () => {
    const probe = armed();
    expect(probe.evaluate(1000 + 1499, 60, CLEAN)).toEqual({ kind: 'pending' });
    expect(probe.isPending).toBe(true);
  });

  it('accepts when FPS improved by at least the configured ratio', () => {
    const probe = armed();
    const verdict = probe.evaluate(1000 + 1500, 42, CLEAN); // 42/40 = 1.05
    expect(verdict).toMatchObject({ kind: 'accepted', fpsRatio: 1.05 });
    expect(probe.isPending).toBe(false);
  });

  it('rejects when FPS did not improve enough, handing back the revert target', () => {
    const probe = armed();
    const verdict = probe.evaluate(1000 + 1500, 41, CLEAN); // 41/40 = 1.025 < 1.05
    expect(verdict).toMatchObject({
      kind: 'rejected',
      probe: { previousDPR: 2.0, probedDPR: 1.8 },
    });
    expect(probe.isPending).toBe(false);
  });

  it('treats a zero baseline as a rejection (ratio 0)', () => {
    const probe = new ProbeController(CONFIG);
    probe.arm({ previousDPR: 2.0, previousFPS: 0, probedDPR: 1.8, startTime: 0 });
    expect(probe.evaluate(1500, 60, CLEAN)).toMatchObject({ kind: 'rejected', fpsRatio: 0 });
  });

  it('void_() drops the pending probe without a verdict', () => {
    const probe = armed();
    probe.void_();
    expect(probe.isPending).toBe(false);
    expect(probe.evaluate(10_000, 60, CLEAN)).toBeNull();
  });

  describe('quality gating', () => {
    it('keeps waiting past the window while the sample is too thin', () => {
      const probe = armed();
      // Few frames over a SHORT span (ramping back up after a reset) —
      // must NOT settle the probe. (Few frames over a LONG span is a
      // different story: that's a genuinely slow scene — see below.)
      const verdict = probe.evaluate(1000 + 1600, 60, {
        sampleCount: 3,
        spanMs: 550,
        suppressed: false,
      });
      expect(verdict).toEqual({ kind: 'pending' });
      expect(probe.isPending).toBe(true);

      // A clean sample arriving later settles normally.
      expect(probe.evaluate(1000 + 2100, 60, CLEAN)).toMatchObject({ kind: 'accepted' });
    });

    it('a low-sample settle showing IMPROVEMENT voids inconclusive, never accepts', () => {
      const probe = armed(); // baseline 40fps
      // A few-frame window can read burst-inflated FPS (gap detection
      // bounds inter-frame gaps, not intra-window variance). A wrong
      // ACCEPT would reset the rejection-backoff streak — the exact
      // protection that quiets hopeless scenes — while inconclusive
      // keeps the DPR identically and learns nothing.
      const verdict = probe.evaluate(1000 + 1600, 60, {
        sampleCount: 6, // below minSamples(8)
        spanMs: 950,
        suppressed: false,
      });
      expect(verdict).toMatchObject({ kind: 'inconclusive' });
      expect(probe.isPending).toBe(false);
    });

    it('settles from a full-span low-fps window (few frames = genuinely slow, not contaminated)', () => {
      const probe = armed(); // baseline 40fps
      // ~5fps scene: the 1s window can never hold minSamples(8) frames.
      // A full-span window is that scene's best obtainable sample and
      // MUST settle — voiding it as inconclusive forever would mean no
      // U-shape floor is ever learned below 8fps and DPR walks
      // unprotected to minDPR.
      const verdict = probe.evaluate(1000 + 1600, 5, {
        sampleCount: 5,
        spanMs: 950,
        suppressed: false,
      });
      expect(verdict).toMatchObject({ kind: 'rejected' }); // 5/40 « 1.05
      expect(probe.isPending).toBe(false);
    });

    it('keeps waiting while the window span is too short', () => {
      const probe = armed();
      const verdict = probe.evaluate(1000 + 1600, 60, {
        sampleCount: 20,
        spanMs: 200,
        suppressed: false,
      });
      expect(verdict).toEqual({ kind: 'pending' });
    });

    it('keeps waiting while samples are load-suppressed', () => {
      const probe = armed();
      const verdict = probe.evaluate(1000 + 1600, 60, { ...CLEAN, suppressed: true });
      expect(verdict).toEqual({ kind: 'pending' });
    });

    it('voids as inconclusive when no clean sample arrives within 2x the window', () => {
      const probe = armed();
      const verdict = probe.evaluate(1000 + 3000, 60, { ...CLEAN, suppressed: true });
      expect(verdict).toMatchObject({ kind: 'inconclusive', probe: { probedDPR: 1.8 } });
      expect(probe.isPending).toBe(false);
    });
  });
});
