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
      // 3 janky frames right after a stall — must NOT settle the probe.
      const verdict = probe.evaluate(1000 + 1600, 60, {
        sampleCount: 3,
        spanMs: 900,
        suppressed: false,
      });
      expect(verdict).toEqual({ kind: 'pending' });
      expect(probe.isPending).toBe(true);

      // A clean sample arriving later settles normally.
      expect(probe.evaluate(1000 + 2100, 60, CLEAN)).toMatchObject({ kind: 'accepted' });
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
