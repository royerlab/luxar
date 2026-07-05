import { describe, it, expect } from 'vitest';
import { BoundsLedger } from '../../../../rendering/adaptive-dpr/bounds-ledger';

const CONFIG = {
  minDPR: 0.5,
  floorTtlMs: 30_000,
  backoffMultiplier: 2,
  backoffMaxTtlMs: 300_000,
  ceilingTtlMs: 60_000,
  punishedAscentWindowMs: 3000,
  punishedAscentThreshold: 2,
};

describe('BoundsLedger — floor', () => {
  it('starts at minDPR and blocks proposals at or below it', () => {
    const ledger = new BoundsLedger(CONFIG);
    expect(ledger.dprFloor).toBe(0.5);
    expect(ledger.blocksScaleDownTo(0.5)).toBe(true);
    expect(ledger.blocksScaleDownTo(0.5005)).toBe(true); // within epsilon
    expect(ledger.blocksScaleDownTo(0.52)).toBe(false);
  });

  it('tightens the floor on a rejection — the exact probed value is then blocked', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordRejection(1.8, 5000);
    expect(ledger.dprFloor).toBe(1.8);
    // The next multiplicative step proposes exactly the floored value;
    // the to-or-below rule must block it (no re-firing the failed probe).
    expect(ledger.blocksScaleDownTo(1.8)).toBe(true);
    expect(ledger.blocksScaleDownTo(1.9)).toBe(false);
  });

  it('decays the floor back to minDPR only after the TTL', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordRejection(1.8, 5000);

    expect(ledger.decayIfExpired(5000 + 30_000)).toBe(false); // exactly TTL — not yet
    expect(ledger.dprFloor).toBe(1.8);

    expect(ledger.decayIfExpired(5000 + 30_001)).toBe(true);
    expect(ledger.dprFloor).toBe(0.5);
    // Already at minDPR — further decay reports nothing.
    expect(ledger.decayIfExpired(100_000)).toBe(false);
  });

  it('reset() forgets the learned floor immediately', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordRejection(1.8, 5000);
    ledger.reset();
    expect(ledger.dprFloor).toBe(0.5);
    expect(ledger.blocksScaleDownTo(1.8)).toBe(false);
    expect(ledger.backoffLevel).toBe(0);
  });
});

describe('BoundsLedger — rejection backoff', () => {
  it('escalates the TTL exponentially on consecutive identical rejections, up to the cap', () => {
    const ledger = new BoundsLedger(CONFIG);
    expect(ledger.recordRejection(1.8, 0)).toBe(30_000); // level 1
    expect(ledger.recordRejection(1.8, 40_000)).toBe(60_000); // level 2
    expect(ledger.recordRejection(1.8, 110_000)).toBe(120_000); // level 3
    expect(ledger.recordRejection(1.8, 240_000)).toBe(240_000); // level 4
    expect(ledger.recordRejection(1.8, 500_000)).toBe(300_000); // capped
    expect(ledger.backoffLevel).toBe(5);
  });

  it('the escalation memory SURVIVES floor expiry — the next identical rejection backs off harder', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordRejection(1.8, 0); // 30s
    expect(ledger.decayIfExpired(31_000)).toBe(true); // floor lifts...
    // ...but the same failed experiment repeated now waits 60s.
    expect(ledger.recordRejection(1.8, 32_000)).toBe(60_000);
  });

  it('a rejection at a DIFFERENT DPR restarts the ladder', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordRejection(1.8, 0);
    ledger.recordRejection(1.8, 40_000); // level 2
    expect(ledger.recordRejection(1.2, 80_000)).toBe(30_000); // new experiment — level 1
    expect(ledger.backoffLevel).toBe(1);
  });

  it('an accepted probe resets the rejection streak', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordRejection(1.8, 0);
    ledger.recordRejection(1.8, 40_000); // level 2
    ledger.recordAcceptance();
    expect(ledger.backoffLevel).toBe(0);
    expect(ledger.recordRejection(1.8, 80_000)).toBe(30_000); // ladder restarted
  });
});

describe('BoundsLedger — ceiling (punished-ascent demotion)', () => {
  it('starts undemoted (null → caller uses native)', () => {
    const ledger = new BoundsLedger(CONFIG);
    expect(ledger.dprCeiling).toBeNull();
  });

  it('demotes to exactly 1.0 after the configured number of punished ascents', () => {
    const ledger = new BoundsLedger(CONFIG);

    ledger.recordAscent(1.4, 0);
    expect(ledger.recordSlowSample(1000)).toBe(false); // punished #1 (< threshold 2)
    expect(ledger.ascentPunishments).toBe(1);
    expect(ledger.dprCeiling).toBeNull();

    ledger.recordAscent(1.3, 5000);
    expect(ledger.recordSlowSample(6000)).toBe(true); // punished #2 → demoted NOW
    expect(ledger.dprCeiling).toBe(1.0);
  });

  it('ignores slow samples outside the punishment window and ascents at or below 1.0', () => {
    const ledger = new BoundsLedger(CONFIG);

    ledger.recordAscent(1.4, 0);
    expect(ledger.recordSlowSample(3001)).toBe(false); // outside 3000ms window
    expect(ledger.ascentPunishments).toBe(0);

    ledger.recordAscent(0.9, 5000); // not above 1.0 — not a luxury ascent
    expect(ledger.recordSlowSample(5500)).toBe(false);
    expect(ledger.ascentPunishments).toBe(0);
  });

  it('a slow sample consumes the ascent — one ascent cannot be punished twice', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordAscent(1.4, 0);
    ledger.recordSlowSample(500);
    ledger.recordSlowSample(600);
    expect(ledger.ascentPunishments).toBe(1);
  });

  it('the demotion expires after its TTL, with backoff escalation on re-demotion', () => {
    const ledger = new BoundsLedger(CONFIG);
    const demoteAt = (t: number) => {
      ledger.recordAscent(1.4, t);
      ledger.recordSlowSample(t + 500);
      ledger.recordAscent(1.4, t + 1000);
      ledger.recordSlowSample(t + 1500); // second punishment → demote
    };

    demoteAt(0); // level 1 → TTL 60s, expires at 61_500
    expect(ledger.dprCeiling).toBe(1.0);
    expect(ledger.decayCeilingIfExpired(61_000)).toBe(false);
    expect(ledger.decayCeilingIfExpired(61_501)).toBe(true);
    expect(ledger.dprCeiling).toBeNull();

    demoteAt(70_000); // level 2 → TTL 120s, expires at 71_500 + 120_000
    expect(ledger.decayCeilingIfExpired(71_500 + 120_000)).toBe(false);
    expect(ledger.decayCeilingIfExpired(71_501 + 120_000)).toBe(true);
  });

  it('content changes soften the ceiling and reset its streaks', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordAscent(1.4, 0);
    ledger.recordSlowSample(500);
    ledger.recordAscent(1.4, 1000);
    ledger.recordSlowSample(1500); // demoted, expires at 61_500

    ledger.softenForContentChange(2000, 5000); // expiry pulled to 7000
    expect(ledger.decayCeilingIfExpired(6999)).toBe(false);
    expect(ledger.decayCeilingIfExpired(7001)).toBe(true);

    // Streak reset: the next demotion starts back at the base TTL.
    ledger.recordAscent(1.4, 10_000);
    ledger.recordSlowSample(10_500);
    ledger.recordAscent(1.4, 11_000);
    ledger.recordSlowSample(11_500);
    expect(ledger.decayCeilingIfExpired(11_500 + 60_000)).toBe(false);
    expect(ledger.decayCeilingIfExpired(11_501 + 60_000)).toBe(true);
  });

  it('reset() clears all ceiling state', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordAscent(1.4, 0);
    ledger.recordSlowSample(500);
    ledger.recordAscent(1.4, 1000);
    ledger.recordSlowSample(1500);
    expect(ledger.dprCeiling).toBe(1.0);

    ledger.reset();
    expect(ledger.dprCeiling).toBeNull();
    expect(ledger.ascentPunishments).toBe(0);
  });
});

describe('BoundsLedger — content-change softening', () => {
  it('pulls the floor expiry forward to at most recheckMs from now', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordRejection(1.8, 0); // expires at 30_000
    ledger.softenForContentChange(1000, 5000); // expiry pulled to 6000

    expect(ledger.decayIfExpired(5000)).toBe(false);
    expect(ledger.decayIfExpired(6001)).toBe(true);
  });

  it('never pushes an already-near expiry further out', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordRejection(1.8, 0); // expires at 30_000
    ledger.softenForContentChange(29_000, 5000); // min(30_000, 34_000) = 30_000
    expect(ledger.decayIfExpired(30_001)).toBe(true);
  });

  it('resets the escalation streak — new content deserves fresh evidence', () => {
    const ledger = new BoundsLedger(CONFIG);
    ledger.recordRejection(1.8, 0);
    ledger.recordRejection(1.8, 40_000); // level 2
    ledger.softenForContentChange(50_000, 5000);
    expect(ledger.backoffLevel).toBe(0);
    expect(ledger.recordRejection(1.8, 60_000)).toBe(30_000);
  });
});
