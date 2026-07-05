import { describe, it, expect } from 'vitest';
import { BoundsLedger } from '../../../../rendering/adaptive-dpr/bounds-ledger';

const CONFIG = {
  minDPR: 0.5,
  floorTtlMs: 30_000,
  backoffMultiplier: 2,
  backoffMaxTtlMs: 300_000,
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
