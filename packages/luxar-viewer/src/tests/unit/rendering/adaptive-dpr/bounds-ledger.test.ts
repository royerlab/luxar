import { describe, it, expect } from 'vitest';
import { BoundsLedger } from '../../../../rendering/adaptive-dpr/bounds-ledger';

const CONFIG = { minDPR: 0.5, floorTtlMs: 30_000 };

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
  });
});
