import { describe, it, expect } from 'vitest';
import { HysteresisTracker } from '../../../../rendering/adaptive-dpr/hysteresis-tracker';

const CONFIG = { hysteresisMs: 3000, graceSamples: 1 };

describe('HysteresisTracker', () => {
  it('fires only after the full hysteresis period of high samples', () => {
    const h = new HysteresisTracker(CONFIG);
    expect(h.recordHigh(0)).toBe(false); // streak starts
    expect(h.recordHigh(1500)).toBe(false);
    expect(h.recordHigh(2999)).toBe(false);
    expect(h.recordHigh(3000)).toBe(true);
  });

  it('tolerates graceSamples consecutive mid-band samples without resetting', () => {
    const h = new HysteresisTracker(CONFIG);
    h.recordHigh(0);
    h.recordMidband(); // one strike — tolerated (grace = 1)
    expect(h.active).toBe(true);
    expect(h.recordHigh(3000)).toBe(true); // streak survived the wobble
  });

  it('resets after more than graceSamples consecutive mid-band samples', () => {
    const h = new HysteresisTracker(CONFIG);
    h.recordHigh(0);
    h.recordMidband();
    h.recordMidband(); // second consecutive strike — streak dies
    expect(h.active).toBe(false);
    expect(h.recordHigh(3000)).toBe(false); // fresh streak
  });

  it('a high sample between mid-band samples resets the strike count', () => {
    const h = new HysteresisTracker(CONFIG);
    h.recordHigh(0);
    h.recordMidband();
    h.recordHigh(500); // strikes reset to 0
    h.recordMidband(); // one strike again — still tolerated
    expect(h.active).toBe(true);
    expect(h.recordHigh(3000)).toBe(true);
  });

  it('a low sample always resets the streak', () => {
    const h = new HysteresisTracker(CONFIG);
    h.recordHigh(0);
    h.recordLow();
    expect(h.active).toBe(false);
    expect(h.recordHigh(5000)).toBe(false);
  });

  it('mid-band samples with no active streak are a no-op', () => {
    const h = new HysteresisTracker(CONFIG);
    h.recordMidband();
    h.recordMidband();
    expect(h.active).toBe(false);
    expect(h.recordHigh(0)).toBe(false); // fresh streak, not poisoned
  });

  it('graceSamples: 0 restores the strict any-sample-resets rule', () => {
    const h = new HysteresisTracker({ hysteresisMs: 3000, graceSamples: 0 });
    h.recordHigh(0);
    h.recordMidband();
    expect(h.active).toBe(false);
  });
});
