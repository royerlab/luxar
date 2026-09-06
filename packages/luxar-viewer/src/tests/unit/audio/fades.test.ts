import { describe, it, expect } from 'vitest';
import { dbToGain, rampGain } from '../../../audio/fades';
import { MIN_FADE_MS } from '../../../types/audio';
import { FakeAudioParam } from '../../mocks/fake-audio-context.mock';

describe('rampGain', () => {
  it('cancels pending automation, holds, then ramps to the target', () => {
    const p = new FakeAudioParam(0.2);
    const end = rampGain(p as unknown as AudioParam, 10, 10, 1, 500);
    expect(p.calls.map((c) => c.method)).toEqual([
      'cancelScheduledValues',
      'setValueAtTime',
      'linearRampToValueAtTime',
    ]);
    expect(p.calls[1]).toMatchObject({ value: 0.2, time: 10 });
    expect(end).toBeCloseTo(10.5);
    expect(p.lastRampTarget()).toBe(1);
  });

  it('floors every ramp at MIN_FADE_MS so a start never clicks', () => {
    const p = new FakeAudioParam(0);
    const end = rampGain(p as unknown as AudioParam, 0, 0, 1, 0);
    expect(end).toBeCloseTo(MIN_FADE_MS / 1000);
  });

  it('a delayed start holds `from` until startAt before ramping', () => {
    const p = new FakeAudioParam(0.7);
    rampGain(p as unknown as AudioParam, 1, 1.8, 0.5, 200, 0);
    // cancel, set(0 @1), set(0 @1.8), ramp(0.5 @2.0)
    expect(p.calls[1]).toMatchObject({ value: 0, time: 1 });
    expect(p.calls[2]).toMatchObject({ value: 0, time: 1.8 });
    expect(p.calls[3]).toMatchObject({ method: 'linearRampToValueAtTime', value: 0.5 });
    expect(p.calls[3].time).toBeCloseTo(2.0);
  });
});

describe('dbToGain', () => {
  it('maps -9 dB to about 0.355 and 0 dB to 1', () => {
    expect(dbToGain(0)).toBe(1);
    expect(dbToGain(-9)).toBeCloseTo(0.355, 2);
    expect(dbToGain(-20)).toBeCloseTo(0.1, 5);
  });
});
