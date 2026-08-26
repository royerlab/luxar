import { describe, expect, it } from 'vitest';
import { COVERAGE_MIN, LIT_FRACTION_MIN, evaluateUnderfill } from '../screenshots/crop-policy';

describe('gallery under-fill policy', () => {
  it('pins the measured floors that catch the original regression', () => {
    expect(COVERAGE_MIN).toBe(0.5);
    expect(LIT_FRACTION_MIN).toBe(0.1);
    expect(
      evaluateUnderfill({
        demoId: 'atp_synthase',
        measurement: { coverage: 0.493, litFraction: 0.24 },
      }).underfilled
    ).toBe(true);
  });

  it('stays quiet at the measured floors and warns immediately below them', () => {
    const atFloor = evaluateUnderfill({
      demoId: 'demo',
      measurement: { coverage: COVERAGE_MIN, litFraction: LIT_FRACTION_MIN },
    });
    expect(atFloor).toEqual({ underfilled: false, message: null });

    const lowSpan = evaluateUnderfill({
      demoId: 'demo',
      measurement: { coverage: COVERAGE_MIN - 0.001, litFraction: 0.5 },
    });
    expect(lowSpan.underfilled).toBe(true);
    expect(lowSpan.message).toContain('49.9% span');
    expect(lowSpan.message).toContain('minimum 50.0%');

    const lowArea = evaluateUnderfill({
      demoId: 'demo',
      measurement: { coverage: 0.9, litFraction: LIT_FRACTION_MIN - 0.001 },
    });
    expect(lowArea.underfilled).toBe(true);
    expect(lowArea.message).toContain('9.9% lit area');
    expect(lowArea.message).toContain('minimum 10.0%');
  });

  it('uses lit area to catch an elongated subject whose span looks full', () => {
    const verdict = evaluateUnderfill({
      demoId: 'thin_subject',
      measurement: { coverage: 0.9, litFraction: 0.09 },
    });

    expect(verdict.underfilled).toBe(true);
    expect(verdict.message).toContain('[thin_subject] under-filled?');
    expect(verdict.message).toContain('90.0% span');
    expect(verdict.message).toContain('9.0% lit area');
  });

  it('reports both failed signals for an almost-empty frame', () => {
    const verdict = evaluateUnderfill({
      demoId: 'empty',
      measurement: { coverage: 0, litFraction: 0 },
    });

    expect(verdict.underfilled).toBe(true);
    expect(verdict.message).toContain('span is below minimum 50.0%');
    expect(verdict.message).toContain('lit area is below minimum 10.0%');
  });
});
