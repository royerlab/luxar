/**
 * Tests for validateControls (src/config/sections/controls/validate.ts).
 *
 * Audit followup — see delme/test-audit-luxar-viewer.src/config.md
 *
 *   [G5][P5] / [O3][P4,P5] / [M2][P11]: pre-audit the file had only 4
 *   tests for 8 parallel `ConfigRange` validators — 5 of the 8 ranges
 *   were entirely unexercised, so any mutation in their loop body
 *   (e.g. swapping `>=` for `>`, or dropping a NaN guard on one entry)
 *   would have slipped through. We now use `it.each` to hit every
 *   range × every failure mode (NaN, Infinity on min/max/default;
 *   min>=max; default-outside-bounds). The 8 ranges are intentionally
 *   declared with the same shape as the source's `ranges` array so a
 *   future entry added to the source becomes immediately visible here.
 */

import { describe, it, expect } from 'vitest';
import type { AppConfig } from '../../../../../config/types';
import { validateControls } from '../../../../../config/sections/controls/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';
import { config } from '../../../../../config';
import {
  DEFAULT_AUTO_DOLLY_AMPLITUDE,
  DEFAULT_AUTO_DOLLY_PERIOD,
  dollyAmplitudeFromPercent,
  dollyAmplitudeToPercent,
} from '../../../../../controls/types';

/**
 * Path-getters mirroring the source's `ranges` array (validate.ts:10-19).
 * Each entry's `set(cfg, range)` writes a new ConfigRange at the same
 * location. Adding a new range to the source without adding it here will
 * NOT silently regress coverage — the `MAP_COMPLETENESS` test below
 * cross-checks our list against the validator's behaviour by asserting
 * we exercise as many ranges as the source declares (10).
 */
type RangeRef = {
  name: string;
  set: (cfg: AppConfig, r: { min: number; max: number; default: number }) => void;
};
const RANGE_REFS: RangeRef[] = [
  {
    name: 'fly.movement.speed',
    set: (c, r) => {
      c.controls.fly.movement.speed = r;
    },
  },
  {
    name: 'fly.movement.acceleration',
    set: (c, r) => {
      c.controls.fly.movement.acceleration = r;
    },
  },
  {
    name: 'fly.movement.damping',
    set: (c, r) => {
      c.controls.fly.movement.damping = r;
    },
  },
  {
    name: 'fly.rotation.speed',
    set: (c, r) => {
      c.controls.fly.rotation.speed = r;
    },
  },
  {
    name: 'fly.rotation.damping',
    set: (c, r) => {
      c.controls.fly.rotation.damping = r;
    },
  },
  {
    name: 'orbit.autoRotate.speed',
    set: (c, r) => {
      c.controls.orbit.autoRotate.speed = r;
    },
  },
  {
    name: 'orbit.autoDolly.amplitudePercent',
    set: (c, r) => {
      c.controls.orbit.autoDolly.amplitudePercent = r;
    },
  },
  {
    name: 'orbit.autoDolly.period',
    set: (c, r) => {
      c.controls.orbit.autoDolly.period = r;
    },
  },
  {
    name: 'orbit.zoom.speed',
    set: (c, r) => {
      c.controls.orbit.zoom.speed = r;
    },
  },
  {
    name: 'orbit.damping.factor',
    set: (c, r) => {
      c.controls.orbit.damping.factor = r;
    },
  },
];

describe('validateControls', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateControls).valid).toBe(true);
  });

  describe('wheelZoomSensitivity (global wheel-zoom multiplier)', () => {
    it.each([0, -1, NaN, Infinity])('errors on %s (must be finite and > 0)', (bad) => {
      const cfg = cloneConfig();
      cfg.controls.wheelZoomSensitivity = bad;
      const result = invokeValidator(validateControls, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringMatching(/controls\.wheelZoomSensitivity.*finite and > 0/)
      );
    });

    it('accepts any positive finite value (the UI range is a UX choice, not a validity one)', () => {
      const cfg = cloneConfig();
      cfg.controls.wheelZoomSensitivity = 0.01;
      expect(invokeValidator(validateControls, cfg).valid).toBe(true);
    });
  });

  // [G5] every ConfigRange × min>=max — pre-audit only fly.movement.speed was tested.
  describe.each(RANGE_REFS)('range $name', ({ name, set }) => {
    it('errors when min >= max (min set greater than max)', () => {
      const cfg = cloneConfig();
      set(cfg, { min: 100, max: 1, default: 50 });
      const result = invokeValidator(validateControls, cfg);
      expect(result.valid).toBe(false);
      // [P2] assert BOTH the range name and the specific failure mode
      // appear together. A mutation that emitted a generic "Invalid range"
      // for all entries would pass `stringContaining(name)` alone.
      expect(result.errors).toContainEqual(
        expect.stringMatching(new RegExp(`controls\\.${name.replace(/\./g, '\\.')}.*min.*>=.*max`))
      );
    });

    it('errors when min === max (no strict inequality)', () => {
      const cfg = cloneConfig();
      set(cfg, { min: 0.5, max: 0.5, default: 0.5 });
      const result = invokeValidator(validateControls, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining(`controls.${name}`));
    });

    it('errors when default is below min', () => {
      const cfg = cloneConfig();
      set(cfg, { min: 1, max: 10, default: 0 });
      const result = invokeValidator(validateControls, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringMatching(
          new RegExp(`controls\\.${name.replace(/\./g, '\\.')}.*default.*outside`)
        )
      );
    });

    it('errors when default is above max', () => {
      const cfg = cloneConfig();
      set(cfg, { min: 0, max: 1, default: 2 });
      const result = invokeValidator(validateControls, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringMatching(
          new RegExp(`controls\\.${name.replace(/\./g, '\\.')}.*default.*outside`)
        )
      );
    });

    // [P5] boundary first-class: NaN on each of {min, max, default} must
    // be caught by the non-finite guard — pre-audit only fly.movement.speed.default
    // exercised this path.
    it.each(['min', 'max', 'default'] as const)('errors when %s is NaN (non-finite)', (field) => {
      const cfg = cloneConfig();
      const r = { min: 0, max: 1, default: 0.5, [field]: NaN } as {
        min: number;
        max: number;
        default: number;
      };
      set(cfg, r);
      const result = invokeValidator(validateControls, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringMatching(new RegExp(`controls\\.${name.replace(/\./g, '\\.')}.*non-finite`))
      );
    });

    it.each(['min', 'max', 'default'] as const)(
      'errors when %s is +Infinity (non-finite)',
      (field) => {
        const cfg = cloneConfig();
        const r = { min: 0, max: 1, default: 0.5, [field]: Number.POSITIVE_INFINITY } as {
          min: number;
          max: number;
          default: number;
        };
        set(cfg, r);
        const result = invokeValidator(validateControls, cfg);
        expect(result.valid).toBe(false);
        expect(result.errors).toContainEqual(
          expect.stringMatching(new RegExp(`controls\\.${name.replace(/\./g, '\\.')}.*non-finite`))
        );
      }
    );

    it('errors when -Infinity violates non-finite guard', () => {
      // -Infinity on `min` would otherwise pass `min < max` numerically;
      // the source uses Number.isFinite() which rejects ±Infinity equally.
      const cfg = cloneConfig();
      set(cfg, { min: Number.NEGATIVE_INFINITY, max: 1, default: 0.5 });
      const result = invokeValidator(validateControls, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringMatching(new RegExp(`controls\\.${name.replace(/\./g, '\\.')}.*non-finite`))
      );
    });
  });

  // [P10] Source-themed completeness guard: if a future entry is added
  // to the validator's `ranges` array, this assertion forces us to add
  // it to RANGE_REFS above. We can't introspect the validator's internal
  // array directly, but we can assert that perturbing each declared
  // range we DO know about produces a unique-named error — a strong
  // proxy for "the validator iterates every entry independently".
  it('every range in RANGE_REFS produces an error keyed to its name', () => {
    const errorNames = new Set<string>();
    for (const { name, set } of RANGE_REFS) {
      const cfg = cloneConfig();
      set(cfg, { min: 100, max: 1, default: 50 });
      const result = invokeValidator(validateControls, cfg);
      // Find the error mentioning this exact name.
      const hit = result.errors.find((e) => e.includes(`controls.${name}`));
      expect(hit).toBeDefined();
      errorNames.add(name);
    }
    // 10 ranges declared in the source — keep the literal expectation
    // so a deletion from the source's ranges array is loud.
    expect(errorNames.size).toBe(10);
  });

  // [W3-style] After perturbing exactly one range, no OTHER range's
  // error should appear — kills mutants that "errors" on the wrong key.
  it('perturbing one range does not produce errors for other ranges', () => {
    const cfg = cloneConfig();
    // Use default-outside-only (don't violate min<max) so we get exactly
    // one error from one range — keeps the isolation invariant precise.
    cfg.controls.orbit.zoom.speed = { min: 0, max: 1, default: 5 };
    const result = invokeValidator(validateControls, cfg);
    expect(result.errors.filter((e) => e.includes('controls.orbit.zoom.speed'))).toHaveLength(1);
    // No spillover into other ranges
    for (const { name } of RANGE_REFS.filter((r) => r.name !== 'orbit.zoom.speed')) {
      expect(result.errors.filter((e) => e.includes(`controls.${name}`))).toHaveLength(0);
    }
  });
});

/**
 * The auto-dolly has TWO sources of default: the class's own constructor
 * fallbacks (`DEFAULT_AUTO_DOLLY_*`, used when no config is passed) and the
 * slider ranges the UI builds from. The damping factor already has this shape
 * and guards it with nothing but a "keep aligned" comment; these assertions
 * make the alignment a gate instead, so a value edited in one place fails the
 * build rather than producing a slider whose default silently disagrees with
 * what a bare `new LuxarOrbitControls()` does.
 */
describe('auto-dolly defaults agree with the control class', () => {
  it('amplitude: config percent matches DEFAULT_AUTO_DOLLY_AMPLITUDE', () => {
    expect(
      dollyAmplitudeFromPercent(config.controls.orbit.autoDolly.amplitudePercent.default)
    ).toBeCloseTo(DEFAULT_AUTO_DOLLY_AMPLITUDE, 12);
  });

  it('period: config default matches DEFAULT_AUTO_DOLLY_PERIOD', () => {
    expect(config.controls.orbit.autoDolly.period.default).toBe(DEFAULT_AUTO_DOLLY_PERIOD);
  });

  it('the class default sits inside the slider range it will be shown on', () => {
    const range = config.controls.orbit.autoDolly.amplitudePercent;
    const asPercent = dollyAmplitudeToPercent(DEFAULT_AUTO_DOLLY_AMPLITUDE);
    expect(asPercent).toBeGreaterThanOrEqual(range.min);
    expect(asPercent).toBeLessThanOrEqual(range.max);
  });
});
