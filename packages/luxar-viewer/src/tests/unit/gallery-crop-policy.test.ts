/**
 * Gallery crop policy — unit tests for the border-lit verdict.
 *
 * The pixel-counting half is not testable here (see `measureBorderLit`); the two
 * pure halves that decide WHICH poses get measured and WHAT the verdict says are.
 * No simulation is needed for the verdict — it is one comparison plus a message,
 * and what actually deserves pinning is the ADVICE: the warning must point at
 * whichever framing knob the demo really used (`fillTarget`, `zoom`, `distance`,
 * or a skipped fill), because suggesting a knob that is not in play is worse than
 * saying nothing. For the pose selection, what deserves pinning is COVERAGE of the
 * rock: sampling only its endpoints misses a subject whose projected extent peaks
 * at an intermediate angle.
 *
 * Every threshold is asserted through the imported constant, never a literal
 * copy, and the floor is additionally exercised at a NON-default value — so
 * raising `BORDER_LIT_MAX` (the documented escape hatch if the warning turns out
 * to be noisy: 25 of the 28 committed tiles are non-zero) neither breaks this
 * suite nor leaves the knob unpinned.
 */

import { describe, it, expect } from 'vitest';
import {
  borderLitPercent,
  borderSampleFrames,
  evaluateBorderLit,
  selectCropVerdictSamples,
  BORDER_LIT_MAX,
  BORDER_SAMPLE_STEP_DEG,
  FILL_TARGET_MIN,
  FILL_TARGET_SUGGEST_STEP,
  type BorderSample,
  type CropFraming,
} from '../screenshots/crop-policy';

/**
 * Perimeter of the 900×900 committed README media (2·900 + 2·900 − 4). NOT the
 * capture geometry — the live check reads a 1080×1080 screenshot, perimeter 4316
 * — but it is the geometry every number quoted in `crop-policy.ts`'s audit was
 * measured at, so the fixtures stay comparable with those doc figures.
 */
const PERIMETER = 2 * 900 + 2 * 900 - 4;

const sample = (label: string, borderLit: number): BorderSample => ({
  label,
  borderLit,
  borderPixels: PERIMETER,
});

/** The default framing path: closed-loop fill, no zoom nudge, no distance. */
const fill = (fillTarget: number): CropFraming => ({ fillTarget, autoFrame: true });

describe('gallery crop policy', () => {
  describe('which orbit poses get measured', () => {
    /** The harness values: ±20° rock over 120 captured frames. */
    const AMP = 20;
    const FRAMES = 120;
    const angleAt = (i: number, n = FRAMES): number => AMP * Math.sin((i / n) * Math.PI * 2);

    it('spans the whole rock — every angle is within one grid step of a measured pose', () => {
      const frames = borderSampleFrames(FRAMES, AMP);
      const angles = frames.map((i) => angleAt(i));
      // The endpoints-only version measured exactly these two and nothing else.
      expect(Math.min(...angles)).toBeCloseTo(-AMP, 6);
      expect(Math.max(...angles)).toBeCloseTo(AMP, 6);
      expect(frames.length).toBeGreaterThan(2);
      // The property that matters: no pose in the sweep is far from a sampled one,
      // so a subject whose projected extent peaks mid-rock cannot hide between
      // samples. Walk the sweep finely and check the nearest measured pose.
      for (let a = -AMP; a <= AMP; a += 0.25) {
        const nearest = Math.min(...angles.map((x) => Math.abs(x - a)));
        expect(nearest).toBeLessThanOrEqual(BORDER_SAMPLE_STEP_DEG);
      }
    });

    it('still includes the two rock extremes, and stays cheap', () => {
      const frames = borderSampleFrames(FRAMES, AMP);
      expect(frames).toContain(FRAMES / 4); // +20°
      expect(frames).toContain((3 * FRAMES) / 4); // −20°
      // One in-page PNG decode per pose, so the grid must stay a handful of poses,
      // not all 120 frames.
      expect(frames.length).toBeLessThanOrEqual(16);
      // Sorted, unique, in range.
      expect([...new Set(frames)]).toEqual(frames);
      expect([...frames].sort((a, b) => a - b)).toEqual(frames);
      expect(frames.every((i) => Number.isInteger(i) && i >= 0 && i < FRAMES)).toBe(true);
    });

    it('degrades sanely on a smoke run with very few frames', () => {
      // GALLERY_ORBIT_FRAMES=4: fewer distinct poses exist than the grid asks for,
      // so it must dedup rather than emit repeats or out-of-range indices.
      const frames = borderSampleFrames(4, AMP);
      expect(frames.length).toBeGreaterThan(0);
      expect(frames.every((i) => i >= 0 && i < 4)).toBe(true);
      expect([...new Set(frames)]).toEqual(frames);
      expect(borderSampleFrames(1, AMP)).toEqual([0]);
      expect(borderSampleFrames(0, AMP)).toEqual([]);
    });

    it('falls back to the endpoints for a coarse step, and never spins on a zero one', () => {
      // A step wider than the sweep leaves one interval: its two ends.
      expect(borderSampleFrames(FRAMES, AMP, 1000)).toEqual([FRAMES / 4, (3 * FRAMES) / 4]);
      // A zero step would be an infinite grid; it is capped at one pose per frame.
      const dense = borderSampleFrames(8, AMP, 0);
      expect(dense.length).toBeLessThanOrEqual(8);
      expect(dense.every((i) => i >= 0 && i < 8)).toBe(true);
    });
  });

  describe('borderLitPercent', () => {
    it('is the perimeter fraction, and 0 for a degenerate empty perimeter', () => {
      expect(borderLitPercent(sample('still', PERIMETER / 2))).toBeCloseTo(50, 6);
      expect(borderLitPercent({ label: 'x', borderLit: 5, borderPixels: 0 })).toBe(0);
    });
  });

  describe('verdict', () => {
    it('judges no-orbit tiles from the published still while preserving normal orbit checks', () => {
      const still = sample('still', 0);
      const croppedOrbit = sample('rock +20°', BORDER_LIT_MAX + 1);

      expect(selectCropVerdictSamples(still, [croppedOrbit], true)).toEqual([still]);
      expect(selectCropVerdictSamples(still, [croppedOrbit], false)).toEqual([still, croppedOrbit]);
      expect(selectCropVerdictSamples(null, [croppedOrbit], true)).toEqual([]);
    });

    it('reports no crop when nothing is lit on the border', () => {
      const samples = [sample('still', 0), sample('rock +20°', 0), sample('rock -20°', 0)];
      const v = evaluateBorderLit({ demoId: 'lorenz', samples, framing: fill(0.95) });
      expect(v.cropped).toBe(false);
      expect(v.message).toBeNull();
      // The worst sample is still reported: the harness logs the count either
      // way, because the NUMBER is the per-tile regression signal.
      expect(v.worst).not.toBeNull();
      expect(v.worst!.borderLit).toBe(0);
    });

    it('handles an empty sample list without throwing', () => {
      const v = evaluateBorderLit({ demoId: 'lorenz', samples: [], framing: fill(0.95) });
      expect(v.cropped).toBe(false);
      expect(v.worst).toBeNull();
      expect(v.message).toBeNull();
    });

    it('flags a single cropped pose among clean ones', () => {
      const count = BORDER_LIT_MAX + 218;
      const samples = [sample('still', 0), sample('rock +20°', count), sample('rock -20°', 0)];
      const v = evaluateBorderLit({
        demoId: 'mesh_isosurface_cells3d',
        samples,
        framing: fill(0.84),
      });
      expect(v.cropped).toBe(true);
      expect(v.worst!.label).toBe('rock +20°');
      const msg = v.message!;
      expect(msg).toContain('mesh_isosurface_cells3d'); // names the demo
      // Count and perimeter as ONE fragment: asserted separately, a message that
      // printed the ratio backwards would pass.
      expect(msg).toContain(`${count}/${PERIMETER}`);
      expect(msg).toContain('rock +20°'); // which pose
      expect(msg).toContain(`${borderLitPercent(samples[1]).toFixed(1)}%`); // ≈6.1%
      // The point of the whole check: the coverage loop cannot see this.
      expect(msg).toMatch(/coverage/i);
      expect(msg.split('\n')).toHaveLength(1); // one line
    });

    it('picks the MAXIMUM sample, not the first or the last', () => {
      const samples = [
        sample('still', BORDER_LIT_MAX + 5),
        sample('rock +20°', BORDER_LIT_MAX + 99),
        sample('rock -20°', BORDER_LIT_MAX + 7),
      ];
      const v = evaluateBorderLit({ demoId: 'demo', samples, framing: fill(0.95) });
      expect(v.worst!.label).toBe('rock +20°');
    });

    it('keeps the FIRST sample on a tie (the still is the reproducible pose)', () => {
      const samples = [
        sample('still', BORDER_LIT_MAX + 42),
        sample('rock +20°', BORDER_LIT_MAX + 42),
      ];
      const v = evaluateBorderLit({ demoId: 'demo', samples, framing: fill(0.95) });
      expect(v.worst!.label).toBe('still');
    });
  });

  describe('the warning floor', () => {
    const at = (floor: number, borderLit: number) =>
      evaluateBorderLit(
        { demoId: 'demo', samples: [sample('still', borderLit)], framing: fill(0.95) },
        floor
      );

    it('does not warn exactly AT the default floor, and does one pixel above it', () => {
      const atFloor = evaluateBorderLit({
        demoId: 'demo',
        samples: [sample('still', BORDER_LIT_MAX)],
        framing: fill(0.95),
      });
      const overFloor = evaluateBorderLit({
        demoId: 'demo',
        samples: [sample('still', BORDER_LIT_MAX + 1)],
        framing: fill(0.95),
      });
      expect(atFloor.cropped).toBe(false);
      expect(atFloor.message).toBeNull();
      expect(overFloor.cropped).toBe(true);
      expect(overFloor.message).not.toBeNull();
    });

    it('honours a RAISED floor — the documented escape hatch', () => {
      // ~1% of the 1080² capture perimeter above the default: the kind of floor
      // the docs suggest if the warning proves noisy. Offset from the constant so
      // the case holds however the default is set.
      const raised = BORDER_LIT_MAX + 43;
      expect(at(raised, raised).cropped).toBe(false);
      expect(at(raised, raised).message).toBeNull();
      expect(at(raised, raised + 1).cropped).toBe(true);
      // ...and the default floor is genuinely stricter than the raised one, which
      // is what proves the parameter is actually consulted.
      expect(at(BORDER_LIT_MAX, raised).cropped).toBe(true);
    });
  });

  describe('advice follows the framing path in effect', () => {
    const cropped = [sample('still', BORDER_LIT_MAX + 300)];

    it('suggests one step lower fillTarget on the closed-loop fill path', () => {
      const effective = 0.84;
      const v = evaluateBorderLit({ demoId: 'demo', samples: cropped, framing: fill(effective) });
      const suggested = Math.round((effective - FILL_TARGET_SUGGEST_STEP) * 100) / 100;
      expect(v.message).toContain(`fillTarget from ${effective} to ${suggested}.`);
    });

    it('takes the fill path when autoFrame is UNSET (the common manifest case)', () => {
      // 57 of the 61 manifest demos omit `autoFrame`, and the harness forwards
      // that `undefined` verbatim — so an `!framing.autoFrame` test here would
      // send almost every demo down the "framing is baked" branch.
      const v = evaluateBorderLit({
        demoId: 'demo',
        samples: cropped,
        framing: { fillTarget: 0.95 },
      });
      expect(v.message).toContain('fillTarget from 0.95');
      expect(v.message).not.toContain('viewer_config');
    });

    it('refuses to suggest a target under the framing floor', () => {
      // Suggesting a step here would print a no-op ("from 0.3 to 0.3") or, lower
      // still, an INVERTED suggestion ("from 0.25 to 0.3") that tightens the crop.
      for (const effective of [FILL_TARGET_MIN, FILL_TARGET_MIN + FILL_TARGET_SUGGEST_STEP / 2]) {
        const v = evaluateBorderLit({ demoId: 'demo', samples: cropped, framing: fill(effective) });
        expect(v.message).toContain('under the framing floor');
        expect(v.message).not.toContain('Try lowering fillTarget from');
      }
      // Just above: one step lands exactly ON the floor, which is allowed — and
      // the boundary is decided on the ROUNDED value that gets printed, not on
      // the float subtraction (0.4 − 0.1 === 0.30000000000000004).
      const edge = evaluateBorderLit({
        demoId: 'demo',
        samples: cropped,
        framing: fill(FILL_TARGET_MIN + FILL_TARGET_SUGGEST_STEP),
      });
      expect(edge.message).toContain(`to ${FILL_TARGET_MIN}.`);
    });

    it('also blames a zoom > 1, which is applied AFTER the framing', () => {
      const v = evaluateBorderLit({
        demoId: 'demo',
        samples: cropped,
        framing: { fillTarget: 0.95, zoom: 3, autoFrame: true },
      });
      expect(v.message).toContain('zoom=3');
      expect(v.message).toContain('fillTarget'); // in ADDITION to, not instead of
      // The DIRECTION matters: telling the operator to raise the zoom would
      // tighten the crop it is warning about.
      expect(v.message).toContain('lower it first');
    });

    it('stays quiet about zoom when it is absent or does not zoom in', () => {
      const absent = evaluateBorderLit({
        demoId: 'demo',
        samples: cropped,
        framing: fill(0.95),
      });
      const out = evaluateBorderLit({
        demoId: 'demo',
        samples: cropped,
        framing: { fillTarget: 0.95, zoom: 0.8, autoFrame: true },
      });
      const unity = evaluateBorderLit({
        demoId: 'demo',
        samples: cropped,
        framing: { fillTarget: 0.95, zoom: 1, autoFrame: true },
      });
      expect(absent.message).not.toContain('zoom');
      expect(out.message).not.toContain('zoom');
      expect(unity.message).not.toContain('zoom');
    });

    it('points at `distance` (and never fillTarget) when the fill was bypassed', () => {
      const v = evaluateBorderLit({
        demoId: 'quantum_orbitals',
        samples: cropped,
        framing: { fillTarget: 0.95, distance: 120, autoFrame: true },
      });
      expect(v.message).toContain('distance=120');
      // RAISE, not lower: the inverted advice would tighten the crop.
      expect(v.message).toContain('RAISE that distance');
      expect(v.message).not.toContain('fillTarget');
    });

    it('says the fill was skipped when autoFrame is off', () => {
      const v = evaluateBorderLit({
        demoId: 'gaia_stars',
        samples: cropped,
        framing: { fillTarget: 0.95, autoFrame: false },
      });
      // The message names both possibilities (an authored camera, or a plain
      // bounds fit when the scene authors no viewer_config); one fragment is
      // enough to identify the branch.
      expect(v.message).toContain('viewer_config');
      expect(v.message).not.toContain('Try lowering fillTarget');
    });

    it('names zoom on the OTHER two paths too — the harness applies it on all three', () => {
      // `protein_landscape` in the manifest is exactly this: autoFrame false AND
      // zoom 3. Its demo script authors no ViewerConfig, so the baked advice
      // alone would point at a camera that does not exist while ignoring the 3×
      // zoom-in applied last.
      const baked = evaluateBorderLit({
        demoId: 'protein_landscape',
        samples: cropped,
        framing: { fillTarget: 0.95, autoFrame: false, zoom: 3 },
      });
      const dist = evaluateBorderLit({
        demoId: 'demo',
        samples: cropped,
        framing: { fillTarget: 0.95, distance: 600, zoom: 2 },
      });
      expect(baked.message).toContain('zoom=3');
      expect(baked.message).toContain('viewer_config');
      expect(dist.message).toContain('zoom=2');
      expect(dist.message).toContain('distance=600');
    });
  });
});
