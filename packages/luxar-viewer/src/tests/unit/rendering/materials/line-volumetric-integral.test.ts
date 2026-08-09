/**
 * Quadrature validation of the volumetric line primitive's sum-family ray
 * integral (issue #1352).
 *
 * `lineVolumetricSumIntegral` mirrors the fragment-shader lane structure
 * (structural parallel / soft-soft / hard-hard / mixed inclusion–exclusion,
 * including the Taylor branches). These tests hold every lane against brute
 * numerical quadrature of the MODEL density over a grid of bend angles,
 * segment lengths, and ray obliquities — so the lane math is proven before
 * either shader backend bakes it in, and any future lane edit that breaks
 * the math fails here rather than in a screenshot.
 */
import { describe, expect, it } from 'vitest';

import { erfPoly } from '../../../../rendering/materials/_shared/erf';
import {
  LINE_PARALLEL_LANE_THRESHOLD,
  lineVolumetricModelDensity,
  lineVolumetricSumIntegral,
  type Vec3,
  type VolumetricSegment,
} from '../../../../rendering/materials/_shared/line-volumetric';

const SIGMA = 1.0;
const INV_NORM = 1 / (SIGMA * Math.sqrt(2 * Math.PI));

function vsub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function vadd(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function vscale(a: Vec3, k: number): Vec3 {
  return [a[0] * k, a[1] * k, a[2] * k];
}
function vnorm(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}
function vnormalize(a: Vec3): Vec3 {
  return vscale(a, 1 / vnorm(a));
}
function vdot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Brute-force Simpson quadrature of the model density along the ray,
 * normalized by σ√2π — the oracle every lane is held against. The
 * integration window is sized from the segment's own support (axial window
 * ± 8σ, radial support ± 8σ/sin(angle) around the closest approach), so
 * near-axial rays get the long window they need.
 */
function quadrature(rayO: Vec3, dRaw: Vec3, seg: VolumetricSegment, steps = 40001): number {
  const rn = vnorm(dRaw);
  const dHat = vscale(dRaw, 1 / rn);
  const axis = vnormalize(vsub(seg.b, seg.a));
  const L = vnorm(vsub(seg.b, seg.a));
  const dw = vdot(dHat, axis);
  const sinAngle = Math.sqrt(Math.max(1 - dw * dw, 1e-12));

  // Closest approach of the ray to the segment midpoint's axis.
  const M = vadd(seg.a, vscale(axis, 0.5 * L));
  const bm = vsub(rayO, M);
  const tC = -(vdot(bm, dHat) - vdot(bm, axis) * dw) / Math.max(1 - dw * dw, 1e-12);

  // Integration window = INTERSECTION of the radial support (±8σ/sin about
  // the closest approach) and the axial window's support in t (s spanning
  // [−8σ, L+8σ]); taking a union instead would spread the fixed sample
  // count over dead space and under-resolve near-axial rays.
  let t0 = tC - (8 * seg.sigma) / sinAngle;
  let t1 = tC + (8 * seg.sigma) / sinAngle;
  if (Math.abs(dw) > 1e-9) {
    const sC = vdot(vsub(vadd(rayO, vscale(dHat, tC)), seg.a), axis);
    const tAtS = (s: number) => tC + (s - sC) / dw;
    const ta = tAtS(-8 * seg.sigma);
    const tb = tAtS(L + 8 * seg.sigma);
    t0 = Math.max(t0, Math.min(ta, tb));
    t1 = Math.min(t1, Math.max(ta, tb));
  }
  if (t1 <= t0) return 0;
  const h = (t1 - t0) / (steps - 1);
  let sum = 0;
  for (let i = 0; i < steps; i++) {
    const p = vadd(rayO, vscale(dHat, t0 + i * h));
    const f = lineVolumetricModelDensity(p, seg);
    const wgt = i === 0 || i === steps - 1 ? 1 : i % 2 === 1 ? 4 : 2;
    sum += wgt * f;
  }
  return ((sum * h) / 3) * INV_NORM;
}

/**
 * Build the bisector-cut normal at an endpoint for a partner leaving the
 * shared vertex in direction `q` (unit). `intoMe` is the unit direction
 * from the shared vertex INTO this segment. n = normalize(q − intoMe), my
 * side negative — the same construction as the vertex stage.
 */
function bisectorNormal(intoMe: Vec3, q: Vec3): Vec3 {
  return vnormalize(vsub(q, intoMe));
}

/**
 * A joint fixture: main segment A→B along +x, partner leaving B at `bend`
 * degrees (0 = straight through). Returns the main segment with a hard cut
 * at B, plus the partner segment with the mirrored hard cut at its start.
 */
function jointAtB(
  L: number,
  bendDeg: number,
  partnerLen: number,
  softOuterEnds = true
): { main: VolumetricSegment; partner: VolumetricSegment } {
  const A: Vec3 = [0, 0, 0];
  const B: Vec3 = [L, 0, 0];
  const rad = (bendDeg * Math.PI) / 180;
  const q: Vec3 = [Math.cos(rad), Math.sin(rad), 0]; // partner direction from B
  const C = vadd(B, vscale(q, partnerLen));
  const nMain = bisectorNormal([-1, 0, 0], q); // intoMe at end B = −x
  const nPartner = bisectorNormal(q, [-1, 0, 0] as Vec3); // partner's intoMe = q
  return {
    main: { a: A, b: B, sigma: SIGMA, cutA: softOuterEnds ? null : undefined, cutB: nMain },
    partner: { a: B, b: C, sigma: SIGMA, cutA: nPartner, cutB: null },
  };
}

/** Rays: a fan of origins/directions that cross the interesting geometry. */
function obliqueRay(target: Vec3, tiltDeg: number, azimuthDeg = 30): { o: Vec3; d: Vec3 } {
  const tilt = (tiltDeg * Math.PI) / 180;
  const az = (azimuthDeg * Math.PI) / 180;
  // Direction tilted `tiltDeg` away from the segment axis (+x): 90 = side-on.
  const d: Vec3 = [Math.cos(tilt), Math.sin(tilt) * Math.cos(az), Math.sin(tilt) * Math.sin(az)];
  // Aim through the target from 50 units away.
  const o = vsub(target, vscale(d, 50));
  return { o, d };
}

describe('lineVolumetricSumIntegral vs quadrature', () => {
  it('normalization anchor: long soft segment, side-on, through the core reads 1', () => {
    const seg: VolumetricSegment = { a: [-500, 0, 0], b: [500, 0, 0], sigma: SIGMA };
    const I = lineVolumetricSumIntegral([0, 0, -50], [0, 0, 1], seg);
    expect(I).toBeCloseTo(1.0, 5);
  });

  it('soft/soft lane matches quadrature across obliquity and length', () => {
    for (const L of [0.5, 1, 3, 10, 40]) {
      const seg: VolumetricSegment = { a: [0, 0, 0], b: [L, 0, 0], sigma: SIGMA };
      for (const tilt of [90, 45, 10, 2, 0.5]) {
        for (const along of [0, 0.5, 1, 1.3]) {
          const target: Vec3 = [L * along, 0.7, 0];
          const { o, d } = obliqueRay(target, tilt);
          const lane = lineVolumetricSumIntegral(o, d, seg);
          const ref = quadrature(o, d, seg);
          expect(Math.abs(lane - ref), `soft L=${L} tilt=${tilt} along=${along}`).toBeLessThan(
            2.5e-3 * Math.max(ref, 0.05)
          );
        }
      }
    }
  });

  it('hard/hard lane (interior segment) matches quadrature at any bend', () => {
    for (const bend of [0, 30, 90, 150]) {
      for (const L of [3, 10]) {
        // Interior segment: hard cuts BOTH ends (straight partners fore and aft).
        const { main } = jointAtB(L, bend, 10);
        const nA = bisectorNormal([1, 0, 0], [-1, 0, 0] as Vec3); // straight partner at A
        const seg: VolumetricSegment = { ...main, cutA: nA };
        for (const tilt of [90, 45, 10]) {
          for (const along of [0.02, 0.5, 0.98]) {
            const target: Vec3 = [L * along, 0.4, 0];
            const { o, d } = obliqueRay(target, tilt);
            const lane = lineVolumetricSumIntegral(o, d, seg);
            const ref = quadrature(o, d, seg);
            expect(
              Math.abs(lane - ref),
              `hard bend=${bend} L=${L} tilt=${tilt} along=${along}`
            ).toBeLessThan(2.5e-3 * Math.max(ref, 0.05));
          }
        }
      }
    }
  });

  describe('mixed lane (one hard cut, one soft cap — chain-end segments)', () => {
    it('matches quadrature for L ≥ 3σ at bends up to 90° and every obliquity', () => {
      for (const bend of [0, 30, 90]) {
        for (const L of [3, 10]) {
          const { main } = jointAtB(L, bend, 10); // soft at A, hard at B
          for (const tilt of [90, 45, 10, 3]) {
            for (const along of [-0.1, 0.05, 0.5, 0.95, 1.05]) {
              const target: Vec3 = [L * along, 0.4, 0];
              const { o, d } = obliqueRay(target, tilt);
              const lane = lineVolumetricSumIntegral(o, d, main);
              const ref = quadrature(o, d, main);
              expect(
                Math.abs(lane - ref),
                `mixed bend=${bend} L=${L} tilt=${tilt} along=${along}`
              ).toBeLessThan(6e-3 * Math.max(ref, 0.05));
            }
          }
        }
      }
    });

    it('sharp bends (≥120°): measured absolute envelope', () => {
      // Past ~90° the bisector plane leans toward the axis, and the
      // inclusion–exclusion residual is governed by the PERPENDICULAR
      // distance from the soft cap's support to the plane — which a longer
      // segment does NOT grow once the plane is nearly axis-parallel. So
      // sharp chain-end bends carry a bounded, measured absolute error
      // concentrated around the free end (still several times better than
      // the spike's ignore-the-cut fallback, whose error at these probes
      // exceeds 0.5), and the envelope is pinned here rather than hidden.
      let worstAbs = 0;
      let worstAt = '';
      for (const bend of [120, 135, 150, 165]) {
        for (const L of [3, 10]) {
          const { main } = jointAtB(L, bend, 10);
          for (const tilt of [90, 45, 10, 3]) {
            for (const along of [-0.1, 0.05, 0.5, 0.95, 1.05]) {
              const { o, d } = obliqueRay([L * along, 0.4, 0], tilt);
              const lane = lineVolumetricSumIntegral(o, d, main);
              const ref = quadrature(o, d, main);
              const abs = Math.abs(lane - ref);
              if (abs > worstAbs) {
                worstAbs = abs;
                worstAt = `bend=${bend} L=${L} tilt=${tilt} along=${along} lane=${lane.toFixed(4)} ref=${ref.toFixed(4)}`;
              }
            }
          }
        }
      }
      console.log(`mixed sharp-bend worstAbs=${worstAbs.toFixed(5)} at ${worstAt}`);
      // Measured 2026-08: 0.212, at bend=165° L=3σ side-on just beyond the
      // free end (an underestimate — the tip dims, never brightens). A sign
      // slip or dropped term in the lane produces ≥ 0.5 here.
      expect(worstAbs, worstAt).toBeLessThan(0.25);
      expect(worstAbs).toBeGreaterThan(1e-4); // sensitivity control
    });

    it('short segments (L < 3σ): bounded underestimate, documented envelope', () => {
      // The dropped inclusion–exclusion term is the cap-complement's mass on
      // the excluded side of the plane — an UNDERestimate that grows as the
      // two end treatments overlap. Intensities are normalized (a side-on
      // core reads 1.0), so the visually meaningful envelope is the ABSOLUTE
      // error. Measure the worst over the adversarial grid and pin it.
      let worstAbs = 0;
      let worstAt = '';
      for (const bend of [0, 60, 120]) {
        for (const L of [0.5, 1, 2]) {
          const { main } = jointAtB(L, bend, 10);
          for (const tilt of [90, 45, 10]) {
            for (const along of [0.05, 0.5, 0.95]) {
              const target: Vec3 = [L * along, 0.3, 0];
              const { o, d } = obliqueRay(target, tilt);
              const lane = lineVolumetricSumIntegral(o, d, main);
              const ref = quadrature(o, d, main);
              const abs = Math.abs(lane - ref);
              if (abs > worstAbs) {
                worstAbs = abs;
                worstAt = `bend=${bend} L=${L} tilt=${tilt} along=${along} lane=${lane.toFixed(4)} ref=${ref.toFixed(4)}`;
              }
              // Never OVERestimates beyond lane/quadrature noise.
              expect(lane, `over bend=${bend} L=${L} tilt=${tilt}`).toBeLessThan(ref + 6e-3);
            }
          }
        }
      }
      console.log(`mixed short-segment worstAbs=${worstAbs.toFixed(5)} at ${worstAt}`);
      expect(worstAbs, worstAt).toBeLessThan(0.2);
      // Sensitivity control: the envelope is a real measurement, not slack.
      expect(worstAbs).toBeGreaterThan(1e-4);
    });

    it('MUTATION: dropping the inclusion–exclusion cap term fails at chain ends', () => {
      // Re-evaluate the L≥3σ grid with the cap term suppressed by moving the
      // soft cap infinitely far away (cutting the same plane but a segment
      // extended 60σ past B) — the spike's old any-soft fallback error shape.
      // If the mixed lane's cap term were dropped, rays near the FREE end
      // would see this value; assert it genuinely differs from quadrature so
      // the previous test has teeth.
      const L = 6;
      const { main } = jointAtB(L, 60, 10); // soft cap at A, hard cut at B
      // Probe just inside the SOFT end A, where the cap term carries the value.
      const { o, d } = obliqueRay([0.05 * L, 0.2, 0], 90);
      const withCap = lineVolumetricSumIntegral(o, d, main);
      const noCapSeg: VolumetricSegment = {
        ...main,
        a: vsub(main.a, [60 * SIGMA, 0, 0] as unknown as Vec3) as Vec3,
      };
      const noCap = lineVolumetricSumIntegral(o, d, noCapSeg);
      const ref = quadrature(o, d, main);
      expect(Math.abs(withCap - ref)).toBeLessThan(4e-3 * Math.max(ref, 0.05));
      expect(Math.abs(noCap - ref)).toBeGreaterThan(0.2 * ref);
    });
  });

  it('straight 2-segment chain with bisector cuts sums to the single rod (linearity)', () => {
    // A→B→C collinear: (soft A, hard B) + (hard B, soft C) must reproduce the
    // single soft A→C segment wherever the rays cross the joint region.
    const L1 = 6;
    const L2 = 8;
    const { main, partner } = jointAtB(L1, 0, L2);
    const whole: VolumetricSegment = { a: [0, 0, 0], b: [L1 + L2, 0, 0], sigma: SIGMA };
    for (const along of [0.5, 0.9, 1.0, 1.1, 1.5]) {
      for (const tilt of [90, 30]) {
        const { o, d } = obliqueRay([L1 * along, 0.3, 0], tilt);
        const sum =
          lineVolumetricSumIntegral(o, d, main) + lineVolumetricSumIntegral(o, d, partner);
        const ref = lineVolumetricSumIntegral(o, d, whole);
        expect(Math.abs(sum - ref), `chain along=${along} tilt=${tilt}`).toBeLessThan(
          5e-3 * Math.max(ref, 0.05)
        );
      }
    }
  });

  it('bent joint partitions seamlessly: me + partner ≈ quadrature of the pair', () => {
    for (const bend of [30, 90]) {
      const L = 8;
      const { main, partner } = jointAtB(L, bend, L);
      // Rays crossing right at the joint, where partition must be exact.
      for (const off of [-1, -0.3, 0, 0.3, 1]) {
        const { o, d } = obliqueRay([L + off, off * 0.4, 0.2], 80, 60);
        const sum =
          lineVolumetricSumIntegral(o, d, main) + lineVolumetricSumIntegral(o, d, partner);
        const ref = quadrature(o, d, main) + quadrature(o, d, partner);
        expect(Math.abs(sum - ref), `joint bend=${bend} off=${off}`).toBeLessThan(
          5e-3 * Math.max(ref, 0.05)
        );
      }
    }
  });

  describe('float32 + erfPoly (the GPU arithmetic) stays accurate across the parallel threshold', () => {
    it('mixed lane near-parallel: both sides of the structural threshold', () => {
      const L = 10;
      const { main } = jointAtB(L, 45, 10);
      // Sweep the ray tilt so A/n² crosses the threshold; the lane must not
      // blow up on either side (the general lane's subtraction is the
      // dangerous side — that is what the threshold exists for).
      for (const sin2 of [1e-7, 1e-6, 3e-6, 3e-5, 1e-4, 1e-3]) {
        const tilt = (Math.asin(Math.sqrt(sin2)) * 180) / Math.PI;
        const { o, d } = obliqueRay([L * 0.6, 0.5, 0], tilt);
        const gpu = lineVolumetricSumIntegral(o, d, main, { erf: erfPoly, f32: true });
        const ref = quadrature(o, d, main);
        expect(
          Math.abs(gpu - ref),
          `sin²=${sin2} (thr=${LINE_PARALLEL_LANE_THRESHOLD})`
        ).toBeLessThan(0.02 * Math.max(ref, 0.05));
      }
    });

    it('all lanes: erfPoly+f32 error stays within the polynomial budget', () => {
      const cases: Array<{ seg: VolumetricSegment; label: string }> = [];
      cases.push({ seg: { a: [0, 0, 0], b: [8, 0, 0], sigma: SIGMA }, label: 'soft' });
      const { main } = jointAtB(8, 60, 10);
      cases.push({ seg: main, label: 'mixed' });
      const nA = bisectorNormal([1, 0, 0], [-1, 0, 0] as Vec3);
      cases.push({ seg: { ...main, cutA: nA }, label: 'hardhard' });
      for (const { seg, label } of cases) {
        for (const tilt of [90, 20, 1]) {
          for (const along of [0.1, 0.6, 1.02]) {
            const { o, d } = obliqueRay([8 * along, 0.4, 0], tilt);
            const gpu = lineVolumetricSumIntegral(o, d, seg, { erf: erfPoly, f32: true });
            const f64 = lineVolumetricSumIntegral(o, d, seg);
            expect(Math.abs(gpu - f64), `${label} tilt=${tilt} along=${along}`).toBeLessThan(
              3e-3 * Math.max(f64, 0.05)
            );
          }
        }
      }
    });
  });

  it('end-on limit: near-axial ray through a soft segment approaches L·G(r)/(σ√2π)', () => {
    const L = 12;
    const seg: VolumetricSegment = { a: [0, 0, 0], b: [L, 0, 0], sigma: SIGMA };
    // Axial ray offset 0.8σ from the core.
    const I = lineVolumetricSumIntegral([-30, 0.8, 0], [1, 0, 0], seg);
    const expected = (Math.exp(-0.5 * 0.64) * L) / (SIGMA * Math.sqrt(2 * Math.PI));
    expect(I).toBeCloseTo(expected, 4);
    // And quadrature agrees.
    expect(quadrature([-30, 0.8, 0], [1, 0, 0], seg)).toBeCloseTo(expected, 3);
  });
});
