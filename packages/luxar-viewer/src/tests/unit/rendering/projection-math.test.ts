import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import {
  focalLengthFromProjection,
  gsplatJacobian,
  isOrthoProjection,
  lineScaleFromProjection,
  pointSizeFactorFromProjection,
  projectCenterPx,
} from '../../../rendering/materials/_shared/projection-math';

const RES: [number, number] = [1280, 720];

function perspective(fovDeg: number, aspect = RES[0] / RES[1], zoom = 1): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera(fovDeg, aspect, 0.1, 1000);
  cam.zoom = zoom;
  cam.updateProjectionMatrix();
  return cam;
}

function ortho(frustumHeight: number, zoom = 1): THREE.OrthographicCamera {
  const halfH = frustumHeight / 2;
  const halfW = (halfH * RES[0]) / RES[1];
  const cam = new THREE.OrthographicCamera(-halfW, halfW, halfH, -halfH, 0.1, 1000);
  cam.zoom = zoom;
  cam.updateProjectionMatrix();
  return cam;
}

/**
 * The historical point-size factor, as the removed CPU helper
 * `computePointSizeFactor` computed it (`fov` carried the frustum height in ortho).
 */
function legacyPointSizeFactor(fov: number, resY: number, isOrtho: boolean): number {
  if (isOrtho) {
    const halfFrustum = fov * 0.5;
    return (2.0 * resY) / halfFrustum;
  }
  return (2.0 * resY) / Math.tan(fov / 2);
}

/**
 * The historical splat focal length, as the removed CPU helper
 * `computeFocalLength` computed it (`fov` carried the frustum height in ortho).
 */
function legacyFocalLength(fov: number, resY: number, isOrtho: boolean): number {
  if (isOrtho) {
    return resY / fov;
  }
  return resY / (2 * Math.tan(fov / 2));
}

/** The historical line pixel-width scale, as `line/material-glsl.ts` computed it. */
function legacyLineScale(fov: number, resY: number, isOrtho: boolean): number {
  const safeFov = Math.max(fov, 1e-4);
  return isOrtho ? (2.0 * resY) / safeFov : resY / Math.max(Math.tan(safeFov * 0.5), 1e-4);
}

/** Central finite difference of `projectCenterPx` along view-space axis `k`. */
function numericColumn(
  P: ArrayLike<number>,
  c: [number, number, number],
  k: number
): [number, number] {
  const h = 1e-4 * Math.max(1, Math.abs(c[k]));
  const plus: [number, number, number] = [...c];
  const minus: [number, number, number] = [...c];
  plus[k] += h;
  minus[k] -= h;
  const a = projectCenterPx(P, plus, RES);
  const b = projectCenterPx(P, minus, RES);
  return [(a[0] - b[0]) / (2 * h), (a[1] - b[1]) / (2 * h)];
}

const VIEW_POINTS: [number, number, number][] = [
  [0, 0, -10],
  [3.5, -2, -25],
  [-7, 4.2, -60],
];

describe('projection-math — agreement with the historical fov helpers', () => {
  for (const fovDeg of [5, 30, 50, 75, 110, 150]) {
    it(`perspective fov ${fovDeg}°`, () => {
      const P = perspective(fovDeg).projectionMatrix.elements;
      const fov = THREE.MathUtils.degToRad(fovDeg);
      expect(isOrthoProjection(P)).toBe(false);
      expect(focalLengthFromProjection(P, RES[1])).toBeCloseTo(
        legacyFocalLength(fov, RES[1], false),
        8
      );
      expect(pointSizeFactorFromProjection(P, RES[1])).toBeCloseTo(
        legacyPointSizeFactor(fov, RES[1], false),
        6
      );
      expect(lineScaleFromProjection(P, RES[1])).toBeCloseTo(
        legacyLineScale(fov, RES[1], false),
        8
      );
    });
  }

  for (const [h, zoom] of [
    [2, 1],
    [40, 1],
    [40, 2.5],
    [0.001, 1],
  ] as const) {
    it(`orthographic frustum height ${h}, zoom ${zoom}`, () => {
      const cam = ortho(h, zoom);
      const P = cam.projectionMatrix.elements;
      const effective = (cam.top - cam.bottom) / cam.zoom; // getOrthoFrustumHeight
      expect(isOrthoProjection(P)).toBe(true);
      const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);
      expect(
        rel(focalLengthFromProjection(P, RES[1]), legacyFocalLength(effective, RES[1], true))
      ).toBeLessThan(1e-12);
      expect(
        rel(
          pointSizeFactorFromProjection(P, RES[1]),
          legacyPointSizeFactor(effective, RES[1], true)
        )
      ).toBeLessThan(1e-12);
      expect(
        rel(lineScaleFromProjection(P, RES[1]), legacyLineScale(effective, RES[1], true))
      ).toBeLessThan(1e-12);
    });
  }

  it('honours perspective zoom, which the fov helpers ignore', () => {
    const P = perspective(50, RES[0] / RES[1], 2).projectionMatrix.elements;
    const unzoomed = legacyFocalLength(THREE.MathUtils.degToRad(50), RES[1], false);
    expect(focalLengthFromProjection(P, RES[1])).toBeCloseTo(2 * unzoomed, 8);
  });

  it('is free of the historical 1e-4 ortho clamp on nanometre-scale frusta', () => {
    const cam = ortho(2e-5);
    const P = cam.projectionMatrix.elements;
    // Exact: 2/h pixels per unit, times resY.
    expect(lineScaleFromProjection(P, RES[1])).toBeCloseTo((2 / 2e-5) * RES[1], 2);
    // The legacy formula clamped h to 1e-4, a 5x narrower line.
    expect(legacyLineScale(2e-5, RES[1], true)).toBeCloseTo((2 / 1e-4) * RES[1], 2);
  });
});

describe('projection-math — gsplat centre and Jacobian', () => {
  it('reproduces the historical symmetric-perspective Jacobian', () => {
    const P = perspective(50).projectionMatrix.elements;
    const fx = legacyFocalLength(THREE.MathUtils.degToRad(50), RES[1], false);
    for (const c of VIEW_POINTS) {
      const z = -c[2];
      const J = gsplatJacobian(P, c, RES);
      expect(J[0][0]).toBeCloseTo(fx / z, 8);
      expect(J[0][1]).toBeCloseTo(0, 12);
      expect(J[1][0]).toBeCloseTo(0, 12);
      expect(J[1][1]).toBeCloseTo(fx / z, 8);
      expect(J[2][0]).toBeCloseTo((fx * c[0]) / (z * z), 8);
      expect(J[2][1]).toBeCloseTo((fx * c[1]) / (z * z), 8);
      // And the historical centre: fx * x / z + res / 2.
      const px = projectCenterPx(P, c, RES);
      expect(px[0]).toBeCloseTo((fx * c[0]) / z + RES[0] / 2, 6);
      expect(px[1]).toBeCloseTo((fx * c[1]) / z + RES[1] / 2, 6);
    }
  });

  it('reproduces the historical orthographic Jacobian', () => {
    const cam = ortho(30);
    const P = cam.projectionMatrix.elements;
    const fx = legacyFocalLength(30, RES[1], true);
    const J = gsplatJacobian(P, [2, -3, -40], RES);
    expect(J[0][0]).toBeCloseTo(fx, 8);
    expect(J[1][1]).toBeCloseTo(fx, 8);
    expect(J[2][0]).toBeCloseTo(0, 12);
    expect(J[2][1]).toBeCloseTo(0, 12);
  });

  const cases: [string, () => THREE.Camera][] = [
    ['symmetric perspective', () => perspective(60)],
    [
      'asymmetric (setViewOffset) perspective',
      () => {
        const cam = perspective(60);
        cam.setViewOffset(RES[0] * 2, RES[1], RES[0], 0, RES[0], RES[1]);
        cam.updateProjectionMatrix();
        return cam;
      },
    ],
    ['zoomed perspective', () => perspective(40, RES[0] / RES[1], 1.7)],
    ['orthographic', () => ortho(25, 1.3)],
    [
      'CubeCamera face (fov -90)',
      () => {
        const target = new THREE.WebGLCubeRenderTarget(16);
        const cube = new THREE.CubeCamera(0.1, 100, target);
        return cube.children[0] as THREE.PerspectiveCamera;
      },
    ],
  ];

  for (const [name, make] of cases) {
    it(`matches finite differences: ${name}`, () => {
      const P = (make() as THREE.PerspectiveCamera).projectionMatrix.elements;
      for (const c of VIEW_POINTS) {
        const J = gsplatJacobian(P, c, RES);
        for (let k = 0; k < 3; k++) {
          const n = numericColumn(P, c, k);
          const scale = Math.max(1, Math.abs(J[k][0]), Math.abs(J[k][1]));
          expect(Math.abs(J[k][0] - n[0]) / scale).toBeLessThan(1e-5);
          expect(Math.abs(J[k][1] - n[1]) / scale).toBeLessThan(1e-5);
        }
      }
    });
  }

  it('flips positions but not sizes under a CubeCamera projection', () => {
    const target = new THREE.WebGLCubeRenderTarget(16);
    const face = new THREE.CubeCamera(0.1, 100, target).children[0] as THREE.PerspectiveCamera;
    const mirror = perspective(90, 1);
    const Pf = face.projectionMatrix.elements;
    const Pm = mirror.projectionMatrix.elements;
    const res: [number, number] = [256, 256];
    // Sizes agree with a normal +90° camera ...
    expect(focalLengthFromProjection(Pf, 256)).toBeCloseTo(focalLengthFromProjection(Pm, 256), 8);
    expect(focalLengthFromProjection(Pf, 256)).toBeGreaterThan(0);
    // ... while positions are point-reflected through the face centre.
    const c: [number, number, number] = [1.5, -0.7, -4];
    const a = projectCenterPx(Pf, c, res);
    const b = projectCenterPx(Pm, c, res);
    expect(a[0]).toBeCloseTo(res[0] - b[0], 6);
    expect(a[1]).toBeCloseTo(res[1] - b[1], 6);
  });
});
