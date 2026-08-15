// @vitest-environment jsdom
/**
 * Unit tests for luxar-orbit-controls/math/pan.ts.
 *
 * Targets audit findings G6 (perspective vs ortho branches entirely
 * untested in isolation) and H4 (linearity / symmetry properties).
 *
 * The pan math mutates a caller-supplied accumulator, so all assertions
 * read the post-call accumulator vector.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  applyPan,
  applyPanLeft,
  applyPanUp,
  type PanCtx,
} from '../../../../../controls/luxar-orbit-controls/math/pan';

/** Build a stub HTMLElement with overridable clientWidth/clientHeight. */
function mockDomElement(width: number, height: number): HTMLElement {
  const el = document.createElement('div');
  Object.defineProperty(el, 'clientWidth', { configurable: true, get: () => width });
  Object.defineProperty(el, 'clientHeight', { configurable: true, get: () => height });
  return el;
}

describe('applyPanLeft', () => {
  it('accumulates -distance * matrix.col0 into out', () => {
    // matrix col0 = world X axis when matrix = identity.
    const out = new THREE.Vector3(1, 2, 3);
    const mat = new THREE.Matrix4(); // identity
    applyPanLeft(out, 5, mat);
    // out += -5 * (1,0,0)
    expect(out.x).toBeCloseTo(1 - 5, 10);
    expect(out.y).toBeCloseTo(2, 10);
    expect(out.z).toBeCloseTo(3, 10);
  });

  it('is linear in distance (H4)', () => {
    const out1 = new THREE.Vector3();
    const out2 = new THREE.Vector3();
    const mat = new THREE.Matrix4();
    applyPanLeft(out1, 3, mat);
    applyPanLeft(out2, 6, mat);
    expect(out2.x).toBeCloseTo(2 * out1.x, 10);
  });

  it('reads matrix column 0 (camera X axis) for non-identity matrices', () => {
    // Rotate matrix 90° around Y — col0 becomes (0, 0, -1).
    const mat = new THREE.Matrix4().makeRotationY(Math.PI / 2);
    const out = new THREE.Vector3();
    applyPanLeft(out, 2, mat);
    // out = -2 * (0, 0, -1) = (0, 0, 2)
    expect(out.x).toBeCloseTo(0, 5);
    expect(out.y).toBeCloseTo(0, 5);
    expect(out.z).toBeCloseTo(2, 5);
  });
});

describe('applyPanUp', () => {
  it('uses matrix col1 (Y axis) when screenSpacePanning is true', () => {
    const out = new THREE.Vector3();
    const mat = new THREE.Matrix4(); // identity → col1 = (0,1,0)
    const camUp = new THREE.Vector3(0, 1, 0);
    applyPanUp(out, 3, mat, camUp, true);
    expect(out.x).toBeCloseTo(0, 10);
    expect(out.y).toBeCloseTo(3, 10);
    expect(out.z).toBeCloseTo(0, 10);
  });

  it('uses cross(cameraUp, col0) when screenSpacePanning is false (world-up pan)', () => {
    // matrix = identity → col0 = (1,0,0).
    // cameraUp = (0,1,0) → cross((0,1,0), (1,0,0)) = (0,0,-1).
    // out += 5 * (0,0,-1) = (0,0,-5).
    const out = new THREE.Vector3();
    const mat = new THREE.Matrix4();
    const camUp = new THREE.Vector3(0, 1, 0);
    applyPanUp(out, 5, mat, camUp, false);
    expect(out.x).toBeCloseTo(0, 5);
    expect(out.y).toBeCloseTo(0, 5);
    expect(out.z).toBeCloseTo(-5, 5);
  });

  it('is linear in distance', () => {
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const mat = new THREE.Matrix4();
    const camUp = new THREE.Vector3(0, 1, 0);
    applyPanUp(a, 2, mat, camUp, true);
    applyPanUp(b, 6, mat, camUp, true);
    expect(b.y).toBeCloseTo(3 * a.y, 10);
  });
});

describe('applyPan — perspective camera branch', () => {
  it('combines left + up pan with the FOV-based scale formula', () => {
    // Formula:
    //   height = 2 * distance * tan(fov/2 * π/180)
    //   left_dist = (deltaX * height * panSpeed) / clientHeight
    //   up_dist   = (deltaY * height * panSpeed) / clientHeight
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();

    const ctx: PanCtx = {
      camera,
      distance: 5,
      panSpeed: 1,
      screenSpacePanning: true,
      domElement: mockDomElement(800, 600),
    };

    const fovRad = (60 * Math.PI) / 180;
    const height = 2 * 5 * Math.tan(fovRad / 2);
    const expectedX = -(100 * height * 1) / 600; // pan-left negates
    const expectedY = (50 * height * 1) / 600;

    const out = new THREE.Vector3();
    applyPan(out, 100, 50, ctx);

    expect(out.x).toBeCloseTo(expectedX, 5);
    expect(out.y).toBeCloseTo(expectedY, 5);
    expect(out.z).toBeCloseTo(0, 5);
  });

  it('pan + un-pan returns the accumulator to zero (H4 round-trip)', () => {
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const ctx: PanCtx = {
      camera,
      distance: 5,
      panSpeed: 1,
      screenSpacePanning: true,
      domElement: mockDomElement(800, 600),
    };
    const out = new THREE.Vector3();
    applyPan(out, 100, 50, ctx);
    applyPan(out, -100, -50, ctx);
    expect(out.x).toBeCloseTo(0, 5);
    expect(out.y).toBeCloseTo(0, 5);
    expect(out.z).toBeCloseTo(0, 5);
  });
});

describe('applyPan — orthographic camera branch', () => {
  it('uses (cam.right - cam.left) / clientWidth scale (ortho branch)', () => {
    const camera = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.zoom = 1;
    camera.updateMatrixWorld();

    const ctx: PanCtx = {
      camera,
      distance: 5,
      panSpeed: 1,
      screenSpacePanning: true,
      domElement: mockDomElement(800, 600),
    };

    // Formula for ortho:
    //   left_dist = (deltaX * (right - left) * panSpeed) / zoom / clientWidth
    //             = (100 * 20 * 1) / 1 / 800 = 2.5
    //   sign convention: pan-left negates → -2.5
    const out = new THREE.Vector3();
    applyPan(out, 100, 0, ctx);
    expect(out.x).toBeCloseTo(-2.5, 5);
  });

  it('camera.zoom scales the result inversely (zoom in → smaller world-space pan)', () => {
    const camera = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.zoom = 1;
    camera.updateMatrixWorld();

    const ctx: PanCtx = {
      camera,
      distance: 5,
      panSpeed: 1,
      screenSpacePanning: true,
      domElement: mockDomElement(800, 600),
    };

    const out1 = new THREE.Vector3();
    applyPan(out1, 100, 0, ctx);

    // Now zoom in 2x.
    camera.zoom = 2;
    const out2 = new THREE.Vector3();
    applyPan(out2, 100, 0, ctx);

    // Zoom doubles → pan distance halves.
    expect(out2.x).toBeCloseTo(out1.x / 2, 5);
  });
});

describe('linearity properties (H4)', () => {
  it('applyPan(2*d) == 2*applyPan(d) for perspective (linearity in deltas)', () => {
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const ctx: PanCtx = {
      camera,
      distance: 5,
      panSpeed: 1,
      screenSpacePanning: true,
      domElement: mockDomElement(800, 600),
    };
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    applyPan(a, 30, 20, ctx);
    applyPan(b, 60, 40, ctx);
    expect(b.x).toBeCloseTo(2 * a.x, 5);
    expect(b.y).toBeCloseTo(2 * a.y, 5);
    expect(b.z).toBeCloseTo(2 * a.z, 5);
  });
});
