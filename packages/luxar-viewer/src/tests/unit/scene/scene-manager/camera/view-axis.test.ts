/**
 * Axis-aligned opening views (`viewer_config.camera.view_axis`, `?viewAxis=`).
 *
 * The contract under test is what a person sees: for the image convention
 * (`up = (0, -1, 0)`, what `luxar gsplat convert` writes for a fitted z-stack)
 * the `xy` preset must put world +X on screen-right and world +Y on screen-DOWN,
 * looking along +Z — the Fiji/napari view — and must do it as a rotation, never
 * a mirror. The default up keeps the historical pose (+Y up, looking along -Z).
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { resolveViewAxisPose } from '../../../../../scene/scene-manager/camera/view-axis';
import { parseViewAxis } from '../../../../../types/view-axis';
import {
  boundsInViewFrame,
  centerOnOrigin,
  fitCameraToBounds,
} from '../../../../../scene/scene-manager/camera/camera-framing';
import type { ControlsManager } from '../../../../../controls/controls-manager';

const XYZ = ['x', 'y', 'z'];
const IMAGE_UP = new THREE.Vector3(0, -1, 0);
const DEFAULT_UP = new THREE.Vector3(0, 1, 0);

/** Components rounded, with -0 folded onto +0 (cross products produce -0). */
function clean(values: readonly number[]): number[] {
  return values.map((v) => Math.round(v * 1e9) / 1e9 + 0);
}

function makeControls(): ControlsManager {
  return {
    setSceneScale: vi.fn(),
    setDistanceLimits: vi.fn(),
    setZoomLimits: vi.fn(),
    setTarget: vi.fn(),
    reinitialize: vi.fn(),
    update: vi.fn(),
    saveState: vi.fn(),
    getFocusTarget: vi.fn(() => new THREE.Vector3()),
  } as unknown as ControlsManager;
}

/** Screen direction (NDC delta) a world displacement from the target maps to. */
function screenDirection(
  camera: THREE.PerspectiveCamera,
  target: THREE.Vector3,
  delta: THREE.Vector3
): { x: number; y: number } {
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();
  const a = target.clone().project(camera);
  const b = target.clone().add(delta).project(camera);
  const [x, y] = clean([
    Math.sign(Math.round((b.x - a.x) * 1e6)),
    Math.sign(Math.round((b.y - a.y) * 1e6)),
  ]);
  return { x, y };
}

function fit(
  pose: { forward: THREE.Vector3; up: THREE.Vector3 },
  target = new THREE.Vector3(1, 2, 3)
): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
  fitCameraToBounds(
    camera,
    makeControls(),
    { min: { x: -4, y: 0, z: 1 }, max: { x: 6, y: 4, z: 5 } },
    { lookAtTarget: target, up: pose.up, viewDirection: pose.forward }
  );
  return camera;
}

describe('parseViewAxis', () => {
  it('accepts the three presets case-insensitively', () => {
    expect(parseViewAxis('xy')).toBe('xy');
    expect(parseViewAxis(' XZ ')).toBe('xz');
    expect(parseViewAxis('yz')).toBe('yz');
  });

  it('rejects anything else', () => {
    expect(parseViewAxis('yx')).toBeNull();
    expect(parseViewAxis('')).toBeNull();
    expect(parseViewAxis(null)).toBeNull();
    expect(parseViewAxis(3)).toBeNull();
  });
});

describe('resolveViewAxisPose', () => {
  it('xy with the default up is the historical pose (look along -Z, +Y up)', () => {
    const pose = resolveViewAxisPose('xy', DEFAULT_UP, XYZ);
    expect(clean(pose?.forward.toArray() ?? [])).toEqual([0, 0, -1]);
    expect(clean(pose?.up.toArray() ?? [])).toEqual([0, 1, 0]);
  });

  it('xy with the image up looks along +Z with y down', () => {
    const pose = resolveViewAxisPose('xy', IMAGE_UP, XYZ);
    expect(clean(pose?.forward.toArray() ?? [])).toEqual([0, 0, 1]);
    expect(clean(pose?.up.toArray() ?? [])).toEqual([0, -1, 0]);
  });

  it('a ?viewAxis override orthogonal to the authored up keeps its convention', () => {
    // Authored for xy (up = -Y); the vertical of xz is Z, orthogonal to it.
    // The dominant component's sign (negative) carries the image convention.
    const pose = resolveViewAxisPose('xz', IMAGE_UP, XYZ);
    expect(clean(pose?.up.toArray() ?? [])).toEqual([0, 0, -1]);
    expect(clean(pose?.forward.toArray() ?? [])).toEqual([0, -1, 0]);
  });

  it('resolves letters by DIMENSION NAME, so column-ordered (z, y, x) dims face XY', () => {
    // A grafted partition keeps its columns: world X is z, world Z is x.
    const pose = resolveViewAxisPose('xy', IMAGE_UP, ['z', 'y', 'x']);
    // right must be world Z (the x dim), up world -Y: forward = up × right = -X.
    expect(clean(pose?.forward.toArray() ?? [])).toEqual([-1, 0, 0]);
    expect(clean(pose?.up.toArray() ?? [])).toEqual([0, -1, 0]);
  });

  it('falls back to world letters when names do not match', () => {
    const pose = resolveViewAxisPose('yz', DEFAULT_UP, ['dim0', 'dim1', 'dim2']);
    // yz: right = +Y, up = +Z ⇒ forward = Z × Y = -X.
    expect(clean(pose?.forward.toArray() ?? [])).toEqual([-1, 0, 0]);
    expect(clean(pose?.up.toArray() ?? [])).toEqual([0, 0, 1]);
  });

  it('returns null when two letters land on one axis', () => {
    expect(resolveViewAxisPose('xy', DEFAULT_UP, ['x', 'x2', 'y'])).not.toBeNull();
    // `x` is drawn along world Y, and the missing `y` falls back to world Y
    // too: no plane.
    expect(resolveViewAxisPose('xy', DEFAULT_UP, ['a', 'x', 'b'])).toBeNull();
  });
});

describe('fitCameraToBounds with an axis preset', () => {
  it('image xy: +X screen-right, +Y screen-DOWN, camera on the -Z side', () => {
    const pose = resolveViewAxisPose('xy', IMAGE_UP, XYZ);
    expect(pose).not.toBeNull();
    const target = new THREE.Vector3(1, 2, 3);
    const camera = fit(pose!, target);
    expect(camera.position.x).toBeCloseTo(1);
    expect(camera.position.y).toBeCloseTo(2);
    expect(camera.position.z).toBeLessThan(3);
    expect(screenDirection(camera, target, new THREE.Vector3(1, 0, 0))).toEqual({ x: 1, y: 0 });
    expect(screenDirection(camera, target, new THREE.Vector3(0, 1, 0))).toEqual({ x: 0, y: -1 });
  });

  it('image xz: +X screen-right, +Z screen-down', () => {
    const pose = resolveViewAxisPose('xz', IMAGE_UP, XYZ);
    const target = new THREE.Vector3(1, 2, 3);
    const camera = fit(pose!, target);
    expect(screenDirection(camera, target, new THREE.Vector3(1, 0, 0))).toEqual({ x: 1, y: 0 });
    expect(screenDirection(camera, target, new THREE.Vector3(0, 0, 1))).toEqual({ x: 0, y: -1 });
  });

  it('default yz: +Y screen-right, +Z screen-up', () => {
    const pose = resolveViewAxisPose('yz', DEFAULT_UP, XYZ);
    const target = new THREE.Vector3(0, 0, 0);
    const camera = fit(pose!, target);
    expect(screenDirection(camera, target, new THREE.Vector3(0, 1, 0))).toEqual({ x: 1, y: 0 });
    expect(screenDirection(camera, target, new THREE.Vector3(0, 0, 1))).toEqual({ x: 0, y: 1 });
  });

  it('fits the face the camera sees: same distance for xy as the historical fit', () => {
    const bounds = { min: { x: -4, y: 0, z: 1 }, max: { x: 6, y: 4, z: 5 } };
    const target = new THREE.Vector3(1, 2, 3);
    const legacy = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    fitCameraToBounds(legacy, makeControls(), bounds, { lookAtTarget: target });
    const preset = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    const pose = resolveViewAxisPose('xy', DEFAULT_UP, XYZ)!;
    fitCameraToBounds(preset, makeControls(), bounds, {
      lookAtTarget: target,
      up: pose.up,
      viewDirection: pose.forward,
    });
    expect(preset.position.distanceTo(target)).toBeCloseTo(legacy.position.distanceTo(target));
    expect(preset.position.toArray()).toEqual(legacy.position.toArray());
  });
});

describe('boundsInViewFrame', () => {
  it('permutes and negates the box into (right, up, back) about the target', () => {
    const pose = resolveViewAxisPose('xz', IMAGE_UP, XYZ)!; // right +X, up -Z, back +Y
    const box = boundsInViewFrame(
      { min: { x: 0, y: 10, z: 20 }, max: { x: 2, y: 14, z: 26 } },
      new THREE.Vector3(1, 12, 23),
      pose.forward,
      pose.up
    );
    expect(box.min.x).toBeCloseTo(-1);
    expect(box.max.x).toBeCloseTo(1);
    expect(box.min.y).toBeCloseTo(-3);
    expect(box.max.y).toBeCloseTo(3);
    expect(box.min.z).toBeCloseTo(-2);
    expect(box.max.z).toBeCloseTo(2);
  });
});

describe('centerOnOrigin with an axis preset', () => {
  it('backs away along -viewDirection and keeps the preset up', () => {
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    camera.position.set(0, 0, 10);
    const pose = resolveViewAxisPose('xy', IMAGE_UP, XYZ)!;
    centerOnOrigin(camera, makeControls(), pose.up, pose.forward);
    expect(clean(camera.position.toArray())).toEqual([0, 0, -10]);
    expect(clean(camera.up.toArray())).toEqual([0, -1, 0]);
  });
});
