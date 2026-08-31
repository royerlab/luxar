import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { AUTO_ROTATE_AXES, type AutoRotateAxis } from '../../../../../controls/types';
import { autoRotateAxisVector } from '../../../../../controls/luxar-orbit-controls/math/auto-rotate';

describe('autoRotateAxisVector', () => {
  /** A tilted orientation, so camera-frame axes differ from world axes. */
  const tilted = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.4, 0.9, -0.3));

  it('gives every token in the vocabulary its own unit axis', () => {
    // A token added to AUTO_ROTATE_AXES without a direction falls back to
    // vertical, which is silent — so assert the six are pairwise DISTINCT
    // (under a tilt, where no camera-frame axis coincides with a world one)
    // rather than merely non-throwing.
    const axes = AUTO_ROTATE_AXES.map((axis) =>
      autoRotateAxisVector(axis, tilted, new THREE.Vector3())
    );
    for (const v of axes) expect(v.length()).toBeCloseTo(1, 6);
    for (let i = 0; i < axes.length; i++) {
      for (let j = i + 1; j < axes.length; j++) {
        expect(axes[i].angleTo(axes[j])).toBeGreaterThan(0.05);
      }
    }
  });

  it.each([
    { axis: 'world-x' as const, expected: [1, 0, 0] },
    { axis: 'world-y' as const, expected: [0, 1, 0] },
    { axis: 'world-z' as const, expected: [0, 0, 1] },
  ])('$axis is a world CONSTANT — orientation cannot move it', ({ axis, expected }) => {
    // The whole point of the world family: the axis does not track the camera,
    // so the subject spins about its own axis instead of precessing. If these
    // were rotated by the orientation they would be camera-frame axes wearing
    // world names.
    expect(autoRotateAxisVector(axis, tilted, new THREE.Vector3()).toArray()).toEqual(expected);
    expect(
      autoRotateAxisVector(axis, new THREE.Quaternion(), new THREE.Vector3()).toArray()
    ).toEqual(expected);
  });

  it('world-y and vertical agree only while the camera is level', () => {
    const level = new THREE.Quaternion();
    expect(autoRotateAxisVector('vertical', level, new THREE.Vector3()).toArray()).toEqual(
      autoRotateAxisVector('world-y', level, new THREE.Vector3()).toArray()
    );
    // …and diverge once it has elevation, which is the reason both exist.
    const camUp = autoRotateAxisVector('vertical', tilted, new THREE.Vector3());
    const worldUp = autoRotateAxisVector('world-y', tilted, new THREE.Vector3());
    expect(camUp.angleTo(worldUp)).toBeGreaterThan(0.1);
  });

  it.each(['__proto__', 'constructor', 'toString'])(
    'falls back to vertical for inherited property name %s',
    (axis) => {
      const out = new THREE.Vector3();

      expect(() =>
        autoRotateAxisVector(axis as AutoRotateAxis, new THREE.Quaternion(), out)
      ).not.toThrow();
      expect(out.toArray()).toEqual([0, 1, 0]);
    }
  );
});
