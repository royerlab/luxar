import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import type { AutoRotateAxis } from '../../../../../controls/types';
import { autoRotateAxisVector } from '../../../../../controls/luxar-orbit-controls/math/auto-rotate';

describe('autoRotateAxisVector', () => {
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
