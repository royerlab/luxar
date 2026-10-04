/**
 * The projected-footprint pick must not override the occupancy metric's
 * camera-inside saturation.
 *
 * With the eye inside a node's box the occupancy metric is `+Infinity`, i.e.
 * the finest level (`lod-selector-math.ts`, scene/README step 4). The footprint
 * rule projects each level's median splat at the box-CENTRE depth, which says
 * nothing about the splats at the eye: honouring it there would show a coarse
 * level with its merged splats filling the view.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import {
  LODGroupRegistry,
  type LODGroupChild,
  type LODGroupEntry,
} from '../../../scene/lod-group-registry';

function selectedLevel(bounds: { min: number[]; max: number[] }): number {
  const camera = new THREE.PerspectiveCamera(60, 4 / 3, 0.1, 100);
  camera.updateMatrixWorld(true);
  const reg = new LODGroupRegistry({
    getCamera: () => camera,
    getViewportSize: () => ({ width: 800, height: 600 }),
    getDisplayDims: () => [0, 1, 2],
  });
  // Coarse → fine; at the box centre's depth (7) the middle level's median
  // splat projects to ~1.5 px, so the footprint rule alone picks level 1.
  const children: LODGroupChild[] = [0.1, 0.02, 0.005].map((medianFootprint, index) => ({
    object: new THREE.Group(),
    coverageFraction: [0, 0.25, 0.5][index],
    positionBounds: bounds,
    medianFootprint,
    footprintDims: [0, 1, 2],
  }));
  const entry: LODGroupEntry = {
    path: '/inside',
    groupObject: new THREE.Group(),
    children,
    selectorMode: 'auto',
    defaultLevel: 0,
    activeChildIndex: 0,
    selector: 'screen-area',
  };
  reg.register(entry);
  reg.evaluatePerFrame();
  return entry.activeChildIndex;
}

describe('LODGroupRegistry — footprint pick with the camera inside the box', () => {
  it('shows the finest level when the eye is inside a stamped node', () => {
    expect(selectedLevel({ min: [-10, -10, -15], max: [10, 10, 1] })).toBe(2);
  });

  it('SENSITIVITY: the same stamps decide from outside the box', () => {
    // Same centre depth, eye outside: the footprint rule is in charge.
    expect(selectedLevel({ min: [-0.1, -0.1, -7.1], max: [0.1, 0.1, -6.9] })).toBe(1);
  });
});
