// @vitest-environment jsdom
/**
 * The eager load (`ViewStateManager.initializeFromDimensions`, used by
 * `loadScene`) and the slider path (`SceneDimsManager.initFromScene` →
 * `simpleDimsToViewState`, used by the post-load `updateAllNDNodes` and every
 * slider event) must describe the SAME initial query, as judged by
 * `viewStatesEqual`. That equality is what lets `updateSceneForDimensions`
 * skip the post-load pass instead of re-streaming every node (the "startup
 * double pass": L0 hits == misses on every 3-D scene before the fix).
 *
 * The two builders used to disagree on two details, both pinned here: a
 * missing `step` (undefined vs `|| 1.0`) and the tolerance of a discrete
 * SPATIAL non-displayed axis (0 vs `maxRadius`).
 */
import * as THREE from 'three';
import { afterEach, describe, expect, it } from 'vitest';

import { simpleDimsToViewState } from '../../../data/dims-to-view-state';
import { viewStatesEqual } from '../../../data/loaders/progressive/view-state-equal';
import { ViewStateManager } from '../../../data/view-state-manager';
import { config } from '../../../config';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';

interface RawDim {
  name: string;
  unit: string;
  range: [number, number];
  display: boolean;
  discrete?: boolean;
  spatial?: boolean;
  step?: number;
  cyclic?: boolean;
}

function sceneWith(dimensions: RawDim[]): THREE.Scene {
  const scene = new THREE.Scene();
  scene.userData.sceneDimensions = { dimensions };
  return scene;
}

/** Both builders, driven from the same scene-dimensions block. */
function bothStates(dimensions: RawDim[]) {
  const eager = ViewStateManager.initializeFromDimensions({ dimensions } as never);
  const scene = sceneWith(dimensions);
  expect(sceneDimsManager.initFromScene(scene)).toBe(true);
  const dims = sceneDimsManager.getDims();
  expect(dims).not.toBeNull();
  const slider = simpleDimsToViewState(dims!, {
    maxRadius: config.dataLoading.spatial.defaultMaxRadius,
    defaultTolerance: config.dataLoading.spatial.defaultTolerance,
  });
  return { eager, slider };
}

const XYZ: RawDim[] = [
  { name: 'x', unit: 'um', range: [0, 100], display: true },
  { name: 'y', unit: 'um', range: [0, 100], display: true },
  { name: 'z', unit: 'um', range: [0, 50], display: true },
];

describe('eager load and slider view-state builders agree', () => {
  afterEach(() => sceneDimsManager.reset());

  it('3-D scene', () => {
    const { eager, slider } = bothStates(XYZ);
    expect(viewStatesEqual(slider, eager)).toBe(true);
  });

  it('4-D scene with a discrete time axis WITHOUT an authored step', () => {
    const { eager, slider } = bothStates([
      ...XYZ,
      { name: 'time', unit: 's', range: [0, 10], display: false, discrete: true },
    ]);
    expect(eager.dimensions?.[3].step).toBe(1.0);
    expect(viewStatesEqual(slider, eager)).toBe(true);
  });

  it('discrete SPATIAL non-displayed axis (a z-plane index) takes the same radius', () => {
    const { eager, slider } = bothStates([
      ...XYZ,
      { name: 'plane', unit: 'px', range: [0, 5], display: false, discrete: true, spatial: true },
    ]);
    expect(eager.tolerance[3]).toBe(slider.tolerance[3]);
    expect(viewStatesEqual(slider, eager)).toBe(true);
  });

  it('continuous non-displayed axis, and a cyclic one with a step', () => {
    const { eager, slider } = bothStates([
      ...XYZ,
      { name: 'c', unit: '', range: [0, 1], display: false },
      {
        name: 'phase',
        unit: 'rad',
        range: [0, 6],
        display: false,
        discrete: true,
        step: 0.5,
        cyclic: true,
      },
    ]);
    expect(viewStatesEqual(slider, eager)).toBe(true);
  });

  it('a moved slider still compares unequal (the skip never swallows a real change)', () => {
    const { eager, slider } = bothStates([
      ...XYZ,
      { name: 'time', unit: 's', range: [0, 10], display: false, discrete: true },
    ]);
    const moved = { ...slider, slicePosition: [...slider.slicePosition] };
    moved.slicePosition[3] = 3;
    expect(viewStatesEqual(moved, eager)).toBe(false);
  });
});
