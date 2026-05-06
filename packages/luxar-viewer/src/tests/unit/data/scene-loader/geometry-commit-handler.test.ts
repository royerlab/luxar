/**
 * Unit tests for the points geometry-commit handler.
 *
 * Strategy: real `THREE.Group` + `THREE.Points` so the
 * getObjectByName lookup is exercised; mocked NodeFactory and
 * GPUBufferPool stubs to detect which code path ran.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import { commitPointsGeometry } from '../../../../data/scene-loader/geometry-commit-handler';
import type { LoadedPointsData } from '../../../../data/data-loader-types';
import type { NodeFactory } from '../../../../rendering/node-factory';

function makeData(pointCount: number, withRadii = false): LoadedPointsData {
  return {
    positions: new Float32Array(pointCount * 3),
    colors: new Uint8Array(pointCount * 3),
    radii: withRadii ? new Float32Array(pointCount) : undefined,
    sharpness: undefined,
    pointCount,
    metadata: {
      bounds: new THREE.Box3(
        new THREE.Vector3(-1, -1, -1),
        new THREE.Vector3(1, 1, 1)
      ),
    },
  } as unknown as LoadedPointsData;
}

function makePoints(name: string): THREE.Points {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
  const material = new THREE.PointsMaterial();
  const points = new THREE.Points(geometry, material);
  points.name = name;
  points.userData = { nodeType: 'points', visiblePointCount: 0 };
  return points;
}

const mockCreatePointsGeometry = vi.fn(() => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(9), 3));
  return g;
});
const mockNodeFactory = {
  createPointsGeometry: mockCreatePointsGeometry,
} as unknown as NodeFactory;

beforeEach(() => {
  mockCreatePointsGeometry.mockClear();
});

describe('commitPointsGeometry', () => {
  it('no-ops when rootGroup is null', () => {
    expect(() =>
      commitPointsGeometry('/p', makeData(5), null, null, mockNodeFactory)
    ).not.toThrow();
  });

  it('no-ops when no points node with the given path exists', () => {
    expect(() =>
      commitPointsGeometry('/missing', makeData(5), new THREE.Group(), null, mockNodeFactory)
    ).not.toThrow();
    expect(mockCreatePointsGeometry).not.toHaveBeenCalled();
  });

  it('updates visiblePointCount on the userData (pool disabled, pointCount=0)', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    commitPointsGeometry('/p', makeData(0), root, null, mockNodeFactory);
    expect(
      (points.userData as { visiblePointCount: number }).visiblePointCount
    ).toBe(0);
  });

  it('uses GPU buffer pool when supplied', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);

    const newGeometry = new THREE.BufferGeometry();
    const gpuBufferPool = {
      acquirePointsGeometry: vi.fn(() => newGeometry),
      updatePointsGeometry: vi.fn(),
    };

    commitPointsGeometry(
      '/p',
      makeData(3),
      root,
      gpuBufferPool as never,
      mockNodeFactory
    );
    expect(gpuBufferPool.acquirePointsGeometry).toHaveBeenCalledTimes(1);
    expect(gpuBufferPool.updatePointsGeometry).toHaveBeenCalledTimes(1);
    expect(points.geometry).toBe(newGeometry);
    expect(mockCreatePointsGeometry).not.toHaveBeenCalled();
  });

  it('falls back to dispose+create when pool disabled and counts differ', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const oldGeom = points.geometry;
    const disposeSpy = vi.spyOn(oldGeom, 'dispose');

    // Existing geometry has 0 positions; new data has 3 → counts differ.
    commitPointsGeometry('/p', makeData(3), root, null, mockNodeFactory);
    expect(disposeSpy).toHaveBeenCalledTimes(1);
    expect(mockCreatePointsGeometry).toHaveBeenCalledTimes(1);
  });

  it('updates attributes in place when pool disabled and counts match', () => {
    const root = new THREE.Group();
    const points = new THREE.Points();
    points.name = '/p';
    points.userData = { nodeType: 'points', visiblePointCount: 0 };
    // 3-point geometry to match the 3-point data.
    const geom = new THREE.BufferGeometry();
    geom.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(9), 3)
    );
    points.geometry = geom;
    root.add(points);

    commitPointsGeometry('/p', makeData(3), root, null, mockNodeFactory);
    // No dispose / no recreate → in-place path.
    expect(mockCreatePointsGeometry).not.toHaveBeenCalled();
    // The same geometry instance is preserved.
    expect(points.geometry).toBe(geom);
    expect(
      (points.userData as { visiblePointCount: number }).visiblePointCount
    ).toBe(3);
  });
});
