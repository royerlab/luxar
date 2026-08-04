/**
 * Unit tests for `updateVisibleCountsInMonitor` — verifies the three
 * geometry types (points / lines / gsplats) are aggregated symmetrically
 * from per-mesh userData and pushed to the monitor port.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { updateVisibleCountsInMonitor } from '../../../../../data/scene-loader/monitor/visible-counts';
import type { SceneLoaderMonitorPort } from '../../../../../data/scene-loader-monitor-port';

function makeMonitor() {
  return {
    updateVisibleCount: vi.fn(),
    updateVisibleCountsByPath: vi.fn(),
  } as unknown as SceneLoaderMonitorPort & {
    updateVisibleCount: ReturnType<typeof vi.fn>;
    updateVisibleCountsByPath: ReturnType<typeof vi.fn>;
  };
}

function meshWith(userData: Record<string, unknown>, name?: string): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.userData = userData;
  if (name) mesh.name = name;
  return mesh;
}

describe('updateVisibleCountsInMonitor', () => {
  it('no-ops when rootGroup or monitor is null', () => {
    const monitor = makeMonitor();
    updateVisibleCountsInMonitor(null, monitor);
    expect(monitor.updateVisibleCount).not.toHaveBeenCalled();

    const group = new THREE.Group();
    updateVisibleCountsInMonitor(group, null);
    // (nothing to assert — just must not throw)
  });

  it('aggregates all three geometry types symmetrically', () => {
    const monitor = makeMonitor();
    const root = new THREE.Group();

    root.add(meshWith({ nodeType: 'points', visiblePointCount: 120 }));
    root.add(meshWith({ nodeType: 'points', visiblePointCount: 80 }));
    root.add(meshWith({ nodeType: 'lines', visibleSegmentCount: 30 }));
    root.add(meshWith({ nodeType: 'gsplats', visibleSplatCount: 1000 }));

    updateVisibleCountsInMonitor(root, monitor);

    expect(monitor.updateVisibleCount).toHaveBeenCalledWith('points', 200);
    expect(monitor.updateVisibleCount).toHaveBeenCalledWith('lines', 30);
    expect(monitor.updateVisibleCount).toHaveBeenCalledWith('gsplats', 1000);
  });

  it('treats missing visible-count userData as zero', () => {
    const monitor = makeMonitor();
    const root = new THREE.Group();
    // Points mesh with no visiblePointCount field.
    root.add(meshWith({ nodeType: 'points' }));

    updateVisibleCountsInMonitor(root, monitor);

    expect(monitor.updateVisibleCount).toHaveBeenCalledWith('points', 0);
    expect(monitor.updateVisibleCount).toHaveBeenCalledWith('lines', 0);
    expect(monitor.updateVisibleCount).toHaveBeenCalledWith('gsplats', 0);
  });

  it('excludes hidden meshes (inactive substitutive-LOD levels)', () => {
    // A kind=lod group loads & commits ALL levels (each gets its
    // visibleSplatCount) but hides all but the active one. Only the
    // visible level must contribute — summing hidden levels would inflate
    // the count ~K×.
    const monitor = makeMonitor();
    const root = new THREE.Group();

    const lodGroup = new THREE.Group();
    lodGroup.userData.kind = 'lod';
    const active = meshWith({ nodeType: 'gsplats', visibleSplatCount: 5000 });
    const hidden1 = meshWith({ nodeType: 'gsplats', visibleSplatCount: 1000 });
    hidden1.visible = false;
    const hidden2 = meshWith({ nodeType: 'gsplats', visibleSplatCount: 2000 });
    hidden2.visible = false;
    lodGroup.add(active, hidden1, hidden2);
    root.add(lodGroup);

    updateVisibleCountsInMonitor(root, monitor);

    // Only the visible level's 5000 — NOT 5000 + 1000 + 2000 = 8000.
    expect(monitor.updateVisibleCount).toHaveBeenCalledWith('gsplats', 5000);
  });

  it('prunes whole hidden subtrees (toggled-off layer group)', () => {
    const monitor = makeMonitor();
    const root = new THREE.Group();

    const hiddenLayer = new THREE.Group();
    hiddenLayer.visible = false; // user toggled the layer off
    hiddenLayer.add(meshWith({ nodeType: 'points', visiblePointCount: 999 }));
    root.add(hiddenLayer);
    root.add(meshWith({ nodeType: 'points', visiblePointCount: 42 }));

    updateVisibleCountsInMonitor(root, monitor);

    expect(monitor.updateVisibleCount).toHaveBeenCalledWith('points', 42);
  });

  it('pushes a per-path breakdown keyed by mesh name (scene-graph path)', () => {
    const monitor = makeMonitor();
    const root = new THREE.Group();

    root.add(meshWith({ nodeType: 'points', visiblePointCount: 120 }, '/pts'));
    root.add(meshWith({ nodeType: 'gsplats', visibleSplatCount: 1000 }, '/splats'));
    const hidden = meshWith({ nodeType: 'gsplats', visibleSplatCount: 500 }, '/hidden');
    hidden.visible = false;
    root.add(hidden);
    // Unnamed mesh: contributes to totals but not to the per-path map.
    root.add(meshWith({ nodeType: 'points', visiblePointCount: 5 }));

    updateVisibleCountsInMonitor(root, monitor);

    expect(monitor.updateVisibleCountsByPath).toHaveBeenCalledTimes(1);
    const map = monitor.updateVisibleCountsByPath.mock.calls[0][0] as Map<string, number>;
    expect(map.get('/pts')).toBe(120);
    expect(map.get('/splats')).toBe(1000);
    expect(map.has('/hidden')).toBe(false);
    expect(map.size).toBe(2);
    expect(monitor.updateVisibleCount).toHaveBeenCalledWith('points', 125);
  });

  it('dispatches on nodeType alone — extra count fields cannot double-count', () => {
    // The reader table is a first-match loop over GEOMETRY_TYPES. All three guards
    // test `userData.nodeType === <literal>`, so at most one can ever match and the
    // iteration order is irrelevant. A mesh carrying every count field must still
    // tally only to the type its `nodeType` names.
    const monitor = makeMonitor();
    const root = new THREE.Group();
    root.add(
      meshWith({
        nodeType: 'lines',
        visiblePointCount: 111,
        visibleSegmentCount: 222,
        visibleSplatCount: 333,
      })
    );

    updateVisibleCountsInMonitor(root, monitor);

    expect(Object.fromEntries(monitor.updateVisibleCount.mock.calls)).toEqual({
      points: 0,
      lines: 222,
      gsplats: 0,
      mesh: 0,
    });
  });

  it('reports every geometry type exactly once per invocation', () => {
    const monitor = makeMonitor();
    const root = new THREE.Group();
    root.add(meshWith({ nodeType: 'points', visiblePointCount: 5 }));
    root.add(meshWith({ nodeType: 'points', visiblePointCount: 6 }));

    updateVisibleCountsInMonitor(root, monitor);

    // One call per type — a type with no meshes must still be reported as 0 so a
    // previous scene's count cannot linger in the HUD.
    expect(monitor.updateVisibleCount).toHaveBeenCalledTimes(4);
    expect(Object.fromEntries(monitor.updateVisibleCount.mock.calls)).toEqual({
      points: 11,
      lines: 0,
      gsplats: 0,
      mesh: 0,
    });
  });

  it('ignores a nodeType outside the geometry vocabulary', () => {
    // No reader matches, so `visible` stays undefined: the mesh contributes to no
    // total and gets no per-path entry (rather than a spurious 0).
    const monitor = makeMonitor();
    const root = new THREE.Group();
    root.add(meshWith({ nodeType: 'volume', visibleVoxelCount: 9 }, '/vol'));
    root.add(meshWith({ nodeType: 'points', visiblePointCount: 4 }, '/pts'));

    updateVisibleCountsInMonitor(root, monitor);

    expect(Object.fromEntries(monitor.updateVisibleCount.mock.calls)).toEqual({
      points: 4,
      lines: 0,
      gsplats: 0,
      mesh: 0,
    });
    const map = monitor.updateVisibleCountsByPath.mock.calls[0][0] as Map<string, number>;
    expect(map.has('/vol')).toBe(false);
    expect(map.get('/pts')).toBe(4);
  });
});
