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
    updateVisiblePoints: vi.fn(),
    updateVisibleSegments: vi.fn(),
    updateVisibleSplats: vi.fn(),
  } as unknown as SceneLoaderMonitorPort & {
    updateVisiblePoints: ReturnType<typeof vi.fn>;
    updateVisibleSegments: ReturnType<typeof vi.fn>;
    updateVisibleSplats: ReturnType<typeof vi.fn>;
  };
}

function meshWith(userData: Record<string, unknown>): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.userData = userData;
  return mesh;
}

describe('updateVisibleCountsInMonitor', () => {
  it('no-ops when rootGroup or monitor is null', () => {
    const monitor = makeMonitor();
    updateVisibleCountsInMonitor(null, monitor);
    expect(monitor.updateVisiblePoints).not.toHaveBeenCalled();

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

    expect(monitor.updateVisiblePoints).toHaveBeenCalledWith(200);
    expect(monitor.updateVisibleSegments).toHaveBeenCalledWith(30);
    expect(monitor.updateVisibleSplats).toHaveBeenCalledWith(1000);
  });

  it('treats missing visible-count userData as zero', () => {
    const monitor = makeMonitor();
    const root = new THREE.Group();
    // Points mesh with no visiblePointCount field.
    root.add(meshWith({ nodeType: 'points' }));

    updateVisibleCountsInMonitor(root, monitor);

    expect(monitor.updateVisiblePoints).toHaveBeenCalledWith(0);
    expect(monitor.updateVisibleSegments).toHaveBeenCalledWith(0);
    expect(monitor.updateVisibleSplats).toHaveBeenCalledWith(0);
  });
});
