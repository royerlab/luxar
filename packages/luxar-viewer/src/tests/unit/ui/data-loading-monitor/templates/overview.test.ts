/**
 * Unit tests for Overview-tab templates.
 */

import { describe, it, expect } from 'vitest';
import {
  renderLoaderItem,
  renderOverviewContent,
  renderSecondaryMetrics,
} from '../../../../../ui/data-loading-monitor/templates/overview';
import type { CacheMetrics, GlobalStats } from '../../../../../types/data-monitor-types';

describe('renderSecondaryMetrics — requests served', () => {
  const memory = { used: 0, limit: 1000 };
  const querySpeed = { avgTime: 0, perSec: 0 };

  it('surfaces totalRequestsServed in the network detail when present', () => {
    const html = renderSecondaryMetrics(memory, querySpeed, {
      bytesTransferred: 100,
      requestCount: 7,
      bandwidth: 50,
      totalBytesServed: 200,
      totalRequestsServed: 42,
    });
    expect(html).toContain('42 reqs');
  });

  it('falls back to requestCount when totalRequestsServed is undefined', () => {
    const html = renderSecondaryMetrics(memory, querySpeed, {
      bytesTransferred: 100,
      requestCount: 7,
      bandwidth: 50,
    });
    expect(html).toContain('7 reqs');
  });
});

// ---------------------------------------------------------------------------
// Hero cards — one per geometry type present, mesh included
//
// The panel used to hard-code three types here, so a scene's mesh triangles
// were never shown (and a mesh-ONLY scene showed a permanent "LOADING …"
// card) even though `visibleByType.mesh` was aggregated every tick.
// ---------------------------------------------------------------------------

/** A `GlobalStats` with every count zero, so each test names only what it needs. */
function statsWith(over: Partial<GlobalStats> = {}): GlobalStats {
  return {
    totalLoaders: 0,
    activeSpatialLoaders: 0,
    totalElementsLoaded: 0,
    totalMemory: 0,
    datasetSize: 0,
    visiblePoints: 0,
    datasetSegments: 0,
    visibleSegments: 0,
    datasetSplats: 0,
    visibleSplats: 0,
    datasetTriangles: 0,
    visibleTriangles: 0,
    totalQueries: 0,
    totalLoads: 0,
    avgQueryTime: 0,
    queriesPerSecond: 0,
    recommendations: [],
    ...over,
  };
}

const NO_CACHE = { totalCacheMemory: 0, memoryLimit: 100 } as CacheMetrics;

describe('renderOverviewContent — per-type hero cards', () => {
  it('shows a mesh card for a mesh-only scene instead of "LOADING"', () => {
    const html = renderOverviewContent(
      statsWith({ datasetTriangles: 1000, visibleTriangles: 250 }),
      NO_CACHE
    );
    expect(html).toContain('VISIBLE TRIANGLES');
    expect(html).toContain('data-field="visible-triangles"');
    expect(html).toContain('25.0% of 1.0K total');
    expect(html).not.toContain('>LOADING<');
    expect(html).not.toContain('Waiting for data');
  });

  it('shows all four types together, each with its own field and label', () => {
    const html = renderOverviewContent(
      statsWith({
        datasetSize: 10,
        visiblePoints: 1,
        datasetSegments: 20,
        visibleSegments: 2,
        datasetSplats: 30,
        visibleSplats: 3,
        datasetTriangles: 40,
        visibleTriangles: 4,
      }),
      NO_CACHE
    );
    for (const label of [
      'VISIBLE POINTS',
      'VISIBLE LINES',
      'VISIBLE SPLATS',
      'VISIBLE TRIANGLES',
    ]) {
      expect(html).toContain(label);
    }
    for (const field of [
      'visible-points',
      'visible-lines',
      'visible-splats',
      'visible-triangles',
    ]) {
      expect(html).toContain(`data-field="${field}"`);
    }
    // The wrapping grid, not a fixed column count: `--cols-3` was emitted for
    // years with no CSS rule behind it, so three cards silently stacked.
    expect(html).toContain('luxar-overview-grid--auto');
    expect(html).not.toContain('luxar-overview-grid--cols-3');
    expect(html).not.toContain('luxar-overview-grid--cols-4');
  });

  it('leaves out the types a scene does not have', () => {
    const html = renderOverviewContent(
      statsWith({ datasetSegments: 20, visibleSegments: 2, datasetTriangles: 40 }),
      NO_CACHE
    );
    expect(html).toContain('VISIBLE LINES');
    expect(html).toContain('VISIBLE TRIANGLES');
    expect(html).not.toContain('VISIBLE POINTS');
    expect(html).not.toContain('VISIBLE SPLATS');
  });

  it('still shows the LOADING placeholder when no type has data', () => {
    const html = renderOverviewContent(statsWith(), NO_CACHE);
    expect(html).toContain('Waiting for data');
    expect(html).toContain('luxar-overview-grid--cols-1');
  });

  it('counts a mesh whose triangles are all outside the current slice', () => {
    // Presence is `total > 0 || visible > 0`: a layer sliced fully out still
    // has a card, reading 0 — dropping it would make the layer vanish from the
    // panel exactly when the user wants to know why nothing is drawn.
    const html = renderOverviewContent(statsWith({ datasetTriangles: 500 }), NO_CACHE);
    expect(html).toContain('VISIBLE TRIANGLES');
    expect(html).toContain('0.0% of 500 total');
  });
});

describe('renderLoaderItem — mesh rows', () => {
  it('labels a mesh loader as mesh/tris, not points/pts', () => {
    const html = renderLoaderItem('/surface', {
      type: 'mesh-whole-node',
      path: '/surface',
      queries: 0,
      loads: 1,
      errors: 0,
      elementsLoaded: 900,
      bytesLoaded: 1024,
      visibleElements: 300,
      avgQueryTime: 0,
      avgLoadTime: 12,
      memoryUsed: 2048,
    });
    expect(html).toContain('>mesh<');
    expect(html).toContain('300 tris');
    expect(html).not.toContain('pts');
  });

  it('shows a loaded whole-node mesh as active with loader-neutral tooltips', () => {
    const html = renderLoaderItem('/surface', {
      type: 'mesh-whole-node',
      path: '/surface',
      queries: 0,
      loads: 1,
      errors: 0,
      elementsLoaded: 900,
      bytesLoaded: 1024,
      visibleElements: 300,
      avgQueryTime: 0,
      avgLoadTime: 12,
      memoryUsed: 2048,
    });
    expect(html).toContain('luxar-loader-item__path luxar-color--success');
    expect(html).toContain('green = has loaded data or answered queries this session');
    expect(html).toContain('CPU memory this loader currently holds for loaded data');
    expect(html).not.toContain('loaded chunks and index data');
  });
});
