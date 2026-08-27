/**
 * The shared per-geometry headline table.
 *
 * Its job is to be the ONE place three surfaces (hero cards, their patcher, the
 * compact badge) read "which types are present and what are they called" from —
 * so these tests pin completeness and the presence rule rather than the
 * wording.
 */

import { describe, expect, it } from 'vitest';
import {
  compactTooltip,
  headlineCounts,
  headlineTooltip,
  presentHeadlineCounts,
} from '../../../../ui/data-loading-monitor/headline-counts';
import { GEOMETRY_TYPES } from '../../../../types/format-contract';
import type { GlobalStats } from '../../../../types/data-monitor-types';

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

describe('headlineCounts', () => {
  it('covers every geometry type, in contract order', () => {
    // Completeness is the point: the monitor's three headline surfaces used to
    // keep their own list of three, and mesh was missing from all of them.
    expect(headlineCounts(statsWith()).map((c) => c.type)).toEqual([...GEOMETRY_TYPES]);
  });

  it('reads each type from its own named GlobalStats pair', () => {
    // `GlobalStats` names its pairs after each type's element noun
    // (`visibleSegments`, not `visibleLines`), so a wrong wiring here would
    // show one type's count under another's label. Distinct values per field
    // are what make that detectable.
    const counts = headlineCounts(
      statsWith({
        datasetSize: 10,
        visiblePoints: 1,
        datasetSegments: 20,
        visibleSegments: 2,
        datasetSplats: 30,
        visibleSplats: 3,
        datasetTriangles: 40,
        visibleTriangles: 4,
      })
    );
    expect(counts.map((c) => [c.type, c.visible, c.total])).toEqual([
      ['points', 1, 10],
      ['lines', 2, 20],
      ['gsplats', 3, 30],
      ['mesh', 4, 40],
    ]);
  });

  it('gives every type a distinct label, field id and unit', () => {
    const counts = headlineCounts(statsWith());
    expect(new Set(counts.map((c) => c.label)).size).toBe(counts.length);
    expect(new Set(counts.map((c) => c.field)).size).toBe(counts.length);
    expect(new Set(counts.map((c) => c.unit)).size).toBe(counts.length);
  });

  it('calls mesh elements triangles, matching the rest of the monitor', () => {
    const mesh = headlineCounts(statsWith()).find((c) => c.type === 'mesh')!;
    expect(mesh.label).toBe('VISIBLE TRIANGLES');
    expect(mesh.field).toBe('visible-triangles');
    expect(mesh.unit).toBe('tris');
  });
});

describe('presentHeadlineCounts', () => {
  it('is empty before anything loads', () => {
    expect(presentHeadlineCounts(statsWith())).toEqual([]);
  });

  it('keeps a type whose elements are all outside the current slice', () => {
    // `total > 0` with `visible === 0`: the layer must stay visible in the
    // panel — that combination is precisely what the user is investigating.
    expect(presentHeadlineCounts(statsWith({ datasetTriangles: 500 })).map((c) => c.type)).toEqual([
      'mesh',
    ]);
  });

  it('keeps a type whose total is not known yet', () => {
    // `visible > 0` with `total === 0`: the window before dataset attrs land.
    expect(presentHeadlineCounts(statsWith({ visibleTriangles: 12 })).map((c) => c.type)).toEqual([
      'mesh',
    ]);
  });

  it('reports several types in contract order, skipping the absent ones', () => {
    expect(
      presentHeadlineCounts(
        statsWith({ visibleTriangles: 12, datasetSize: 5, visiblePoints: 5 })
      ).map((c) => c.type)
    ).toEqual(['points', 'mesh']);
  });
});

describe('tooltips', () => {
  it('name the type in prose, and differ between the two surfaces', () => {
    expect(headlineTooltip('mesh triangles')).toContain('How many mesh triangles are on screen');
    // Capitalized for the badge, since it opens the sentence there.
    expect(compactTooltip('mesh triangles')).toMatch(/^Mesh triangles currently on screen/);
  });
});
