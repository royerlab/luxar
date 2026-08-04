/**
 * Unit tests for the pure helpers in data-loading-monitor/templates.ts.
 *
 * Targets the small zero-dependency utilities (formatters, color
 * classifiers, percentage math) that live alongside the larger
 * HTML-template renderers. The renderers themselves are dominated
 * by string concatenation — covering them in unit tests adds little
 * value past the helpers.
 *
 * No mocks, no DOM. Each function takes primitives and returns a
 * primitive.
 */

import { describe, it, expect } from 'vitest';
import {
  calculateReuseRate,
  formatBytes,
  formatNumber,
  getCacheHitRateColorClass,
  getCacheMemoryColorClass,
  getColorClass,
  getReuseRateColorClass,
  renderMetricCard,
  renderProgressBar,
  renderStatGrid,
  renderSceneGraphTree,
  renderSecondaryMetrics,
  summariseLodStates,
  lodChipContent,
  countAdditiveNodes,
  nodeStatsContent,
  activeLevelRole,
} from '../../../ui/data-loading-monitor/templates';
import type {
  SceneGraphState,
  SceneGraphNode,
  LODProgressState,
  NodeDrawOrder,
} from '../../../types/data-monitor-types';

describe('getColorClass', () => {
  it('prefixes the semantic name with luxar-color--', () => {
    expect(getColorClass('success')).toBe('luxar-color--success');
    expect(getColorClass('warning')).toBe('luxar-color--warning');
    expect(getColorClass('error')).toBe('luxar-color--error');
    expect(getColorClass('muted')).toBe('luxar-color--muted');
    expect(getColorClass('dimmed')).toBe('luxar-color--dimmed');
  });
});

describe('formatNumber', () => {
  it('formats values < 1000 as plain numbers', () => {
    expect(formatNumber(0)).toBe('0');
    expect(formatNumber(1)).toBe('1');
    expect(formatNumber(999)).toBe('999');
  });

  it('uses K suffix from 1e3 to 1e6', () => {
    expect(formatNumber(1000)).toBe('1.0K');
    expect(formatNumber(1500)).toBe('1.5K');
    expect(formatNumber(999999)).toBe('1000.0K');
  });

  it('uses M suffix from 1e6 to 1e9', () => {
    expect(formatNumber(1_000_000)).toBe('1.0M');
    expect(formatNumber(2_500_000)).toBe('2.5M');
  });

  it('uses B suffix from 1e9 onward', () => {
    expect(formatNumber(1_000_000_000)).toBe('1.0B');
    expect(formatNumber(7_300_000_000)).toBe('7.3B');
  });
});

describe('formatBytes', () => {
  it('formats values < 1KB as plain B with no decimals', () => {
    expect(formatBytes(0)).toBe('0B');
    expect(formatBytes(1)).toBe('1B');
    expect(formatBytes(999)).toBe('999B');
  });

  it('uses KB suffix', () => {
    expect(formatBytes(1000)).toBe('1.0KB');
    expect(formatBytes(2500)).toBe('2.5KB');
  });

  it('uses MB suffix', () => {
    expect(formatBytes(1_000_000)).toBe('1.0MB');
    expect(formatBytes(50_000_000)).toBe('50.0MB');
  });

  it('uses GB suffix', () => {
    expect(formatBytes(1_000_000_000)).toBe('1.0GB');
    expect(formatBytes(2_500_000_000)).toBe('2.5GB');
  });
});

describe('calculateReuseRate', () => {
  it('returns 0 when both counts are 0', () => {
    expect(calculateReuseRate(0, 0)).toBe(0);
  });

  it('returns 0 when reuses is 0', () => {
    expect(calculateReuseRate(10, 0)).toBe(0);
  });

  it('returns 100 when allocations is 0 (everything was a reuse)', () => {
    expect(calculateReuseRate(0, 10)).toBe(100);
  });

  it('returns the right percentage for a mixed workload', () => {
    expect(calculateReuseRate(1, 1)).toBe(50);
    expect(calculateReuseRate(1, 9)).toBe(90);
    expect(calculateReuseRate(9, 1)).toBe(10);
    expect(calculateReuseRate(75, 25)).toBe(25);
  });
});

describe('getReuseRateColorClass', () => {
  it('classifies ≥80 as success, ≥50 as warning, else error', () => {
    expect(getReuseRateColorClass(100)).toBe('luxar-color--success');
    expect(getReuseRateColorClass(80)).toBe('luxar-color--success');
    expect(getReuseRateColorClass(79.9)).toBe('luxar-color--warning');
    expect(getReuseRateColorClass(50)).toBe('luxar-color--warning');
    expect(getReuseRateColorClass(49.9)).toBe('luxar-color--error');
    expect(getReuseRateColorClass(0)).toBe('luxar-color--error');
  });
});

describe('getCacheHitRateColorClass', () => {
  it('uses strict-greater-than thresholds (80 maps to warning, not success)', () => {
    expect(getCacheHitRateColorClass(100)).toBe('luxar-color--success');
    expect(getCacheHitRateColorClass(80.1)).toBe('luxar-color--success');
    expect(getCacheHitRateColorClass(80)).toBe('luxar-color--warning');
    expect(getCacheHitRateColorClass(50.1)).toBe('luxar-color--warning');
    expect(getCacheHitRateColorClass(50)).toBe('luxar-color--error');
    expect(getCacheHitRateColorClass(0)).toBe('luxar-color--error');
  });
});

describe('getCacheMemoryColorClass', () => {
  it('classifies ≤60 as success, ≤80 as warning, else error (low pressure is good)', () => {
    expect(getCacheMemoryColorClass(0)).toBe('luxar-color--success');
    expect(getCacheMemoryColorClass(60)).toBe('luxar-color--success');
    expect(getCacheMemoryColorClass(60.1)).toBe('luxar-color--warning');
    expect(getCacheMemoryColorClass(80)).toBe('luxar-color--warning');
    expect(getCacheMemoryColorClass(80.1)).toBe('luxar-color--error');
    expect(getCacheMemoryColorClass(100)).toBe('luxar-color--error');
  });
});

describe('renderMetricCard', () => {
  it('renders title, value, and subtitle when provided', () => {
    const html = renderMetricCard('Loaded', 42, '90% reuse');
    expect(html).toContain('Loaded');
    expect(html).toContain('42');
    expect(html).toContain('90% reuse');
  });

  it('omits the title block when title is empty', () => {
    const html = renderMetricCard('', 7);
    expect(html).not.toContain('luxar-metric-card__title');
  });

  it('omits the subtitle block when subtitle is undefined', () => {
    const html = renderMetricCard('Total', 100);
    expect(html).not.toContain('luxar-metric-card__subtitle');
  });

  it('writes data-field attributes when dataField is supplied', () => {
    const html = renderMetricCard('Total', 100, 'subtitle', '', 'medium', 'totals');
    expect(html).toContain('data-field="totals"');
    expect(html).toContain('data-field="totals-sub"');
  });

  it('applies size modifier to both block class and value class', () => {
    expect(renderMetricCard('a', 1, undefined, '', 'small')).toContain('luxar-metric-card--small');
    expect(renderMetricCard('a', 1, undefined, '', 'small')).toContain(
      'luxar-metric-card__value--small'
    );
  });
});

describe('renderProgressBar', () => {
  it('clamps the fill width to 100%', () => {
    expect(renderProgressBar(150)).toContain('width: 100%');
  });

  it('writes the literal percentage when within [0, 100]', () => {
    expect(renderProgressBar(42)).toContain('width: 42%');
  });

  it('uses the default progress color when no override is given', () => {
    // 42% → 42 ≤ 60 → success
    expect(renderProgressBar(42)).toContain('luxar-color--success');
  });

  it('applies the override color class when supplied', () => {
    expect(renderProgressBar(42, 'luxar-color--custom')).toContain('luxar-color--custom');
  });

  it('omits the label block when label is undefined', () => {
    expect(renderProgressBar(50)).not.toContain('luxar-progress-bar__label');
  });
});

describe('renderStatGrid', () => {
  it('uses the column-count modifier capped at 3', () => {
    expect(renderStatGrid([{ label: 'a', value: 1 }])).toContain('luxar-stat-grid--cols-1');
    expect(
      renderStatGrid([
        { label: 'a', value: 1 },
        { label: 'b', value: 2 },
      ])
    ).toContain('luxar-stat-grid--cols-2');
    const five = Array.from({ length: 5 }, (_, i) => ({ label: `l${i}`, value: i }));
    expect(renderStatGrid(five)).toContain('luxar-stat-grid--cols-3');
  });

  it('renders one item block per stat', () => {
    const html = renderStatGrid([
      { label: 'first', value: 1 },
      { label: 'second', value: 2 },
    ]);
    const matches = html.match(/luxar-stat-grid__item/g) ?? [];
    expect(matches.length).toBe(2);
    expect(html).toContain('first');
    expect(html).toContain('second');
  });

  it('applies the per-item colorClass when supplied', () => {
    const html = renderStatGrid([{ label: 'a', value: 1, colorClass: 'luxar-color--success' }]);
    expect(html).toContain('luxar-color--success');
  });
});

describe('lodChipContent', () => {
  it('renders the active substitutive level', () => {
    const node = { kind: 'lod', lodGroupChildCount: 3 } as SceneGraphNode;
    const state: LODProgressState = {
      kind: 'lod',
      levelCount: 3,
      activeLevel: 1,
      selector: 'auto',
    };
    expect(lodChipContent(node, state)).toEqual({
      text: 'L2/3',
      title: 'Active substitutive level 2 of 3 — only this level is rendered',
    });
  });

  it('renders additive loaded/total with refining spinner + streaming dot', () => {
    const node = { additiveSublods: 4 } as SceneGraphNode;
    const state: LODProgressState = {
      kind: 'additive',
      loaded: 2,
      total: 4,
      refining: true,
      lastAllResident: false,
    };
    const c = lodChipContent(node, state)!;
    expect(c.text).toContain('LOD 2/4');
    expect(c.text).toContain('◌'); // streaming (not all resident)
    expect(c.text).toContain('⏳'); // refining
  });

  it('shows "–" (not a guessed level) before the first provider poll', () => {
    const node = { kind: 'lod', lodGroupChildCount: 2 } as SceneGraphNode;
    expect(lodChipContent(node, undefined)).toEqual({
      text: 'L–/2',
      title: 'Substitutive LOD group with 2 levels — active level not yet reported',
    });
  });

  it('shows "–" for an additive node with no live loader state (inactive level)', () => {
    const node = { additiveSublods: 6 } as SceneGraphNode;
    const c = lodChipContent(node, undefined)!;
    expect(c.text).toBe('LOD –/6');
    expect(c.title).toContain('not streaming');
  });

  it('always explains the residency dot in the tooltip, including when refinement is done', () => {
    const node = { additiveSublods: 4 } as SceneGraphNode;
    const done: LODProgressState = {
      kind: 'additive',
      loaded: 4,
      total: 4,
      refining: false,
      lastAllResident: true,
    };
    const c = lodChipContent(node, done)!;
    expect(c.text).toContain('●');
    expect(c.title).toContain('● = fully cache-resident');

    const streaming = lodChipContent(node, { ...done, lastAllResident: false })!;
    expect(streaming.text).toContain('◌');
    expect(streaming.title).toContain('◌ = streaming from network');
  });

  it('shows the committed energy fraction e(k) with a didactic tooltip when stamped', () => {
    const node = { additiveSublods: 6 } as SceneGraphNode;
    const state: LODProgressState = {
      kind: 'additive',
      loaded: 2,
      total: 6,
      refining: true,
      lastAllResident: false,
      energy: 0.72,
    };
    const c = lodChipContent(node, state)!;
    expect(c.text).toContain('LOD 2/6 ~72%');
    expect(c.title).toContain('~72% of the level');
    expect(c.title).toContain('energy-ordered streaming');

    // Unstamped (legacy) datasets: no percentage, no energy tooltip clause.
    const legacy = lodChipContent(node, { ...state, energy: undefined })!;
    expect(legacy.text).not.toContain('%');
    expect(legacy.title).not.toContain('energy');
  });

  it('returns null for a plain node with no LOD dimension', () => {
    expect(lodChipContent({ type: 'points' } as SceneGraphNode, undefined)).toBeNull();
  });
});

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

describe('summariseLodStates', () => {
  it('summarises substitutive groups, additive nodes, and refinement', () => {
    const states = new Map<string, LODProgressState>([
      ['/a', { kind: 'lod', levelCount: 3, activeLevel: 0 }],
      ['/b', { kind: 'additive', loaded: 1, total: 4, refining: true }],
    ]);
    expect(summariseLodStates(states)).toBe('1 substitutive · 1 additive · refining 1');
  });

  it('returns empty string for no states', () => {
    expect(summariseLodStates(undefined)).toBe('');
    expect(summariseLodStates(new Map())).toBe('');
  });

  it('reconciles live additive count against the tree total ("x/y additive active")', () => {
    const states = new Map<string, LODProgressState>([
      ['/lod', { kind: 'lod', levelCount: 5, activeLevel: 0 }],
      ['/lod/child_0', { kind: 'additive', loaded: 4, total: 4, refining: false }],
    ]);
    expect(summariseLodStates(states, 5)).toBe('1 substitutive · 1/5 additive active');
    // Matching totals keep the plain wording.
    expect(summariseLodStates(states, 1)).toBe('1 substitutive · 1 additive');
  });
});

describe('countAdditiveNodes', () => {
  it('counts nodes with additiveSublods > 1 across the tree', () => {
    const root: SceneGraphNode = {
      path: '/',
      name: 'LOD',
      type: 'group',
      kind: 'lod',
      lodGroupChildCount: 2,
      children: [
        {
          path: '/child_0',
          name: 'child_0',
          type: 'gsplats',
          additiveSublods: 4,
          children: [],
        },
        {
          path: '/child_1',
          name: 'child_1',
          type: 'gsplats',
          additiveSublods: 6,
          children: [],
        },
      ],
    };
    expect(countAdditiveNodes(root)).toBe(2);
    expect(countAdditiveNodes(null)).toBe(0);
  });
});

describe('nodeStatsContent', () => {
  it('appends per-node visible counts symmetrically for all three geometry types', () => {
    const points = {
      type: 'points',
      pointCount: 1000,
      visiblePointCount: 250,
      children: [],
    } as unknown as SceneGraphNode;
    expect(nodeStatsContent(points)!.title).toContain('250 visible after slicing');

    const lines = {
      type: 'lines',
      segmentCount: 500,
      visibleSegmentCount: 100,
      children: [],
    } as unknown as SceneGraphNode;
    expect(nodeStatsContent(lines)!.title).toContain('100 visible after slicing');

    const gsplats = {
      type: 'gsplats',
      splatCount: 2000,
      visibleSplatCount: 700,
      children: [],
    } as unknown as SceneGraphNode;
    expect(nodeStatsContent(gsplats)!.title).toContain('700 visible after slicing');
  });

  it('omits the visible suffix when unknown or equal to the total', () => {
    const node = { type: 'gsplats', splatCount: 2000, children: [] } as unknown as SceneGraphNode;
    expect(nodeStatsContent(node)!.title).not.toContain('visible');
    const same = {
      type: 'gsplats',
      splatCount: 2000,
      visibleSplatCount: 2000,
      children: [],
    } as unknown as SceneGraphNode;
    expect(nodeStatsContent(same)!.title).not.toContain('visible');
  });

  it('suppresses the child-count badge on specialized groups (kind badge covers it)', () => {
    const child = { type: 'gsplats', splatCount: 1, children: [] } as unknown as SceneGraphNode;
    const plain = { type: 'group', children: [child] } as unknown as SceneGraphNode;
    expect(nodeStatsContent(plain)!.text).toBe('1');
    const lod = {
      type: 'group',
      kind: 'lod',
      lodGroupChildCount: 1,
      children: [child],
    } as unknown as SceneGraphNode;
    expect(nodeStatsContent(lod)).toBeNull();
  });
});

describe('activeLevelRole', () => {
  // Exported so the monitor's per-tick level-row patcher shares this
  // exact derivation with the initial render (no drifting inline copy).
  it('marks the active level and dims the others', () => {
    const state: LODProgressState = { kind: 'lod', levelCount: 3, activeLevel: 1 };
    expect(activeLevelRole(state, 0)).toBe('inactive');
    expect(activeLevelRole(state, 1)).toBe('active');
    expect(activeLevelRole(state, 2)).toBe('inactive');
  });

  it('returns undefined before the first provider poll (no state / no activeLevel)', () => {
    expect(activeLevelRole(undefined, 0)).toBeUndefined();
    expect(activeLevelRole({ kind: 'lod', levelCount: 3 }, 0)).toBeUndefined();
  });

  it('returns undefined for non-substitutive state kinds', () => {
    expect(activeLevelRole({ kind: 'additive', loaded: 1, total: 2 }, 0)).toBeUndefined();
    expect(activeLevelRole({ kind: 'partition', partCount: 4 }, 0)).toBeUndefined();
  });
});

describe('renderSceneGraphTree — kind badges', () => {
  function tree(root: SceneGraphNode): SceneGraphState {
    return {
      root,
      totalNodes: 1,
      nodesByType: { points: 0, lines: 0, gsplats: 1, mesh: 0 },
      totalByType: { points: 0, lines: 0, gsplats: 0, mesh: 0 },
      visibleByType: { points: 0, lines: 0, gsplats: 0, mesh: 0 },
    };
  }

  it('header reports LAYER counts (nodesByType), not element counts', () => {
    // The header renders "N points, N lines, N gsplats" as *layer* counts, and its
    // tooltip says so explicitly ("not element counts") because "5 gsplats"
    // otherwise reads as 5 splats. `nodesByType` and `totalByType` are both
    // `GeometryCounters` and so are trivially swappable at the call site — this
    // pins which one the header reads.
    const state: SceneGraphState = {
      root: { path: '/', name: 'Scene', type: 'scene', children: [] },
      totalNodes: 3,
      nodesByType: { points: 2, lines: 0, gsplats: 1, mesh: 0 },
      // Deliberately disjoint from nodesByType so reading the wrong record shows.
      totalByType: { points: 90000, lines: 5000, gsplats: 70000, mesh: 0 },
      visibleByType: { points: 1, lines: 2, gsplats: 3, mesh: 0 },
    };
    const html = renderSceneGraphTree(state, new Set(['/']), new Map());

    expect(html).toContain('2 points');
    expect(html).toContain('1 gsplats');
    // A type with no layers is omitted entirely, even though it has elements.
    expect(html).not.toContain('0 lines');
    expect(html).not.toContain('5000 lines');
    // Element totals must never appear in the layer-count header.
    expect(html).not.toContain('90000 points');
    expect(html).not.toContain('70000 gsplats');
    expect(html).toContain('not element counts'); // the disambiguating tooltip
  });

  it('renders a "K LODs" badge + active-level chip for a kind=lod group', () => {
    const root: SceneGraphNode = {
      path: '/lod',
      name: 'lod',
      type: 'group',
      kind: 'lod',
      lodGroupChildCount: 3,
      children: [],
    };
    const lodStates = new Map<string, LODProgressState>([
      ['/lod', { kind: 'lod', levelCount: 3, activeLevel: 2, selector: 'auto' }],
    ]);
    const html = renderSceneGraphTree(tree(root), new Set(['/lod']), lodStates);
    expect(html).toContain('3 LODs');
    expect(html).toContain('luxar-scene-graph__badge--kind');
    expect(html).toContain('data-lod-path="/lod"');
    expect(html).toContain('L3/3');
  });

  it('renders an "N parts" badge for a kind=partition group', () => {
    const root: SceneGraphNode = {
      path: '/part',
      name: 'part',
      type: 'group',
      kind: 'partition',
      partCount: 4,
      children: [],
    };
    const html = renderSceneGraphTree(tree(root), new Set(), new Map());
    expect(html).toContain('4 parts');
  });

  describe('draw-order chip', () => {
    const gsplatsNode: SceneGraphNode = {
      path: '/cloud',
      name: 'cloud',
      type: 'gsplats',
      children: [],
    };

    it('renders bucket + renderOrder when live state exists', () => {
      const drawOrderStates = new Map<string, NodeDrawOrder>([
        ['/cloud', { bucket: 'transparent', depthWrite: false, renderOrder: 3 }],
      ]);
      const html = renderSceneGraphTree(tree(gsplatsNode), new Set(), new Map(), drawOrderStates);
      expect(html).toContain('data-draworder-path="/cloud"');
      expect(html).toContain('#3 transparent');
    });

    it('renders a persistent EMPTY chip slot for a drawable node without state', () => {
      // A node hidden at structural-render time (toggled-off layer, inactive
      // substitutive-LOD level) has no provider state. The per-tick updater
      // only patches existing `.luxar-scene-graph__draworder` elements, so
      // the empty slot must exist or the chip could never appear once the
      // node becomes visible.
      const html = renderSceneGraphTree(tree(gsplatsNode), new Set(), new Map(), new Map());
      // Empty slot: no text, no tooltip — nothing fabricated.
      expect(html).toContain('data-draworder-path="/cloud" title=""></span>');
    });

    it('renders no chip slot for a non-drawable node (group)', () => {
      const group: SceneGraphNode = {
        path: '/grp',
        name: 'grp',
        type: 'group',
        children: [],
      };
      const html = renderSceneGraphTree(tree(group), new Set(), new Map(), new Map());
      expect(html).not.toContain('data-draworder-path');
    });
  });
});

describe('nodeStatsContent — per-type element counts in the scene tree', () => {
  const node = (o: Record<string, unknown>) =>
    ({ path: '/n', name: 'n', type: 'group', children: [], hasSpatialIndex: false, ...o }) as never;

  it('reports a MESH in triangles, with vertices in the tooltip', () => {
    // Mesh had no arm here, so a mesh row rendered a blank count while points, lines
    // and gsplats all showed one — and `faceCount` was populated by the converter for
    // nothing. Triangles, not vertices: the drawn-primitive convention `lines` follows
    // in reporting segments.
    const r = nodeStatsContent(node({ type: 'mesh', faceCount: 1200, vertexCount: 640 }));
    expect(r).not.toBeNull();
    expect(r!.text).toBe(formatNumber(1200));
    expect(r!.title).toContain('1,200 triangles');
    expect(r!.title).toContain('640 vertices');
  });

  it('omits the vertex clause when the store did not record one', () => {
    const r = nodeStatsContent(node({ type: 'mesh', faceCount: 8 }));
    expect(r!.title).toBe('8 triangles');
  });

  it('returns null for a mesh with no faceCount rather than rendering a blank row', () => {
    expect(nodeStatsContent(node({ type: 'mesh' }))).toBeNull();
  });

  it('still reports the other three types in their own units', () => {
    // Anti-vacuity for the arm order: adding the mesh branch must not shadow these.
    expect(nodeStatsContent(node({ type: 'gsplats', splatCount: 5 }))!.title).toContain(
      'Gaussian splats'
    );
    expect(nodeStatsContent(node({ type: 'lines', segmentCount: 7 }))!.title).toContain(
      'line segments'
    );
  });
});
