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
} from '../../../ui/data-loading-monitor/templates';
import type {
  SceneGraphState,
  SceneGraphNode,
  LODProgressState,
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
      title: 'Active substitutive level 2 of 3',
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

  it('falls back to structural counts before the first provider poll', () => {
    const node = { kind: 'lod', lodGroupChildCount: 2 } as SceneGraphNode;
    expect(lodChipContent(node, undefined)).toEqual({
      text: 'L1/2',
      title: 'Active substitutive level 1 of 2',
    });
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
  it('summarises LOD groups, additive nodes, and refinement', () => {
    const states = new Map<string, LODProgressState>([
      ['/a', { kind: 'lod', levelCount: 3, activeLevel: 0 }],
      ['/b', { kind: 'additive', loaded: 1, total: 4, refining: true }],
    ]);
    expect(summariseLodStates(states)).toBe('1 LOD group · 1 additive · refining 1');
  });

  it('returns empty string for no states', () => {
    expect(summariseLodStates(undefined)).toBe('');
    expect(summariseLodStates(new Map())).toBe('');
  });
});

describe('renderSceneGraphTree — kind badges', () => {
  function tree(root: SceneGraphNode): SceneGraphState {
    return {
      root,
      totalNodes: 1,
      pointsNodes: 0,
      linesNodes: 0,
      gsplatsNodes: 1,
      totalPoints: 0,
      visiblePoints: 0,
      totalSegments: 0,
      visibleSegments: 0,
      totalSplats: 0,
      visibleSplats: 0,
    };
  }

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
});
