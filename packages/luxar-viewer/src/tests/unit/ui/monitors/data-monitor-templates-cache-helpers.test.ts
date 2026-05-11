/**
 * R3: focused unit tests for the cache-tab helper functions exported
 * from `data-monitor-templates.ts`. These guard the small pure helpers
 * used by both the initial render and the per-tick patcher so changes
 * to the cache-tab UI stay in sync across both paths.
 */

import { describe, it, expect } from 'vitest';
import {
  CACHE_BADGE_COLOR,
  formatValidationMode,
  formatLastValidated,
  l2ErrorTotal,
  renderCacheStatusBadges,
  getCacheHitRateColorClass,
  getCacheHitRateColorClassWithGuard,
  renderCacheContent,
} from '../../../../ui/monitors/data-monitor-templates';
import type {
  CacheMetrics,
  CacheStatusBadge,
  GlobalStats,
} from '../../../../types/data-monitor-types';

function makeGlobalStats(): GlobalStats {
  return {
    totalLoaders: 0,
    activeSpatialLoaders: 0,
    activeFallbackLoaders: 0,
    totalPoints: 0,
    totalMemory: 0,
    datasetSize: 0,
    visiblePoints: 0,
    datasetSegments: 0,
    visibleSegments: 0,
    datasetSplats: 0,
    visibleSplats: 0,
    totalQueries: 0,
    totalLoads: 0,
    totalCacheHits: 0,
    totalPointsLoaded: 0,
    totalMemoryUsed: 0,
    globalCacheHitRate: 0,
    avgQueryTime: 0,
    queriesPerSecond: 0,
    recommendations: [],
  };
}

function makeCacheMetrics(overrides: Partial<CacheMetrics> = {}): CacheMetrics {
  return {
    totalCacheMemory: 0,
    memoryLimit: 1024,
    memoryPercent: 0,
    totalEntries: 0,
    totalAccesses: 0,
    recentHitRate: 0,
    evictionsTotal: 0,
    avgEntrySize: 0,
    reuseRatio: 0,
    hitsPerSecond: 0,
    missesPerSecond: 0,
    avgAccessTime: 0,
    queriesPerSec: 0,
    loadsPerSec: 0,
    bandwidth: 0,
    ...overrides,
  };
}

describe('CACHE_BADGE_COLOR', () => {
  it('maps every CacheStatusBadge to a CSS class', () => {
    const allBadges: CacheStatusBadge[] = [
      'cache-enabled',
      'no-cache',
      'disabled-config',
      'opfs-unavailable',
      'quota-constrained',
      'unvalidated-external-dataset',
      'cache-errors-detected',
      'provider-missing',
    ];
    for (const b of allBadges) {
      expect(CACHE_BADGE_COLOR[b]).toMatch(/^luxar-color--/);
    }
  });

  it('uses success color for cache-enabled, error for errors-detected', () => {
    expect(CACHE_BADGE_COLOR['cache-enabled']).toMatch(/success/);
    expect(CACHE_BADGE_COLOR['cache-errors-detected']).toMatch(/error/);
    expect(CACHE_BADGE_COLOR['provider-missing']).toMatch(/error/);
    expect(CACHE_BADGE_COLOR['quota-constrained']).toMatch(/warning/);
    expect(CACHE_BADGE_COLOR['unvalidated-external-dataset']).toMatch(/warning/);
  });
});

describe('renderCacheStatusBadges', () => {
  it('returns empty string for empty/undefined input', () => {
    expect(renderCacheStatusBadges(undefined)).toBe('');
    expect(renderCacheStatusBadges([])).toBe('');
  });

  it('renders one pill span per badge', () => {
    const html = renderCacheStatusBadges(['cache-enabled', 'quota-constrained']);
    expect(html).toContain('data-badge="cache-enabled"');
    expect(html).toContain('data-badge="quota-constrained"');
    expect(html.match(/<span/g)?.length).toBe(2);
  });

  it('includes the color class for each badge', () => {
    const html = renderCacheStatusBadges(['cache-errors-detected']);
    expect(html).toContain('luxar-color--error');
  });
});

describe('formatValidationMode', () => {
  it('formats content-hash as "Content Hash"', () => {
    expect(formatValidationMode('content-hash')).toBe('Content Hash');
  });

  it('formats ttl as "TTL"', () => {
    expect(formatValidationMode('ttl')).toBe('TTL');
  });

  it('formats none as "None"', () => {
    expect(formatValidationMode('none')).toBe('None');
  });

  it('returns em-dash for undefined', () => {
    expect(formatValidationMode(undefined)).toBe('—');
  });
});

describe('formatLastValidated', () => {
  it('returns "Never" for null/undefined', () => {
    expect(formatLastValidated(null)).toBe('Never');
    expect(formatLastValidated(undefined)).toBe('Never');
  });

  it('returns "Never" for NaN / Infinity (defensive)', () => {
    expect(formatLastValidated(NaN)).toBe('Never');
    expect(formatLastValidated(Infinity)).toBe('Never');
  });

  it('returns a formatted timestamp for a valid millisecond value', () => {
    const ts = new Date('2026-05-10T14:00:00Z').getTime();
    const result = formatLastValidated(ts);
    // Locale-dependent format — just check the year is present and
    // we got more than the "Never" placeholder.
    expect(result).not.toBe('Never');
    expect(result).toContain('2026');
  });
});

describe('l2ErrorTotal', () => {
  it('returns 0 when l2 is undefined', () => {
    expect(l2ErrorTotal(undefined)).toBe(0);
  });

  it('returns 0 when all counters are zero or missing', () => {
    expect(l2ErrorTotal({ size: 0, count: 0, reads: 0, writes: 0, misses: 0 })).toBe(0);
  });

  it('sums the four health counters', () => {
    expect(
      l2ErrorTotal({
        size: 0,
        count: 0,
        reads: 0,
        writes: 0,
        misses: 0,
        quotaWriteSkipped: 1,
        writeFailures: 2,
        corruptedEntries: 3,
        metadataParseFailures: 4,
      })
    ).toBe(10);
  });

  it('treats missing counters as zero (older provider compatibility)', () => {
    expect(
      l2ErrorTotal({
        size: 0,
        count: 0,
        reads: 0,
        writes: 0,
        misses: 0,
        quotaWriteSkipped: 5,
        // writeFailures, corruptedEntries, metadataParseFailures all undefined
      })
    ).toBe(5);
  });
});

// S3: no-data guard so initial render uses dimmed instead of error
// when there are no accesses yet — matches the incremental updater.
describe('getCacheHitRateColorClassWithGuard (S3)', () => {
  it('returns dimmed color when totalAccesses === 0', () => {
    expect(getCacheHitRateColorClassWithGuard(0, 0)).toMatch(/dimmed/);
    // Same response regardless of rate value when nothing has been accessed.
    expect(getCacheHitRateColorClassWithGuard(99, 0)).toMatch(/dimmed/);
  });

  it('returns error color when totalAccesses > 0 and rate is ≤ 50', () => {
    expect(getCacheHitRateColorClassWithGuard(0, 1)).toMatch(/error/);
    expect(getCacheHitRateColorClassWithGuard(50, 100)).toMatch(/error/);
  });

  it('returns warning color when rate is in (50, 80]', () => {
    expect(getCacheHitRateColorClassWithGuard(60, 100)).toMatch(/warning/);
    expect(getCacheHitRateColorClassWithGuard(80, 100)).toMatch(/warning/);
  });

  it('returns success color when rate > 80', () => {
    expect(getCacheHitRateColorClassWithGuard(90, 100)).toMatch(/success/);
    expect(getCacheHitRateColorClassWithGuard(100, 100)).toMatch(/success/);
  });

  it('matches getCacheHitRateColorClass for any non-zero totalAccesses', () => {
    // The guard is purely a no-data wrapper; once totalAccesses > 0
    // the two functions must agree.
    for (const rate of [0, 25, 51, 65, 80.01, 95]) {
      expect(getCacheHitRateColorClassWithGuard(rate, 1)).toBe(getCacheHitRateColorClass(rate));
    }
  });
});

describe('renderCacheContent status rows in non-full cache views', () => {
  it('renders the no-cache badge in the disabled ?no-cache view', () => {
    const html = renderCacheContent(
      makeGlobalStats(),
      makeCacheMetrics({
        enabled: false,
        telemetryState: { kind: 'disabled-no-cache' },
        status: ['no-cache'],
      })
    );

    expect(html).toContain('Caching disabled by ?no-cache');
    expect(html).toContain('data-field="cache-status-row"');
    expect(html).toContain('data-badge="no-cache"');
  });

  it('renders the disabled-config badge in the config-disabled view', () => {
    const html = renderCacheContent(
      makeGlobalStats(),
      makeCacheMetrics({
        enabled: false,
        telemetryState: { kind: 'disabled-config' },
        status: ['disabled-config'],
      })
    );

    expect(html).toContain('Caching disabled by configuration');
    expect(html).toContain('data-badge="disabled-config"');
  });

  it('renders status badges in the enabled fallback/loading view', () => {
    const html = renderCacheContent(
      makeGlobalStats(),
      makeCacheMetrics({
        enabled: true,
        telemetryState: { kind: 'enabled' },
        status: ['cache-enabled', 'provider-missing'],
        // No l1/l2: exercises the fallback "Loading cache statistics" view.
        l1: undefined,
        l2: undefined,
      })
    );

    expect(html).toContain('Loading cache statistics');
    expect(html).toContain('data-badge="cache-enabled"');
    expect(html).toContain('data-badge="provider-missing"');
  });
});
