// @vitest-environment jsdom
/**
 * R3: focused unit tests for the cache-tab helper functions exported
 * from `data-loading-monitor/templates/cache.ts`. These guard the small pure
 * helpers used by both the initial render and the per-tick patcher so
 * changes to the cache-tab UI stay in sync across both paths.
 *
 * AUDIT NOTE (ui.md C6): `renderCacheContent layout guards` block pins
 * exact CSS class names (`luxar-cache-section__metrics--cols-3` etc.)
 * and uses rendered HTML header strings ("L0 DECOMPRESSED CACHE" / "L1
 * MEMORY CACHE" / "L2 OPFS CACHE") as `indexOf` anchors. A pure UI
 * rename (e.g. shortening "MEMORY CACHE" to "MEMORY") would fail these
 * tests without any real bug. Accepted as a deliberate layout-
 * regression pin for now; if the UI strings churn frequently consider
 * extracting them into named constants both source and test import.
 */

import { describe, it, expect } from 'vitest';
import {
  CACHE_BADGE_COLOR,
  formatValidationMode,
  formatLastValidated,
  lastValidatedLabel,
  lastValidatedTooltip,
  l2ErrorTotal,
  renderCacheStatusBadges,
  validationModeTooltip,
  getCacheHitRateColorClass,
  getCacheHitRateColorClassWithGuard,
  CACHE_WARMUP_ACCESSES,
  renderCacheContent,
  CACHE_SECTION_KEYS,
} from '../../../../../ui/data-loading-monitor/templates/cache';
import type {
  CacheMetrics,
  CacheStatusBadge,
  GlobalStats,
} from '../../../../../types/data-monitor-types';

function makeGlobalStats(): GlobalStats {
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
    droppedElements: 0,
    totalQueries: 0,
    totalLoads: 0,
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

  it('formats archive-etag as "Archive ETag"', () => {
    expect(formatValidationMode('archive-etag')).toBe('Archive ETag');
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

describe('archive-etag validation presentation', () => {
  it('describes validation against the whole archive', () => {
    expect(validationModeTooltip('archive-etag')).toContain('Archive-ETag validation');
    expect(validationModeTooltip('archive-etag')).toContain('whole store');
    expect(lastValidatedTooltip('archive-etag')).toContain("archive's ETag");
  });

  it('labels archive checks as validation rather than a cache baseline', () => {
    expect(lastValidatedLabel('archive-etag')).toBe('Last Validated');
    expect(lastValidatedLabel('ttl')).toBe('Cached Since');
  });

  it('includes archive validation in the initial cache-health tooltips', () => {
    const host = document.createElement('div');
    host.innerHTML = renderCacheContent(
      makeGlobalStats(),
      makeCacheMetrics({
        enabled: true,
        l0: { size: 0, count: 0, hits: 0, misses: 0, evictions: 0, hitRate: 0 },
        l1: { size: 0, count: 0, hits: 0, misses: 0, evictions: 0 },
        l2: { size: 0, count: 0, reads: 0, writes: 0, misses: 0 },
        health: { validationMode: 'archive-etag' },
      })
    );

    const labels = host.querySelectorAll('.luxar-cache-health__label');
    expect(labels[0]?.getAttribute('title')).toContain('Archive ETag');
    expect(labels[1]?.getAttribute('title')).toContain('source-validated datasets');
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

// S3: no-data guard so initial render uses dimmed instead of error
// while the cache is still warming up — matches the incremental updater.
describe('getCacheHitRateColorClassWithGuard (S3)', () => {
  it('returns dimmed color while warming up (below CACHE_WARMUP_ACCESSES)', () => {
    expect(getCacheHitRateColorClassWithGuard(0, 0)).toMatch(/dimmed/);
    // Same response regardless of rate — a handful of first-touch
    // lookups carries no signal (the cache HAS to miss before it hits).
    expect(getCacheHitRateColorClassWithGuard(99, 0)).toMatch(/dimmed/);
    expect(getCacheHitRateColorClassWithGuard(0, CACHE_WARMUP_ACCESSES - 1)).toMatch(/dimmed/);
  });

  it('returns error color when warmed up and rate is ≤ 50', () => {
    expect(getCacheHitRateColorClassWithGuard(0, CACHE_WARMUP_ACCESSES)).toMatch(/error/);
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

  it('matches getCacheHitRateColorClass once warmed up', () => {
    // The guard is purely a warm-up wrapper; at or past the warm-up
    // threshold the two functions must agree.
    for (const rate of [0, 25, 51, 65, 80.01, 95]) {
      expect(getCacheHitRateColorClassWithGuard(rate, CACHE_WARMUP_ACCESSES)).toBe(
        getCacheHitRateColorClass(rate)
      );
    }
  });
});

describe('renderCacheContent status rows in non-full cache views', () => {
  it('renders the no-cache badge in the disabled ?noCache view', () => {
    const html = renderCacheContent(
      makeGlobalStats(),
      makeCacheMetrics({
        enabled: false,
        telemetryState: { kind: 'disabled-no-cache' },
        status: ['no-cache'],
      })
    );

    expect(html).toContain('Caching disabled by ?noCache');
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

/**
 * Layout guards for the full L0/L1/L2 cache view. These tests pin
 * the visual contract surfaced in `renderCacheContent`: shared column
 * grid across all sections, the no-limit progress-bar label, and the
 * styled EFFECTIVE-HIT-RATE footer.
 */
describe('renderCacheContent layout guards (full L0/L1/L2 view)', () => {
  function makeFullCacheMetrics(overrides: Partial<CacheMetrics> = {}): CacheMetrics {
    return makeCacheMetrics({
      enabled: true,
      telemetryState: { kind: 'enabled' },
      status: ['cache-enabled'],
      l0: { size: 845_900, count: 54, hits: 54, misses: 54, evictions: 0, hitRate: 0.5 },
      l1: { size: 694_000, count: 58, hits: 45, misses: 113, evictions: 0 },
      l2: { size: 694_000, count: 58, reads: 58, writes: 0, misses: 12 },
      ...overrides,
    });
  }

  /**
   * Slice the rendered HTML into the text region between two section
   * headers so we can assert the cols class belongs to *that* section
   * (not the next one further down).
   */
  function sliceBetween(html: string, from: string, to: string): string {
    const start = html.indexOf(from);
    const end = to ? html.indexOf(to) : html.length;
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    return html.slice(start, end);
  }

  it('renders the L0 section with the 3-card grid class (and not cols-4)', () => {
    const html = renderCacheContent(makeGlobalStats(), makeFullCacheMetrics());
    const region = sliceBetween(html, 'L0 DECOMPRESSED CACHE', 'L1 MEMORY CACHE');
    expect(region).toContain('luxar-cache-section__metrics--cols-3');
    expect(region).not.toContain('luxar-cache-section__metrics--cols-4');
  });

  it('renders the L1 section with the 3-card grid class (and not cols-4)', () => {
    const html = renderCacheContent(makeGlobalStats(), makeFullCacheMetrics());
    const region = sliceBetween(html, 'L1 MEMORY CACHE', 'L2 OPFS CACHE');
    expect(region).toContain('luxar-cache-section__metrics--cols-3');
    expect(region).not.toContain('luxar-cache-section__metrics--cols-4');
  });

  it('renders the L2 section with the 4-card grid class (and not cols-3)', () => {
    const html = renderCacheContent(makeGlobalStats(), makeFullCacheMetrics());
    // L2 is followed by CACHE HEALTH in the rendered template.
    const region = sliceBetween(html, 'L2 OPFS CACHE', 'CACHE HEALTH');
    expect(region).toContain('luxar-cache-section__metrics--cols-4');
    expect(region).not.toContain('luxar-cache-section__metrics--cols-3');
  });

  it('progress-bar label reads "X% of Y limit" when memoryLimit > 0', () => {
    const html = renderCacheContent(
      makeGlobalStats(),
      makeFullCacheMetrics({
        totalCacheMemory: 1024,
        memoryLimit: 4096,
        memoryPercent: 25,
      })
    );

    expect(html).toContain('25% of');
    expect(html).toContain('limit');
    expect(html).not.toContain('no memory limit configured');
  });

  it('progress-bar label reads "no memory limit configured" when memoryLimit is 0', () => {
    const html = renderCacheContent(
      makeGlobalStats(),
      makeFullCacheMetrics({
        totalCacheMemory: 2_200_000,
        memoryLimit: 0,
        memoryPercent: 0,
      })
    );

    expect(html).toContain('no memory limit configured');
    // The misleading zero-limit label must not slip back in.
    expect(html).not.toContain('0% of 0B limit');
    expect(html).not.toContain('of 0B');
  });

  it('renders the EFFECTIVE HIT RATE footer when effectiveDemandHitRate is defined', () => {
    const html = renderCacheContent(
      makeGlobalStats(),
      makeFullCacheMetrics({ effectiveDemandHitRate: 1.0 })
    );

    // Footer is wrapped in its own styled container so it reads as
    // part of the TOTAL card rather than unstyled stray text. The
    // label is a static span; only the value span carries the
    // data-field patched by the per-tick updater.
    expect(html).toMatch(/<div[^>]*class="luxar-cache-total__demand"/);
    expect(html).toContain('EFFECTIVE HIT RATE');
    const host = document.createElement('div');
    host.innerHTML = html;
    const value = host.querySelector('[data-field="cache-effective-hitrate"]');
    expect(value?.textContent?.trim()).toBe('100.0%');
  });

  it('omits the EFFECTIVE HIT RATE footer when effectiveDemandHitRate is undefined', () => {
    const html = renderCacheContent(
      makeGlobalStats(),
      makeFullCacheMetrics({ effectiveDemandHitRate: undefined })
    );

    expect(html).not.toContain('luxar-cache-total__demand');
    expect(html).not.toContain('EFFECTIVE HIT RATE');
  });

  // Regression: the L2 HIT RATE tooltip contains a literal double-quote
  // (`"—"`). It is interpolated into `title="${tooltip}"` in
  // renderCacheSection, so an unescaped quote closes the attribute early
  // and truncates the tooltip (the tail leaks as bogus DOM attributes).
  // Parse the rendered HTML and read the *effective* title the browser
  // sees — pre-fix it stops at "...network download." and the closing
  // phrase is lost.
  it('keeps the full L2 hit-rate tooltip intact despite its embedded quotes', () => {
    const html = renderCacheContent(makeGlobalStats(), makeFullCacheMetrics());
    const host = document.createElement('div');
    host.innerHTML = html;
    // The field renders twice (compact header summary + full metric
    // card); target the card copy for the tooltip check.
    const valueEl = host.querySelector('.luxar-metric-card__value[data-field="l2-hitrate"]');
    expect(valueEl).not.toBeNull();
    const card = valueEl!.closest('.luxar-metric-card') as HTMLElement;
    expect(card).not.toBeNull();
    // The whole tooltip survives, including the quoted em-dash and the
    // trailing clause that a broken attribute would have dropped.
    expect(card.title).toContain('An L2 miss is the only case that costs a network download.');
    expect(card.title).toContain('"—" = nothing has fallen through to L2 yet');
    // And the raw HTML must carry the escaped quote, never a bare one
    // inside the attribute value.
    expect(html).toContain('&quot;—&quot; = nothing has fallen through to L2 yet');
  });

  it('surfaces OPFS read gate activity and cancellations', () => {
    const html = renderCacheContent(
      makeGlobalStats(),
      makeFullCacheMetrics({
        l2: {
          size: 1,
          count: 1,
          reads: 2,
          writes: 3,
          misses: 4,
          canceledReads: 5,
          activeReads: 6,
          queuedReads: 7,
        },
      })
    );

    expect(html).toContain('5 canceled');
    expect(html).toContain('6 active · 7 queued');
  });

  // Collapsible sections: each cache section can collapse to a compact
  // one-line header summary that mirrors every metric via the SAME
  // data-field keys as the full cards, so the per-tick patcher keeps
  // both views current with a single pass.
  describe('collapsible sections', () => {
    function renderFull(collapsed?: ReadonlySet<string>): HTMLElement {
      const html = renderCacheContent(
        makeGlobalStats(),
        makeFullCacheMetrics({
          slice: { size: 1024, count: 3, hits: 5, misses: 2, evictions: 1, hitRate: 0.71 },
        }),
        collapsed
      );
      const host = document.createElement('div');
      host.innerHTML = html;
      return host;
    }

    it('collapses every section by default (compact resting state)', () => {
      const host = renderFull();
      const sections = host.querySelectorAll('.luxar-cache-section');
      expect(sections.length).toBe(4); // slice + l0 + l1 + l2
      sections.forEach((s) => {
        expect(s.classList.contains('luxar-cache-section--collapsed')).toBe(true);
      });
    });

    it('respects an explicit collapsed-set (expanded sections lack the modifier)', () => {
      const host = renderFull(new Set(['l1']));
      const l1 = host.querySelector('.luxar-cache-section[data-section="l1"]');
      const l0 = host.querySelector('.luxar-cache-section[data-section="l0"]');
      expect(l1?.classList.contains('luxar-cache-section--collapsed')).toBe(true);
      expect(l0?.classList.contains('luxar-cache-section--collapsed')).toBe(false);
    });

    it('every section header carries the toggle action + its section key', () => {
      const host = renderFull();
      for (const key of CACHE_SECTION_KEYS) {
        const header = host.querySelector(
          `.luxar-cache-section[data-section="${key}"] .luxar-cache-section__header`
        ) as HTMLElement | null;
        expect(header).not.toBeNull();
        expect(header!.dataset.action).toBe('toggleCacheSection');
        expect(header!.dataset.sectionKey).toBe(key);
      }
    });

    it('duplicates every metric data-field into the compact summary (value + hit-rate sub)', () => {
      const host = renderFull();
      const fields = [
        's-size',
        's-hitrate',
        's-evictions',
        'l0-size',
        'l0-hitrate',
        'l0-evictions',
        'l1-size',
        'l1-hitrate',
        'l1-evictions',
        'l2-size',
        'l2-hitrate',
        'l2-io',
        'l2-errors',
      ];
      for (const field of fields) {
        // Every VALUE renders twice: compact summary span + full metric
        // card. Subtitles ride along in the summary only for hit-rate
        // metrics (the hits·miss split is live data); the others would
        // just truncate at one-line width and stay on the cards/tooltips.
        expect(host.querySelectorAll(`[data-field="${field}"]`).length).toBe(2);
        const expectedSubCopies = field.includes('hitrate') ? 2 : 1;
        expect(host.querySelectorAll(`[data-field="${field}-sub"]`).length).toBe(expectedSubCopies);
        expect(
          host.querySelector(`.luxar-cache-section__summary [data-field="${field}"]`)
        ).not.toBeNull();
      }
    });

    it('summary values render the same text and color class as the cards', () => {
      const host = renderFull();
      const summaryVal = host.querySelector(
        '.luxar-cache-section__summary [data-field="l0-hitrate"]'
      ) as HTMLElement;
      const cardVal = host.querySelector(
        '.luxar-metric-card__value[data-field="l0-hitrate"]'
      ) as HTMLElement;
      expect(summaryVal.textContent?.trim()).toBe(cardVal.textContent?.trim());
      const colorOf = (el: HTMLElement) =>
        Array.from(el.classList).find((c) => c.startsWith('luxar-color--'));
      expect(colorOf(summaryVal)).toBe(colorOf(cardVal));
    });
  });
});
