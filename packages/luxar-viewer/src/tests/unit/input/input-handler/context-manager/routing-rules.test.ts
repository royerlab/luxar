/**
 * Unit tests for the pure context-routing helpers.
 *
 * No DOM, no keyboard, no manager — just the allow/block filter and the
 * priority sort, exercised against synthetic configs.
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import {
  isKeyAllowedInContext,
  sortContextsByPriority,
  type KeyFilterConfig,
  type PriorityConfig,
} from '../../../../../input/input-handler/context-manager/routing-rules';

describe('isKeyAllowedInContext', () => {
  it('allows any key when neither filter is set', () => {
    expect(isKeyAllowedInContext('a', {})).toBe(true);
    expect(isKeyAllowedInContext('Escape', {})).toBe(true);
    expect(isKeyAllowedInContext('', {})).toBe(true);
  });

  it('rejects keys present in blockedKeys', () => {
    const cfg: KeyFilterConfig = { blockedKeys: ['w', 'a', 's', 'd'] };
    expect(isKeyAllowedInContext('w', cfg)).toBe(false);
    expect(isKeyAllowedInContext('a', cfg)).toBe(false);
    expect(isKeyAllowedInContext('e', cfg)).toBe(true);
  });

  it('only allows keys present in allowedKeys when allowedKeys is set', () => {
    const cfg: KeyFilterConfig = { allowedKeys: ['[', ']'] };
    expect(isKeyAllowedInContext('[', cfg)).toBe(true);
    expect(isKeyAllowedInContext(']', cfg)).toBe(true);
    expect(isKeyAllowedInContext('w', cfg)).toBe(false);
  });

  it('treats an empty allowedKeys array as "block everything"', () => {
    expect(isKeyAllowedInContext('w', { allowedKeys: [] })).toBe(false);
    expect(isKeyAllowedInContext('Escape', { allowedKeys: [] })).toBe(false);
  });

  it('blockedKeys takes precedence over allowedKeys when both list the key', () => {
    const cfg: KeyFilterConfig = {
      allowedKeys: ['x', 'y'],
      blockedKeys: ['x'],
    };
    expect(isKeyAllowedInContext('x', cfg)).toBe(false);
    expect(isKeyAllowedInContext('y', cfg)).toBe(true);
  });

  it('rejects keys not in allowedKeys even if blockedKeys does not list them', () => {
    const cfg: KeyFilterConfig = {
      allowedKeys: ['x'],
      blockedKeys: ['z'],
    };
    expect(isKeyAllowedInContext('y', cfg)).toBe(false);
  });
});

describe('sortContextsByPriority', () => {
  it('returns an empty array for an empty map', () => {
    expect(sortContextsByPriority(new Map<string, PriorityConfig>())).toEqual([]);
  });

  it('orders entries by descending priority', () => {
    const m = new Map<string, PriorityConfig>([
      ['low', { priority: 1 }],
      ['high', { priority: 100 }],
      ['mid', { priority: 50 }],
    ]);
    const sorted = sortContextsByPriority(m).map(([k]) => k);
    expect(sorted).toEqual(['high', 'mid', 'low']);
  });

  it('treats missing priority as 0', () => {
    const m = new Map<string, PriorityConfig>([
      ['no-priority', {}],
      ['high', { priority: 10 }],
      ['negative', { priority: -5 }],
    ]);
    const sorted = sortContextsByPriority(m).map(([k]) => k);
    expect(sorted).toEqual(['high', 'no-priority', 'negative']);
  });

  it('preserves insertion order among equal priorities (stable sort)', () => {
    const m = new Map<string, PriorityConfig>([
      ['a', { priority: 1 }],
      ['b', { priority: 1 }],
      ['c', { priority: 1 }],
    ]);
    const sorted = sortContextsByPriority(m).map(([k]) => k);
    expect(sorted).toEqual(['a', 'b', 'c']);
  });

  it('excludes the currently-active context when one is supplied', () => {
    const m = new Map<string, PriorityConfig>([
      ['nav', { priority: 10 }],
      ['fly', { priority: 5 }],
      ['modal', { priority: 100 }],
    ]);
    const sorted = sortContextsByPriority(m, 'modal').map(([k]) => k);
    expect(sorted).toEqual(['nav', 'fly']);
  });

  it('passes through every entry when the excluded key is not in the map', () => {
    const m = new Map<string, PriorityConfig>([
      ['nav', { priority: 10 }],
      ['fly', { priority: 5 }],
    ]);
    const sorted = sortContextsByPriority(m, 'absent').map(([k]) => k);
    expect(sorted).toEqual(['nav', 'fly']);
  });

  // input.md [H3][P12] fast-check property test: sortContextsByPriority
  // must be idempotent (sorting an already-sorted list yields the same
  // order) and stable (ties preserve insertion order). Without these
  // pinned, a future refactor that swapped to an unstable sort would
  // silently re-order ties and surface only as a UX glitch.
  it('[property] idempotent: sort(sort(x)) === sort(x)', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(fc.string({ minLength: 1, maxLength: 5 }), fc.integer({ min: -100, max: 100 })),
          { minLength: 0, maxLength: 20 }
        ),
        (entries) => {
          // De-duplicate keys (Map collapses duplicates by key); preserve insertion order.
          const seen = new Set<string>();
          const unique = entries.filter(([k]) => !seen.has(k) && (seen.add(k), true));
          const m = new Map<string, PriorityConfig>(unique.map(([k, p]) => [k, { priority: p }]));
          const once = sortContextsByPriority(m);
          const twice = sortContextsByPriority(new Map(once));
          if (once.length !== twice.length) return false;
          for (let i = 0; i < once.length; i++) {
            if (once[i][0] !== twice[i][0]) return false;
          }
          return true;
        }
      ),
      { numRuns: 200 }
    );
  });

  it('[property] stable: ties preserve insertion order', () => {
    fc.assert(
      fc.property(
        // Generate a list of unique keys, all with priority=42 (all-tie).
        fc.uniqueArray(fc.string({ minLength: 1, maxLength: 5 }), {
          minLength: 1,
          maxLength: 12,
        }),
        (keys) => {
          const m = new Map<string, PriorityConfig>(keys.map((k) => [k, { priority: 42 }]));
          const sorted = sortContextsByPriority(m).map(([k]) => k);
          // Stable sort over all-equal priorities = insertion order.
          for (let i = 0; i < keys.length; i++) {
            if (sorted[i] !== keys[i]) return false;
          }
          return true;
        }
      ),
      { numRuns: 200 }
    );
  });

  it('[property] descending priority: i-th result has priority >= (i+1)-th', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(fc.string({ minLength: 1, maxLength: 5 }), fc.integer({ min: -100, max: 100 })),
          { minLength: 0, maxLength: 20 }
        ),
        (entries) => {
          const seen = new Set<string>();
          const unique = entries.filter(([k]) => !seen.has(k) && (seen.add(k), true));
          const m = new Map<string, PriorityConfig>(unique.map(([k, p]) => [k, { priority: p }]));
          const sorted = sortContextsByPriority(m);
          for (let i = 1; i < sorted.length; i++) {
            const prev = sorted[i - 1][1].priority ?? 0;
            const cur = sorted[i][1].priority ?? 0;
            if (prev < cur) return false;
          }
          return true;
        }
      ),
      { numRuns: 200 }
    );
  });
});
