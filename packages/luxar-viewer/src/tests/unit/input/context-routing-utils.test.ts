/**
 * Unit tests for the pure context-routing helpers.
 *
 * No DOM, no keyboard, no manager — just the allow/block filter and the
 * priority sort, exercised against synthetic configs.
 */

import { describe, it, expect } from 'vitest';
import {
  isKeyAllowedInContext,
  sortContextsByPriority,
  type KeyFilterConfig,
  type PriorityConfig,
} from '../../../input/context-routing-utils';

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
});
