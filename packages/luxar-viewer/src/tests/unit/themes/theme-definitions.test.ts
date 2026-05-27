/**
 * Theme-definition tests [themes.md/G1,G2,G3,G4][P1][P10]
 *
 * Themes.md lists four theme-definition modules as orphans (no direct test):
 *   - dark.theme.ts
 *   - light.theme.ts
 *   - frosted-glass.theme.ts (the DEFAULT fallback)
 *   - liquid-glass.theme.ts
 *
 * This file pins the structural contract every theme must satisfy:
 *   - kebab-case `id`
 *   - human-readable `name`
 *   - canonical palette sub-objects (background/text/semantic/interactive/
 *     border/visualization) with their required keys
 *   - typography / spacing / effects / zIndex sub-objects with their canonical
 *     keys
 *   - no two themes share an `id`
 *
 * The themes are plain data; a regression that drops or renames any of these
 * fields would silently break the ThemeManager's `--luxar-*` CSS variable
 * injection. Closing these orphans here is cheaper than running mutation
 * testing against every theme literal.
 */

import { describe, it, expect } from 'vitest';
import {
  darkTheme,
  lightTheme,
  frostedGlassTheme,
  liquidGlassTheme,
  type Theme,
} from '../../../themes';

const ALL_THEMES: Array<{ name: string; theme: Theme; expectedId: string }> = [
  { name: 'dark', theme: darkTheme, expectedId: 'dark' },
  { name: 'light', theme: lightTheme, expectedId: 'light' },
  { name: 'frosted-glass', theme: frostedGlassTheme, expectedId: 'frosted-glass' },
  { name: 'liquid-glass', theme: liquidGlassTheme, expectedId: 'liquid-glass' },
];

const KEBAB_CASE = /^[a-z]+(-[a-z]+)*$/;

describe('Theme definitions [themes.md/G1-G4][P1]', () => {
  describe.each(ALL_THEMES)('$name theme', ({ theme, expectedId }) => {
    it('exposes the expected id (kebab-case)', () => {
      expect(theme.id).toBe(expectedId);
      expect(theme.id).toMatch(KEBAB_CASE);
    });

    it('has a non-empty human-readable name', () => {
      expect(typeof theme.name).toBe('string');
      expect(theme.name.length).toBeGreaterThan(0);
      // Names should not be the raw id (UI-visible label).
      expect(theme.name).not.toBe(theme.id);
    });

    it('declares the canonical color sub-objects', () => {
      expect(Object.keys(theme.colors).sort()).toEqual([
        'background',
        'border',
        'interactive',
        'semantic',
        'text',
        'visualization',
      ]);
    });

    it('background palette has primary/secondary/tertiary/overlay (all four distinct)', () => {
      const bg = theme.colors.background;
      expect(typeof bg.primary).toBe('string');
      expect(typeof bg.secondary).toBe('string');
      expect(typeof bg.tertiary).toBe('string');
      expect(typeof bg.overlay).toBe('string');
      // No empty strings sneaking in.
      for (const v of [bg.primary, bg.secondary, bg.tertiary, bg.overlay]) {
        expect(v.length).toBeGreaterThan(0);
      }
      // themes.md G11: a degenerate theme where all four backgrounds collapse
      // to the same value would pass typeof/length checks but render flat. The
      // four background layers must remain pairwise distinct.
      expect(new Set([bg.primary, bg.secondary, bg.tertiary, bg.overlay]).size).toBe(4);
    });

    it('text palette has primary/secondary/muted', () => {
      const t = theme.colors.text;
      expect(typeof t.primary).toBe('string');
      expect(typeof t.secondary).toBe('string');
      expect(typeof t.muted).toBe('string');
      // primary and secondary must differ — degenerate themes get caught here.
      expect(t.primary).not.toBe(t.secondary);
    });

    it('semantic palette has success/warning/error/info', () => {
      const s = theme.colors.semantic;
      expect(typeof s.success).toBe('string');
      expect(typeof s.warning).toBe('string');
      expect(typeof s.error).toBe('string');
      expect(typeof s.info).toBe('string');
    });

    it('interactive palette has default/hover/active/focus/disabled', () => {
      const i = theme.colors.interactive;
      expect(typeof i.default).toBe('string');
      expect(typeof i.hover).toBe('string');
      expect(typeof i.active).toBe('string');
      expect(typeof i.focus).toBe('string');
      expect(typeof i.disabled).toBe('string');
    });

    it('border palette has default/subtle/strong', () => {
      const b = theme.colors.border;
      expect(typeof b.default).toBe('string');
      expect(typeof b.subtle).toBe('string');
      expect(typeof b.strong).toBe('string');
    });

    it('visualization palette has hot/warm/cold/neutral', () => {
      const v = theme.colors.visualization;
      expect(typeof v.hot).toBe('string');
      expect(typeof v.warm).toBe('string');
      expect(typeof v.cold).toBe('string');
      expect(typeof v.neutral).toBe('string');
    });

    it('typography exposes fontFamily/fontSize/fontWeight/lineHeight', () => {
      expect(Object.keys(theme.typography).sort()).toEqual([
        'fontFamily',
        'fontSize',
        'fontWeight',
        'lineHeight',
      ]);
      expect(typeof theme.typography.fontFamily.base).toBe('string');
      expect(typeof theme.typography.fontFamily.mono).toBe('string');
      expect(typeof theme.typography.fontSize.base).toBe('string');
      expect(typeof theme.typography.fontWeight.normal).toBe('number');
      expect(typeof theme.typography.lineHeight.normal).toBe('number');
    });

    it('spacing scale has the 8-px grid keys (0,1,2,3,4,5,6,8,10,12,16,20)', () => {
      const expected = ['0', '1', '2', '3', '4', '5', '6', '8', '10', '12', '16', '20'];
      const actual = Object.keys(theme.spacing).sort(
        (a, b) => Number(a) - Number(b)
      );
      expect(actual).toEqual(expected);
      const spacingRecord = theme.spacing as unknown as Record<string, string>;
      for (const k of expected) {
        expect(typeof spacingRecord[k]).toBe('string');
      }
    });

    it('effects has borderRadius/shadow/blur/opacity/transition', () => {
      expect(Object.keys(theme.effects).sort()).toEqual([
        'blur',
        'borderRadius',
        'opacity',
        'shadow',
        'transition',
      ]);
      // borderRadius canonical keys
      expect(Object.keys(theme.effects.borderRadius).sort()).toEqual([
        'full',
        'lg',
        'md',
        'none',
        'sm',
      ]);
      // shadow canonical keys
      expect(Object.keys(theme.effects.shadow).sort()).toEqual(['lg', 'md', 'sm', 'xl']);
    });

    it('zIndex exposes the canonical 5 layers in monotonic order', () => {
      const z = theme.zIndex;
      expect(typeof z.base).toBe('number');
      expect(typeof z.dropdown).toBe('number');
      expect(typeof z.modal).toBe('number');
      expect(typeof z.popover).toBe('number');
      expect(typeof z.tooltip).toBe('number');
      // Stacking order must increase strictly.
      expect(z.base).toBeLessThan(z.dropdown);
      expect(z.dropdown).toBeLessThan(z.modal);
      expect(z.modal).toBeLessThan(z.popover);
      expect(z.popover).toBeLessThan(z.tooltip);
    });
  });

  describe('cross-theme invariants', () => {
    it('no two themes share an id', () => {
      const ids = ALL_THEMES.map(({ theme }) => theme.id);
      expect(new Set(ids).size).toBe(ids.length);
    });

    it('no two themes share a display name', () => {
      const names = ALL_THEMES.map(({ theme }) => theme.name);
      expect(new Set(names).size).toBe(names.length);
    });

    it('every theme uses the same font-size scale keys (xs..3xl)', () => {
      const expected = ['2xl', '3xl', 'base', 'lg', 'md', 'sm', 'xl', 'xs'];
      for (const { theme } of ALL_THEMES) {
        expect(Object.keys(theme.typography.fontSize).sort()).toEqual(expected);
      }
    });

    it('every theme uses the same fontWeight keys (normal..bold)', () => {
      const expected = ['bold', 'medium', 'normal', 'semibold'];
      for (const { theme } of ALL_THEMES) {
        expect(Object.keys(theme.typography.fontWeight).sort()).toEqual(expected);
      }
    });
  });
});
