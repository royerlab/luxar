/**
 * WCAG AA contrast pins for the two glass themes (issue #1513).
 *
 * `frosted-glass` (default theme) and `liquid-glass` both paint text over a
 * translucent DARK panel tint, so the panel's actual on-screen color depends
 * on whatever scene the user has loaded behind it — a value we don't
 * control. The worst case a scene can produce is pure white. Pre-fix, over
 * that backdrop:
 *   frosted-glass composited to #6b6d71, giving primary/secondary/muted
 *     4.88:1 / 3.73:1 / 2.56:1 — secondary and muted below WCAG AA's 4.5:1
 *     floor for normal text (muted, used for meaning-bearing icons, was
 *     also below the 3:1 non-text floor).
 *   liquid-glass's `::after` layer painted its tint at an effective 0.385
 *     alpha (a 0.55 tint further multiplied by an `opacity: 0.7` on the
 *     same rule — see readLiquidGlassEffectiveTint below), compositing to
 *     rgb(157) and giving primary/secondary/muted 2.61:1 / 2.19:1 / 1.73:1
 *     — ALL THREE below AA, not just secondary/muted.
 *
 * The fix: frosted-glass's `background.secondary` tint moved 0.65 -> 0.75;
 * liquid-glass's tint moved 0.55 -> 0.68 AND its `::after` layer's
 * `opacity: 0.7` multiplier was deleted so the tint's own alpha is what's
 * painted; BOTH themes' `text.secondary` / `text.muted` alphas moved
 * 0.75 -> 0.85 / 0.5 -> 0.72. Measured over the worst-case pure-white
 * composite, the new numbers are:
 *   frosted-glass: primary 6.8:1, secondary 5.8:1, muted 4.7:1
 *   liquid-glass:  primary 7.3:1, secondary 6.2:1, muted 5.0:1
 * All clear AA; this file pins those numbers so a future edit to either
 * theme's tint or text alphas can't silently regress below 4.5:1 again.
 *
 * liquid-glass's panel tint isn't a theme token at all — it's the
 * `--luxar-glass-tint` CSS custom property declared in liquid-glass.css and
 * painted by `.luxar-glass-surface::after` (the theme's own
 * `background.secondary` token is a translucent WHITE, used elsewhere, not
 * what text sits on). This test reads that CSS variable's declared value
 * AND the `::after` rule's `opacity` straight out of the stylesheet —
 * rather than hardcoding a tint here — so a re-introduced opacity
 * multiplier (the regression this test guards against — the bug itself was
 * already caught in review, before this test existed) fails the test
 * instead of silently drifting from what's rendered.
 *
 * text.disabled is deliberately NOT asserted: it is a tracked gap (UI_DESIGN_GUIDE.md
 * §15.2), not a WCAG-exempt case — several of its live uses are metric
 * captions and status labels, not disabled controls, so WCAG 1.4.3's
 * disabled-control exemption does not actually cover them.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { frostedGlassTheme } from '../../../themes/themes/frosted-glass.theme';
import { liquidGlassTheme } from '../../../themes/themes/liquid-glass.theme';
import { stripComments, ruleBody } from '../styles/_helpers/css-text';

const HERE = dirname(fileURLToPath(import.meta.url));
const LIQUID_GLASS_CSS_PATH = resolve(HERE, '../../../styles/themes/liquid-glass.css');

interface Rgba {
  r: number; // 0-255
  g: number; // 0-255
  b: number; // 0-255
  a: number; // 0-1
}

/** Parse a `rgba(r, g, b, a)` (or `rgb(r, g, b)`, alpha defaults to 1) string. */
function parseRgba(css: string): Rgba {
  const m = css.match(/rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)/);
  if (!m) {
    throw new Error(`Could not parse rgba() color from: "${css}"`);
  }
  return {
    r: Number(m[1]),
    g: Number(m[2]),
    b: Number(m[3]),
    a: m[4] === undefined ? 1 : Number(m[4]),
  };
}

/** Alpha-composite `fg` over an opaque `bg` (bg.a is ignored), "over" operator. */
function compositeOver(fg: Rgba, bg: Rgba): Rgba {
  return {
    r: fg.a * fg.r + (1 - fg.a) * bg.r,
    g: fg.a * fg.g + (1 - fg.a) * bg.g,
    b: fg.a * fg.b + (1 - fg.a) * bg.b,
    a: 1,
  };
}

/** WCAG sRGB channel linearization (0-255 input -> linear 0-1). */
function linearizeChannel(c255: number): number {
  const c = c255 / 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** WCAG relative luminance of an opaque color. */
function relativeLuminance(color: Rgba): number {
  const r = linearizeChannel(color.r);
  const g = linearizeChannel(color.g);
  const b = linearizeChannel(color.b);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG contrast ratio between two opaque colors, order-independent. */
function contrastRatio(a: Rgba, b: Rgba): number {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  const lighter = Math.max(la, lb);
  const darker = Math.min(la, lb);
  return (lighter + 0.05) / (darker + 0.05);
}

const WHITE: Rgba = { r: 255, g: 255, b: 255, a: 1 };
const BLACK: Rgba = { r: 0, g: 0, b: 0, a: 1 };

const AA_NORMAL_TEXT = 4.5;

const AFTER_SELECTOR = "[data-theme='liquid-glass'] .luxar-glass-surface::after";
const SURFACE_SELECTOR = "[data-theme='liquid-glass'] .luxar-glass-surface";

/**
 * Count how many times `selector` opens a rule in `css` (comment-stripped).
 *
 * `ruleBody` (below/`_helpers/css-text.ts`) returns only the FIRST rule
 * matching a selector, but CSS applies the LAST — so a plain duplicate rule
 * appended later in the file (no `@media`/`@supports` needed) would
 * silently win at render time while this test kept reading the earlier,
 * unrelated one. Used to fail closed on that case instead of assuming
 * `ruleBody`'s first match is the only one.
 */
function countRuleOccurrences(css: string, selector: string): number {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`(^|[^\\w-])${escaped}\\s*\\{`, 'gm');
  return (css.match(re) ?? []).length;
}

/**
 * Find every declaration in a CSS rule body whose property name matches
 * `propName` exactly (case-insensitive), in source order.
 *
 * Splits on `;` and compares the exact declaration name — NOT a custom
 * property like `--something-opacity` and NOT a differently-hyphenated
 * longhand like `backdrop-filter` (a different property from `filter`),
 * both of which are different declarations entirely. `ruleBody` already
 * strips the surrounding `{`/`}`, so the final declaration needs no
 * trailing `;` either — exactly the spelling the original #1513 bug used.
 */
function findDeclarations(body: string, propName: string): string[] {
  const values: string[] = [];
  for (const decl of body.split(';')) {
    const colon = decl.indexOf(':');
    if (colon === -1) continue;
    if (decl.slice(0, colon).trim().toLowerCase() === propName) {
      values.push(decl.slice(colon + 1).trim());
    }
  }
  return values;
}

/**
 * Find the `opacity` declaration (if any) in a CSS rule body, returning its
 * raw (unparsed) value, or `null` if the rule has no `opacity` declaration
 * at all.
 *
 * Takes the LAST matching declaration, not the first: CSS resolves a
 * property declared more than once in the same rule body to whichever
 * occurrence comes last (e.g. `opacity: 1; opacity: 0.7;` renders at 0.7),
 * so reading the first would silently miss a re-declaration that overrides
 * an earlier, harmless-looking one.
 */
function findOpacityDeclaration(body: string): string | null {
  const decls = findDeclarations(body, 'opacity');
  return decls.length > 0 ? decls[decls.length - 1] : null;
}

/**
 * Parse an `opacity` declaration's value strictly into a 0-1 multiplier.
 * Handles a plain number (`0.7`), a percentage (`70%`), and an optional
 * trailing `!important` on either. Anything else — a `var()`, a keyword
 * (`inherit`/`initial`/`unset`), or any other unparseable value — throws
 * rather than silently defaulting to "no multiplier": that default is only
 * legitimate when there is NO `opacity` declaration at all, which is
 * `findOpacityDeclaration` returning `null`, not a value this function
 * can't parse.
 */
function parseOpacityValue(rawValue: string): number {
  const value = rawValue.replace(/!\s*important\s*$/i, '').trim();
  const percent = value.match(/^([\d.]+)%$/);
  if (percent) {
    return Number(percent[1]) / 100;
  }
  const plain = value.match(/^([\d.]+)$/);
  if (plain) {
    return Number(plain[1]);
  }
  throw new Error(
    `Unrecognized "opacity" value "${rawValue}" on the ${AFTER_SELECTOR} rule ` +
      `in ${LIQUID_GLASS_CSS_PATH}. This parser only understands a plain number ` +
      '("0.7") or a percentage ("70%"), each optionally followed by ' +
      '"!important" — teach parseOpacityValue about this new spelling rather ' +
      'than silently treating it as "no multiplier" (that is the exact class ' +
      'of bug this test exists to catch).'
  );
}

/**
 * Read the tint that `.luxar-glass-surface::after` actually PAINTS —
 * `--luxar-glass-tint`'s declared alpha multiplied by that rule's own
 * `opacity` (default 1 if — and ONLY if — the rule has no `opacity`
 * declaration at all; see `findOpacityDeclaration`/`parseOpacityValue`).
 *
 * Element `opacity` multiplies a whole layer, tint included, so reading the
 * custom property alone is structurally blind to it: an earlier pass of
 * this same fix raised the tint to 0.65 while the same `::after` rule still
 * carried its old `opacity: 0.7`, so only 0.65 x 0.7 = 0.455 ever reached
 * the compositor — a real regression a bare-variable version of this test
 * would have stayed green through (and would stay green if an opacity
 * multiplier were reintroduced later). So this reads the rule body itself,
 * via the shared `stripComments`/`ruleBody` helpers (comment-blind regexes
 * are exactly what let the multiplier hide), and multiplies the two
 * together — failing CLOSED (throwing) on any spelling of `opacity` it
 * can't confidently parse to 1, rather than failing OPEN (silently treating
 * an unparsed declaration as absent). Handles `opacity: 0.7;`,
 * `opacity: 0.7 !important;`, `opacity: 70%;`, and a bare `opacity: 0.7`
 * with no trailing `;` (the original bug's exact spelling, as the rule's
 * last declaration).
 *
 * Covered after the round-3 hardening (all fail CLOSED, i.e. throw rather
 * than silently pass):
 *   - a plain duplicate `::after` rule anywhere else in the file (not just
 *     inside an `@media`/`@supports` block) — `countRuleOccurrences` demands
 *     exactly one match;
 *   - `opacity` declared more than once in the rule body — the LAST
 *     declaration wins, matching the CSS cascade, not the first;
 *   - a `filter` declaration on the `::after` rule (e.g. `filter:
 *     opacity(70%)` is an exact synonym for the deleted #1513 bug) or a
 *     `mix-blend-mode` declaration (changes how the tint composites in ways
 *     this test's plain alpha-over-white model can't account for);
 *   - an `opacity` on the PARENT `.luxar-glass-surface` rule, which would
 *     multiply the same tint layer a second time on top of anything on
 *     `::after` itself.
 *
 * Known gap: none of the above reaches inside an `@media` or `@supports`
 * block — `ruleBody`/`countRuleOccurrences` are selector-text scans, not
 * full CSS parsers, so any of `opacity`/`filter`/`mix-blend-mode`
 * re-declared for the same selector inside a conditional block is still
 * invisible to this function. Closing that would mean teaching the shared
 * `ruleBody` helper (used by other style tests too) to parse and merge
 * conditional blocks, which is more risk than this fix's scope justifies —
 * flagging it here instead of claiming a guarantee this function doesn't
 * give.
 */
function readLiquidGlassEffectiveTint(): Rgba {
  const raw = readFileSync(LIQUID_GLASS_CSS_PATH, 'utf8');
  const css = stripComments(raw);

  const m = css.match(/--luxar-glass-tint:\s*([^;]+);/);
  if (!m) {
    throw new Error(
      `--luxar-glass-tint declaration not found in ${LIQUID_GLASS_CSS_PATH}. ` +
        'This test reads the panel tint straight out of the stylesheet so it ' +
        'cannot drift from what the CSS actually paints — if the variable was ' +
        'renamed or removed, update this test alongside it.'
    );
  }
  const tint = parseRgba(m[1]);

  const afterOccurrences = countRuleOccurrences(css, AFTER_SELECTOR);
  if (afterOccurrences !== 1) {
    throw new Error(
      `Expected exactly one "${AFTER_SELECTOR}" rule in ${LIQUID_GLASS_CSS_PATH}, ` +
        `found ${afterOccurrences}. ruleBody() reads only the FIRST rule matching ` +
        'this selector, but CSS applies the LAST — a plain duplicate rule ' +
        '(appended later in the file, no `@media`/`@supports` needed) would ' +
        'silently win at render time while this test kept reading the earlier ' +
        'one. Merge or remove the duplicate, or update this test if a ' +
        'legitimate reason to split the rule ever appears.'
    );
  }

  const body = ruleBody(css, AFTER_SELECTOR);
  if (!body) {
    throw new Error(
      `Could not find the rule "${AFTER_SELECTOR}" in ${LIQUID_GLASS_CSS_PATH}. ` +
        "This test reads that rule's `opacity` (if any) so an element-level " +
        'opacity multiplier on the tint layer cannot silently escape ' +
        'detection again (issue #1513) — if the selector changed, update this ' +
        'test alongside it.'
    );
  }
  const backgroundMatch = body.match(/background:\s*var\(--luxar-glass-tint\)/);
  if (!backgroundMatch) {
    throw new Error(
      `The rule "${AFTER_SELECTOR}" in ${LIQUID_GLASS_CSS_PATH} no longer paints ` +
        '`background: var(--luxar-glass-tint)` — update this test alongside it.'
    );
  }
  if (findDeclarations(body, 'filter').length > 0) {
    throw new Error(
      `The rule "${AFTER_SELECTOR}" in ${LIQUID_GLASS_CSS_PATH} declares a ` +
        '`filter` — e.g. `filter: opacity(70%)` is an exact synonym for the ' +
        'deleted #1513 `opacity` bug and would silently divide this tint again ' +
        'without this test noticing. Remove it, or teach this test about the ' +
        'new spelling if it is ever legitimately needed.'
    );
  }
  if (findDeclarations(body, 'mix-blend-mode').length > 0) {
    throw new Error(
      `The rule "${AFTER_SELECTOR}" in ${LIQUID_GLASS_CSS_PATH} declares a ` +
        "`mix-blend-mode` — that can change how the tint composites onto what's " +
        "beneath it in ways this test's plain alpha-over-white model does not " +
        'account for. Remove it, or teach this test about it if it is ever ' +
        'legitimately needed.'
    );
  }

  const surfaceBody = ruleBody(css, SURFACE_SELECTOR);
  if (findDeclarations(surfaceBody, 'opacity').length > 0) {
    throw new Error(
      `The parent rule "${SURFACE_SELECTOR}" in ${LIQUID_GLASS_CSS_PATH} declares ` +
        'an `opacity` — an opacity on the surface root would multiply the same ' +
        'tint layer a second time (independently of anything on `::after` ' +
        'itself) and silently divide the contrast floor this test pins. Remove ' +
        'it, or teach this test about the new intent if it is ever legitimately ' +
        'needed.'
    );
  }

  const opacityDecl = findOpacityDeclaration(body);
  const opacity = opacityDecl === null ? 1 : parseOpacityValue(opacityDecl);

  return { ...tint, a: tint.a * opacity };
}

describe('glass-contrast helpers self-check', () => {
  it('black-on-white is 21:1 and the ratio is order-independent', () => {
    expect(contrastRatio(BLACK, WHITE)).toBeCloseTo(21, 1);
    expect(contrastRatio(WHITE, BLACK)).toBeCloseTo(21, 1);
  });

  it('composites a 50% white over black to mid-gray', () => {
    const result = compositeOver({ r: 255, g: 255, b: 255, a: 0.5 }, BLACK);
    expect(result.r).toBeCloseTo(127.5, 5);
    expect(result.g).toBeCloseTo(127.5, 5);
    expect(result.b).toBeCloseTo(127.5, 5);
  });
});

describe('frosted-glass worst-case (pure-white scene) contrast', () => {
  const panel = compositeOver(parseRgba(frostedGlassTheme.colors.background.secondary), WHITE);

  it('composites the panel tint to the expected worst-case gray', () => {
    // rgba(28, 30, 36, 0.75) over white
    expect(panel.r).toBeCloseTo(84.75, 1);
    expect(panel.g).toBeCloseTo(86.25, 1);
    expect(panel.b).toBeCloseTo(90.75, 1);
  });

  const primary = contrastRatio(
    compositeOver(parseRgba(frostedGlassTheme.colors.text.primary), panel),
    panel
  );
  const secondary = contrastRatio(
    compositeOver(parseRgba(frostedGlassTheme.colors.text.secondary), panel),
    panel
  );
  const muted = contrastRatio(
    compositeOver(parseRgba(frostedGlassTheme.colors.text.muted), panel),
    panel
  );

  it('text.primary clears AA (~6.8:1)', () => {
    expect(primary).toBeGreaterThanOrEqual(AA_NORMAL_TEXT);
    expect(primary).toBeCloseTo(6.8, 1);
  });

  it('text.secondary clears AA (~5.8:1)', () => {
    expect(secondary).toBeGreaterThanOrEqual(AA_NORMAL_TEXT);
    expect(secondary).toBeCloseTo(5.8, 1);
  });

  it('text.muted clears AA (~4.7:1)', () => {
    expect(muted).toBeGreaterThanOrEqual(AA_NORMAL_TEXT);
    expect(muted).toBeCloseTo(4.7, 1);
  });

  it('preserves the rank order: primary > secondary > muted', () => {
    expect(primary).toBeGreaterThan(secondary);
    expect(secondary).toBeGreaterThan(muted);
  });
});

describe('frosted-glass dark-scene direction (documented, not independently guarded)', () => {
  // Over a dark/black scene the panel tint is even darker, so this composite
  // is componentwise <= the pure-white one pinned above for every one of
  // these (light) text colors — the bright-scene case is the stricter one
  // for any text alpha in use here. An "all three ranks clear AA" check here
  // would therefore pass against any alphas that already clear the bright
  // case (it did against the PRE-#1513 values too), so it wouldn't guard
  // anything; this block exists only to document the numbers. Measured:
  // panel rgb(21, 22.5, 27), primary/secondary/muted 16.3:1 / 13.1:1 / 9.7:1.
  // The rank order is the one thing here that isn't implied by the
  // bright-scene pins, so that's the only assertion kept.
  const panel = compositeOver(parseRgba(frostedGlassTheme.colors.background.secondary), BLACK);

  const primary = contrastRatio(
    compositeOver(parseRgba(frostedGlassTheme.colors.text.primary), panel),
    panel
  );
  const secondary = contrastRatio(
    compositeOver(parseRgba(frostedGlassTheme.colors.text.secondary), panel),
    panel
  );
  const muted = contrastRatio(
    compositeOver(parseRgba(frostedGlassTheme.colors.text.muted), panel),
    panel
  );

  it('preserves the rank order', () => {
    expect(primary).toBeGreaterThan(secondary);
    expect(secondary).toBeGreaterThan(muted);
  });
});

describe('liquid-glass worst-case (pure-white scene) contrast', () => {
  const tint = readLiquidGlassEffectiveTint();
  const panel = compositeOver(tint, WHITE);

  it('reads the effective alpha as 0.68 (tint alpha x rule opacity)', () => {
    // Pins the multiplication itself, not just its downstream composite —
    // this is the exact number a reintroduced `opacity` on the ::after rule
    // would silently change (issue #1513).
    expect(tint.a).toBeCloseTo(0.68, 2);
  });

  it('composites the effective tint to the expected worst-case gray', () => {
    // rgba(0, 0, 0, 0.68) over white, opacity multiplier folded in already
    expect(panel.r).toBeCloseTo(81.6, 1);
    expect(panel.g).toBeCloseTo(81.6, 1);
    expect(panel.b).toBeCloseTo(81.6, 1);
  });

  const primary = contrastRatio(
    compositeOver(parseRgba(liquidGlassTheme.colors.text.primary), panel),
    panel
  );
  const secondary = contrastRatio(
    compositeOver(parseRgba(liquidGlassTheme.colors.text.secondary), panel),
    panel
  );
  const muted = contrastRatio(
    compositeOver(parseRgba(liquidGlassTheme.colors.text.muted), panel),
    panel
  );

  it('text.primary clears AA (~7.3:1)', () => {
    expect(primary).toBeGreaterThanOrEqual(AA_NORMAL_TEXT);
    expect(primary).toBeCloseTo(7.3, 1);
  });

  it('text.secondary clears AA (~6.2:1)', () => {
    expect(secondary).toBeGreaterThanOrEqual(AA_NORMAL_TEXT);
    expect(secondary).toBeCloseTo(6.2, 1);
  });

  it('text.muted clears AA (~5.0:1)', () => {
    expect(muted).toBeGreaterThanOrEqual(AA_NORMAL_TEXT);
    expect(muted).toBeCloseTo(5.0, 1);
  });

  it('preserves the rank order: primary > secondary > muted', () => {
    expect(primary).toBeGreaterThan(secondary);
    expect(secondary).toBeGreaterThan(muted);
  });
});

describe('liquid-glass dark-scene direction (documented, not independently guarded)', () => {
  // Same reasoning as the frosted-glass block above: the black-scene
  // composite is componentwise <= the pure-white one pinned above for this
  // (light) tint and text colors, so the bright-scene case is the stricter
  // one for any text alpha in use here — an "all three ranks clear AA" check
  // here would pass against any alphas that already clear the bright case
  // (it did against the PRE-#1513 values too) and so wouldn't guard
  // anything. Measured: panel rgb(0, 0, 0) (opaque black under the tint),
  // primary/secondary/muted 18.8:1 / 14.8:1 / 10.5:1. Only the rank order
  // isn't implied by the bright-scene pins.
  const tint = readLiquidGlassEffectiveTint();
  const panel = compositeOver(tint, BLACK);

  const primary = contrastRatio(
    compositeOver(parseRgba(liquidGlassTheme.colors.text.primary), panel),
    panel
  );
  const secondary = contrastRatio(
    compositeOver(parseRgba(liquidGlassTheme.colors.text.secondary), panel),
    panel
  );
  const muted = contrastRatio(
    compositeOver(parseRgba(liquidGlassTheme.colors.text.muted), panel),
    panel
  );

  it('preserves the rank order', () => {
    expect(primary).toBeGreaterThan(secondary);
    expect(secondary).toBeGreaterThan(muted);
  });
});
