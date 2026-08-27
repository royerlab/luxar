/**
 * CSS-shape guards for the two §5.1 glass-surface constraints that issue
 * #1483 fixed. Both were text-level facts about a stylesheet that no other
 * test could see, so they are pinned here by scanning the source directly
 * (same technique as `data-loading-monitor-css.test.ts`).
 *
 *   1. `.luxar-dimension-sliders` scrolled on its own glass root
 *      (`overflow-y: auto` + `max-height`), which clips the glass layers —
 *      they paint at negative z-index with `inset: 0` behind the root
 *      (§5.1.2). Scrolling now lives on a `__scroll` wrapper (§7.4), and the
 *      panel — a floating standing surface — sits on the `dropdown` tier
 *      rather than the baseline in-canvas-widget layer (§3.5).
 *   2. The toast keeps its `transition: opacity` dismiss fade, which is legal
 *      only because it is no longer a glass surface (§5.1.3). The DOM half of
 *      that pin — that the faded element carries no `luxar-glass-surface` —
 *      lives in `tests/unit/ui/toast.test.ts`; here we pin the readability
 *      consequence: liquid-glass must paint the dark tint the glass `::after`
 *      layer used to supply, since that theme's `--luxar-bg-secondary` is a
 *      translucent WHITE.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { stripComments, stripMediaQueries, ruleBody } from './_helpers/css-text';

const HERE = dirname(fileURLToPath(import.meta.url));
const STYLES = resolve(HERE, '../../../styles');

/** Read a stylesheet with comments and media queries stripped. */
function loadCss(relativePath: string): string {
  return stripMediaQueries(stripComments(readFileSync(resolve(STYLES, relativePath), 'utf8')));
}

describe('dimension-sliders.css — §5.1.2 scroll delegation', () => {
  const css = loadCss('components/dimension-sliders.css');
  // `ruleBody` is non-global: this is the BASE `.luxar-dimension-sliders` rule
  // only. A theme-scoped re-declaration (`[data-theme='light']
  // .luxar-dimension-sliders { overflow: hidden }`) would not be seen here, so
  // read the case titles below as "the base rule", not as unconditional.
  const root = ruleBody(css, '.luxar-dimension-sliders');

  it('has a .luxar-dimension-sliders root rule to inspect', () => {
    expect(root).not.toBe('');
  });

  // Any value that establishes a scroll container OR clips kills the glass
  // layers — §5.1.2 spells out `hidden` by name ("clips them dead"), and
  // `clip`/`scroll` are the same failure. Rejecting only auto/scroll would let
  // the canonical violation through.
  it.each(['overflow', 'overflow-x', 'overflow-y'])(
    'the glass root declares no clipping or scrolling %s',
    (property) => {
      expect(root).not.toMatch(new RegExp(`${property}:\\s*(auto|scroll|hidden|clip)`));
    }
  );

  it('the glass root declares overflow: visible explicitly (§5.1.2)', () => {
    expect(root).toMatch(/overflow:\s*visible/);
  });

  it('the root keeps a height bound of its own, which is what the wrapper caps against', () => {
    // Not a §5.1.2 claim, and not about the number (240px today): the shipped
    // §7.4 variant puts the single bound on the flex root and lets the
    // wrapper's `flex: 1; min-height: 0` cap against it.
    expect(root).toMatch(/max-height:\s*\S+/);
  });

  it('the root is the flex column that makes that bound reach the wrapper', () => {
    // The other half of the same mechanism, and the one nothing else catches:
    // without `display: flex; flex-direction: column` here, the wrapper's
    // `flex: 1; min-height: 0` is inert, so it grows to its content instead of
    // capping — and since the root is `overflow: visible`, the rows then paint
    // outside the panel. `ui/dimension-sliders.ts` guards the same property
    // against being overwritten inline; this guards the declaration itself.
    expect(root).toMatch(/display:\s*flex/);
    expect(root).toMatch(/flex-direction:\s*column/);
  });

  it('the floating panel sits on the dropdown tier, not the baseline layer (§3.5)', () => {
    expect(root).toMatch(/z-index:\s*var\(--luxar-z-dropdown\)/);
    expect(root).not.toMatch(/z-index:\s*var\(--luxar-z-base\)/);
  });

  it('a __scroll wrapper owns the scrolling (§7.4)', () => {
    const scroll = ruleBody(css, '.luxar-dimension-sliders__scroll');
    expect(scroll).not.toBe('');
    expect(scroll).toMatch(/overflow-y:\s*auto/);
    // flex:1 + min-height:0 is what makes the root's max-height the cap.
    expect(scroll).toMatch(/flex:\s*1/);
    expect(scroll).toMatch(/min-height:\s*0/);
    // Quiet scrollbar, never accent-colored (§6.5).
    expect(scroll).toMatch(/scrollbar-color:\s*var\(--luxar-border-strong\)\s+transparent/);
  });

  // A scroll container clips ink overflow at its PADDING edge, so the panel's
  // padding has to be inside the scroller: on the root it would cut the
  // :focus-visible ring off any descendant flush with the scrollport (the play
  // button, an outer-column dropdown) — WCAG 2.4.7. The two arrangements are
  // otherwise pixel-identical, so nothing else catches a "tidy-up" that moves
  // it back. Split in two so each half fails by name.
  it('the panel padding is declared on the __scroll wrapper', () => {
    expect(ruleBody(css, '.luxar-dimension-sliders__scroll')).toMatch(
      /padding:\s*var\(--luxar-spacing-\d+\)/
    );
  });

  it('the glass root declares no padding of its own', () => {
    expect(root).not.toMatch(/padding:/);
  });
});

describe('dimension-sliders.css — bounded single-line labels (#2188)', () => {
  const css = loadCss('components/dimension-sliders.css');

  it('keeps the panel height responsive while exposing common dimension counts', () => {
    expect(ruleBody(css, '.luxar-dimension-sliders')).toMatch(
      /max-height:\s*min\([^;]*\d+(?:\.\d+)?vh[^;]*\)/
    );
  });

  it('protects the title and ellipsizes the status on one line', () => {
    expect(ruleBody(css, '.luxar-dimension-sliders__header')).toMatch(
      /gap:\s*var\(--luxar-spacing-\d+\)/
    );

    const title = ruleBody(css, '.luxar-dimension-sliders__title');
    expect(title).toMatch(/flex:\s*0\s+0\s+auto/);
    expect(title).toMatch(/white-space:\s*nowrap/);

    const status = ruleBody(css, '.luxar-dimension-sliders__status');
    expect(status).toMatch(/min-width:\s*0/);
    expect(status).toMatch(/white-space:\s*nowrap/);
    expect(status).toMatch(/overflow:\s*hidden/);
    expect(status).toMatch(/text-overflow:\s*ellipsis/);
    expect(status).toMatch(/text-align:\s*right/);
  });

  it('protects slider names and ellipsizes long values', () => {
    const name = ruleBody(css, '.luxar-dimension-slider__name');
    expect(name).toMatch(/flex:\s*0\s+0\s+auto/);
    expect(name).toMatch(/white-space:\s*nowrap/);

    const value = ruleBody(css, '.luxar-dimension-slider__value');
    expect(value).toMatch(/white-space:\s*nowrap/);
    expect(value).toMatch(/overflow:\s*hidden/);
    expect(value).toMatch(/text-overflow:\s*ellipsis/);
    expect(value).toMatch(/text-align:\s*right/);
  });
});

/**
 * #2193 hardened the row against a long VALUE and left the other side open. The
 * row's label is the DIMENSION NAME out of the store, so it is
 * dataset-controlled too: `flex: 0 0 auto; white-space: nowrap` with no width
 * bound let a 62-char name take a 400px panel's entire row (name 385px, value
 * 0px, row overflowing by 34px → a horizontal scrollbar inside `__scroll`).
 *
 * These are CSS-text facts nothing else can see. jsdom does no layout, so the
 * unit suite cannot observe the overflow; the geometry half lives in
 * `tests/e2e/nd-navigation.spec.ts`.
 */
describe('dimension-sliders.css — the row survives a long dimension NAME (follow-up to #2193)', () => {
  const css = loadCss('components/dimension-sliders.css');

  it('reserves the value floor once, on the row', () => {
    // Declared on `__label` rather than duplicated as two literals: the name's
    // max-width and the value's min-width must reference the SAME number or the
    // row either overflows (name floor too small) or wastes width.
    expect(ruleBody(css, '.luxar-dimension-slider__label')).toMatch(
      /--luxar-dim-value-floor:\s*\d+px/
    );
  });

  it('caps the name at the row minus that floor, and ellipsizes it', () => {
    const name = ruleBody(css, '.luxar-dimension-slider__name');
    expect(name).toMatch(
      /max-width:\s*calc\(\s*100%\s*-\s*var\(--luxar-dim-value-floor\)\s*-\s*var\(--luxar-spacing-\d+\)\s*\)/
    );
    // Without these the cap would CLIP the name mid-glyph instead of
    // ellipsising it — bounded, but indistinguishable from the bug it fixes.
    expect(name).toMatch(/overflow:\s*hidden/);
    expect(name).toMatch(/text-overflow:\s*ellipsis/);
  });

  it('keeps the name unshrinkable so a long VALUE still cannot squeeze it', () => {
    // The regression a first attempt at this fix actually produced: making the
    // name `flex: 0 1 auto` bounded it, but flex shrink is weighted by base
    // size, so a 73-char value then shrank the name from 53px to 25px —
    // reintroducing precisely what #2193/§7.6 exists to prevent. `max-width` is
    // load-bearing here, not stylistic.
    expect(ruleBody(css, '.luxar-dimension-slider__name')).not.toMatch(/flex:\s*0\s+[1-9]/);
  });

  it('holds the floor from the value side too', () => {
    expect(ruleBody(css, '.luxar-dimension-slider__value')).toMatch(
      /min-width:\s*var\(--luxar-dim-value-floor\)/
    );
  });

  it('lets the panel floor yield to a narrower viewport', () => {
    // A flat `min-width: 400px` under `left: 50%; translateX(-50%)` put the
    // panel at left -6px / right 386px in a 380px window — off BOTH edges, and
    // unreachable because the panel is `position: fixed`.
    const root = ruleBody(css, '.luxar-dimension-sliders');
    expect(root).toMatch(/min-width:\s*min\([^;]*100vw[^;]*var\(--luxar-spacing-10\)[^;]*\)/);
    expect(root).not.toMatch(/min-width:\s*\d+px/);
  });

  it('ellipsizes compact categorical labels and toggle values on one line', () => {
    expect(ruleBody(css, '.luxar-dimension-dropdown')).toMatch(/min-width:\s*0/);

    for (const selector of ['.luxar-dimension-dropdown__label', '.luxar-dimension-toggle']) {
      const rule = ruleBody(css, selector);
      expect(rule).toMatch(/white-space:\s*nowrap/);
      expect(rule).toMatch(/overflow:\s*hidden/);
      expect(rule).toMatch(/text-overflow:\s*ellipsis/);
    }
  });
});

describe('toast.css — §5.1.3 root fade is legal because the toast is not glass', () => {
  const css = loadCss('components/toast.css');

  it('keeps the dismiss opacity transition', () => {
    // Documented as intentional: ui/toast.ts drives opacity 1 → 0 on this very
    // element. Legal only while the element is not a glass surface — see
    // tests/unit/ui/toast.test.ts for that half.
    expect(ruleBody(css, '.luxar-toast')).toMatch(/transition:\s*opacity\s/);
  });
});

describe('liquid-glass.css — the un-glassed toast needs the dark tint', () => {
  const css = loadCss('themes/liquid-glass.css');

  it('paints .luxar-toast the shared glass tint under liquid-glass', () => {
    // Without the glass ::after tint, white text would land on this theme's
    // translucent-WHITE --luxar-bg-secondary over a bright render. It has to be
    // `--luxar-glass-tint` specifically, not just any dark rgba(): that
    // variable is the theme's single AA-verified contrast floor (0.68, pinned
    // by tests/unit/themes/glass-contrast.test.ts). The literal every surface
    // in this class used to carry, rgba(0, 0, 0, 0.55), composites to rgb(115)
    // over a white scene and puts text-primary at 4.48:1 — under AA, and the
    // exact drift #1513 introduced the variable to remove.
    const body = ruleBody(css, "[data-theme='liquid-glass'] .luxar-toast");
    expect(body).not.toBe('');
    expect(body).toMatch(/background:\s*var\(--luxar-glass-tint\)/);
  });
});
