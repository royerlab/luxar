# Luxar Viewer UI Design Guide

**This document is the authoritative ground truth for the Luxar viewer's visual
design.** Every new UI surface must follow it; every deliberate deviation must
be documented here. When code and this guide disagree, one of them is wrong —
fix the code or fix the guide in the same PR, never let them drift silently.

- Scope: the TypeScript viewer (`packages/luxar-viewer`) — panels, overlays,
  dialogs, the control rail, and every widget drawn over the WebGL canvas.
- Not in scope: console/log output (see `CONSOLE_OUTPUT_STYLE.md`), code
  comments (see `JSDOC_STYLE_GUIDE.md`), rendered scene content.
- Verified against the codebase: 2026-08-11.
- Everything here describes the tree **as it stands**. Where a rule's
  implementation only exists in an open PR, it is marked **(pending #N)** —
  never write a claim in the present tense against an unmerged change; drop
  the marker when that PR lands.

---

## 1. Design philosophy — the "quiet instrument"

Luxar's UI language is called the **quiet instrument**: the scientific data is
the interface, and the chrome is a precision instrument that recedes until
needed. Seven principles govern every surface:

1. **The data is the interface.** Panels are translucent glass over the live
   canvas; idle chrome fades (the control rail rests at `opacity: 0.55`,
   `0.12` when collapsed, `0` in fullscreen — waking on hover/movement).
   Never occlude data with opaque decoration.
2. **One material.** Every floating panel is built from the same surface
   recipe (§7.1) so the whole UI reads as a single physical material. Do not
   invent per-panel chrome.
3. **Color is semantics, never decoration.** Green means healthy, amber means
   warning, red means error, blue `highlight` means interactive/selected.
   Identity tints are banned (a points-count is not "points-colored"; it is
   neutral, and dimmed at zero). See §6.
4. **Precision typography.** Metric values are mono + `tabular-nums` so they
   tick without jitter; labels are a 10px letterspaced micro-voice; hierarchy
   comes from weight/brightness, not size explosions. See §8.
5. **Stroke iconography.** Line icons drawn with `currentColor` strokes —
   never emoji, never filled clip-art. See §9.
6. **Motion is acknowledgment, not spectacle.** One-shot staggered reveals,
   fast fades, transform-only entry pops. Everything honors
   `prefers-reduced-motion` (WCAG 2.3.3). See §10.
7. **Keyboard is first-class.** Focus-visible rings, real `<kbd>` chips in
   help text, arrow-key navigation in lists, Escape always closes the
   topmost surface.

The reference implementations of this language are the **control rail**
(`src/ui/control-rail.ts` + `src/styles/components/control-rail.css`), the
**data-loading monitor** (`src/styles/components/data-loading-monitor.css`,
whose end-of-file "Refinement layer" block is the original manifesto), the
**help overlay**, and — for the modal tier specifically — the **dataset
browser** (`src/ui/dataset-browser.ts` + `styles/components/dataset-browser.css`).

---

## 2. Where the truth lives

| Concern | Source of truth |
| --- | --- |
| Design tokens (`--luxar-*`) | `src/themes/theme-manager.ts` → `themeToCSSVariables()` — a hand-written flat map; nothing is algorithmically derived |
| Token values per theme | `src/themes/themes/{dark,light,frosted-glass,liquid-glass}.theme.ts` |
| Token schema | `src/themes/types.ts` (`Theme` interface) |
| Component styles | `src/styles/components/*.css` — one file per UI surface |
| Custom GUI library styles | `src/ui/gui/styles/{gui,controller,folder}.css` |
| Glass theme overrides | `src/styles/themes/{frosted-glass,liquid-glass}.css` |
| Liquid-glass SVG filter | `src/themes/glass-filters.ts` (its `defaultGlassParams` are authoritative — CSS comments describing them have historically gone stale) |
| Rail/panel icons | `src/ui/control-rail/icons.ts` (`RAIL_ICONS`) |
| Monitor icons | `src/ui/data-loading-monitor/templates/primitives.ts` (`MONITOR_ICONS`) |
| Dataset-browser icons | `src/ui/dataset-browser/icons.ts` (`BROWSER_ICONS`) |
| Shared panel header/close recipes | `src/styles/base/utilities.css` (`.luxar-panel-header`, `.luxar-panel-close`; #1508 adds `.luxar-panel-filter`, `.luxar-panel-pop`) |
| Shared context-menu widget (pending #1508) | `src/ui/overlay-widgets/context-menu.ts` + `styles/components/context-menu.css` |
| Modal focus trap | `src/ui/help-overlay/focus-trap.ts` (`trapFocus`) — shared by help overlay and error overlay |
| Native `<select>` chrome | `src/styles/components/select-menu.css` — the single place `<option>` colors may be styled |

Two CSS entry points (see `src/styles/README.md`):

- `index.css` — **embed-safe library entry**. Every rule is scoped to a
  `.luxar-*` class or `[data-theme=…]`. Imports utilities, all component CSS,
  the GUI library styles, and the glass theme overrides.
- `standalone.css` — global host-page chrome (reset, typography, layout).
  Only the standalone app imports it; embedders never get their `body`
  clobbered. **Consequence:** the global `:focus-visible` ring lives in
  `reset.css` and therefore does NOT reach embedded consumers — any component
  that can receive focus must declare its own `:focus-visible` style (§12).

---

## 3. Design tokens

All colors, spacing, and effects are CSS custom properties with the
`--luxar-` prefix, injected at runtime by `ThemeManager` as **inline styles on
`document.documentElement`** (not a `:root {}` stylesheet rule — they carry
inline-style specificity) — with one sanctioned exception registered in
§15.6: `--luxar-glass-tint` is the only `--luxar-` property declared in a
stylesheet rather than by `ThemeManager` (a separate, unregistered case,
`--luxar-overlay-transition-duration`, is a per-element runtime value
`overlay-manager.ts` sets directly on an element, not a theme token at all).
Component CSS must reference tokens, never hardcoded values (sanctioned
exceptions are registered in §15.6).

The full vocabulary is **87 variables** (count them with
`grep -c "'--luxar" src/themes/theme-manager.ts` — that reports 88, one of
which is the `startsWith('--luxar-')` guard in the theme-wipe loop, not a
token). Dark-theme values shown; other themes override per §4.

### 3.1 Color tokens

| Token | Dark value | Use |
| --- | --- | --- |
| `--luxar-bg-primary` | `#111111` | App/page background |
| `--luxar-bg-secondary` | `rgba(30, 30, 30, 0.95)` | **Panel surface** (the material of §7.1) |
| `--luxar-bg-tertiary` | `rgba(0, 0, 0, 0.3)` | Nested/inset elements, `<code>`/`<kbd>` chips |
| `--luxar-bg-overlay` | `rgba(0, 0, 0, 0.5)` | Modal scrims |
| `--luxar-text-primary` | `#e0e0e0` | Main text |
| `--luxar-text-secondary` | `#888888` | Labels, captions, section titles |
| `--luxar-text-muted` | `rgba(255, 255, 255, 0.6)` | Placeholder, micro-labels, idle icons |
| `--luxar-text-disabled` | `rgba(255, 255, 255, 0.4)` | Disabled text, dimmed-at-zero values |
| `--luxar-text-inverse` | `#111111` | Text on inverted backgrounds |
| `--luxar-success` | `#4CAF50` | Semantic: healthy/OK; the tick motif; focus (via border-focus) |
| `--luxar-warning` | `#FFC107` | Semantic: warning |
| `--luxar-error` | `#f44336` | Semantic: error |
| `--luxar-info` | `#2196F3` | Semantic: informational; layers-panel selection |
| `--luxar-highlight` | `#00a0ff` | **The interactive accent** — active/selected/current states (§6) |
| `--luxar-interactive-default` | `rgba(255, 255, 255, 0.1)` | Resting input/button fill |
| `--luxar-interactive-hover` | `rgba(255, 255, 255, 0.15)` | Hover fill |
| `--luxar-interactive-active` | `rgba(255, 255, 255, 0.2)` | Pressed fill |
| `--luxar-interactive-focus` | `rgba(76, 175, 80, 0.3)` | Focus fill |
| `--luxar-interactive-disabled` | `rgba(255, 255, 255, 0.05)` | Disabled fill |
| `--luxar-border-default` | `rgba(255, 255, 255, 0.1)` | **The hairline** — every panel border |
| `--luxar-border-subtle` | `rgba(255, 255, 255, 0.05)` | Internal dividers |
| `--luxar-border-strong` | `rgba(255, 255, 255, 0.2)` | Emphasized borders, quiet scrollbar thumbs |
| `--luxar-border-focus` | `rgba(76, 175, 80, 0.5)` | Focus rings/borders |
| `--luxar-menu-bg` / `--luxar-menu-fg` | `#1e1e1e` / `#e0e0e0` | Native `<option>` popups (deliberately **opaque** in all themes — OS layers can't blur) |
| `--luxar-menu-active-bg` / `--luxar-menu-active-fg` | `#2a2a2a` / `#ffffff` | Selected option |
| `--luxar-viz-hot` / `-warm` / `-cold` / `-neutral` | `#ff6b6b` / `#FFC107` / `#4CAF50` / `#9E9E9E` | Data-intensity readouts only |

Note the naming quirks (these are exactly as emitted; there is no deeper
scheme): `colors.semantic.*` drops the `semantic-` segment
(`--luxar-success`, not `--luxar-semantic-success`); `colors.menu.text` →
`--luxar-menu-fg` and `colors.menu.activeText` → `--luxar-menu-active-fg` (the
only renamed keys); `visualization` abbreviates to `viz`; `background` to `bg`.

### 3.2 Typography tokens

| Token | Value | Notes |
| --- | --- | --- |
| `--luxar-font-base` | system-UI stack (theme-specific, §4) | All UI text |
| `--luxar-font-mono` | `"Monaco", "Menlo", "Ubuntu Mono", monospace` (dark) | Values, paths, `<kbd>` |
| `--luxar-font-display` | `"Inter", -apple-system, sans-serif` (dark) | Rarely used |
| `--luxar-text-xs` … `--luxar-text-4xl` | `10, 11, 13, 14, 16, 18, 20, 28, 32px` | Keys: `xs sm base md lg xl 2xl 3xl 4xl`; **`base` = 13px** is the panel default |
| `--luxar-font-normal/medium/semibold/bold` | `400/500/600/700` | |
| `--luxar-line-tight/normal/relaxed` | `1.2 / 1.4 / 1.6` (dark, light) | Glass themes use `1.2 / 1.5 / 1.7` |

⚠️ **Two namespace collisions to know about** (harmless — key sets are
disjoint — but confusing when scanning): `--luxar-text-*` carries both text
*colors* (`primary/secondary/muted/disabled/inverse`) and font *sizes*
(`xs…4xl`); `--luxar-font-*` carries both font *families*
(`base/mono/display`) and *weights* (`normal/medium/semibold/bold`).

### 3.3 Spacing tokens

| Key | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| px | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32 | 40 |

⚠️ **The key is HALF the pixel value** (`--luxar-spacing-10` = **20px**, not
10px), and the keys are not contiguous (`7, 9, 11, 13–15, 17–19` don't
exist). This is the single most common authoring mistake.

### 3.4 Effect tokens

| Token | Dark value | Notes |
| --- | --- | --- |
| `--luxar-radius-none/sm/md/lg/full` | `0 / 4 / 8 / 12 / 9999px` | **There is no `--luxar-radius-xs`** — referencing it silently falls back |
| `--luxar-shadow-sm/md/lg/xl` | `0 1px 2px …0.2` / `0 4px 12px …0.3` / `0 8px 24px …0.4` / `0 12px 48px …0.5` | liquid-glass bakes `inset` glows into the same tokens |
| `--luxar-blur-none/sm/md/lg` | `none / blur(4px) / blur(10px) / blur(20px)` | Values are **complete `blur()` functions** — write `backdrop-filter: var(--luxar-blur-md)`, never `blur(var(--luxar-blur-md))` |
| `--luxar-opacity-disabled/secondary/hover/full` | `0.4 / 0.6 / 0.8 / 1` | |
| `--luxar-transition-fast/normal/slow` | `all 0.1s ease` / `all 0.2s ease` / `all 0.3s ease` | Full shorthands — `transition: var(--luxar-transition-fast)` |

### 3.5 Z-index tokens

| Token | Value | Layer |
| --- | --- | --- |
| `--luxar-z-base` | 100 | Baseline layer — in-canvas widgets and frameless overlays. The prescribed migrations are still pending, so today it holds the opposite: the data monitor, which belongs a tier up (§15.4), while the widgets that belong here sit on `tooltip` (§15.1) |
| `--luxar-z-dropdown` | 1000 | Docked panels, the rail |
| `--luxar-z-modal` | 2000 | Modal dialogs (+ scrim at `calc(var(--luxar-z-modal) - 1)`) |
| `--luxar-z-popover` | 3000 | Rail popovers/flyouts, first-run hint |
| `--luxar-z-tooltip` | 4000 | Tooltips |

Use the tokens. The historical raw z-indexes still in the tree (§15.1) are
debt, not precedent.

To sit just off a tier, offset it — `calc(var(--luxar-z-…) ± N)` is the
sanctioned idiom (#1478 standardized it), never a fresh literal. The three
shipped uses: the modal scrim at `calc(var(--luxar-z-modal) - 1)`
(`dataset-browser.css:18`), the debug console at
`calc(var(--luxar-z-base) + 50)` (`debug-console.css:31` — just above the
baseline layer, deliberately below every panel), and the toast at
`calc(var(--luxar-z-tooltip) + 1000)` (`toast.css:28` — deliberately above
*every* tier in the table, because a transient notice must not be occluded).
Keep the offset readable as an intent ("just above/below tier X"); if you need
a whole new band, that is a token, not a `calc`. All three are registered in
§15.6.

**A second, parallel z-scale exists in TypeScript**, and the table above does
not predict it on its own. `config.ui.zIndex`
(`src/config/sections/ui/data.ts`) declares eleven numeric layers; four are
written straight onto `el.style.zIndex` at runtime (marked ● —
`layers-panel.ts:346`, `recording-panel.ts:168`, `rendering-controls.ts:189`,
and `performance-monitor.ts:87`, which writes `statsMonitor`, *not* the
`performanceMonitor` key), the rest are currently unread:

| `config.ui.zIndex` key | Value | Nearest token tier |
| --- | --- | --- |
| `dimensionSliders`, `performanceMonitor` | 100 | `base` (100) — `dimensionSliders` is one of the unread keys; its stylesheet is on `--luxar-z-dropdown` (#1483), so this row does not describe where the panel paints |
| `debugConsole` | 150 | just above `base` |
| `datasetBrowser`, `loading`, `error` | 1000 | `dropdown` (1000) |
| `help` | 1001 | just above `dropdown` |
| ● `recordingPanel`, `layersPanel` | 1500 | **between** `dropdown` and `modal` |
| ● `renderingControls` | 1999 | **between** `dropdown` and `modal` |
| ● `statsMonitor` | 2000 | `modal` (2000) |

Those live values are normative: a new left-docked panel spelled
`--luxar-z-dropdown` (1000) paints *underneath* the layers, recording and
rendering panels, which share that same dock (§7.5). **Do not out-stack them.**
A new rail-docked panel joins the exclusive-dock handshake of §7.5
(`closeOtherLeftPanels` in `core/app/init/build-rail-items.ts:74`), so rail
activation only ever leaves one of them open; the R/L/T shortcuts deliberately
bypass the handshake, and a pair stacked that way is ordered by the
`config.ui.zIndex` values above, not by the tokens.

**The tokens cannot express "just above the incumbents", so do not try.** The
two scales overlap, and the space the token tiers appear to leave is already
occupied: `calc(var(--luxar-z-modal) - 1)` resolves to 1999, which is exactly
`renderingControls` — so it *ties* an incumbent, and a tie is settled by DOM
order, not by intent — and is also exactly the value the modal scrim already
claims (`dataset-browser.css:18`, prescribed by §7.2 and registered in §15.6).
There is nothing left between 1999 and `modal` (2000). The correct answer is to
join the exclusive dock and not stack at all. A surface that genuinely must
coexist with the incumbents has to pick its value against the *live*
`config.ui.zIndex` numbers in the table above rather than against a token tier,
and register that value in §15.6 together with the reason it cannot dock. Until
the two scales are reconciled (§15.1), read both before picking a tier.

---

## 4. The four themes

Registered in this order (which is also the theme-picker order): `dark`,
`light`, `frosted-glass`, `liquid-glass`. **Default: `frosted-glass`.**

**What is identical across all four:** `spacing`, `zIndex`, `fontSize`,
`fontWeight`. **Everything else varies.** Do not treat `dark.theme.ts` as
canonical for line-heights, font families, radii, shadows, blur, opacity, or
transitions:

| Group | dark / light | frosted-glass | liquid-glass |
| --- | --- | --- | --- |
| `lineHeight` | 1.2 / 1.4 / 1.6 | 1.2 / 1.5 / 1.7 | 1.2 / 1.5 / 1.7 |
| `fontFamily.base` | Helvetica/Segoe/Roboto stack | SF Pro Display stack | Segoe UI stack |
| `borderRadius` sm/md/lg | 4 / 8 / 12px | 8 / 12 / 18px | 12 / 18 / 28px |
| `blur` sm/md/lg | 4 / 10 / 20px | **12 / 32 / 64px** | **1 / 2 / 3px** |
| `shadow` | opaque dark | lighter, two-layer soft | two-layer + **`inset` glow** |
| `transition` easing | `ease` | `cubic-bezier(0.4, 0, 0.2, 1)` | `cubic-bezier(0.4, 0, 0.2, 1)` |
| `opacity.disabled` | 0.4 | 0.3 | 0.3 |

Key color differences:

| | dark | light | frosted-glass | liquid-glass |
| --- | --- | --- | --- | --- |
| `bg.secondary` (panel) | `rgba(30,30,30,0.95)` | `rgba(250,250,250,0.95)` | `rgba(28,30,36,0.75)` — a **dark frost** (contrast floor, see below) | `rgba(255,255,255,0.15)` + a dark `::after` |
| `highlight` | `#00a0ff` (blue) | `#0277bd` (blue) | `rgba(0,160,255,1)` (= `#00a0ff`) | `rgba(0,160,255,1)` (= `#00a0ff`) |
| `border.focus` | green `rgba(76,175,80,0.5)` | green | **blue** `rgba(0,122,255,0.6)` | **blue** `rgba(0,122,255,0.5)` |
| `interactive.*` base | white alpha | black alpha | white alpha | bluish-gray `rgba(120,120,128,…)` |
| `menu.background` | `#1e1e1e` | `#ffffff` | `#1a1a1a` (opaque!) | `#1a1a1a` (opaque!) |

Design consequences:

- **The accent is the brand blue in three of four themes.** `highlight` is
  `#00a0ff` in dark, frosted-glass and liquid-glass, and a darker `#0277bd` in
  light — the correct contrast direction on white. It is still a *token*, not a
  constant: never hardcode the hex, and never rely on a specific luminance
  (the light-theme value is much darker than the other three).
- **Never assume the focus ring is green** — it is green in dark/light and
  blue in the glass themes. Always use `--luxar-border-focus` /
  `--luxar-interactive-focus`; never hardcode a green.
- **Never assume 13px math based on `px` line-heights** — line-height tokens
  are unitless multipliers and theme-dependent.
- Anything using `box-shadow: var(--luxar-shadow-lg)` silently gains an inner
  glow in liquid-glass; that is intended.
- **Frosted-glass panels are a dark frost, and that is a contrast floor.** The
  panel tint must keep text legible over ANY scene, so `bg.secondary` is a 75%
  dark layer under the blur — the same contrast-protection role liquid-glass's
  dark `::after` plays (liquid-glass paints its tint via the
  `--luxar-glass-tint` CSS custom property, `rgba(0, 0, 0, 0.68)`, not a theme
  token — see §15.6 for why it lives in the stylesheet rather than as a
  `ThemeManager` token). Composited over the worst case (a pure-white scene),
  frosted-glass measures **6.8:1 / 5.8:1 / 4.7:1** for `text-primary` /
  `text-secondary` / `text-muted`, and liquid-glass measures **7.3:1 / 6.2:1 /
  5.0:1** for the same three ranks — all six clear WCAG AA (4.5:1) **for a bare
  panel surface at full opacity, on those three ranks**; a `bg.tertiary` inset
  lift and a surface-level `opacity` are two exceptions this claim does not
  cover — a third, unrelated gap affects `text-disabled`, which sits outside
  the three ranks asserted here (§15.2).
  (Issue #1513: pre-fix, frosted-glass's panel composited to `#6b6d71`, giving
  secondary/muted 3.73:1 / 2.56:1 — both below AA, and muted also below the
  3:1 non-text floor for a meaning-bearing icon. Liquid-glass's `::after`
  layer additionally carried a stray `opacity: 0.7` multiplier on top of its
  own 0.55 tint — an effective 0.385 alpha that composited to `rgb(157)` and
  gave primary/secondary/muted 2.61:1 / 2.19:1 / 1.73:1, all three below AA,
  not only secondary/muted; deleting that multiplier is as much a part of
  this fix as the alpha bumps.) With all three ranks clearing AA on a bare
  panel, the `secondary`/`muted` split is now a pure typographic hierarchy
  (§8.3's quiet-instrument micro-voice) rather than a contrast cliff — on a
  bare panel, pick a rank for visual weight, not to dodge a legibility floor
  (an inset still has one — §15.2). **Never lighten a
  glass panel tint, shrink a glass text alpha, or add an `opacity` to the
  `::after` tint layer, without re-running the worst-case composite** —
  `tests/unit/themes/glass-contrast.test.ts` pins these six numbers and (for
  liquid-glass) the tint's own effective alpha, and fails if any regresses
  below 4.5:1 or an opacity multiplier reappears. Its dark-scene `describe`
  blocks document the numbers in a comment (frosted-glass **16.3:1 / 13.1:1 /
  9.7:1**, liquid-glass **18.8:1 / 14.8:1 / 10.5:1**) but can't independently
  guard that direction: for white-ish text over a darker panel the
  bright-scene case above is the stricter one for any text alpha in use
  here, so those checks only assert the rank order stays
  `primary > secondary > muted`.
- Blur tokens differ radically per theme by design: frosted-glass IS its blur;
  liquid-glass barely blurs because refraction + tint do the work.

### 4.1 Theme mechanics (invariants)

- `ThemeManager` (singleton) sets `data-theme="<id>"` on
  `document.documentElement`, then injects all tokens as inline styles there.
  Theme-conditional CSS keys off `[data-theme='…'] .luxar-…` selectors.
- Persistence: `localStorage` key **`luxar.theme`** (`StorageKeys.theme`).
  Resolution order at boot: saved preference → `frosted-glass`; a `?theme=`
  URL param is applied after boot **and overwrites the saved preference**.
  Invalid ids degrade to a warning, leaving the previous theme.
- `setTheme` is fail-atomic: DOM is updated first; only on success are the
  current-theme pointer, `localStorage`, and observers updated.
- Theme switch wipes every inline `--luxar-*` on `<html>` — never park custom
  state in a `--luxar-`-prefixed inline variable.
- Liquid-glass refraction layers are injected on the next animation frame
  (plus a `MutationObserver` + rAF for panels created later) — a freshly
  created glass panel is fully glassed ~2 frames after insertion. Don't
  screenshot/measure it earlier.

---

## 5. The glass-surface system

Both glass themes key off **one marker class: `luxar-glass-surface`**. A panel
opts in by adding it to its root element at creation. There is no per-panel
selector list to maintain — `frosted-glass.css`, `liquid-glass.css`, and the
refraction injector in `glass-filters.ts` all read the same class.

- **Opt in**: self-contained bordered panels (rail, GUI panels, help overlay,
  layers panel, monitor, dataset browser, debug console, dimension sliders,
  resolution indicator, error overlay, scene-identity banner, rail
  flyout/popover).
- **Opt out**, three kinds: frameless in-canvas widgets (scale bar, colormap
  legend) are deliberately NOT glass — they use drop-shadows instead of a
  panel material; transient cursor popovers (context menus, §7.8) are
  *framed* but still not glassed, because they live and die with the cursor;
  and two over-canvas badges — the toast and the REC pill — are framed on the
  badge variant of the recipe (§7.1) but likewise not glassed. **Being a badge
  is not the criterion**, and the two have different-strength reasons. For the
  toast, un-glassing is *mandatory*: its dismiss animation is an `opacity` fade
  on its own root, which §5.1.3 forbids on a glass surface. The REC pill's reason
  is weaker — it is simply not put on the panel material
  (`recording-panel.css:85-97`, a hairline-bordered pill in the untokenised 9999
  band that must beat unknown host chrome, §15.6); it does not fade its root
  (only `transition: border-color`, with the pulse on its inner `__dot`), so it
  *could* be glassed. The resolution indicator shows why the toast's reason is
  the load-bearing one: a badge that IS glassed (opt-in list above) and ramps
  `opacity` on its root anyway — drift (§15.4), not a member of this kind.
  These last two kinds take the liquid-glass
  dark tint explicitly — `background: var(--luxar-glass-tint)` in
  `liquid-glass.css`, never a hardcoded `rgba()` of their own (§15.6) — since
  without the glass `::after` layer they would paint that theme's
  translucent-white `--luxar-bg-secondary` under white text.
- **Nested surfaces must de-glass**: a GUI mounted inside an already-glass
  rail popover removes the class (`rail-panels/popover-gui.ts`) — never
  double-glass.

Recipes (do not re-implement; shown for understanding):

- `frosted-glass`: single rule — `isolation: isolate` +
  `backdrop-filter: var(--luxar-blur-md) saturate(1.5)` (→ `blur(32px)`).
- `liquid-glass`: three stacked layers behind the content — a **real DOM**
  `.luxar-glass-refraction` div (z:-3, gradient source pixels), `::before`
  (z:-2, `backdrop-filter: blur(2px) saturate(180%)` +
  `filter: url(#luxar-liquid-refraction)` SVG displacement/chromatic
  aberration), `::after` (z:-1, dark tint `var(--luxar-glass-tint)` =
  `rgba(0, 0, 0, 0.68)` + four inset bevel shadows). The `::after` layer
  carries **no `opacity` multiplier, and must not gain one** — the tint's own
  alpha IS the theme's contrast floor (§4), and an element-level `opacity`
  would silently divide it back down (issue #1513's root cause).

### 5.1 Hard constraints on every glass surface

These are load-bearing; violating any of them visibly breaks a theme:

1. **`::before` and `::after` are RESERVED.** The glass themes claim both
   pseudo-elements of every `.luxar-glass-surface`. Decorations like popover
   arrows must be **real child elements** (see
   `control-rail.css` `__flyout-arrow` / `__popover-arrow`). The tick motif
   (§8.3) is safe because it sits on an inner element, never the surface root.
2. **The surface root must keep `overflow: visible`.** The glass layers paint
   at negative z-index with `inset: 0`; `overflow: hidden` clips them dead.
   Therefore: panel root = `overflow: visible` + `max-height`, and scrolling
   is delegated to an inner `__scroll` wrapper (§7.4).
3. **Never attach entry/exit `opacity` animations to the surface root.**
   Transient opacity ramps on panel open/close are the recorded liquid-glass
   regression class — four component files carry "animation removed" comments
   from exactly this bug. Entry motion must be transform-only (§10.2); fade
   the *scrim* or an inner wrapper, not the panel. Note the boundary of this
   rule: a *steady* translucent resting state with opacity transitions
   between rest points is proven in production — the control rail lives at
   `opacity: 0.55/0.12/1` and renders its glass correctly in every theme
   (fractional opacity does create a compositing context, but a stable one;
   it is the transient animation-time churn that broke the SVG filter).
4. **SVG filters don't work reliably on pseudo-elements** — that's why the
   refraction layer is a real div. Don't "simplify" it back into a pseudo.
5. **Native `<option>` popups can't be glassed** (OS-layer rendering, no
   `backdrop-filter`). All `<select>`/`<option>`/`<optgroup>` chrome lives in
   `select-menu.css` using the opaque `--luxar-menu-*` tokens. A new
   `<select>` = add its selector there; never restyle options per component.

---

## 6. Color semantics — the rules

This is the most drift-prone area of the codebase; the rules below are the
intended, authoritative direction (existing violations are cataloged in
§15.1–15.5).

| Color | Meaning | Correct uses |
| --- | --- | --- |
| `--luxar-highlight` | **Interactive accent**: active, selected, current | Rail active button (20% `color-mix` fill + 3px left pip), active chips/segments (`22%` fill + `50%` border mix), "current item" rings, type badges (the dataset browser's file-type chip: 10% fill + 40% border mix + full-accent text) |
| `--luxar-success` | Semantically **good/healthy** + the house motif | Status ticks (§8.3), healthy metrics, LOADED state, cache-hit-good; focus rings *via the focus tokens only* |
| `--luxar-warning` / `--luxar-error` / `--luxar-info` | Their names | Alarms and information only. `info` additionally marks the layers-panel selection |
| `--luxar-interactive-*` | Neutral fills | Resting/hover/pressed backgrounds of ALL controls — hover feedback is a neutral fill change, not a color change |
| `--luxar-viz-*` | Data intensity | Readouts describing data (hot/warm/cold/neutral), never UI chrome |

Rules:

1. **Hover ≠ accent.** Hovering neutral chrome uses
   `--luxar-interactive-hover`. The accent appears only when something is
   *on/selected/current*.
2. **Accent fills are `color-mix` tints, not solid fills.** The house pattern:
   `background: color-mix(in srgb, var(--luxar-highlight) 20%, transparent)`
   with border at 40–50% mix and text/icon at full accent. Solid accent
   buttons are reserved for singular primary actions.
3. **No identity tints.** Geometry-type counts, node names, layer rows are
   neutral. State (0 → dimmed) may change brightness, not hue.
4. **Green is not "interactive".** Sliders, checkboxes, input focus borders and
   scrollbars were all green historically; they are now neutral + highlight, and
   nothing should go back. The only sanctioned greens are semantics (healthy /
   LOADED / good), the tick motif, and the theme-owned focus tokens.
5. **Scrollbars are quiet**: `scrollbar-color: var(--luxar-border-strong)
   transparent`, 6px webkit width, transparent track, `radius-full` thumb —
   never accent-colored.
6. Text on accent/semantic fills: use `--luxar-text-primary` over tinted
   (`color-mix`) fills; literal `white` is acceptable only over *solid*
   semantic fills (e.g. white-on-red REC dot) and must carry a comment.

---

## 7. Surfaces

### 7.1 The panel surface recipe (the "one material")

Every floating panel:

```css
background:      var(--luxar-bg-secondary);
border:          1px solid var(--luxar-border-default);  /* crisp hairline */
border-radius:   var(--luxar-radius-lg);                 /* large soft corners */
box-shadow:      var(--luxar-shadow-lg);
backdrop-filter: var(--luxar-blur-md);                   /* + -webkit- prefix */
overflow:        visible;                                /* glass constraint §5.1 */
color:           var(--luxar-text-primary);
font-family:     var(--luxar-font-base);
font-size:       var(--luxar-text-base);                 /* 13px */
```

…plus `luxar-glass-surface` on the root element in TS — except for §5's three
opt-outs (frameless in-canvas widgets, transient cursor popovers such as
context menus, §7.8, and the two un-glassed over-canvas badges, toast and REC
pill), which take the recipe without the glass class. Do NOT add per-theme
`box-shadow` rings on top (removed deliberately in PR #447).

**Tier variants:**

| Variant | Radius | Shadow | Blur | z |
| --- | --- | --- | --- | --- |
| Panel (default) | `lg` | `lg` | `md` | `dropdown` |
| Modal dialog | `lg` | **`xl`** | `md` | `modal` (+ scrim) |
| Popover / flyout / tooltip | `lg` (tooltip `md`) | `xl` (tooltip `lg`) | `md` | `popover` / `tooltip` |
| Transient badge / toast | **`md`** | `md`–`lg` | `sm`–`md` | context |

The toast is the one deliberate outlier in that last column — for its value and
the reason for it, see §3.5.

### 7.2 Modals

Modals additionally get (the dataset browser is the reference):

- A **scrim**: sibling element, `position: fixed; inset: 0;
  background: var(--luxar-bg-overlay); z-index: calc(var(--luxar-z-modal) - 1)`,
  click-to-close, `aria-hidden="true"`, opacity fade-in (the scrim may fade;
  the panel may not — §5.1.3).
- Transform-only entry pop on the panel (§10.2).
- `role="dialog"`, `aria-modal="true"`, `aria-labelledby` pointing at the
  title element. Escape closes (wired through the InputHandler).
- **Focus management, all three parts.** `aria-modal="true"` only *asserts* to
  assistive tech that the rest of the page is unavailable; it does not stop Tab
  from walking behind the panel, and neither does a scrim. So a modal must
  (1) place initial focus inside itself on open, (2) contain Tab/Shift+Tab, and
  (3) return focus to the opener on close. Use the shared
  `trapFocus(container)` (`src/ui/help-overlay/focus-trap.ts`) — it does all
  three, and its returned cleanup releases the listener and restores focus.
  Don't hand-roll another. An asynchronously-arriving initial focus (the
  dataset browser focuses its filter field only once the listing resolves) must
  check the user hasn't focused something else first (§12).

### 7.3 Headers

The header row and its close button are **two shared classes in
`styles/base/utilities.css`**, not per-panel CSS. Add them alongside the
panel's own BEM class and do not restate what they declare:

- `.luxar-panel-header` — flex row, `space-between`, 6px down to the hairline
  rule (`border-bottom: 1px solid var(--luxar-border-default)`), 10px from the
  rule to the content. One material: no per-panel `border-strong` emphasis
  variants. Genuinely structural extras (a panel's own horizontal padding, a
  drag cursor) stay per-panel.
- `.luxar-panel-close` — 28×28 `display: grid; place-items: center`,
  `radius-md`, `text-muted` → `text-primary` with an `interactive-hover` fill,
  a **15px stroke ✕** (the §9.1 rail icon contract — never the text `×`
  glyph), a `:focus-visible` ring, and a reduced-motion guard.

Titles are `--luxar-text-lg`/`--luxar-font-bold`, or the §8.3 tick-motif
micro-header for quiet-instrument surfaces.

Adopters: help overlay, layers panel, GUI, debug console, monitor. The dataset
browser predates the extraction and still spells both out in its own file —
and the copies have already drifted from the originals (a focus ring without
the mandatory fallback, no reduced-motion guard, a header row that restates
half the shared one; see §15.1) — so copy the classes, not that file's header
block.

### 7.4 Scroll containers

Because the surface root must stay `overflow: visible`, scrolling always lives
in an inner wrapper:

```css
.luxar-<panel>__scroll {
  overflow-y: auto;
  max-height: <bound>;               /* e.g. 82vh or calc(100vh - 40px) */
  scrollbar-width: thin;
  scrollbar-color: var(--luxar-border-strong) transparent;  /* quiet, §6.5 */
}
.luxar-<panel>__scroll::-webkit-scrollbar { width: 6px; }
.luxar-<panel>__scroll::-webkit-scrollbar-track { background: transparent; }
.luxar-<panel>__scroll::-webkit-scrollbar-thumb {
  background: var(--luxar-border-strong);
  border-radius: var(--luxar-radius-full);
}
```

**Shipped variant — bound on the root.** Three panels put the height bound on
the root instead and make it a flex column (`max-height` +
`display: flex; flex-direction: column`), driving the scrolling child with
`flex: 1; min-height: 0` so it caps against that bound: the help overlay
(`help-overlay.css:19-21` root, `:51-61` wrapper), the dimension sliders
(`dimension-sliders.css`, #1483), and the data monitor (`overflow: visible` root
at `data-loading-monitor.css:24`, the `--expanded` bound at `:62-67`, capped by
`.luxar-monitor-detailed` `:88-89` and `.luxar-data-monitor__content`
`:139-144`). The point is that the root owns the panel's single height bound and
the scroller caps against it, so the glass root stays `overflow: visible`
without the bound being duplicated on two boxes. It also lets a panel pin
content *outside* the scroll area — the help overlay keeps its header and filter
row there, and the monitor its header and tabs; the dimension sliders put
everything, header included, inside the wrapper.

Two consequences. The root's `display` must not be overwritten inline or the cap
is lost (see `ui/dimension-sliders.ts::setVisible`). And the panel's padding must
be *inside* the scrollport whenever a focusable descendant sits flush with that
scrollport's edge: a scroll container clips ink overflow at its padding edge, so
a focus ring on a flush child is cut off if the slack is on the root instead —
which is why the sliders moved their padding onto the wrapper (#1483) while the
monitor, whose scrollers have no flush focusable edge, keeps its padding on the
root (`:65`).

### 7.5 The control rail and its satellites

The rail is the canonical interactive surface; its patterns generalize:

- **Rail buttons**: 38×38, `display: grid; place-items: center`, `radius-md`,
  muted → primary on hover with `interactive-hover` fill; active =
  `color-mix(highlight 20%)` fill + a 3px×(button-18px) rounded left pip in
  `--luxar-highlight`; disabled = `opacity: 0.3` + `pointer-events: none`.
- **Chips** (flyout/popover icon buttons): 40×40, same states + a 45%-mix
  highlight border when active.
- **Tooltips**: surface recipe at `radius-md`, `padding: 5px 9px`,
  `--luxar-text-base`, appear on `:hover` and `:focus-visible`, suppressed
  while the button's popover is open (`[aria-expanded='true']`), `<kbd>` hint
  in mono + `text-secondary`. Flyouts near the viewport bottom flip their
  chip-tips upward.
- **Popover/flyout arrows are real children** (§5.1.1): 12×12 rotated square
  painted `--luxar-bg-secondary` with two hairline borders.
- **One rail gutter: `left: 73px`.** Docked panels, popovers/flyouts and the
  first-run hint all share it — it is the popovers' own computed left edge (the
  rail box including its border, plus their 10px gap), so whichever surface is
  open its left edge lands in exactly the same place. Docked panels get it as
  an `!important` override of inline positioning (`control-rail.css:547`,
  sanctioned by §13 and registered in §15.6); the draggable debug console gets
  the same default *without* `!important` so dragging still wins.
  **The rule is a hardcoded selector pair** — only `.luxar-gui` and
  `.luxar-layers-panel`, each under `.luxar-has-control-rail`
  (`control-rail.css:542-543`) — so a new docked surface that is neither of
  those silently gets no gutter and opens under the rail. Adding its class to
  that selector pair is part of docking it, not an afterthought. (The recording
  panel is already covered: `ui/gui/gui.ts:87` builds it as a `.luxar-gui`, and
  `recording-panel.ts:161` only adds a second class alongside.)
- **The rail's item buttons live in `.luxar-control-rail__items`**, a
  `display: contents` wrapper on fine pointers that becomes the rail's scroll
  box under `(pointer: coarse)` (§11.5). Popovers, flyouts, the footer and the
  collapse handle stay outside it, so scrolling never clips them.
- **The rail's left dock is exclusive** — Rendering, Layers and Recording all
  open at that one position, so activating any of them from the rail (or
  opening a rail popover) closes the others rather than stacking
  (`closeOtherLeftPanels`, `core/app/init/build-rail-items.ts:74`). Keyboard
  shortcuts (R/L/T) deliberately bypass this, so panels can still be stacked
  on purpose — and a stacked pair is then ordered by `config.ui.zIndex`, not
  by the tokens (§3.5). A new rail-anchored surface must join this handshake
  rather than out-stacking the incumbents.

### 7.6 GUI controller rows (custom GUI library, `src/ui/gui/`)

Row anatomy: flex, `min-height: 20px`, `gap: 4px`, `radius-sm`, hover =
`interactive-hover`; label column `flex: 0 0 45%` at `--luxar-text-sm` with
ellipsis; widgets fill the rest. Sliders: 3px track in `interactive-default`,
14px round thumb; number inputs: 52px, mono, right-aligned, spinners stripped;
checkboxes 16px via `accent-color`; buttons full-width `interactive-default` +
hairline. Folders indent children by 12px behind a `border-subtle` left rule.
Thumbs and `accent-color` are `--luxar-highlight`; the mono values are
`text-primary` (instrument voice, not an accent); focus goes through
`--luxar-border-focus`.

For any flex row that pairs a fixed label with a dataset-controlled value,
protect the label with `flex: 0 0 auto; white-space: nowrap` and give the value
`min-width: 0`, single-line ellipsis, and a tooltip carrying the full text. This
keeps long values from shrinking or wrapping the label and prevents horizontal
scrollbars inside bounded panels.

**First ask whether the label is really fixed.** The recipe above protects one
side because the other is dataset-controlled; when *both* sides are, protecting
one just moves the overflow to it. A dimension-slider row is the case in point —
its label is the dimension name out of the store — and a 62-char name took a
400px panel's whole row, leaving the value 0px wide and giving `__scroll` a
horizontal scrollbar. So a dataset-controlled label additionally needs:

- a reserved floor for the value, declared once on the row
  (`--luxar-dim-value-floor`) and consumed from both ends: the label takes
  `max-width: calc(100% - <floor> - <gap>)`, the value `min-width: <floor>`;
- its own `overflow: hidden; text-overflow: ellipsis`;
- an unconditional `title`, since it can now be truncated.

Bound such a label with `max-width`, never by making it shrinkable: flex shrink
is weighted by base size, so a shrinkable label is one a long *value* can shrink
— reintroducing exactly what the first paragraph prevents. Reserve the floor on
the **right-aligned** side where possible; a box wider than its glyphs is
invisible there, so the reservation is free whenever the value is short.

### 7.7 The panel filter row (pending #1508)

Type-to-filter for list-bearing panels is **one shared recipe**,
`.luxar-panel-filter` (+ `__icon`, `__input`) in `styles/base/utilities.css` —
the dataset browser's search anatomy generalized: a `position: relative`
wrapper, a 13px stroke search icon (§9.1 contract) absolutely placed at the
left, and an input on `interactive-default` that moves to the
`--luxar-border-focus` border (with the documented fallback — the token is
optional on `ThemeBorderColors`) and `interactive-hover` fill on focus.
Behavioral contract, uniform across adopters (layers panel, help overlay):

- **Filtering hides, never rebuilds.** Rows keep their DOM and get a
  `--filtered` modifier (`display: none`); indices and listeners stay valid,
  and keyboard list navigation must skip hidden rows. The modifier must be
  declared AFTER the base row rule — equal specificity means source order
  decides, and the wrong order leaves the filter toggling a class that does
  nothing.
- **Zero matches shows a note** ("No layers match." / "No shortcuts match."),
  centered, `text-muted`, `--luxar-text-sm` — a silently collapsed list reads
  as broken. In a panel that outlives its list — one that re-renders the list
  container in place (the layers panel's `renderList()` clears it with
  `innerHTML = ''`) — the note must live OUTSIDE that container so a rebuild
  never wipes it; a surface rebuilt whole per open (the help overlay) may keep
  it inline.
- **Escape is two-stage**: with a query it clears and stays (stopPropagation);
  empty, it falls through to the panel's own close.
- **Keystrokes must not leak** to global shortcuts while the input has focus.
- Panels whose list is usually short gate the row on a threshold (the layers
  panel shows it above 8 layers) — a filter over four rows is noise.

### 7.8 Context menus (pending #1508)

Cursor-anchored right-click menus are **one shared widget**,
`openContextMenu()` in `src/ui/overlay-widgets/context-menu.ts` +
`styles/components/context-menu.css` — do not hand-roll another (the
dimension-slider menu predates it; §15.5). Surface: the popover tier of §7.1
(`bg-secondary`, hairline `border-strong`, `radius-md`, `shadow-xl`,
`blur-md`, `z-popover`), transform-only entry pop. Deliberately **not** a
`luxar-glass-surface`: menus are transient cursor popovers (the same
precedent as the dimension menu), which is also what makes the root's
`overflow-y: auto` legal (§5.1.2 binds glass roots only).

Non-negotiables the widget already implements — a new menu gets them by
construction:

- Full menu ARIA: `role="menu"` / `menuitem` / `menuitemradio` +
  `aria-checked`; `aria-haspopup="menu"` + `aria-expanded` on submenu openers
  AND on the opener element in the invoking panel.
- Roving focus: ArrowUp/Down wrap, Home/End, disabled items skipped; focus
  returns to the opener on close; Escape closes one level, Tab dismisses.
- One side-flyout submenu level (ArrowRight opens, ArrowLeft retracts).
- Viewport containment is position AND size: the menu clamps into the
  viewport and caps its height to it (`overflow-y: auto`), so a long submenu
  (the full colormap list) scrolls instead of pinning off-screen; keyboard
  focus scrolls items into view.
- Dismissal: Escape, outside `pointerdown` (attached next tick so the opening
  right-click doesn't self-dismiss), opener re-invocation. One menu at a
  time, module-wide.
- Radio state: an 8px dot, hollow on `border-strong`, filled
  `--luxar-highlight` when checked (interactive accent, §6 — checked state is
  a selection, not a health signal).
- Invoking panels `preventDefault()` the native menu over glass, offer
  Shift+F10 / the ContextMenu key as the keyboard path, and right-click
  selects the row under the cursor unless it is already in the selection
  (Finder/napari convention).

---

## 8. Typography

### 8.1 Scale usage

- Panel body/default: `--luxar-text-base` (13px).
- Labels/controls: `--luxar-text-sm` (11px).
- Micro-labels/hints/badges: `--luxar-text-xs` (10px).
- Panel titles: `--luxar-text-lg` (16px) bold, or the §8.3 micro-header.
- Anything ≥ `--luxar-text-2xl` is exceptional (legacy close-glyphs, hero
  numbers). Hierarchy comes from weight and brightness, not size jumps.

### 8.2 Numerals — the instrument voice

Every live metric value:

```css
font-family: var(--luxar-font-mono);
font-variant-numeric: tabular-nums;   /* stable ticking, no jitter */
/* large hero values additionally: letter-spacing: -0.02em */
```

### 8.3 Micro-labels and the tick motif

Three-rank hierarchy (from the monitor's refinement layer):

1. **Section title** — the house micro-header:
   `--luxar-text-xs`/`semibold`, `letter-spacing 0.06em`, `text-transform:
   uppercase`, `color: var(--luxar-text-secondary)`, laid out
   `inline-flex; gap: var(--luxar-spacing-3)` with the **status tick**
   `::before`: `width: 3px; height: 10px; border-radius: 1px;
   background: var(--luxar-success); opacity: 0.55`. Optionally a 13px stroke
   icon. (Dialog titles may scale this up to `--luxar-text-base` with a
   3×12px tick — the dataset browser's title is the shipped example.)
2. **Metric label** — 10px / 600 / `letter-spacing: 0.08em` /
   `--luxar-text-muted` (usually uppercase).
3. **Tertiary/summary label** — `medium` weight, `--luxar-text-disabled`.

Letter-spacing is always **em-based**, never `px` — `0.06em` for the
micro-header (`0.09em` where a dialog title scales it up), `0.08em` for metric
labels, and `0.02em`–`0.05em` for the monitor's tighter ranks.

### 8.4 `<kbd>` chips

Real `<kbd>` elements: mono, `--luxar-text-xs`, `line-height: 1`,
`background: var(--luxar-bg-tertiary)`, hairline `border-default`,
`border-radius: 4px`, `padding: 3px 5px` (1px 5px in dense hint contexts),
`white-space: nowrap`. In shortcut tables: fixed keys column
(`flex: 0 0 118px`, right-justified) + `text-secondary` description,
baseline-aligned.

The `4px` is a **sanctioned literal, not an oversight** (§15.6): `radius-sm`
is 4px in dark/light but 8px in frosted-glass and 12px in liquid-glass, which
on an ~18px-tall chip is a pill, not a keycap. Chip corners are deliberately
theme-invariant.

### 8.5 `<code>` chips

Inline paths/extensions: mono, `--luxar-text-xs`,
`background: var(--luxar-bg-tertiary)`, `padding: 1px 4px`, `border-radius:
3px` — the same theme-invariant-corner exception as §8.4, one step tighter
because the chip has no border to carry the shape.

---

## 9. Iconography

**Never emoji.** All icons are inline-SVG line drawings painted by
`currentColor` so they inherit text color and theme automatically.

### 9.1 The rail contract (default for all new icons)

Markup: `<svg viewBox="0 0 24 24" aria-hidden="true">` + geometry only — no
inline presentation attributes. Paint comes from CSS:

```css
stroke: currentColor;
fill: none;
stroke-width: 1.7;
stroke-linecap: round;
stroke-linejoin: round;
```

Sets: `RAIL_ICONS` (`src/ui/control-rail/icons.ts`) and `BROWSER_ICONS`
(`src/ui/dataset-browser/icons.ts`). New icons: draw on the
24-grid with ~2px optical margins, single stroke weight, no fills (a filled
dot ≤2.5px radius is acceptable as an accent), and check the existing sets
first to avoid glyph collisions (e.g. fullscreen deliberately avoids corner
brackets because "fit" owns them).

### 9.2 The monitor micro-contract (dense data UIs only)

`MONITOR_ICONS`: `viewBox="0 0 14 14"`, presentation attributes inline
(`fill="none" stroke="currentColor" stroke-width="1.5" …`), class
`luxar-micon`. Use only inside dense metric surfaces.

### 9.3 Rendered size ladder

`32px`/`28px`/`26px` hero glyphs (empty states, error dialog, the monitor's
all-clear) · `19px` rail buttons · `18px` chips · `17px` list rows · `16px`
scene-identity banner (§15.1) · `15px` GUI/layers rows and the shared
`.luxar-panel-close` ✕ · `14px` inline affordances (the breadcrumb edit
pencil) · `13px` section titles & micon default · `12px` tabs/inline and
scene-graph node glyphs · `9px` compact alerts. Icons at ≤13px may carry
`opacity: 0.75–0.9` at rest.

---

## 10. Motion

### 10.1 Micro-interactions

Use the transition tokens (`fast` for hover/focus color+fill changes,
`normal` for structural changes, `slow` for opacity/visibility of whole
surfaces). Prefer transitioning specific properties (`background-color 0.1s
ease`) over `all` in hot paths (long lists).

### 10.2 Entry/exit of surfaces

- Panels/modals: **transform-only** scale-settle, e.g.
  `scale(0.975) → scale(1)` over `0.15–0.2s cubic-bezier(0.2, 0.7, 0.2, 1)`.
  Never animate `opacity` on a glass surface root (§5.1.3).
  **(pending #1508)** This is a shared recipe: add `.luxar-panel-pop`
  (`styles/base/utilities.css`) beside the panel's BEM class — CSS animations
  restart on `display: none → block`, so toggled panels re-pop with zero JS.
  A surface whose resting transform is not identity (the rail flyout's
  `translateY(-50%)`, the dimension sliders' `translateX(-50%)`) must NOT use
  the generic class — `animation` composes by replacing the `transform`
  channel, which would yank it out of position for a frame; give it a
  composed keyframe in its own CSS (`translate…(-50%) scale(…)`) instead.
- Scrims and genuinely **non-glass** transients (badges, frameless in-canvas
  widgets): opacity fades are fine. The toast is one of them — it is
  deliberately un-glassed for exactly this reason (§5's third opt-out), so its
  `transition: opacity 0.3s ease` root fade is legal. The rule it illustrates
  is the constraint's boundary, not an exemption from it: a *glass* surface
  still must not fade its own root — fade an inner wrapper or move with
  transform, or drop the glass class as the toast does and take the theme tint
  instead.
- The rail collapse/expand animates `opacity 0.3s ease` alongside
  `transform 0.28s cubic-bezier(0.2, 0.7, 0.2, 1)`. The rail is the sanctioned
  steady-translucency exception to §5.1.3: its resting state is already
  fractional opacity (a stable compositing context, verified rendering
  correctly under liquid-glass), and it transitions between rest points
  rather than fading in from nothing on entry. Do not cite it as precedent
  for entry fades on ordinary panels.

### 10.3 Staggered reveals (data panels)

The monitor pattern: content fades/slides in once per *user-initiated* paint —
`opacity: 0; translateY(4px)` → none over `0.18–0.22s ease`, with
`nth-child` delays in 0.03s steps. **Gate the animation class one-shot** so
live data refreshes never replay it.

### 10.4 Reduced motion — mandatory

Every animation and transition a component introduces must be disabled under
`@media (prefers-reduced-motion: reduce)` (use `animation: none` /
`transition: none`; `!important` is sanctioned here to beat shorthands).
`transition` is not inherited — guard each element that declares one.

---

## 11. Layout & placement

| Surface | Placement |
| --- | --- |
| Control rail | Left edge, vertically centered, `left: 12px` |
| Left-docked panels (GUI, rendering/recording, layers, debug console) | `left: 73px` beside the rail, one at a time (§7.5); debug console draggable |
| Rail popovers/flyouts | `left: calc(100% + 10px)` off the rail (= the same 73px gutter), arrow pointing back |
| Dimension sliders | Bottom-center, 80% width, max 800px |
| Toast | Bottom-center, transient |
| Scene-identity banner | Top-center, `top: 12px`, standing (not transient) |
| Data monitor | Top-right region (tabs + compact pill) |
| Help overlay | Top-right, 400px, max-height 82vh |
| Modals (dataset browser, errors) | Viewport-centered, `translate(-50%,-50%)`, ~600px, max 90vw/80vh |
| Scale bar / colormap legend / resolution indicator | Bottom corners, frameless or badge |

Panels define explicit widths (rail popover body 264px, layers 260px, help
400px, monitor per-tab) — content adapts inside; panels don't reflow the
composition — **except under `(pointer: coarse)`**, where each fixed width is
clamped to the viewport with `min(<desktop width>, calc(100vw - margins))` and
the `vh` bounds become `dvh` (§11.5). On a fine pointer the values above are
exact.

### 11.5 Touch and coarse pointers

The viewer is used on phones and tablets (iPhone and iPad included) as well as
desktops. The rule that keeps the two from fighting: **touch adaptation is
keyed on the pointer, never on viewport width**, and it lives in **one file** —
`styles/components/coarse-pointer.css`, whose top level contains only `@media`
blocks on `(pointer: coarse)`, `(any-hover: none)` and `(any-hover: hover)`. A
narrow desktop window is not a touch device and a landscape tablet is not
narrow. `tests/unit/styles/coarse-pointer-css.test.ts` enforces the contract:
nothing outside media blocks, pointer-feature preludes only, no pointer/hover
media features in any other stylesheet, and the load-bearing clamps present.

- **Two features, two meanings.** `(pointer: coarse)` = the primary pointer is a
  finger: viewport clamps, safe areas, tap-friendly targets. `(any-hover: none)` =
  no pointer can hover: hover-revealed affordances are dead, so the rail does
  not idle-dim (`opacity: 1`) and fullscreen keeps it findable (`0.35`). An
  iPad with a trackpad matches the first and not the second and keeps its
  hover behaviour; do not collapse the two into one query.
- **Clamps, not reflow.** Fixed widths become `min(<desktop>, calc(100vw -
  margins))`; `vh` heights become `dvh` inside `@supports (height: 100dvh)`
  (iOS Safari's `vh` is the large viewport, so a `70vh` panel bottom-clips
  under the visible toolbar). The `vh` fallback stays outside `@supports` — in
  the component file when it already has one, otherwise in the plain
  `(pointer: coarse)` block.
- **Safe areas.** `index.html` declares `viewport-fit=cover`; every
  edge-anchored surface adds its matching `env(safe-area-inset-*)`. The rail
  gutter keeps the docking rule's sanctioned `!important` (§15.6).
- **The rail scrolls its items, never its root.** Popovers, flyouts and the
  footer are children of the rail root, so `overflow` on the root would clip
  them. The buttons live in `.luxar-control-rail__items`, `display: contents`
  on fine pointers (layout-transparent) and a `min-height: 0; overflow: hidden
  auto` scroll box under `(pointer: coarse)`, so a ~600px rail fits a ~340px
  landscape phone without horizontal panning. Its children and the root's
  collapse/footer controls do not shrink. Item tooltips are suppressed there
  because they cannot escape the scroll clip; the collapse handle remains
  outside the wrapper and keeps its label. Under `(any-hover: none)` a tip shows on
  keyboard focus only (`:focus-visible`), and the first-run hint names the tap
  and the press-and-hold instead of "Hover". `RailOverlay` anchors popovers with a root-relative rect, not
  `offsetTop`, so a scrolled wrapper still points the arrow at its button; the
  unit test stubs both rects to guard that distinction.
- **Buttons are `touch-action: manipulation`** (no 300 ms double-tap delay,
  no page zoom on a double-tap over UI). Canvas `touch-action: none` and gesture
  ownership are planned separately.
- **Hit sizes.** Under `(pointer: coarse)` the primary controls (rail buttons,
  chips, panel close) are 44px through a LOCAL `--luxar-hit-min` custom property
  set on the component roots — not a theme token, because the tokens are
  TS-generated across four theme files and a touch-only size is not a theme
  decision. Dense secondary controls (layer eye, play and step buttons) are
  36px; range thumbs use a matching hit band (24px normally, 28px for the
  two-thumb range slider) while the drawn track stays thin; checkboxes 24px.
- **16px inputs.** Every text/number/select inside a panel is `font-size: 16px`
  under a coarse pointer: below that iOS Safari zooms the page into a focused
  field and never zooms back. Numeric inputs also declare `inputmode="decimal"`
  and filter fields `inputmode="search"` (inert on desktop, so ungated in JS).
- **Press, not hover.** Under `(any-hover: none)` a tap leaves an element in a
  sticky `:hover` until the next tap elsewhere, so the hover styling is put
  back to the rest state and `:active` carries the response (a translucent
  highlight fill). The hover rule you restate must be the element's own rest
  state, so keep this list short and exact.
- **Coarse-only affordances are gated in JS on `getInputProfile()`**, never on
  width: a momentary **Hide panels** rail item (the on-screen Escape), the help
  and monitor joining the docked panels' one-surface exclusivity, `◀ ▶` step
  buttons and a tappable name chip on each dimension slider (the finger's `[ ]`
  and `1–9` keys), and the Home popover captioning on `pointerdown` for
  touch-like pointers. Elements that only exist on coarse pointers may take
  their base styling in `coarse-pointer.css`. The Fullscreen chip is gated on
  the real capability (`document.fullscreenEnabled`, absent on iPhone Safari),
  not on the device.

---

## 12. Interaction states & accessibility

Every interactive element defines, in this order:

1. **Rest** — muted foreground (`--luxar-text-muted|secondary`), transparent
   or `interactive-default` fill.
2. **Hover** — `--luxar-text-primary` + `interactive-hover` fill. Reveal-on-
   hover affordances (directory chevrons) go from `opacity: 0` → `0.7`.
3. **Active/selected** — highlight accent per §6.2.
4. **Focus** — `:focus-visible { outline: 2px solid var(--luxar-border-focus);
   outline-offset: 1–2px }` (negative offset inside dense lists). The global
   `:focus-visible` rule lives in `reset.css`, which ships via
   `standalone.css` alone (§2), so the **embed-safe** baseline is instead the
   `.luxar-glass-surface :focus-visible` rule in `base/utilities.css`. That is
   `index.css`'s first import, so it deliberately loses on cascade order to
   every later equal-specificity component rule — richer component styles
   still win. It covers any control inside a glass panel; a surface that is
   *not* a glass surface (frameless in-canvas widgets, §5) must still declare
   its own ring. Because `border.focus` is **optional** in the `Theme`
   interface (`themes/types.ts`), spell the ring with the dark-theme fallback
   — `var(--luxar-border-focus, rgba(76, 175, 80, 0.5))`, as `reset.css` and
   `control-rail.css` do: an undefined token makes the whole `outline`
   declaration invalid, which resolves to *no* ring and, being
   higher-specificity, suppresses the one `reset.css` would have drawn. (All
   four shipped themes define it, so this only bites a new theme — which is why
   seven rings in the tree, the embed-safe baseline itself among them, have
   been able to ship *without* the fallback they prescribe without anyone
   noticing; that is logged as drift in §15.2.) Text inputs
   may substitute a `--luxar-border-focus` border-color switch. Never
   `outline: none` without a visible replacement — a `tabindex="-1"`
   focus-trap *container* (the help overlay's root) is the one sanctioned
   exception, and it must be plain, not `!important`, so descendant rings
   survive.
5. **Disabled** — `opacity: 0.3–0.4`, `cursor: not-allowed` or `default`,
   `pointer-events: none` where semantics allow.

Further requirements:

- Semantic roles: `role="dialog"`+`aria-modal` on modals; `role="button"` +
  `tabIndex=0` + Enter/Space activation on div-buttons; `aria-label` on every
  icon-only control; `aria-hidden="true"` on decorative SVGs and separators;
  `aria-current` for breadcrumb position; `aria-expanded` on popover anchors;
  `aria-live="polite"` on status bars.
- Keyboard: Escape closes the topmost surface; lists support
  ArrowUp/Down/Home/End with `preventDefault()` even at the boundaries;
  a filter field hands off to its list via ArrowDown.
- **Modal surfaces trap and restore focus** — initial focus inside, Tab
  contained, focus returned to the opener on close, via the shared `trapFocus`
  (§7.2). `aria-modal` and a scrim do neither on their own.
- Never steal focus asynchronously: an auto-focus that fires after an await
  must first check the user hasn't focused something else.
- Reduced motion per §10.4. Color is never the only signal (pair with dimming,
  ticks, text) — and note that satisfying *that* rule says nothing about
  contrast: text still needs its own ratio (§4, §15.2). On a bare glass panel
  all three text ranks now clear AA against a worst-case backdrop (§4); two
  things sit outside that guarantee: a `bg.tertiary` inset surface, where
  only `text-muted` falls short of AA (`text-primary`/`text-secondary` still
  clear it there) (§15.2), and a surface-level `opacity` (e.g. the control
  rail idling at 0.55), which multiplies text and panel together and drops
  even `text-primary` well below AA — a case the token-level guarantee never
  covered in the first place (§4).

---

## 13. Naming & file conventions

- **Prefix everything** `luxar-` (embed safety).
- **BEM-ish**: `.luxar-<block>__<element>--<modifier>`
  (`.luxar-dataset-browser__file-item--current`). State classes: `.is-active`,
  `.is-collapsed`, `.is-awake`, `.is-fullscreen`, `.is-leaving` (rail family)
  or `--modifier` (BEM family) — stay consistent within a block.
- **Ids are for E2E/aria plumbing** (`#luxar-dataset-browser-title`), not
  styling hooks. Tests pin ids, classes, and user-visible strings — renaming
  any of them is a breaking change to the test suite; grep first.
- One CSS file per surface in `styles/components/`, header comment stating
  what it styles and which TS file drives it; keep the README table in
  `styles/README.md` in sync.
- Tokens only. A literal standing in for a token *value* needs both an inline
  comment justifying it and a line in §15.6 (§16). Two shapes are outside that
  rule and need only the comment: a `var(--token, <literal>)` fallback (§12.4
  mandates one on every focus ring, and 114 ship across the tree), and the
  §6.6 white-on-solid-semantic case (`recording-panel.css:202`). The model
  citizens of the registered kind: `select-menu.css`'s `#999999` data-URI
  chevron, the perf monitor's 7px micro-caption, and `colormap-legend.css`'s
  `border-radius: 2px` — deliberately half of `--luxar-radius-sm`, because
  `--luxar-radius-xs` does not exist (§3.4).
- `!important` policy — exactly three sanctioned categories: reduced-motion
  overrides; the documented rail-docking/nesting overrides in
  `control-rail.css`; state-forcing in `overlay-layer.css` that must beat
  inline styles. All three are registered in §15.6; anything else is a smell.
- Icons live in per-domain `icons.ts` modules exporting
  `Record<string, string>` of SVG strings.

---

## 14. Checklist for a new UI surface

1. Root: surface recipe (§7.1) + `luxar-glass-surface` (unless it is one of
   §5's three opt-outs — frameless by design, a transient cursor popover, §7.8,
   or one of the two un-glassed over-canvas badges, the toast and the REC pill).
   Being a badge is not itself the criterion: a badge that ramps `opacity` on
   its own root MUST be un-glassed (§5.1.3, the toast); one that does not may go
   either way (the REC pill is not glassed, the resolution indicator is — and
   ramps anyway, which is why §15.4 logs it as drift).
   + `overflow: visible` + inner `__scroll` wrapper if it scrolls.
2. Correct tier: panel / modal (+scrim) / popover / badge (§7.1 table) with
   token z-index — and if it docks beside the rail, it joins the exclusive
   dock rather than out-stacking the incumbents (§3.5, §7.5).
3. Header per §7.3 — add `.luxar-panel-header` + `.luxar-panel-close` rather
   than restating them; quiet-instrument surfaces use the tick micro-header.
4. All values tokens; spacing keys double-checked (§3.3 half-pixel trap).
5. Accent usage per §6 (highlight = interactive; green = semantic/tick/focus
   tokens only; `color-mix` tints, not solids).
6. Numerals mono+tabular; labels in the micro-voice (§8).
7. Icons: stroke SVG per §9 contract; no emoji; check for glyph collisions.
8. Motion: transform-only entry; one-shot reveals; full
   `prefers-reduced-motion` guard (§10).
9. States: rest/hover/active/focus-visible/disabled all defined (§12);
   component-level `:focus-visible` ring present, spelled with the fallback —
   `var(--luxar-border-focus, rgba(76, 175, 80, 0.5))` (§12.4).
10. A11y roles/labels/live-regions per §12; Escape wired; arrow keys in lists;
    a modal traps focus and restores it to the opener (`trapFocus`, §7.2).
11. No new `<option>` styling outside `select-menu.css`; no pseudo-elements on
    the glass root; no opacity animation on the glass root (§5.1).
12. Both glass themes eyeballed (default theme is frosted-glass; liquid-glass
    is the stress test), plus light theme; screenshots verified before PR.
13. New strings/ids/classes cross-checked against E2E specs (`grep` the
    `src/tests/e2e/` tree).
14. This guide updated if the surface introduces a new reusable pattern, and
    every sanctioned exception it needs (literal, `!important`, off-tier
    z-index) registered in §15.6 (§16).
15. Touch: a fixed width or `vh` bound gets its `(pointer: coarse)` clamp in
    `components/coarse-pointer.css` (never inline in the component file, never
    keyed on width alone); a bottom- or edge-anchored surface adds its
    `env(safe-area-inset-*)` there too (§11.5).

---

## 15. Known drift (documented debt — do not copy)

The code listed in §15.1–15.5 contradicts this guide. It is listed so nobody
mistakes it for precedent; migrate opportunistically when touching these files.
(§15.6 is the opposite list — sanctioned exceptions that stay.) (Inventory
verified 2026-08-11.) A modernization campaign closed most of it: #1476
(a11y), #1478 (token hygiene), #1479 (emoji→stroke icons), #1472 (the dataset
browser) and #1480 (accent migration + the shared panel recipes) have all
landed, so what they actually fixed is deleted below —
**green-as-interactive, emoji-in-the-DOM and the phantom radius token are gone
from the tree entirely.** What those tranches did not reach stays listed, token
hygiene included. Everything below is live on `main`; each entry names the PR
that will close it where one exists, and the entry goes away as that PR merges
(§16).

### 15.1 Hardcoded values / phantom tokens

- Two raw layer z-indexes are left: `5` on the `overlay-layer.css`
  `.luxar-overlay` container and the scene-identity banner's inline `10000`
  (its own bullet below). Both are *global* layers (the overlay container
  holds screen-space overlays between the canvas and all UI), not local
  stacking indexes, and the first sits below `--luxar-z-base` — a tier the
  scale doesn't name. Give each a token (or a comment justifying the literal)
  when those files are next touched. NOT drift: the recording panel's
  `9999/10000/100000` stay literal by design and small local stacking indexes
  (`1/2/10` inside a positioned parent) are not layer tokens — both registered
  in §15.6.
- **The frameless in-canvas widgets sit on the tooltip tier.** The scale bar
  (`scale-bar.css:35`), the colormap legend (`colormap-legend.css:13`) and the
  resolution indicator (`resolution-indicator.css:67`) all resolve to
  `--luxar-z-tooltip` (4000), so they paint over every panel and modal — the
  opposite of what §11 ("bottom corners, frameless") implies and of the `base`
  tier their role calls for. The first two also carry a stale
  `var(--luxar-z-tooltip, 900)` fallback: 900 was never a tier value, and it is
  inert because the token is always defined. Move them to `--luxar-z-base` when
  next touching those files.
- **The whole `config.ui.zIndex` scale is a second, untokenized layer system.**
  `src/config/sections/ui/data.ts` declares eleven numeric layers (tabulated in
  §3.5); `ui/layers/layers-panel.ts:346`, `ui/performance-monitor.ts:87`,
  `ui/recording-panel.ts:168` and `ui/rendering-controls.ts:189` write four of
  them straight onto `el.style.zIndex`, and
  `styles/components/recording-panel.css:7-9` names the scale in its header
  comment. Three of the live values (`layersPanel`/`recordingPanel` 1500,
  `renderingControls` 1999) fall between the `dropdown` and `modal` token
  tiers, so the §3.5 token table is not sufficient on its own to say what
  paints over what. Move these onto the tokens when next touching those panels
  (and drop the seven unread keys); new surfaces use the tokens.
- The recording **start-confirm** dialog (`recording-panel.css`; it is rendered
  by `ui/recording-panel/session.ts:308,315`, whose title and primary button
  both read "Start … Recording") is on the surface tokens now but still isn't a
  `luxar-glass-surface`, blurs with a raw `blur(2px)` instead of a blur token,
  spaces itself in raw px, and carries a literal `rgba(255,255,255,0.98)`
  light-theme background. (The REC indicator pill itself is on the badge
  variant of the recipe — that half is done.)
- The dataset browser restates `.luxar-panel-close` (§7.3) as its own
  `__close-btn` (`dataset-browser.css:111-143`) instead of adding the shared
  class, and the copy has since drifted from the original in three ways: its
  `:focus-visible` outline (`:141`) omits the mandatory `--luxar-border-focus`
  fallback the shared rule carries — under a theme that leaves `border.focus`
  undefined that is *no ring at all*, not a duplicate ring (§12.4, §15.2); its
  `svg` rule omits `display: block`; and its `transition` (`:122`) is not
  covered by the file's reduced-motion block, whose one rule (`:768-773`) lists
  only `.luxar-dataset-browser`, `-scrim` and `__skeleton-row`. Fold the shared
  class in when next touching that file.
- The dataset browser's header row is per-panel CSS where `utilities.css` calls
  `.luxar-panel-header` "THE header treatment for every panel" and §7.3 / §14.3
  both say to add it rather than restate it. Exactly one thing genuinely
  diverges: the hairline is pushed down to the tagline banner (`:148`) so title
  and tagline read as one block. The rest (`display: flex; justify-content:
  space-between; align-items: center`, `:84-86`) is a verbatim restatement of
  three of the shared class's six declarations. The migration is the shared
  class plus a local `border-bottom: none` **and** `margin-bottom: 0`:
  `.luxar-panel-header` (`base/utilities.css:461-468`) also carries
  `padding-bottom: var(--luxar-spacing-3)` and `margin-bottom:
  var(--luxar-spacing-5)` (10px). The header's own padding shorthand (`:83`)
  already overrides the first, but nothing overrides the second, and the
  tagline banner follows the header directly (`:146-148`), so adopting the
  class bare would open a 10px gap under the title row. Not another
  hand-rolled header either way.
- The scene-identity banner (`ui/scene-identity-banner.ts`) takes only its
  material from the shared system (it does set
  `className = 'luxar-glass-surface'`, `:74`) and spells everything else —
  layout, color, type — inline in `style.cssText`, rather than in a
  `styles/components/*.css` file on the surface recipe: raw `rgba()`
  backgrounds and borders, `z-index: 10000` (§3.5), literal `color: #fff`
  (not the §6.6 case — this is over a translucent tint, not a solid semantic
  fill), `font: 13px system-ui` instead of the font/size tokens, and a
  `border-radius: 8px` literal. Its two glyphs also inline their
  presentation attributes at `width/height="16"` on a 24-grid `viewBox`,
  which is neither the §9.1 rail contract (geometry only, CSS paints) nor the
  §9.2 micon one — a new banner-like surface should get a component
  stylesheet and the §9.1 contract, not copy this.
- Assorted raw `rgba()` duplicating tokens: the GUI library's
  `rgba(0, 0, 0, …)` control fills (`ui/gui/styles/controller.css`); the debug
  console's `rgba(0, 0, 0, 0.3)` filter and content backgrounds (`:107`,
  `:151` — that is exactly `--luxar-bg-tertiary`), its scrollbar track (`:163`)
  and its `rgba(255, 255, 255, 0.05)` message hover (`:199`); the error
  dialog's guidance panel (`:141`), `<code>` chip (`:172`, again `bg-tertiary`)
  and `<kbd>` chip (`:180`); in the monitor, hairlines at
  `rgba(255, 255, 255, 0.1)` (`:1189`, `:1202` = `border-default`) and `0.05`
  (`:1219` = `border-subtle`) plus two *backgrounds* at the same `0.1` (the
  progress-bar track `:300` and the scene-graph badge `:934`, both
  `interactive-default`); and `color: white` in overlay-layer and
  dimension-sliders, plus the same literal spelled `color: #fff` at
  `recording-panel.css:270,278`. (#1478 converted the debug console's
  warn/error row tints, the monitor's active-level and kind badges and the
  loading-indicator chrome — those are done.) **The cited lines are
  representative, not exhaustive** — every file named here has uncited
  siblings, in both kinds of rule: light-theme overrides
  (`debug-console.css:264,268,272,276`, `error-dialog.css:202,214`) and base
  rules (`data-loading-monitor.css:1046,1056,1157,1246`). Grep `rgba(` in a
  file before declaring it migrated and deleting this entry.
- The tick-less legacy `.luxar-section-title` recipe in
  `data-loading-monitor.css` (§8.3 rank 1 is the current one).

### 15.2 Accessibility gaps

- **No modal in the tree contains focus except via `trapFocus`, and the dataset
  browser doesn't use it.** The help overlay and error overlay do (§7.2); the
  dataset browser places initial focus on its filter field but leaves Tab free
  to walk out behind the panel, and does not restore focus to the opener on
  close. Its `aria-modal="true"` therefore over-promises. Wiring `trapFocus`
  into `open()`/`close()` is the fix — #1508 does exactly that; this entry
  goes away when it merges.
- **The embed-safe focus baseline does not follow §12.4's own fallback rule.**
  A sweep of `styles/**/*.css` + `ui/**/*.css` for an `outline:` naming
  `--luxar-border-focus` finds exactly seven rings spelled without the
  fallback: the baseline `.luxar-glass-surface :focus-visible`
  (`base/utilities.css:450`), the dataset browser's five
  (`dataset-browser.css:141,243,302,525,727`) and the GUI slider's
  (`ui/gui/styles/controller.css:92`). Since `border.focus` is optional in the
  `Theme` interface (`themes/types.ts:81`), a theme that omits it gets no ring
  from any of them — and the baseline is precisely the rule that is supposed to
  guarantee one to embedders. `.luxar-panel-close` (`utilities.css:501`),
  `reset.css` and `control-rail.css` spell the fallback correctly and are the
  pattern. All four shipped themes define the token, so nothing is broken
  today.
- **`bg.tertiary` inset surfaces on the glass themes are still below AA for
  `text-muted`.** #1513 closed the panel-level gap (§4) by darkening the glass
  tints and lifting `text.secondary`/`text.muted`, but a `bg.tertiary` inset
  lift is painted ON TOP of the panel and is *lighter* than it, which erodes
  some of that margin back. Frosted-glass: `rgba(255, 255, 255, 0.07)` over
  the worst-case 0.75-tint panel lands near `rgb(97, 98, 102)`, giving
  `text-muted` ~4.1:1 — still under 4.5:1. Liquid-glass: `rgba(255, 255, 255,
  0.1)` over the worst-case (now 0.68-alpha) `--luxar-glass-tint` panel lands
  near `rgb(99, 99, 99)`, giving `text-muted` ~4.0:1. Closing this too would
  need a markedly darker frost/tint than #1513's numbers, trading away
  headroom the panel-level fix deliberately kept modest. Until then,
  `text-muted` is the one rank that should not carry load-bearing copy on a
  `bg.tertiary` inset surface.
- **`text-disabled` is used as a live rank-3 label, not only for disabled
  controls.** `.luxar-text-disabled` (`base/utilities.css:173`) and direct
  `--luxar-text-disabled` uses total ~16 sites: 13 in
  `data-loading-monitor.css` (e.g. `.luxar-metric-card__subtitle` at `:288`
  and `.luxar-progress-bar__label` at `:330`, both live metric captions, not
  disabled state) and 2 in `debug-console.css` (`:181`, `:203`), plus the
  class definition itself. Over the new worst-case panels this token measures
  ~2.09:1 (frosted) / ~2.15:1 (liquid) — well under AA. WCAG 1.4.3 exempts
  genuinely disabled controls, but these are not disabled controls. Either
  those sites move up a rank, or §8.3 stops listing a disabled-named token as
  a live text rank — recorded here; the token itself is not raised, since
  that would blur the actual disabled affordance.
- **Element `opacity` on a whole surface multiplies text and panel together,
  and is invisible to a token-level contrast check.** `.luxar-control-rail`
  idles at `opacity: 0.55` (`control-rail.css:29`), which drops even
  `text-primary` to ~2.45:1 (frosted-glass) / ~2.52:1 (liquid-glass) over a
  worst-case white scene until hover / `.is-awake` restores full opacity.
  This is sanctioned by §5.1's
  steady-translucency exception (§5.1.3, §10.2), but §4's "all six clear AA"
  claim must not be read as covering it — this is the same blind spot that
  hid #1513's liquid-glass `opacity: 0.7` bug on the `::after` tint layer.

(#1476 closed the ring/reduced-motion gaps — the embed-safe
`.luxar-glass-surface :focus-visible` baseline (§12.4), the GUI slider's missing
ring — both of those are also two of the seven no-fallback rings faulted in the
bullet above — the help overlay's `!important` outline suppression, and the
reduced-motion gaps in the GUI library, layers panel, toast, debug console and
overlay fade — and #1472 added the dataset browser's reduced-motion block, the
last stylesheet that lacked one.)

### 15.3 Divergent contracts (tolerated, bounded)

- Two icon contracts exist by design (§9.1 rail vs §9.2 monitor micro) — do
  not invent a third.
- `error-overlay.ts` keeps its one warning glyph as a module-local
  `ALERT_ICON` const rather than an `icons.ts` module (§13). Fine for a
  single glyph; a second one there means promoting it to a module.
  (`scene-identity-banner.ts` already carries two in a module-local `ICONS`
  record — the case that rule is about; see §15.1.)
- Layers-panel selection uses `--luxar-info`; everything else uses
  `--luxar-highlight`. New selection UIs use highlight.

### 15.4 Glass-constraint (§5.1) and layer-tier violations

- The resolution indicator ramps `opacity` on its own `luxar-glass-surface`
  root — the class is added at `ui/resolution-indicator.ts:48`, and
  `resolution-indicator.css:70` runs `animation: luxar-resolution-fade-in 0.3s
  ease-out forwards` on that root, with the `--hidden` modifier (`:78`) swapping
  in `luxar-resolution-fade-out`; both keyframes (`:12-32`) ramp `opacity` 0→1 /
  1→0. It is the remaining §5.1.3 violation, so under liquid-glass its
  refraction layers ride the ramp with it. Fix by moving the ramp to an inner
  wrapper, or by dropping the glass class as the toast did (#1483) and taking
  the theme's dark tint explicitly. Until then, do not cite it as precedent
  (§10.2).
- One standing panel sits on the baseline tier rather than the `dropdown`
  layer its placement implies (§3.5): `data-loading-monitor.css:12`,
  `--luxar-z-base`. It is the only component stylesheet left on that token,
  which is why §3.5's table names it.

### 15.5 Private context menu predating the shared widget

The dimension-slider animation menu (`dimension-sliders.ts::showContextMenu`,
`.luxar-dimension-slider__context-menu`) hand-rolls the cursor-anchored menu
that §7.8's shared widget now owns: right algorithm (clamped fixed position,
next-tick outside-click, Escape) but zero menu ARIA, no roving focus, no
focus return, and an opacity `luxar-fade-in` entry where §10.2 wants
transform-only (legal only because the menu is not a glass surface, §7.8).
Migrate it onto `openContextMenu()` when next touched — carefully: its BEM
classes and behavior are pinned by `dimension-animation.spec.ts`
(`src/tests/e2e/`), so the E2E pins move in the same PR. Until then it is not
the pattern to copy (§7.8 is).

### 15.6 Sanctioned exceptions (not drift — do not migrate)

Unlike §15.1–15.5, these are deliberate and stay. They are registered here
because §16 requires every sanctioned `!important`, off-tier z-index and
token-substituting literal to be written down (§13 exempts `var()` fallbacks
and the §6.6 white-on-solid case — those need only their inline comment).
Deleting an entry means the exception itself went away, not that it was
migrated.

- Literals standing in for token values (§13): `select-menu.css`'s `#999999`
  data-URI chevron (a data-URI cannot read a custom property), the perf
  monitor's 7px micro-caption (`performance-monitor.css:62`, below the
  `--luxar-text-xs` floor by design), `colormap-legend.css`'s
  `border-radius: 2px` on the gradient bar (`:50`) — half of
  `--luxar-radius-sm`, because 4px corners on a 12px-tall pixelated bar read as
  a pill and `--luxar-radius-xs` does not exist (§3.4) — and the
  `<kbd>`/`<code>` chip radii, deliberately theme-invariant (§8.4/§8.5).
- `!important` (§13's three categories): the reduced-motion overrides that need
  one, which exist in exactly eight stylesheets (`base/utilities.css` and, in
  `components/`, `colormap-legend`, `data-loading-monitor`, `dimension-sliders`,
  `error-dialog`, `recording-panel`, `resolution-indicator`, `scale-bar`);
  every other reduced-motion block in the tree spells a plain
  `animation: none` / `transition: none` and needs no override, so `!important`
  is not automatic there. Also sanctioned: the rail-docking gutter
  `left: 73px !important` (`control-rail.css:547`, §7.5) — restated with the
  safe-area inset under `(pointer: coarse)` in `coarse-pointer.css` (§11.5) — and the
  popover-nesting overrides that unpin a GUI mounted inside a popover
  (`control-rail.css:388-391`); and the state-forcing rules in
  `overlay-layer.css:33-34` that must beat inline styles.
- Off-tier z-indexes via `calc()` (§3.5): the modal scrim at
  `calc(var(--luxar-z-modal) - 1)` (`dataset-browser.css:18`), the debug
  console at `calc(var(--luxar-z-base) + 50)` (`debug-console.css:31`), the
  toast at `calc(var(--luxar-z-tooltip) + 1000)` (`toast.css:28`).
- Untokenized z-index magnitudes with a stated reason: the recording panel's
  `9999/10000/100000` (`recording-panel.css:80,132,225`), whose whole point is
  to beat unknown third-party host UI (documented in that file's header). Small
  local stacking indexes (`1/2/10` inside a positioned parent) are not layer
  values at all and need no entry.
- **A stylesheet-declared custom property, not a `ThemeManager` token** (§3):
  `--luxar-glass-tint` (`styles/themes/liquid-glass.css`, under the
  `[data-theme='liquid-glass']` selector) is the dark tint painted by
  `.luxar-glass-surface::after` — liquid-glass's own internal implementation
  detail, with every consumer living in that same file: the `::after` rule
  itself, and every non-`.luxar-glass-surface` surface that would otherwise
  paint white text over too-light a background on a bright scene (currently
  the recording indicator, cursor-anchored context menus, control-rail
  tooltips, the control-rail first-run hint, and the toast (#1483) — all
  white-on-white over the bare canvas — plus the recording confirmation
  dialog and the offline capture overlay, which sit over the
  `--luxar-bg-overlay` scrim and so are merely too-light rather than literally
  white-on-white). A new member of that list spells
  `var(--luxar-glass-tint)`, not a fresh `rgba()` literal: the old duplicated
  `rgba(0, 0, 0, 0.55)` is 4.48:1 against `text-primary` over a white scene,
  which is what #1513 was. It stays a plain
  stylesheet declaration rather
  than moving into `ThemeManager`'s `themeToCSSVariables()` because it is
  this one theme's CSS-layer implementation detail — a tint painted by a
  pseudo-element — not a member of the `Theme` interface, so it does not
  belong in the token vocabulary.

---

## 16. Keeping this guide authoritative

- Any PR that changes tokens (`theme-manager.ts`, `*.theme.ts`), the glass
  system, an icon contract, or a shared recipe **must update this guide in
  the same PR**.
- Any new sanctioned exception — a literal standing in for a token value,
  an `!important`, an off-tier z-index — must be added to **§15.6** with its
  justification. (`var()` fallbacks and the §6.6 case are exempt; §13.)
- When migrating drift out of **§15.1–15.5**, delete the entry. §15.6 is not
  drift — leave it alone unless the exception itself goes away.
- Cross-references: `src/styles/README.md` (CSS architecture),
  `src/themes/README.md` (theme system mechanics),
  `docs/guides/developer/CONSOLE_OUTPUT_STYLE.md` (console voice),
  `docs/guides/user/VIEWER_GUIDE.md` (user-facing behavior).
