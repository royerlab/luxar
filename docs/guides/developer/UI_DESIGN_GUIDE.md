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
| Monitor icons | `src/ui/data-loading-monitor/templates.ts` (`MONITOR_ICONS`) |
| Dataset-browser icons | `src/ui/dataset-browser/icons.ts` (`BROWSER_ICONS`) |
| Shared panel header/close recipes | `src/styles/base/utilities.css` (`.luxar-panel-header`, `.luxar-panel-close`) |
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
inline-style specificity). Component CSS must reference tokens, never
hardcoded values (sanctioned exceptions are listed in §15).

The full vocabulary is **87 variables** (count them with a grep over
`themeToCSSVariables` when in doubt). Dark-theme values shown; other themes
override per §4.

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
| `--luxar-z-base` | 100 | In-canvas widgets (scale bar, legend) |
| `--luxar-z-dropdown` | 1000 | Docked panels, the rail |
| `--luxar-z-modal` | 2000 | Modal dialogs (+ scrim at `calc(var(--luxar-z-modal) - 1)`) |
| `--luxar-z-popover` | 3000 | Rail popovers/flyouts, first-run hint |
| `--luxar-z-tooltip` | 4000 | Tooltips |

Use the tokens. The historical raw z-indexes still in the tree (§15) are debt,
not precedent.

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
| `bg.secondary` (panel) | `rgba(30,30,30,0.95)` | `rgba(250,250,250,0.95)` | `rgba(28,30,36,0.65)` — a **dark frost** (contrast floor, see below) | `rgba(255,255,255,0.15)` + a dark `::after` |
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
  panel tint must keep text legible over ANY scene, so `bg.secondary` is a ~65%
  dark layer under the blur — the same contrast-protection role liquid-glass's
  dark `::after` plays. Composited over the worst case (a pure-white scene) the
  panel lands near `#6b6d71`, which gives roughly **4.9:1 against
  `text-primary`** (clears WCAG AA 4.5:1 for the 13px body text), **~3.7:1
  against `text-secondary`** and **~2.5:1 against `text-muted`**. So: body copy
  and any load-bearing value go in `text-primary`; `secondary`/`muted` are for
  labels, hints and idle icons that are not the only carrier of meaning (§12).
  Never lighten a glass panel tint without re-running that worst-case
  composite.
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
  toast, resolution indicator, error overlay, scene-identity banner, rail
  flyout/popover).
- **Opt out**: frameless in-canvas widgets (scale bar, colormap legend) are
  deliberately NOT glass — they use drop-shadows instead of a panel material.
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
  aberration), `::after` (z:-1, dark tint `rgba(0,0,0,0.55)` + four inset
  bevel shadows).

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
intended, authoritative direction (existing violations are cataloged in §15).

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

…plus `luxar-glass-surface` on the root element in TS. Do NOT add per-theme
`box-shadow` rings on top (removed deliberately in PR #447).

**Tier variants:**

| Variant | Radius | Shadow | Blur | z |
| --- | --- | --- | --- | --- |
| Panel (default) | `lg` | `lg` | `md` | `dropdown` |
| Modal dialog | `lg` | **`xl`** | `md` | `modal` (+ scrim) |
| Popover / flyout / tooltip | `lg` (tooltip `md`) | `xl` (tooltip `lg`) | `md` | `popover` / `tooltip` |
| Transient badge / toast | **`md`** | `md`–`lg` | `sm`–`md` | context |

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
browser predates the extraction and still spells both out in its own file with
matching values (§15.1) — copy the classes, not that file's header block.

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
  an `!important` override of inline positioning
  (sanctioned, see §15.1); the draggable debug console gets the same default
  *without* `!important` so dragging still wins.
- **The rail's left dock is exclusive** — Rendering, Layers and Recording all
  open at that one position, so activating any of them from the rail (or
  opening a rail popover) closes the others rather than stacking. Keyboard
  shortcuts (R/L/T) deliberately bypass this, so panels can still be stacked
  on purpose. A new rail-anchored surface must join this handshake.

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

### 8.5 `<code>` chips

Inline paths/extensions: mono, `--luxar-text-xs`,
`background: var(--luxar-bg-tertiary)`, `padding: 1px 4px`, radius 3–4px.

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
- Scrims and genuinely **non-glass** transients (badges, frameless in-canvas
  widgets): opacity fades are fine. **The toast is not one of them** — it
  carries `luxar-glass-surface` (`ui/toast.ts:17`), so §5.1.3 governs its
  root: a new transient of that shape fades an inner wrapper or moves with
  transform. The shipped toast fades its own root; that is drift (§15.4), not
  the pattern to copy.
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
composition.

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
   four shipped themes define it, so this only bites a new theme.) Text inputs
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
  ticks, text).

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
- Tokens only; each sanctioned literal needs an inline comment justifying it
  (`select-menu.css`'s `#999999` data-URI chevron and the perf monitor's
  7px micro-caption are the model citizens).
- `!important` policy — exactly three sanctioned categories: reduced-motion
  overrides; the documented rail-docking/nesting overrides in
  `control-rail.css`; state-forcing in `overlay-layer.css` that must beat
  inline styles. Anything else is a smell.
- Icons live in per-domain `icons.ts` modules exporting
  `Record<string, string>` of SVG strings.

---

## 14. Checklist for a new UI surface

1. Root: surface recipe (§7.1) + `luxar-glass-surface` (unless frameless by
   design) + `overflow: visible` + inner `__scroll` wrapper if it scrolls.
2. Correct tier: panel / modal (+scrim) / popover / badge (§7.1 table) with
   token z-index.
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
   component-level `:focus-visible` ring present.
10. A11y roles/labels/live-regions per §12; Escape wired; arrow keys in lists;
    a modal traps focus and restores it to the opener (`trapFocus`, §7.2).
11. No new `<option>` styling outside `select-menu.css`; no pseudo-elements on
    the glass root; no opacity animation on the glass root (§5.1).
12. Both glass themes eyeballed (default theme is frosted-glass; liquid-glass
    is the stress test), plus light theme; screenshots verified before PR.
13. New strings/ids/classes cross-checked against E2E specs (`grep` the
    `src/tests/e2e/` tree).
14. This guide updated if the surface introduces a new reusable pattern.

---

## 15. Known drift (documented debt — do not copy)

The following existing code contradicts this guide. It is listed so nobody
mistakes it for precedent; migrate opportunistically when touching these
files. (Inventory verified 2026-08-11.) A modernization campaign closed most of
it: #1476 (a11y), #1479 (emoji→stroke icons), #1472 (the dataset browser) and
#1480 (accent migration + the shared panel recipes) have all landed, so their
entries are deleted below — **green-as-interactive and emoji-in-the-DOM are
gone from the tree entirely.** #1478 (token hygiene) is the one tranche still
open. Everything below is live on `main`; each entry names the PR that will
close it where one exists, and the entry goes away as that PR merges.

### 15.1 Hardcoded values / phantom tokens — largely pending #1478

- `colormap-legend.css` references the **non-existent** `--luxar-radius-xs`
  (falls back to its literal).
- Raw z-indexes: `150` (debug console), `10000` (toast, dimension-slider
  menu), `5` (the `overlay-layer.css` container — a global layer, not a local
  stacking index) — tokenized by #1478. NOT drift (correction to the original
  inventory): the recording panel's `9999/10000/100000` stay literal by
  design (their magnitude beats unknown third-party host UI — documented in
  the file header), and small local stacking indexes (`1/2/10` inside a
  positioned parent) are not layer tokens.
- The recording **stop-confirm** dialog (`recording-panel.css`) is on the
  surface tokens now but still isn't a `luxar-glass-surface`, blurs with a raw
  `blur(2px)` instead of a blur token, spaces itself in raw px, and carries a
  literal `rgba(255,255,255,0.98)` light-theme background. (The REC indicator
  pill itself is on the badge variant of the recipe — that half is done.)
- The dataset browser restates `.luxar-panel-header` / `.luxar-panel-close`
  (§7.3) in its own file instead of adding the shared classes. The values
  match, so this is duplication rather than a visual break; fold it in when
  next touching that file.
- The scene-identity banner (`ui/scene-identity-banner.ts`) styles itself
  entirely from inline `style.cssText` rather than a
  `styles/components/*.css` file on the surface recipe: raw `rgba()`
  backgrounds and borders, `z-index: 10000` (§3.5), literal `color: #fff`
  (§6.6), `font: 13px system-ui` instead of the font/size tokens, and a
  `border-radius: 8px` literal. Its two glyphs also inline their
  presentation attributes at `width/height="16"` on a 24-grid `viewBox`,
  which is neither the §9.1 rail contract (geometry only, CSS paints) nor the
  §9.2 micon one — a new banner-like surface should get a component
  stylesheet and the §9.1 contract, not copy this.
- Assorted raw `rgba()` duplicating tokens: the GUI library's
  `rgba(0, 0, 0, …)` control fills (`ui/gui/styles/controller.css`),
  debug-console warn/error tints, monitor hairlines and the
  `rgba(120,170,255,…)` kind-badge/active-level blue, error-dialog
  spinner chrome, `color: white` in overlay-layer and dimension-sliders.
- The tick-less legacy `.luxar-section-title` recipe in
  `data-loading-monitor.css` (§8.3 rank 1 is the current one).

### 15.2 Accessibility gaps

- **No modal in the tree contains focus except via `trapFocus`, and the dataset
  browser doesn't use it.** The help overlay and error overlay do (§7.2); the
  dataset browser places initial focus on its filter field but leaves Tab free
  to walk out behind the panel, and does not restore focus to the opener on
  close. Its `aria-modal="true"` therefore over-promises. Wiring `trapFocus`
  into `open()`/`close()` is the fix.

(#1476 closed the ring/reduced-motion gaps — the embed-safe
`.luxar-glass-surface :focus-visible` baseline (§12.4), the GUI slider's missing
ring, the help overlay's `!important` outline suppression, and the
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

### 15.4 Glass-constraint violations (§5.1)

- The toast fades `opacity` on its own `luxar-glass-surface` root
  (`toast.css` `transition: opacity 0.3s ease`, driven by
  `ui/toast.ts:19,24`) — the one surface still doing what §5.1.3 forbids, so
  under liquid-glass its refraction layers ride the fade with it. Fix by
  moving the fade to an inner wrapper (or dropping the glass class); until
  then, do not cite it as precedent (§10.2).
- `.luxar-dimension-sliders` is a glass surface whose root sets
  `overflow-y: auto` instead of delegating to an inner `__scroll` wrapper
  (§5.1.2/§7.4), and sits at `--luxar-z-base` rather than the `dropdown`
  layer its placement implies (§3.5).

---

## 16. Keeping this guide authoritative

- Any PR that changes tokens (`theme-manager.ts`, `*.theme.ts`), the glass
  system, an icon contract, or a shared recipe **must update this guide in
  the same PR**.
- Any new sanctioned exception (literal value, `!important`, extra z-index)
  must be added to §15 with its justification.
- When migrating drift out of §15, delete the entry.
- Cross-references: `src/styles/README.md` (CSS architecture),
  `src/themes/README.md` (theme system mechanics),
  `docs/guides/developer/CONSOLE_OUTPUT_STYLE.md` (console voice),
  `docs/guides/user/VIEWER_GUIDE.md` (user-facing behavior).
