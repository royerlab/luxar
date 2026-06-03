# luxar-viewer/src/styles/base

Foundation stylesheets that sit beneath the viewer's component and theme
layers. `layout.css` and `typography.css` are global (host-page chrome)
and are imported only by `standalone.css`. `utilities.css` is fully
namespaced under `.luxar-*` and ships with the embed-safe `index.css`
bundle.

## Files

- `layout.css` — Full-viewport `html`/`body`/`#app` rules, the
  `.luxar-container` width helper, the `.luxar-grid` /
  `.luxar-grid-cols-{1..4}` grid helpers, and the dark-theme
  `::-webkit-scrollbar` + Firefox `scrollbar-color` styling. Global —
  standalone app only.
- `typography.css` — `body` font defaults, `h1`–`h6` sizes via
  `--luxar-text-*`, monospace styling for `code`/`pre`/`kbd`/`samp`, and
  a `::selection` override. Global — standalone app only.
- `utilities.css` — Tailwind-like atomic helpers, all prefixed with
  `luxar-`. Embed-safe: every selector is scoped.

## Utility groups (`utilities.css`)

| Group       | Examples                                                                   |
| ----------- | -------------------------------------------------------------------------- |
| Flexbox     | `luxar-flex`, `luxar-flex-col`, `luxar-items-center`                       |
| Justify     | `luxar-justify-start` ... `luxar-justify-around`                           |
| Gap         | `luxar-gap-0` ... `luxar-gap-12`                                           |
| Padding     | `luxar-p-0` ... `luxar-p-12`                                               |
| Margin      | `luxar-m-{0,2,4,6,8}`, `luxar-mb-{0,2,4,6,8,12}`                           |
| Text color  | `luxar-text-{primary,secondary,muted,disabled,success,warning,error,info}` |
| Text size   | `luxar-text-{xs,sm,base,md,lg,xl,2xl,3xl}`                                 |
| Text weight | `luxar-text-{normal,medium,semibold,bold}`                                 |
| Text align  | `luxar-text-{left,center,right}`                                           |
| Font family | `luxar-font-base`, `luxar-font-mono`                                       |
| Surface     | `luxar-surface-{primary,secondary,tertiary,elevated}`                      |
| Border      | `luxar-border`, `luxar-border-subtle`, `luxar-border-strong`               |
| Radius      | `luxar-rounded-{none,sm,md,lg,full}`                                       |
| Shadow      | `luxar-shadow-{sm,md,lg,xl}`                                               |
| Backdrop    | `luxar-blur-{sm,md,lg}`                                                    |
| Transition  | `luxar-transition`, `luxar-transition-{fast,slow}`                         |
| Display     | `luxar-hidden`, `luxar-block`, `luxar-inline-block`                        |
| Position    | `luxar-relative`, `luxar-absolute`, `luxar-fixed`                          |
| Overflow    | `luxar-overflow-{hidden,auto,y-auto}`                                      |
| Cursor      | `luxar-cursor-{pointer,default}`                                           |
| Opacity     | `luxar-opacity-{disabled,secondary,hover,full}`                            |
| Z-index     | `luxar-z-{base,dropdown,modal,popover,tooltip}`                            |
| Animation   | `luxar-animate-fade-in` (`@keyframes luxar-fade-in`, 0.15s ease)           |

All values resolve to `--luxar-*` custom properties injected by the
`ThemeManager` — no hardcoded colors, spacing, or radii.

## Conventions

- `layout.css` and `typography.css` are **global** (they style the bare
  `html`, `body`, `h1`–`h6`, `::-webkit-scrollbar`, etc.) and therefore
  ship only through `standalone.css`. Embedders must not import them.
- `utilities.css` is **embed-safe** — every selector starts with
  `.luxar-`, so it is loaded by the library entry `index.css`.
- The `@media (prefers-reduced-motion: reduce)` block at the bottom of
  `utilities.css` neutralises `luxar-animate-fade-in`; this is the only
  place `!important` appears in this folder, by design.

## See Also

- [`../README.md`](../README.md) — Full CSS architecture and which
  bundle each file ships in.
- [`../../themes/`](../../themes/) — Source of the `--luxar-*` custom
  properties these utilities consume.
