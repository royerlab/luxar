# luxar-viewer/src/styles/themes

CSS overrides for the two glass-effect themes. These stylesheets are imported
by `../index.css` and are **applied last** so they win over base component
styles when the matching `[data-theme=...]` attribute is set on `:root`.

> Not to be confused with [`src/themes/`](../../themes/README.md) — that is
> the TypeScript `ThemeManager` (singleton, CSS custom-property injector,
> SVG glass filter pipeline, four `Theme` objects). The files here are the
> _pure-CSS_ visual overrides those two glass themes need on top of the
> custom properties; the `dark` and `light` themes need no override CSS.

## Files

| File                | Activated when `data-theme=` | What it does                                                                                                                                                                                                                                                                                                                               |
| ------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `frosted-glass.css` | `frosted-glass`              | Applies `backdrop-filter: var(--luxar-blur-md) saturate(1.5)` plus `isolation: isolate` to every glass-capable surface. (Native `<option>` colors are now centralized in `styles/components/select-menu.css`, driven by the `--luxar-menu-*` tokens.)                                                                                       |
| `liquid-glass.css`  | `liquid-glass`               | Three-layer effect: a real `.luxar-glass-refraction` DOM element behind, `::before` for backdrop blur + `url(#luxar-liquid-refraction)` SVG filter, `::after` for the dark tint and inset shine. Tunes inner controls (GUI inputs, dataset browser items, debug console, layers panel, etc.) to be transparent so the glass shows through. |

## Glass-capable surfaces

Both stylesheets target the same set of top-level panel selectors so the
effect is uniform:

```
.luxar-help-overlay        .luxar-data-monitor
.luxar-error-dialog        .luxar-dimension-sliders
.luxar-dataset-browser     .luxar-layers-panel
.luxar-debug-console       .luxar-gui
```

## Relationship to `src/themes/`

- `src/themes/` (TypeScript) registers four `Theme` objects and injects
  their colors/spacing/effects as `--luxar-*` custom properties on `:root`,
  and (for `liquid-glass`) injects the `<svg>` filter referenced here by
  `url(#luxar-liquid-refraction)`. See `glass-filters.ts` over there for
  the filter pipeline (`blurRadius`, `refractionScale`,
  `chromaticStrength`, `specularIntensity`).
- `src/styles/themes/` (this folder, CSS) carries the override rules that
  cannot be expressed as custom properties — stacking-context isolation,
  pseudo-element layering, `filter: url(...)` references, and per-component
  transparency tweaks for the liquid-glass aesthetic.

The two pieces are coupled: switching the body's `data-theme` attribute
(done by `ThemeManager.setTheme()`) is what activates the matching
selectors in these files.

## See Also

- [`../README.md`](../README.md) — overall styles architecture
  (`index.css` vs `standalone.css`, base/, components/)
- [`../../themes/README.md`](../../themes/README.md) — `ThemeManager`,
  `Theme` interface, SVG glass-filter pipeline
- [`../components/`](../components/README.md) — per-component base styles
  these overrides modify
