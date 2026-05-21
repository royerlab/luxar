# Panels

Reserved namespace for a future panel-framework refactor.

This folder is currently **empty** — no source files live here. The
modal and floating panels that sit above the WebGL canvas (dataset
browser, dimension sliders, debug console, recording panel, layers,
rendering controls, help overlay) currently live one level up in
`packages/luxar-viewer/src/ui/`, each as a `<name>.ts` entry file with
a sibling `<name>/` folder for pure helpers.

When a shared panel base class or registry is introduced, it will land
here. Until then this README exists only so the cross-reference from
`../README.md` (`[panels/](./panels/README.md) — Shared panel framework
used by multiple top-level panels.`) does not dangle.

## See Also

- [`../README.md`](../README.md) — UI package overview, including the
  current list of individual panels and their entry points.
- [`../dataset-browser.ts`](../dataset-browser.ts),
  [`../dimension-sliders.ts`](../dimension-sliders.ts),
  [`../debug-console.ts`](../debug-console.ts) — actual panel
  implementations.
