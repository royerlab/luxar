# `input-handler/key-bindings/`

Keyboard binding table split per context.

- `register-all.ts` — public entry point + the three `KeyBindings*` types
  (deps, commands, panel getters). Imported by the orchestrator.
- `navigation-bindings.ts` — every NAVIGATION-context application binding
  (orbit-mode UI shortcuts: H, P, R, V, I, F, C, B, T, G, N,
  M (cycle data loading monitor), O, [, ],
  digit keys 1-9, Space, Escape, Ctrl+L, Ctrl+Shift+S). The colormap
  legend (J), overlays (U), and layers panel (L) keys are read from
  `config.input.keyboard.shortcuts` rather than hard-coded, so they
  track the user's config.
- `fly-bindings.ts` — FLY_CONTROLS-context fly-mode bindings (W/A/S/D/Q/E ×
  modifier combinations, arrow look keys ± Shift, Shift speed-boost).
- `animation-shortcuts.ts` — `AnimationShortcuts` class for K / Home /
  End / Shift+↑ / Shift+↓ (registered on NAVIGATION context after the
  animation manager is constructed; not wired by `registerAllKeyBindings`).

The split is structural — `registerAllKeyBindings` wires the FOV
hold gate, the navigation bindings, and the fly bindings at startup;
`AnimationShortcuts.register()` is called separately from the
`InputHandler` once the animation manager exists.
