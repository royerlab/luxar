# `input-handler/key-bindings/`

Keyboard binding table split per context.

- `register-all.ts` — public entry point + the three `KeyBindings*` types
  (deps, commands, panel getters). Imported by the orchestrator.
- `navigation-bindings.ts` — every NAVIGATION-context application binding
  (orbit-mode UI shortcuts: H, P, R, V, I, F, C, B, T, G, N,
  M (cycle data loading monitor), O, [, ],
  digit keys 1-9, Space, Escape, Ctrl+L, Ctrl+Shift+S). The named
  navigation shortcuts are driven by `config.input.keyboard.shortcuts`
  rather than hard-coded, so they track the user's config; the
  debug-console binding additionally applies a Ctrl modifier
  structurally at the binding site. A handful with no config entry stay
  hard-coded: dimension navigation (`[` / `]`), digit keys 1-9,
  recording panel (T), screenshot (G), data monitor (M), Escape, and the
  Ctrl+Shift+S state export.
- `fly-bindings.ts` — FLY_CONTROLS-context fly-mode bindings (W/A/S/D/Q/E ×
  modifier combinations, arrow look keys ± Shift, Shift speed-boost).
- `animation-shortcuts.ts` — `AnimationShortcuts` class for K / Home /
  End / Shift+↑ / Shift+↓ (registered on NAVIGATION context after the
  animation manager is constructed; not wired by `registerAllKeyBindings`).

The split is structural — `registerAllKeyBindings` wires the
navigation bindings and the fly bindings at startup;
`AnimationShortcuts.register()` is called separately from the
`InputHandler` once the animation manager exists.

Ctrl/⌘+wheel FOV-vs-zoom exclusivity is not a key binding: each wheel
handler reads the wheel event's own live modifier flags
(`luxar-orbit-controls/input/pointer.ts`,
`luxar-fly-controls/input/wheel.ts`), so there is no held-modifier
state to track here.
