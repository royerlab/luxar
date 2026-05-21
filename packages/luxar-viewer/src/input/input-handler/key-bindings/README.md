# `input-handler/key-bindings/`

Keyboard binding table split per context.

- `register-all.ts` — public entry point + the three `KeyBindings*` types
  (deps, commands, panel getters). Imported by the orchestrator.
- `fov-hold-gate.ts` — Ctrl/Meta hold counter that disables wheel zoom
  while either modifier is held (so Ctrl+wheel only adjusts FOV).
  Includes the `blur` + `visibilitychange` listeners that reset the
  counter when the page loses focus.
- `navigation-bindings.ts` — every NAVIGATION-context application binding
  (orbit-mode UI shortcuts: H, P, R, V, F, C, B, L, M, N, O, T, G, [, ],
  digit keys, Space, Escape, Ctrl+L, Ctrl+Shift+S).
- `fly-bindings.ts` — FLY_CONTROLS-context fly-mode bindings (WASD ×
  modifier combinations, arrow look keys, Shift speed-boost).
- `animation-shortcuts.ts` — `AnimationShortcuts` class for K / Home /
  End / Shift+↑ / Shift+↓ (registered on NAVIGATION context after the
  animation manager is constructed).

Behavior is byte-for-byte identical to the pre-split inline registration
that used to live in `InputHandler`; the split is structural.
