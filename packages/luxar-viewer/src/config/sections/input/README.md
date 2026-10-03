# input

Input handling configuration slice. Owns the global keyboard shortcut map (panel toggles, camera recenter, control/inertial/cinematic mode toggles). Movement and dimension-navigation keys live only in the binding registry.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal and `types.ts` defines the interface. This section has no `validate.ts` — a shortcut map has no cross-field invariant to check.

## Contents

- `data.ts` — `inputConfig: InputConfig`. Defines the `keyboard.shortcuts` map (15 named bindings, e.g. `toggleHelp: 'h'`, `toggleDebugConsole: 'l'` — the Ctrl modifier is applied structurally at the binding site — `recenterCamera: 'f'`).
- `types.ts` — `InputConfig` interface; `keyboard.shortcuts` is a fixed-key record.

## Public API

- `inputConfig` — re-exported through `../../index.ts` into `AppConfig.input`.
- `InputConfig` — re-exported through `../../types.ts`.
