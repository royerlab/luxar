# input

Input handling configuration slice. Owns the default sensitivity for adjustments and the global keyboard shortcut map (panel toggles, camera recenter, control/inertial/cinematic mode toggles). Movement and dimension-navigation keys live only in the binding registry.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher.

## Contents

- `data.ts` — `inputConfig: InputConfig`. Defines `defaultSensitivity` and the `keyboard.shortcuts` map (15 named bindings, e.g. `toggleHelp: 'h'`, `toggleDebugConsole: 'l'` — the Ctrl modifier is applied structurally at the binding site — `recenterCamera: 'f'`).
- `types.ts` — `InputConfig` interface; `keyboard.shortcuts` is a fixed-key record.
- `validate.ts` — `validateInput(config, errors, warnings)`. Errors on non-finite `defaultSensitivity`; warns when it falls outside the typical `(0, 1]` band (helpful range hint `0.01-0.5`).

## Public API

- `inputConfig` — re-exported through `../../index.ts` into `AppConfig.input`.
- `InputConfig` — re-exported through `../../types.ts`.
- `validateInput` — called from `../../validation.ts`.
