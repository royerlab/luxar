# animation

Animation loop configuration slice. Owns the idle-timeout used by the render loop to pause continuous rendering when no input or scene change has occurred — a power-saving knob, not the dimension-animation playback engine (that lives in the sibling `dimension-animation/` slice).

Conforms to the section pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal and `types.ts` defines the interface. This slice has no `validate.ts` — its single field is an unconstrained millisecond count consumed directly by the render loop.

## Contents

- `data.ts` — `animationConfig: AnimationConfig`. Default: `idleTimeoutMs: 2000` (pause the animation loop after two seconds of idleness to save power).
- `types.ts` — `AnimationConfig` interface, a one-field shape (`idleTimeoutMs: number`).

## Public API

- `animationConfig` — re-exported through `../../index.ts` into `AppConfig.animation`.
- `AnimationConfig` — re-exported through `../../types.ts`.
