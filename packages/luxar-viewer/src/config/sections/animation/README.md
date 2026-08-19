# animation

Animation loop configuration slice. Owns the idle-timeout used by the render loop to pause continuous rendering when no input or scene change has occurred — a power-saving knob, not the dimension-animation playback engine (that lives in the sibling `dimension-animation/` slice) — plus the frame-pacing thresholds the loop uses to yield the main thread back between pathologically slow frames.

Conforms to the section pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal and `types.ts` defines the interface. This slice has no `validate.ts` — every field is an unconstrained millisecond count (or a boolean) consumed directly by the render loop.

## Contents

- `data.ts` — `animationConfig: AnimationConfig`. Defaults:
  - `idleTimeoutMs: 2000` — pause the animation loop after two seconds of idleness to save power.
  - `pacing.enabled: true` — frame pacing on; `false` restores the historical back-to-back `requestAnimationFrame` loop.
  - `pacing.slowFrameMs: 250` — frame cost above which a frame counts as slow; the loop paces once TWO consecutive frames exceed it. 250 ms is 4 fps, a scene already far past any interactive threshold, so every healthy frame rate stays on the untouched fast path. The trigger is the frame PERIOD (the loop's wedge spends its second outside JS, where a body span reads ~2 ms), so a FOREIGN main-thread task of that size — a GC pause, a shader compile, a burst of chunk decodes — is charged to the loop too. An ISOLATED one is still never paced, but that is the streak rule in the controller rather than this threshold; SUSTAINED foreign work does pace, which is what you want during a heavy load. See the comment on the field in `data.ts`.
  - `pacing.maxCooldownMs: 250` — ceiling on the inserted gap, which also bounds the worst-case added latency of a `requestRender()` that arrives just after a cooldown starts.
- `types.ts` — `AnimationConfig`: `idleTimeoutMs` plus a nested `pacing` block (`enabled`, `slowFrameMs`, `maxCooldownMs`), inline in the interface as in the sibling `dimension-animation/` slice.

## Public API

- `animationConfig` — re-exported through `../../index.ts` into `AppConfig.animation`.
- `AnimationConfig` — re-exported through `../../types.ts`.

## See also

- `../../../scene/animation/animation-controller.ts` — the only consumer; its class JSDoc documents why pacing exists (#1724), why a STREAK of slow frames is required rather than a single one, why the cooldown is a bounded fraction of the frame cost rather than a constant, and why it is armed from a zero-delay hop (a timer armed at the frame's start would already be overdue and insert no gap).
