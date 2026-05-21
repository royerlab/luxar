# `input-handler/commands/`

Command bodies the orchestrator delegates to. The non-trivial commands
take a narrow `*Ctx` object built by a `makeXxxCtx()` private method on
`InputHandler` — never `this` — so each command is unit-testable
against a small typed surface. Pure helpers (`focus-utils`,
`data-monitor-cycle`) take their inputs directly.

- `panel-coordinator.ts` — `PanelCoordinator` class. Owns the
  priority-ordered "close all panels" flow used by Escape, plus the
  recording-priority short-circuit and the fullscreen-defer rule.
- `viewer-state-export.ts` — `exportViewerState` body. Captures full
  viewer state (camera, rendering, dimensions, animation) and copies
  it to the clipboard. Also stores on `window.__luxarDebug` for
  programmatic access. Triggered by Ctrl+Shift+S.
- `control-mode.ts` — `toggleControlMode` (Orbit → Fly → Ortho cycle,
  V key) + `toggleInertialMode` (I key, fly only) + `nextControlType`
  (pure cycle helper).
- `focus-utils.ts` — `isTypingInInput` + `isFocusOnSceneCanvas`. Pure
  DOM-focus helpers used by the orchestrator's typing-context gate and
  the Space-key fullscreen gate.
- `data-monitor-cycle.ts` — `cycleDataMonitor`. Single eventBus emit +
  log; pulled out so the orchestrator only owns the binding wiring.
