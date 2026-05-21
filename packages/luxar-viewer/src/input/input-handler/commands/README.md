# `input-handler/commands/`

Command bodies the orchestrator delegates to. Each function takes a
narrow `*Ctx` object built by a `makeXxxCtx()` method on the
`InputHandler` — never `this`. Event-emission sites (panel-cycle,
panel-hide, open-dataset-browser) stay at the call site to preserve
the live event surface.

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
