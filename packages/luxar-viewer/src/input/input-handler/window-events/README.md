# `input-handler/window-events/`

Window- and document-level listeners that the orchestrator wires once
during `init()`.

- `window-event-handler.ts` — `WindowEventHandler` class. Owns the
  three global listeners: `resize` (forwards to
  `SceneManager.updateSize()`), `wheel` (intercepts Ctrl/Meta+wheel for
  FOV; orbit/ortho controls own the regular zoom), and
  `fullscreenchange` (toggles canvas inline styles + runs one rAF
  `updateSize()` after the viewport transition).
- `fullscreen-toggle.ts` — `toggleFullscreen` body. Requests
  fullscreen on `document.documentElement`; falls back to the WebGL
  canvas if the document request rejects; exits otherwise.
