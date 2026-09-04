#### Remote-control embedder API: `flyTo`, rendering and layer setters, `getViewerState`, `camera-changed`

A Luxar viewer can now be driven from an external program without the built-in
UI. `LuxarApp.flyTo(pose, { durationMs, easing })` transitions smoothly to a
camera pose captured with `getCameraPose()`, interpolating in the orbit
parameterisation (the focus target moves linearly while the viewing direction is
slerped and the distance log-interpolated) so the camera arcs around the scene
instead of cutting through it, and lands on the pose exactly. Any input on the
canvas or keyboard cancels the flight where it is — the visitor's own hand always
wins — and the promise reports whether the flight completed.

`setRenderingSettings(patch)` rides the exact path an authored `viewer_config`
takes at load, so anything a scene author can bake a controller can set live, with
the same validation and side-effects; `setLayer(path, patch)` and `getLayers()`
expose the Layers panel's per-layer appearance through the panel's own
state-manager and apply-engine route; `getViewerState()` bundles dataset, camera,
slice position, rendering and layers in one call; and a new `camera-changed`
event streams the pose as it moves. `docs/guides/specs/REMOTE_CONTROL_SPEC.md`
records the design and the follow-on phases (WebSocket hub in `luxar serve`,
authored story waypoints and kiosk permissions in `viewer_config`).
