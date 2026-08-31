#### The turntable can spin about any screen or scene axis

The Navigation popover's orbit section gains a `Rotation Axis` dropdown with
two families. Camera frame: `Vertical` (the screen-up axis auto-rotation
always used, and still the default), `Horizontal` (screen-right: the scene
tumbles over the top), `View axis` (the view direction: a pure roll, in which
the camera never moves at all). World frame: `World X` / `World Y` /
`World Z`, a fixed scene axis — the classic turntable, which spins a subject
about its own axis at any camera elevation, where a camera-frame `Vertical`
turntable makes that axis precess instead (a spin plus a wobble). The two
frames coincide exactly while the camera is level, so switching between them
there changes nothing. Only the world family takes letters, which is what a
letter means everywhere else in the repo (the gallery harness's per-demo
`orbitUp`); a bare letter for the camera frame would read as a *data* axis in
an nD scientific viewer.

No axis can destabilize the orbit: a camera-frame axis is invariant under its
own rotation, a world axis is a constant with no feedback at all, and because
`camera.up` is derived from the orientation quaternion the horizontal tumble
passes over the top indefinitely with no pole flip. The one degenerate case is
benign — a world axis parallel to the view direction leaves the camera where it
is and rolls the image, exactly as `View axis` does.

Scenes can author it from Python as `ViewerConfig(auto_rotate=True,
auto_rotate_axis="world-y")`, and the choice persists per scene with the rest
of the rendering settings. Turntable *recording* picks it up for free:
`applyOrbitRotation(angle)` now defaults to the configured axis instead of
hardcoding screen-up, so an exported turntable cannot rotate unlike the
preview it was set up from.
