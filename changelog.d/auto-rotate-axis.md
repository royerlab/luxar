#### The turntable can spin about any screen axis

The Navigation popover's orbit section gains a `Rotation Axis` dropdown —
`Vertical` (the screen-up axis auto-rotation always used, and still the
default), `Horizontal` (screen-right: the scene tumbles over the top), or
`View axis` (the view direction: a pure roll, in which the camera never
moves at all). The axes are named in the camera frame rather than as
x/y/z on purpose: a bare letter reads as a *data* axis in an nD scientific
viewer, and the gallery harness already spends `'x' | 'y' | 'z'` on world
axes. Each axis is invariant under its own rotation, so all three are
stable turntables rather than drifts, and because `camera.up` is derived
from the orientation quaternion, the horizontal tumble passes over the top
indefinitely with no pole flip.

When a fixed scene axis approaches the view direction, the selector's
axis-aligned screen box of the projected bounds is not roll-invariant, so a
scene near an LOD threshold can breathe between levels. See the
[viewer guide](../docs/guides/user/VIEWER_GUIDE.md) for details.

Scenes can author it from Python as `ViewerConfig(auto_rotate=True,
auto_rotate_axis="view")`, and the choice persists per scene with the rest
of the rendering settings. Turntable *recording* picks it up for free:
`applyOrbitRotation(angle)` now defaults to the configured axis instead of
hardcoding screen-up, so an exported turntable cannot rotate unlike the
preview it was set up from.
