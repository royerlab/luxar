"""Composing an opening camera pose for the demos' cinematic 35 mm lens.

Every demo enables ``cinematic_mode``, and the preset the viewer expands from it
includes a lens: a 35 mm barrel distortion AND the 35 mm field of view that
distortion belongs to (63°, against the viewer's 47° default). ``camera.fov`` and
``camera.fov_preset`` are expanded as ONE unit, so a scene that pins either gets
neither from the preset — it keeps a 50 mm framing while still receiving 35 mm
distortion, which is two different lenses in one image. (Seventeen demos do pin an
``fov`` of their own — 38° to 50° where it is a literal, computed in six — and
those pins all predate the cinematic look. They are exactly that mismatch:
their framing is safe, their lens is mixed. Unifying them is #1862.)

So a demo that states its own distance pins neither, and composes its pose for
63° instead. Such a pose specifies a DISTANCE, not a framing, and at a fixed
distance 47° → 63° scales the subject to 0.71x linear — HALF its screen area.

An AUTO-FRAMED scene cannot be fixed from here, and is worth understanding
before reading the helpers below as a general answer. ``calculateCameraDistance``
does divide by ``tan(fov / 2)``, but the viewer widens the lens AFTER it frames:
``SceneManager.loadSceneData`` auto-frames while the camera still holds the 47°
default (``applyZarrViewerConfig`` sets position/target/up, never fov), and the
preset's fov only arrives later, when ``load-dataset.ts`` calls
``RenderingControls.applyZarrDefaults``. Nothing re-frames after that. So an
auto-framed cinematic scene sits at the 47° fit distance behind a 63° lens and
opens ~1.4x looser than it used to — the subject filling 0.53 of the half-frame
where the fit put 0.75. ``docs/guides/user/VIEWER_GUIDE.md`` documents this
("wider than the default auto-framing"), and closing it means applying the
expanded fov before the fit, in the viewer, not here — #1861.

When the preset FOV is applied (that is, the scene has no stored rendering
settings), an authored position suppresses auto-framing entirely, so the poses
composed through this module frame exactly as the arithmetic below says. Stored
settings can restore another FOV while the authored position still applies;
fixing that viewer-side mismatch is also part of #1861.

Two ways to compose for the wider lens, and a demo should use whichever it
already thinks in:

* Derive the distance from the field of view (the better rule — a perspective
  camera subtends a sphere of radius R at ``asin(R / D)``): read
  :data:`CINEMATIC_FOV_DEG` and the distance follows.
* Keep an empirically tuned distance and pull it in: :func:`pull_in`.

Both are the same statement — "this pose was composed for 63°" — and neither
leaves a bare 0.71 in a demo for a later reader to decode.
"""

from __future__ import annotations

import math

#: Vertical FOV, in degrees, that the cinematic preset expands for a scene which
#: pins neither ``camera.fov`` nor ``camera.fov_preset`` — the ``35mm`` entry of
#: the viewer's ``camera.fovPresets``. Vertical because the value reaches
#: ``THREE.PerspectiveCamera.fov``, whose FOV is vertical, notwithstanding the
#: "horizontal FOV" wording on the preset table.
CINEMATIC_FOV_DEG = 63.0

#: The viewer's own default vertical FOV (its ``50mm Normal`` preset), which is
#: what a demo pose composed before the demos opted into the cinematic look.
VIEWER_DEFAULT_FOV_DEG = 47.0

#: Distance scale that holds a subject's on-screen size fixed while the vertical
#: FOV widens from :data:`VIEWER_DEFAULT_FOV_DEG` to :data:`CINEMATIC_FOV_DEG`:
#: ``tan(23.5°) / tan(31.5°)`` ≈ 0.709. Derived rather than written out so the
#: two FOVs above stay the single source of truth.
FRAMING_PULL_IN = math.tan(math.radians(VIEWER_DEFAULT_FOV_DEG / 2.0)) / math.tan(
    math.radians(CINEMATIC_FOV_DEG / 2.0)
)


def pull_in(
    position: tuple[float, float, float],
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Move a pose toward its target so 63° frames what 47° used to.

    Scaling is about ``target``, NOT about the world origin: a pose's distance is
    measured from what it looks at, so scaling an off-origin pose about the
    origin would swing the camera as well as move it — changing the view
    direction the demo chose, which is usually the load-bearing half.

    Only the framing is preserved, not the image: a wider lens at a shorter
    distance foreshortens more, and near geometry grows relative to far. That is
    the look being opted into; it is worth a glance at the result rather than
    trust in the arithmetic.

    Args:
        position: Camera position the demo composed for the viewer's 47° default.
        target: Point the camera looks at, and the point the pull-in is about.

    Returns:
        The position that frames the same subject at :data:`CINEMATIC_FOV_DEG`.
    """
    return (
        target[0] + (position[0] - target[0]) * FRAMING_PULL_IN,
        target[1] + (position[1] - target[1]) * FRAMING_PULL_IN,
        target[2] + (position[2] - target[2]) * FRAMING_PULL_IN,
    )
