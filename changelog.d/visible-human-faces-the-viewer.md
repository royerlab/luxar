#### The Visible Human opens facing you, head up

The demo authored no camera at all, so the viewer auto-framed on the bounding box
with world `+Y` up and its default direction. This volume is not axis-aligned to
those defaults in any orientation that reads as a person, so the subject opened
lying on its side.

The fit's centers are in the source volume's index order, which is not the `x` /
`y` / `z` the scene declares, so the anatomy has to be read off the columns:

- column 0 is the axial slice index. NLM cuts the Visible Human Male from the
  head down, so the index grows *inferiorly* — superior is `-x`.
- column 1 is the image row. Axial cryosection photographs put anterior at the
  top of the frame, row 0 — anterior is `-y`.
- column 2 is the image column, left-right, and the widest axis (shoulders).

Both readings are confirmed against the built cloud rather than assumed: the head
protrudes along `-col0`, and looking down `col0` shows a head with the shoulders
spread along `col2`. The three axis extents (635 / 450 / 895 in resampled voxels,
about 479 / 339 / 676 mm) are consistent with a head-and-shoulders block and with
nothing else.

So the scene now authors a camera standing off along `-y`, in front of the face,
looking back at the centroid with `up = -x`. Distance is derived from the actual
half-extents rather than fixed, fitting the larger of height and shoulder width
into a 45° field with 15% air, and the near/far planes are derived from the same
bounding radius. Rendering the resulting view off the built cloud gives an
upright, front-on head-and-shoulders bust.

Note that the shipped scene keeps its old framing until it is regenerated, and
this demo cannot currently regenerate cheaply: its fast path is disabled (#1670,
the shipped colors sidecar does not correspond to the shipped fit), so a rebuild
means the full ~1.1 GB cryosection download and refit.
