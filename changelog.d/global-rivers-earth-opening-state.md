#### Rivers of Earth opens centred on the planet's centre, one stop down

Two fixes to the `global_rivers_earth` opening state, both authored in the scene
so they travel with the demo rather than living in one browser's local state.

The orbit pivot is now the centre of the Earth. With no authored camera target
the viewer pivots on the metadata bounding-box centre, and relief exaggeration
puts that ~1 unit off the origin — the deepest trenches and the highest peaks are
not antipodal, so the box is not centred on the sphere it contains. The globe is
generated about the origin by construction (`lonlat_to_xyz` measures every radius
from it), so the demo now states `camera=CameraConfig(target=(0, 0, 0))`. A target
alone does not pin the camera: the viewer still auto-frames the distance and only
preserves the pivot, so the opening shot stays a fitted, face-on whole globe.

Exposure is authored at `-1.0`. The terrain went from a 0.05-opacity backdrop to
an opaque full-opacity shell and the rivers from gain 1.6 to 2.63, which left the
neutral `0.0` default clipping the lit hemisphere.
