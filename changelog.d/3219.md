#### OME-TIFF and ImageJ TIFF axes select the channel and timepoint

An OME-TIFF or ImageJ TIFF records its axis order (for example `ZCYX`), but
`--channel` and `--timepoint` on one still demanded `--axes` and claimed the file
had no discoverable channel or time axes. When the metadata was not written by a
known Python writer (in practice Bio-Formats, Fiji or a vendor exporter), the
loader now uses the declared order for an index given without `--axes`, as the
OME-Zarr path already did, and always logs every axis with its size
(`z=65, c=7, y=624, x=924`).

Files written by tifffile, bioio or aicsimageio are not trusted: these fill in
default axes when their caller names none (a CZYX array saved with `imagej=True`
reads back as ZCYX, a z-stack as CYX, and bioio's `OmeTiffWriter.save` writes
CZYX), and a file whose axes were named looks the same on disk. Their signature
(OME `Creator` starting `tifffile`, `bioio` or `aicsimageio`, ImageJ
`ImageJ=1.11a`) therefore keeps `--axes` required, as does an axis letter other than T/C/Z/Y/X or an index for an
axis the file does not declare. Those refusals show the declared order and sizes.
A load without an index ignores the declaration, `--axes` always overrides it, and
a TIFF that cannot be read reports why instead of claiming it declares nothing. An
out-of-range `--channel` or `--timepoint` on a zarr store read positionally (no
`--axes`) is now a clear error rather than a zarr bounds exception, and a
negative one is refused there too instead of silently indexing from the end.

#### Usage errors on volume selection no longer print a traceback

`luxar gsplat fit data.npy out.gsplats.zarr --channel 2` printed a clear message
and then the same message again as a Python traceback. A channel, timepoint or
`--axes` request the input cannot satisfy now raises `VolumeSelectionError` (a
`ValueError` subclass), which `fit`, `cal` and `compare` report as a usage error
(exit 2) with no traceback. Other exceptions are unchanged.

#### Plain log output under NO_COLOR or when redirected

arbol coloured every line, so `luxar demo run cloud > log 2>&1` filled the log
with escape codes and `NO_COLOR` was ignored. Luxar now turns arbol's colour off
when `NO_COLOR` is set or stdout is redirected to a file or pipe, and keeps it on
under `FORCE_COLOR`. The policy runs at `import luxar` and at the start of every
CLI command. A notebook keeps its colours, since its output stream is not a
redirected file. The fit's amplitude histogram follows the same switch. The
developer scripts under `scripts/` use arbol directly and still print colour.

#### Install, remote viewing and skill snippet fixes

The README now says that the CUDA build of PyTorch is about 3 GB to download and
about 6 GB installed, and how to view a scene from a headless or remote machine:
an `ssh -L` tunnel, or `--host 0.0.0.0`, which serves the data and viewer
unauthenticated to anyone who can reach the machine. The Python
examples in the agent skills use `device="auto"` instead of `device="cuda"`, so
they also run on a CPU-only or Apple machine. The multichannel exposure snippet in
the visualization skill now imports `CameraConfig` and `ViewerConfig` and shows
how the camera reaches the scene. The CLI and gsplats READMEs open with a short
pointer to the three-command path from a volume to the viewer.
