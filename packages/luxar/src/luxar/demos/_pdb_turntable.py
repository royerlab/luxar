"""Render a rotating PDB structure to a transparent WebM turntable with PyMOL.

Used by ``demo_esm3_protein_stories`` to show a representative structure next to
each story's panel. This is a deliberate SPECIAL CASE in the demos' dependency
policy: PyMOL open-source is not pip-installable (conda-forge or Homebrew only),
so it is not in :data:`luxar.demos.INSTALL_SPECS`. The gate here therefore
prints an install hint naming both routes and lets the demo continue WITHOUT
turntables rather than raising — the stories scene is complete without them.

Pipeline (all cached under ``~/.cache/luxar/pdb_turntables``):

1. Fetch ``<ID>.pdb`` and the entry title from RCSB (``files.rcsb.org`` and
   ``data.rcsb.org``).
2. Drive PyMOL headless (``pymol -cq <script>``) to ray-trace ``frames`` PNG
   frames with a transparent background, turning 360°/frames per frame — at the
   default 900 frames that is 0.4° per frame, a slow 30 s turn at 30 fps (the
   owner found one turn in 6 s far too fast; 0.4° per displayed frame is well
   below the step that reads as judder at overlay size). The
   style is a minimalist matte "clay" molecular surface in pastel shades of the
   story's own colour (see :func:`pymol_script`).
3. Encode with ffmpeg to VP9 WebM **with an alpha channel** (``yuva420p``) and
   keep frame 0 as a PNG poster (the Safari fallback: it cannot decode alpha
   WebM).

Renders are keyed by ``(pdb_id, STYLE_VERSION, frames, size, colour)``; bump
``STYLE_VERSION`` when the PyMOL style changes so cached turntables re-render.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from arbol import aprint

#: Bump when :func:`pymol_script` changes so cached turntables re-render.
STYLE_VERSION = 4
DEFAULT_FRAMES = 900
DEFAULT_FPS = 30
# 768 px covers the overlay's ~26% of a 4K kiosk width (~1000 px) at a 1.3x
# upscale; ray-tracing cost scales with pixels, and 1024 px cost 1.5x more.
DEFAULT_SIZE = 768
# Parallel PyMOL processes; each gets cpu_count // DEFAULT_JOBS ray threads.
# PyMOL's tracer scales sub-linearly with threads (12 threads = 5.8x, 4 = 3.4x
# on an M-series Mac), so three 4-thread jobs out-render one 12-thread job 1.7x.
DEFAULT_JOBS = 3
# Structures above this many atoms take the coarser molecular surface: the
# surface dominates ray-tracing time (about 1.8x per quality step) and a
# 50k-atom complex reads the same at overlay size. Below it, surface_quality 0
# was measured indistinguishable from quality 1 at one seventh of the cost.
# Antialiasing stays at 1 everywhere (2 cost 1.4x): the clip is shown smaller
# than it is rendered, and that downscale smooths the edges.
LARGE_STRUCTURE_ATOMS = 15_000
#: Colour used when a structure has no story colour (a soft grey-violet).
DEFAULT_COLOR: tuple[float, float, float] = (0.69, 0.42, 0.85)
RGB = tuple[float, float, float]

RCSB_FILE_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

PYMOL_INSTALL_HINT = (
    "PyMOL open-source is not on PyPI. Install it with one of:\n"
    "    brew install pymol                          # macOS, Homebrew\n"
    "    conda install -c conda-forge pymol-open-source\n"
    "and make sure `pymol` is on PATH (or importable by this interpreter)."
)
FFMPEG_INSTALL_HINT = (
    "ffmpeg (with libvpx-vp9) is needed to encode the turntables:\n"
    "    brew install ffmpeg        # or: pip install imageio-ffmpeg"
)


@dataclass(frozen=True)
class TurntableAssets:
    """One rendered structure: the alpha WebM, its poster frame, and the entry title."""

    pdb_id: str
    webm: Path
    poster: Path
    title: str
    frames: int
    fps: int


# ----------------------------------------------------------------------------
# Tool discovery
# ----------------------------------------------------------------------------


def find_pymol() -> Optional[list[str]]:
    """Command prefix that runs PyMOL headless, or ``None`` when unavailable.

    Prefers a ``pymol`` executable on PATH (Homebrew / conda); falls back to
    ``python -m pymol`` when the module is importable by THIS interpreter.
    """
    exe = shutil.which("pymol")
    if exe:
        return [exe, "-cq"]
    try:
        import pymol  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-m", "pymol", "-cq"]


def find_ffmpeg() -> Optional[str]:
    """Path to an ffmpeg binary: PATH first, then the imageio-ffmpeg wheel."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001 — absent module or broken wheel alike
        return None


# ----------------------------------------------------------------------------
# Pure pieces (unit-tested)
# ----------------------------------------------------------------------------


def color_hex(color: RGB) -> str:
    """``#rrggbb`` for an RGB triple in [0, 1] (clamped)."""
    return "#" + "".join(f"{round(min(1.0, max(0.0, c)) * 255):02x}" for c in color)


def cache_key(
    pdb_id: str,
    frames: int,
    size: int,
    style_version: int = STYLE_VERSION,
    color: RGB = DEFAULT_COLOR,
) -> str:
    """Stable file stem for one render configuration (colour included)."""
    digest = hashlib.sha1(
        f"{pdb_id.upper()}|{style_version}|{frames}|{size}|{color_hex(color)}".encode()
    ).hexdigest()
    return f"{pdb_id.upper()}_v{style_version}_{frames}f_{size}px_{digest[:8]}"


def chain_palette(base: RGB, n: int) -> list[RGB]:
    """``n`` pastel clay shades around the story hue, one per chain.

    Only the HUE is taken from the story colour: lightness is pinned to
    0.53-0.67 and saturation to 0.45-0.55 so a neon highlight colour (the
    scene's clusters are lit additively) becomes a matte clay, and neighbouring
    chains differ by a few degrees of hue and a step of lightness.
    """
    h, _l, _s = colorsys.rgb_to_hls(*(min(1.0, max(0.0, c)) for c in base))
    shades: list[RGB] = []
    for i in range(max(n, 1)):
        t = 0.0 if n <= 1 else i / (n - 1) - 0.5
        r, g, b = colorsys.hls_to_rgb(
            (h + 0.06 * t) % 1.0, 0.60 + 0.14 * t, 0.55 - 0.1 * abs(t)
        )
        shades.append((r, g, b))
    return shades


def pymol_script(
    pdb_path: Path,
    frames_dir: Path,
    *,
    frames: int,
    size: int,
    threads: int = 4,
    color: RGB = DEFAULT_COLOR,
) -> str:
    """The PyMOL Python script for one turntable.

    Style — a minimalist "clay" render: the smooth, opaque molecular surface of
    the polymer alone (no cartoon, ligands or ions), each chain in a pastel shade
    of the story colour (:func:`chain_palette`), soft three-light studio
    lighting with low specular, ray-traced shadows and ambient occlusion, and a
    transparent background. One ``turn y`` per frame around the oriented view;
    ``zoom(complete=1)`` plus an open clipping slab keep a long complex inside
    the square frame at every angle (a plain zoom fits the default 4:3 viewport
    and clipped photosystem II's ends).

    Cost is ray-tracing the surface, so quality follows structure size
    (``LARGE_STRUCTURE_ATOMS``): ``surface_quality 0`` below it (measured
    indistinguishable from quality 1 at one seventh of the cost), ``-1`` above;
    ``antialias 1`` throughout. Ambient occlusion and shadows are nearly free on
    an opaque surface. Hydrogens (NMR entries) are removed.
    """
    step = 360.0 / frames
    shades = chain_palette(color, 8)
    return "\n".join(
        [
            "from pymol import cmd",
            f"cmd.set('max_threads', {int(threads)})",
            f"cmd.load({str(pdb_path)!r}, 'mol')",
            "cmd.remove('solvent')",
            "cmd.remove('hydro')",
            "cmd.hide('everything')",
            f"shades = {[tuple(round(c, 4) for c in s) for s in shades]!r}",
            "chains = cmd.get_chains('mol and polymer') or ['']",
            "for i, chain in enumerate(chains):",
            "    # Spread the shades over the chains present (few chains -> the",
            "    # middle shades; many -> the whole ramp), repeating past eight.",
            "    k = round(i * (len(shades) - 1) / max(len(chains) - 1, 1))",
            "    cmd.set_color('story_%d' % i, list(shades[k % len(shades)]))",
            "    cmd.color('story_%d' % i, \"polymer and chain '%s'\" % chain)",
            "n_atoms = cmd.count_atoms('mol and polymer')",
            f"large = n_atoms > {LARGE_STRUCTURE_ATOMS}",
            "cmd.show('surface', 'polymer')",
            "cmd.set('surface_color', -1)",
            "cmd.set('transparency', 0.0)",
            "cmd.set('surface_quality', -1 if large else 0)",
            "cmd.set('antialias', 1)",
            "cmd.set('light_count', 3)",
            "cmd.set('light', [-0.4, -0.6, -1.0])",
            "cmd.set('light2', [0.8, 0.3, -1.0])",
            "cmd.set('ambient', 0.35)",
            "cmd.set('direct', 0.55)",
            "cmd.set('reflect', 0.25)",
            "cmd.set('spec_reflect', 0.12)",
            "cmd.set('spec_power', 30)",
            "cmd.set('shininess', 12)",
            "cmd.set('ray_shadows', 1)",
            "cmd.set('ambient_occlusion_mode', 1)",
            "cmd.set('ambient_occlusion_scale', 22)",
            "cmd.set('ambient_occlusion_smooth', 12)",
            "cmd.set('depth_cue', 0)",
            "cmd.set('ray_opaque_background', 0)",
            "cmd.set('ray_trace_mode', 0)",
            "cmd.set('field_of_view', 22)",
            "cmd.orient('mol and polymer')",
            "cmd.zoom('mol and polymer', buffer=2.0, complete=1)",
            "cmd.clip('slab', 10000)",
            f"for i in range({frames}):",
            f"    cmd.png({str(frames_dir)!r} + '/frame_%04d.png' % i, width={size}, height={size}, dpi=-1, ray=1)",
            f"    cmd.turn('y', {step!r})",
        ]
    )


def ffmpeg_command(
    ffmpeg: str, frames_dir: Path, out_webm: Path, *, fps: int
) -> list[str]:
    """VP9 WebM with alpha from ``frame_%04d.png``. ``-auto-alt-ref 0`` is required for alpha."""
    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-auto-alt-ref",
        "0",
        "-b:v",
        "0",
        "-crf",
        "32",
        "-row-mt",
        "1",
        "-an",
        str(out_webm),
    ]


# ----------------------------------------------------------------------------
# Fetch + render
# ----------------------------------------------------------------------------


def fetch_pdb(pdb_id: str, cache_dir: Path) -> tuple[Path, str]:
    """Download ``<ID>.pdb`` and the entry title into the cache (idempotent)."""
    pdb_id = pdb_id.upper()
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = cache_dir / f"{pdb_id}.pdb"
    meta_path = cache_dir / f"{pdb_id}.json"
    if not pdb_path.exists():
        with urllib.request.urlopen(
            RCSB_FILE_URL.format(pdb_id=pdb_id), timeout=60
        ) as r:  # noqa: S310
            pdb_path.write_bytes(r.read())
    if not meta_path.exists():
        try:
            with urllib.request.urlopen(
                RCSB_ENTRY_URL.format(pdb_id=pdb_id), timeout=60
            ) as r:  # noqa: S310
                entry = json.loads(r.read())
            title = str(entry.get("struct", {}).get("title", pdb_id))
        except Exception:  # noqa: BLE001 — the title is decoration
            title = pdb_id
        meta_path.write_text(json.dumps({"title": title}))
    title = json.loads(meta_path.read_text()).get("title", pdb_id)
    return pdb_path, title


def render_turntable(
    pdb_id: str,
    cache_dir: Path,
    *,
    frames: int = DEFAULT_FRAMES,
    fps: int = DEFAULT_FPS,
    size: int = DEFAULT_SIZE,
    threads: int = 4,
    color: RGB = DEFAULT_COLOR,
    pymol: Optional[Sequence[str]] = None,
    ffmpeg: Optional[str] = None,
) -> TurntableAssets:
    """Render (or reuse) one structure's turntable. Raises on tool failure."""
    pdb_id = pdb_id.upper()
    pymol_cmd = list(pymol) if pymol is not None else find_pymol()
    ffmpeg_exe = ffmpeg if ffmpeg is not None else find_ffmpeg()
    if pymol_cmd is None:
        raise RuntimeError(PYMOL_INSTALL_HINT)
    if ffmpeg_exe is None:
        raise RuntimeError(FFMPEG_INSTALL_HINT)

    pdb_path, title = fetch_pdb(pdb_id, cache_dir)
    stem = cache_key(pdb_id, frames, size, color=color)
    webm = cache_dir / f"{stem}.webm"
    poster = cache_dir / f"{stem}.png"
    if webm.exists() and poster.exists():
        return TurntableAssets(pdb_id, webm, poster, title, frames, fps)

    with tempfile.TemporaryDirectory(prefix=f"luxar_turntable_{pdb_id}_") as tmp:
        frames_dir = Path(tmp) / "frames"
        frames_dir.mkdir()
        script = Path(tmp) / "render.py"
        script.write_text(
            pymol_script(
                pdb_path,
                frames_dir,
                frames=frames,
                size=size,
                threads=threads,
                color=color,
            )
        )
        aprint(
            f"PyMOL: ray-tracing {frames} frames of {pdb_id} at {size}px "
            f"({threads} threads) …"
        )
        subprocess.run([*pymol_cmd, str(script)], check=True, capture_output=True)  # noqa: S603
        rendered = sorted(frames_dir.glob("frame_*.png"))
        if len(rendered) != frames:
            raise RuntimeError(
                f"PyMOL produced {len(rendered)} of {frames} frames for {pdb_id}"
            )
        aprint(f"ffmpeg: encoding {pdb_id} → VP9 alpha WebM at {fps} fps …")
        subprocess.run(
            ffmpeg_command(ffmpeg_exe, frames_dir, webm, fps=fps), check=True
        )  # noqa: S603
        shutil.copyfile(rendered[0], poster)
    return TurntableAssets(pdb_id, webm, poster, title, frames, fps)


def render_turntables(
    pdb_ids: Sequence[str],
    cache_dir: Path,
    *,
    frames: int = DEFAULT_FRAMES,
    fps: int = DEFAULT_FPS,
    size: int = DEFAULT_SIZE,
    jobs: int = DEFAULT_JOBS,
    colors: Optional[Mapping[str, RGB]] = None,
) -> dict[str, TurntableAssets]:
    """Render several structures in parallel PyMOL processes.

    ``colors`` maps a PDB id to the RGB triple (in [0, 1]) whose hue the
    structure is rendered in — the demo passes each story's highlight colour so
    the turntable matches its cluster; ids without one get ``DEFAULT_COLOR``.

    Returns the assets that rendered, keyed by PDB id. When PyMOL or ffmpeg is
    missing, prints the install hint ONCE and returns an empty dict — the caller
    builds its scene without turntables. A structure that fails to render is
    reported and skipped; the others still return. The machine's cores are
    split evenly across the ``jobs`` PyMOL processes.
    """
    pymol_cmd = find_pymol()
    ffmpeg_exe = find_ffmpeg()
    jobs = max(1, jobs)
    cores = render_threads()
    if pymol_cmd is None or ffmpeg_exe is None:
        aprint("⚠️  Turntable videos skipped:")
        for line in (
            PYMOL_INSTALL_HINT if pymol_cmd is None else FFMPEG_INSTALL_HINT
        ).splitlines():
            aprint(f"   {line}")
        return {}

    palette = {k.upper(): v for k, v in (colors or {}).items()}

    def one(pdb_id: str, threads: int) -> Optional[TurntableAssets]:
        try:
            return render_turntable(
                pdb_id,
                cache_dir,
                frames=frames,
                fps=fps,
                size=size,
                threads=threads,
                color=palette.get(pdb_id.upper(), DEFAULT_COLOR),
                pymol=pymol_cmd,
                ffmpeg=ffmpeg_exe,
            )
        except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
            detail = ""
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                detail = (
                    ": " + e.stderr.decode(errors="replace").strip().splitlines()[-1]
                )
            aprint(f"⚠️  Turntable for {pdb_id} failed{detail or f': {e}'} — skipped")
            return None

    # Small structures parallelise well; large ones are memory-bound and slow
    # each other down (three side by side rendered a frame of photosystem II 4x
    # slower than one alone with every core), so they run one at a time, last.
    small: list[str] = []
    large: list[str] = []
    for pdb_id in pdb_ids:
        try:
            n_atoms = structure_atoms(pdb_id, cache_dir)
        except (OSError, ValueError) as e:
            aprint(f"⚠️  Turntable for {pdb_id} failed: {e} — skipped")
            continue
        (large if n_atoms > LARGE_STRUCTURE_ATOMS else small).append(pdb_id)

    results: list[Optional[TurntableAssets]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results.extend(pool.map(lambda i: one(i, max(1, cores // jobs)), small))
    results.extend(one(pdb_id, cores) for pdb_id in large)
    return {a.pdb_id: a for a in results if a is not None}


def structure_atoms(pdb_id: str, cache_dir: Path) -> int:
    """ATOM + HETATM record count of a (cached, fetched on demand) PDB entry."""
    pdb_path, _ = fetch_pdb(pdb_id, cache_dir)
    with pdb_path.open("rb") as f:
        return sum(1 for line in f if line.startswith((b"ATOM", b"HETATM")))


def render_threads() -> int:
    """Ray-tracing threads for a render that owns the machine.

    PyMOL splits each frame into per-thread strips and waits for the slowest,
    so on Apple silicon a strip scheduled onto an efficiency core stalls the
    frame: use the performance-core count there, every logical CPU elsewhere.
    """
    if sys.platform == "darwin":
        try:
            out = subprocess.run(  # noqa: S603
                ["/usr/sbin/sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ""
        if out.isdigit() and int(out) > 0:
            return int(out)
    return os.cpu_count() or 4
