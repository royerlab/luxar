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
   default 360 frames that is exactly 1° per frame, a 6 s loop at 60 fps.
3. Encode with ffmpeg to VP9 WebM **with an alpha channel** (``yuva420p``) and
   keep frame 0 as a PNG poster (the Safari fallback: it cannot decode alpha
   WebM).

Renders are keyed by ``(pdb_id, STYLE_VERSION, frames, size)``; bump
``STYLE_VERSION`` when the PyMOL style changes so cached turntables re-render.
"""

from __future__ import annotations

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
from typing import Optional, Sequence

from arbol import aprint

#: Bump when :func:`pymol_script` changes so cached turntables re-render.
STYLE_VERSION = 2
DEFAULT_FRAMES = 360
DEFAULT_FPS = 60
# 768 px covers the overlay's ~26% of a 4K kiosk width (~1000 px) at a 1.3x
# upscale; ray-tracing cost scales with pixels, and 1024 px cost 1.5x more.
DEFAULT_SIZE = 768
# Parallel PyMOL processes; each gets cpu_count // DEFAULT_JOBS ray threads.
# PyMOL's tracer scales sub-linearly with threads (12 threads = 5.8x, 4 = 3.4x
# on an M-series Mac), so three 4-thread jobs out-render one 12-thread job 1.7x.
DEFAULT_JOBS = 3
# Above this many non-solvent, non-metal hetero atoms the cofactors are hidden
# and only metals are drawn as spheres: photosystem II's ~17k chlorophyll atoms
# otherwise bury the protein under green spheres. Hemes, nucleotides and sugars
# on ordinary complexes stay well under it.
HETATM_SPHERE_LIMIT = 1500
# Structures above this many atoms take the coarser molecular surface
# (surface_quality -2 instead of -1): the surface dominates ray-tracing time
# (about 1.8x per quality step) and the translucent shell reads the same at
# overlay size.
LARGE_STRUCTURE_ATOMS = 15_000

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


def cache_key(
    pdb_id: str, frames: int, size: int, style_version: int = STYLE_VERSION
) -> str:
    """Stable file stem for one render configuration."""
    digest = hashlib.sha1(
        f"{pdb_id.upper()}|{style_version}|{frames}|{size}".encode()
    ).hexdigest()
    return f"{pdb_id.upper()}_v{style_version}_{frames}f_{size}px_{digest[:8]}"


def pymol_script(
    pdb_path: Path, frames_dir: Path, *, frames: int, size: int, threads: int = 4
) -> str:
    """The PyMOL Python script for one turntable.

    Style: cartoon coloured by chain over a translucent grey molecular surface,
    ligands and cofactors (hemes, nucleotides, sugars) as element-coloured
    spheres — metals only on a cofactor-crowded complex (``HETATM_SPHERE_LIMIT``)
    — nucleic acids as cartoon, no shadows or fog, and a transparent background.
    One ``turn y`` per frame around the oriented view.

    Cost is ray-tracing, so the knobs that were measured to matter are set
    cheap: ``antialias 1`` (2 cost 1.9x), ``ray_trace_mode 0`` (the outline mode
    cost 1.5x and looked busier), and a surface quality picked by structure size
    (``LARGE_STRUCTURE_ATOMS``). Hydrogens (NMR entries) are removed.
    """
    step = 360.0 / frames
    return "\n".join(
        [
            "from pymol import cmd, util",
            f"cmd.set('max_threads', {int(threads)})",
            f"cmd.load({str(pdb_path)!r}, 'mol')",
            "cmd.remove('solvent')",
            "cmd.remove('hydro')",
            "cmd.hide('everything')",
            "cmd.dss('mol')",
            "cmd.show('cartoon', 'polymer')",
            "cmd.set('cartoon_ring_mode', 3)",
            "cmd.set('cartoon_transparency', 0.0)",
            "util.cbc('polymer', first_color=2)",
            "n_het = cmd.count_atoms('hetatm and not solvent and not metals')",
            f"if n_het <= {HETATM_SPHERE_LIMIT}:",
            "    cmd.show('spheres', 'hetatm and not solvent')",
            "else:",
            "    cmd.show('spheres', 'hetatm and metals')",
            "util.cbag('hetatm')",
            "cmd.show('surface', 'polymer')",
            "cmd.set('surface_color', 'grey85')",
            "cmd.set('transparency', 0.72)",
            "n_atoms = cmd.count_atoms('mol')",
            f"cmd.set('surface_quality', -2 if n_atoms > {LARGE_STRUCTURE_ATOMS} else -1)",
            "cmd.set('ambient', 0.28)",
            "cmd.set('direct', 0.45)",
            "cmd.set('reflect', 0.35)",
            "cmd.set('spec_reflect', 0.6)",
            "cmd.set('spec_power', 80)",
            "cmd.set('ray_shadows', 0)",
            "cmd.set('depth_cue', 0)",
            "cmd.set('ray_opaque_background', 0)",
            "cmd.set('ray_trace_mode', 0)",
            "cmd.set('antialias', 1)",
            "cmd.set('field_of_view', 22)",
            "cmd.orient('mol')",
            "cmd.zoom('mol', buffer=1.5)",
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
    stem = cache_key(pdb_id, frames, size)
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
                pdb_path, frames_dir, frames=frames, size=size, threads=threads
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
) -> dict[str, TurntableAssets]:
    """Render several structures in parallel PyMOL processes.

    Returns the assets that rendered, keyed by PDB id. When PyMOL or ffmpeg is
    missing, prints the install hint ONCE and returns an empty dict — the caller
    builds its scene without turntables. A structure that fails to render is
    reported and skipped; the others still return. The machine's cores are
    split evenly across the ``jobs`` PyMOL processes.
    """
    pymol_cmd = find_pymol()
    ffmpeg_exe = find_ffmpeg()
    jobs = max(1, jobs)
    threads = max(1, (os.cpu_count() or jobs) // jobs)
    if pymol_cmd is None or ffmpeg_exe is None:
        aprint("⚠️  Turntable videos skipped:")
        for line in (
            PYMOL_INSTALL_HINT if pymol_cmd is None else FFMPEG_INSTALL_HINT
        ).splitlines():
            aprint(f"   {line}")
        return {}

    def one(pdb_id: str) -> Optional[TurntableAssets]:
        try:
            return render_turntable(
                pdb_id,
                cache_dir,
                frames=frames,
                fps=fps,
                size=size,
                threads=threads,
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

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(one, pdb_ids))
    return {a.pdb_id: a for a in results if a is not None}
