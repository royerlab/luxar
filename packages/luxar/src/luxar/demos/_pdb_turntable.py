"""Render a slowly turning PDB structure to a transparent WebM turntable.

Used by ``demo_esm3_protein_stories`` to show a representative structure next to
each story's panel. Two tools take part:

* **PyMOL open-source** computes the molecular surface. This is a deliberate
  SPECIAL CASE in the demos' dependency policy: PyMOL is not pip-installable
  (conda-forge or Homebrew only), so it is not in
  :data:`luxar.demos.INSTALL_SPECS`. The gate prints an install hint naming both
  routes and lets the demo continue WITHOUT turntables rather than raising — the
  stories scene is complete without them.
* **moderngl** (pip, in the ``demos`` extra) shades it on the GPU, offscreen —
  see :mod:`luxar.demos._clay_renderer`. PyMOL's own ray tracer was the first
  implementation and took five and a half hours for the ten structures (2-6 s a
  frame on a laptop, no GPU path); the GPU renders a frame in about a
  millisecond and the whole set in the time ffmpeg takes to encode it.

Pipeline (all cached under ``~/.cache/luxar/pdb_turntables``):

1. Fetch ``<ID>.pdb`` and the entry title from RCSB (``files.rcsb.org`` and
   ``data.rcsb.org``).
2. PyMOL headless (``pymol -cq <script>``) exports the molecular surface of the
   polymer, one mesh per chain (OBJ, re-saved as compressed float32 ``.npz``),
   into ``<ID>_mesh_v<MESH_VERSION>_q<Q>/``
   (cached: a style change re-renders without recomputing surfaces).
3. The GPU renderer stands the assembly on its longest principal axis and
   draws ``frames`` frames of one turn — matte clay in pastel shades of the
   story colour, soft lights, screen-space ambient occlusion, transparent
   background — streaming them into ffmpeg.
4. ffmpeg encodes an opaque VP9 WebM as a **stacked alpha matte** — the colour
   on top, the alpha channel as a grey matte below, twice as tall — which the
   viewer recombines in a shader (``Scene.add_video(alpha_matte="stacked")``).
   A VP9 alpha plane was the first encoding; Safari / WKWebView decode it and
   drop the alpha, so the exported kiosk app showed black squares. Frame 0 is
   kept as a transparent PNG poster.

At the default 900 frames that is 0.4° per frame, a slow 30 s turn at 30 fps
(the owner found one turn in 6 s far too fast). Renders are keyed by
``(pdb_id, STYLE_VERSION, frames, size, colour)``; bump ``STYLE_VERSION`` when
the look changes and ``MESH_VERSION`` when the surface export changes.
"""

from __future__ import annotations

import colorsys
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from arbol import aprint

from luxar.demos._clay_renderer import (
    MODERNGL_INSTALL_HINT,
    ClayRenderer,
    load_obj,
    meshes_from_files,
    render_turntable_video,
    save_mesh_npz,
)

#: Bump when the LOOK changes (shading, palette, frame count semantics) so
#: cached turntables re-render.
#: 7: the clip is a stacked alpha matte (colour over a grey matte, twice as
#: tall, opaque VP9) instead of a VP9 alpha plane — Safari / WKWebView drop the
#: alpha plane, so the exported kiosk app showed black squares.
STYLE_VERSION = 7
#: Bump when the PyMOL surface export changes so cached meshes are recomputed.
MESH_VERSION = 2
DEFAULT_FRAMES = 900
DEFAULT_FPS = 30
# 768 px covers the overlay's ~26% of a 4K kiosk width (~1000 px) at a 1.3x
# upscale.
DEFAULT_SIZE = 768
# Structures above this many atoms take PyMOL surface_quality 0 instead of 1:
# the GPU does not care about triangle counts, but PyMOL's surface computation
# does, and a 50k-atom complex reads the same at overlay size.
LARGE_STRUCTURE_ATOMS = 15_000
#: +1 spins so the front of the structure moves to the right — the same sense
#: as the scene's world-y auto-rotation beside it, as the owner sees it.
TURN_DIRECTION = 1.0
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
    """One rendered structure: the stacked-matte WebM, its poster, and the entry title."""

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


def has_moderngl() -> bool:
    """Whether the GPU renderer's one pip dependency is importable."""
    return importlib.util.find_spec("moderngl") is not None


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


def mesh_dir_name(pdb_id: str, quality: int, mesh_version: int = MESH_VERSION) -> str:
    """Cache directory stem for one structure's exported surface meshes."""
    return f"{pdb_id.upper()}_mesh_v{mesh_version}_q{quality}"


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


def surface_quality_for(n_atoms: int) -> int:
    """PyMOL ``surface_quality`` by structure size (1 fine, 0 default)."""
    return 0 if n_atoms > LARGE_STRUCTURE_ATOMS else 1


def pymol_surface_script(pdb_path: Path, mesh_dir: Path, *, quality: int) -> str:
    """The PyMOL Python script that exports the molecular surface, per chain.

    Solvent and hydrogens are removed, then for every chain of the polymer the
    surface of THAT chain alone is shown and saved as ``chain_<i>.obj`` (PyMOL's
    OBJ export writes whatever is drawn, with no colours, so per-chain colouring
    means per-chain files). ``chains.json`` lists the chain ids in file order.
    """
    return "\n".join(
        [
            "import json",
            "from pymol import cmd",
            f"cmd.load({str(pdb_path)!r}, 'mol')",
            "cmd.remove('solvent')",
            "cmd.remove('hydro')",
            "cmd.hide('everything')",
            f"cmd.set('surface_quality', {int(quality)})",
            "chains = cmd.get_chains('mol and polymer') or ['']",
            "for i, chain in enumerate(chains):",
            "    cmd.hide('everything')",
            "    cmd.show('surface', \"polymer and chain '%s'\" % chain)",
            f"    cmd.save({str(mesh_dir)!r} + '/chain_%d.obj' % i)",
            f"with open({str(mesh_dir)!r} + '/chains.json', 'w') as f:",
            "    json.dump(chains, f)",
        ]
    )


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


def _cached_turntable_assets(
    pdb_id: str,
    cache_dir: Path,
    *,
    frames: int,
    fps: int,
    size: int,
    color: RGB,
) -> Optional[TurntableAssets]:
    """Return complete cached assets without requiring render dependencies."""
    pdb_id = pdb_id.upper()
    stem = cache_key(pdb_id, frames, size, color=color)
    webm = cache_dir / f"{stem}.webm"
    poster = cache_dir / f"{stem}.png"
    if not (webm.exists() and poster.exists()):
        return None

    title = pdb_id
    meta_path = cache_dir / f"{pdb_id}.json"
    if meta_path.exists():
        try:
            title = str(json.loads(meta_path.read_text()).get("title", pdb_id))
        except (OSError, ValueError, TypeError):
            pass
    return TurntableAssets(pdb_id, webm, poster, title, frames, fps)


def structure_atoms(pdb_id: str, cache_dir: Path) -> int:
    """ATOM + HETATM record count of a (cached, fetched on demand) PDB entry."""
    pdb_path, _ = fetch_pdb(pdb_id, cache_dir)
    with pdb_path.open("rb") as f:
        return sum(1 for line in f if line.startswith((b"ATOM", b"HETATM")))


def export_surface_meshes(
    pdb_id: str,
    cache_dir: Path,
    *,
    quality: int,
    pymol: Sequence[str],
) -> list[Path]:
    """PyMOL surface export into the mesh cache (reused when complete).

    Returns the per-chain OBJ paths in chain order. Raises on a PyMOL failure
    or an incomplete export.
    """
    pdb_id = pdb_id.upper()
    pdb_path, _ = fetch_pdb(pdb_id, cache_dir)
    mesh_dir = cache_dir / mesh_dir_name(pdb_id, quality)
    chains_file = mesh_dir / "chains.json"
    if chains_file.exists():
        chains = json.loads(chains_file.read_text())
        paths = [mesh_dir / f"chain_{i}.npz" for i in range(len(chains))]
        if all(p.exists() for p in paths):
            return paths
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    mesh_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=f"luxar_surface_{pdb_id}_") as tmp:
        script = Path(tmp) / "surface.py"
        script.write_text(pymol_surface_script(pdb_path, mesh_dir, quality=quality))
        aprint(f"PyMOL: computing the surface of {pdb_id} (quality {quality}) …")
        subprocess.run([*pymol, str(script)], check=True, capture_output=True)  # noqa: S603
    if not chains_file.exists():
        raise RuntimeError(f"PyMOL wrote no chain list for {pdb_id}")
    chains = json.loads(chains_file.read_text())
    objs = [mesh_dir / f"chain_{i}.obj" for i in range(len(chains))]
    missing = [p.name for p in objs if not p.exists()]
    if missing:
        raise RuntimeError(f"PyMOL exported no surface for {pdb_id}: {missing}")
    # PyMOL can only write OBJ text (tens of MB per chain); keep the meshes as
    # compressed float32 arrays instead, a tenth of the size and quicker to load.
    paths = []
    for obj in objs:
        pos, nrm = load_obj(obj)
        npz = obj.with_suffix(".npz")
        save_mesh_npz(npz, pos, nrm)
        obj.unlink()
        paths.append(npz)
    return paths


def render_turntable(
    pdb_id: str,
    cache_dir: Path,
    *,
    frames: int = DEFAULT_FRAMES,
    fps: int = DEFAULT_FPS,
    size: int = DEFAULT_SIZE,
    color: RGB = DEFAULT_COLOR,
    pymol: Optional[Sequence[str]] = None,
    ffmpeg: Optional[str] = None,
    renderer: Optional[ClayRenderer] = None,
) -> TurntableAssets:
    """Render (or reuse) one structure's turntable. Raises on tool failure.

    A ``renderer`` may be shared across structures (one GL context); without
    one a renderer is created for this call.
    """
    pdb_id = pdb_id.upper()
    cached = _cached_turntable_assets(
        pdb_id,
        cache_dir,
        frames=frames,
        fps=fps,
        size=size,
        color=color,
    )
    if cached is not None:
        return cached

    pymol_cmd = list(pymol) if pymol is not None else find_pymol()
    ffmpeg_exe = ffmpeg if ffmpeg is not None else find_ffmpeg()
    if pymol_cmd is None:
        raise RuntimeError(PYMOL_INSTALL_HINT)
    if ffmpeg_exe is None:
        raise RuntimeError(FFMPEG_INSTALL_HINT)
    if not has_moderngl():
        raise RuntimeError(MODERNGL_INSTALL_HINT)

    pdb_path, title = fetch_pdb(pdb_id, cache_dir)
    stem = cache_key(pdb_id, frames, size, color=color)
    webm = cache_dir / f"{stem}.webm"
    poster = cache_dir / f"{stem}.png"

    quality = surface_quality_for(structure_atoms(pdb_id, cache_dir))
    files = export_surface_meshes(pdb_id, cache_dir, quality=quality, pymol=pymol_cmd)
    meshes = meshes_from_files(files, chain_palette(color, len(files)))
    aprint(
        f"GPU: rendering {frames} frames of {pdb_id} at {size}px "
        f"({len(files)} chains, {sum(len(m.positions) for m in meshes) // 3:,} "
        f"triangles) → stacked-matte VP9 WebM at {fps} fps …"
    )
    render_turntable_video(
        meshes,
        frames=frames,
        fps=fps,
        size=size,
        webm=webm,
        poster=poster,
        ffmpeg=ffmpeg_exe,
        renderer=renderer,
        turn_direction=TURN_DIRECTION,
    )
    return TurntableAssets(pdb_id, webm, poster, title, frames, fps)


def render_turntables(
    pdb_ids: Sequence[str],
    cache_dir: Path,
    *,
    frames: int = DEFAULT_FRAMES,
    fps: int = DEFAULT_FPS,
    size: int = DEFAULT_SIZE,
    colors: Optional[Mapping[str, RGB]] = None,
) -> dict[str, TurntableAssets]:
    """Render several structures with one shared GPU renderer.

    ``colors`` maps a PDB id to the RGB triple (in [0, 1]) whose hue the
    structure is rendered in — the demo passes each story's highlight colour so
    the turntable matches its cluster; ids without one get ``DEFAULT_COLOR``.

    Returns cached or newly rendered assets keyed by PDB id. Missing render
    dependencies only skip uncached structures; complete cached assets remain
    usable. A structure that fails to render is reported and skipped; the
    others still return.
    """
    palette = {k.upper(): v for k, v in (colors or {}).items()}
    results: dict[str, TurntableAssets] = {}
    pending: list[str] = []
    for pdb_id in pdb_ids:
        normalized = pdb_id.upper()
        cached = _cached_turntable_assets(
            normalized,
            cache_dir,
            frames=frames,
            fps=fps,
            size=size,
            color=palette.get(normalized, DEFAULT_COLOR),
        )
        if cached is None:
            pending.append(normalized)
        else:
            results[normalized] = cached
    if not pending:
        return results

    pymol_cmd, ffmpeg_exe, hint = _resolve_render_tools()
    if hint is not None:
        aprint("⚠️  Turntable videos skipped:")
        for line in hint.splitlines():
            aprint(f"   {line}")
        return results
    try:
        renderer = ClayRenderer(size)
    except Exception as e:  # noqa: BLE001 — no GL context on this machine
        aprint(f"⚠️  Turntable videos skipped: no GPU context ({e})")
        return results

    try:
        for pdb_id in pending:
            try:
                a = render_turntable(
                    pdb_id,
                    cache_dir,
                    frames=frames,
                    fps=fps,
                    size=size,
                    color=palette.get(pdb_id.upper(), DEFAULT_COLOR),
                    pymol=pymol_cmd,
                    ffmpeg=ffmpeg_exe,
                    renderer=renderer,
                )
            except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
                detail = ""
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    detail = (
                        ": "
                        + (e.stderr.decode(errors="replace").strip().splitlines()[-1])
                    )
                aprint(
                    f"⚠️  Turntable for {pdb_id} failed{detail or f': {e}'} — skipped"
                )
                continue
            results[a.pdb_id] = a
    finally:
        renderer.release()
    return results


def _resolve_render_tools() -> tuple[Optional[list[str]], Optional[str], Optional[str]]:
    pymol_cmd = find_pymol()
    ffmpeg_exe = find_ffmpeg()
    if pymol_cmd is None:
        return None, ffmpeg_exe, PYMOL_INSTALL_HINT
    if ffmpeg_exe is None:
        return pymol_cmd, None, FFMPEG_INSTALL_HINT
    if not has_moderngl():
        return pymol_cmd, ffmpeg_exe, MODERNGL_INSTALL_HINT
    return pymol_cmd, ffmpeg_exe, None
