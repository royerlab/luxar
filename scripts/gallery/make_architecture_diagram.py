#!/usr/bin/env python3
"""Draw the README's two diagrams (SVG via drawsvg, PNG via rsvg-convert).

``--which pipeline`` (``build()``) is the "how Luxar works" figure:
four stages left to right, describe the scene in Python, compile it once, the
``.luxar.zarr`` archive on any static host, explore it in the browser.
``--which layers`` (``build_layers()``) is the Architecture figure: the Python
package's modules on the left, the viewer's on the right, the archive between
them. ``--which`` defaults to ``both``. Each is drawn twice, for GitHub's dark
and light themes, and the README embeds both through a ``<picture>`` element.

The published PNGs were authored on macOS with "Helvetica Neue" and "Menlo";
rsvg-convert resolves whatever fontconfig offers, so a rebuild on another
machine renders the same content with different fonts and a different hash.
Treat the hosted files as the reference and re-publish only from a Mac.

    hatch run python scripts/gallery/make_architecture_diagram.py -o out/                 # both figures, dark + light
    hatch run python scripts/gallery/make_architecture_diagram.py --which layers -o out/  # architecture layers
    hatch run python scripts/gallery/make_architecture_diagram.py --theme dark --scale 2 -o out/

The PNGs are named after their manifest asset (``architecture-dark.png``,
``layers-light.png``, ...) so ``publish_media.py --record-by-stem`` records them
under the right name; they are hosted by content hash on
data.luxarviewer.dev/media rather than committed.
"""

# mypy: allow-untyped-calls, allow-untyped-defs
# drawsvg ships no type information; the drawing helpers are plain SVG builders.
from __future__ import annotations

import argparse
import math
import random  # decorative jitter only  # nosec B311
import shutil
import subprocess  # nosec B404: shells out to rsvg-convert with a fixed argv
import sys
from dataclasses import dataclass
from pathlib import Path

import drawsvg as dw  # type: ignore[import-untyped]

W, H = 1240, 536
SANS = "Helvetica Neue"
MONO = "Menlo"


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    card: str
    card_edge: str
    text: str
    muted: str
    faint: str
    accent: str
    accent_soft: str
    arrow: str
    code_kw: str
    code_fn: str
    code_str: str
    code_num: str
    canvas: str
    points: str = "#3b82f6"
    lines: str = "#10b981"
    mesh: str = "#f97316"
    splats: str = "#a855f7"


DARK = Theme(
    name="dark",
    bg="#0d1117",
    card="#161b22",
    card_edge="#30363d",
    text="#e6edf3",
    muted="#9198a1",
    faint="#6e7681",
    accent="#79c0ff",
    accent_soft="#1f3b5c",
    arrow="#6e7681",
    code_kw="#ff7b72",
    code_fn="#d2a8ff",
    code_str="#a5d6ff",
    code_num="#79c0ff",
    canvas="#05070c",
)
LIGHT = Theme(
    name="light",
    bg="#ffffff",
    card="#f6f8fa",
    card_edge="#d0d7de",
    text="#1f2328",
    muted="#59636e",
    faint="#8c959f",
    accent="#0969da",
    accent_soft="#ddf4ff",
    arrow="#8c959f",
    code_kw="#cf222e",
    code_fn="#8250df",
    code_str="#0a3069",
    code_num="#0550ae",
    canvas="#05070c",
)


def text(
    d,
    x,
    y,
    s,
    size=13,
    fill="#000",
    weight="normal",
    anchor="start",
    family=SANS,
    italic=False,
):
    s = s.replace("  ", "\u00a0\u00a0")  # SVG collapses runs of spaces; keep alignment
    d.append(
        dw.Text(
            s,
            size,
            x,
            y,
            fill=fill,
            font_family=family,
            font_weight=weight,
            text_anchor=anchor,
            font_style="italic" if italic else "normal",
        )
    )


def card(d, t, x, y, w, h, title, subtitle=None):
    d.append(
        dw.Rectangle(x, y, w, h, rx=12, fill=t.card, stroke=t.card_edge, stroke_width=1)
    )
    text(d, x + 18, y + 30, title, 17, t.text, "bold")
    if subtitle:
        text(d, x + 18, y + 50, subtitle, 12, t.muted)


def arrow(d, t, x1, y1, x2, y2, width=1.6):
    m = dw.Marker(-0.5, -0.5, 1.0, 0.5, scale=8, orient="auto")
    m.append(dw.Lines(-0.5, -0.4, 0.5, 0, -0.5, 0.4, close=True, fill=t.arrow))
    d.append(dw.Line(x1, y1, x2, y2, stroke=t.arrow, stroke_width=width, marker_end=m))


# --- glyphs -----------------------------------------------------------------


def glyph_points(d, t, x, y, s):
    rng = random.Random(3)  # nosec B311
    for _ in range(16):
        d.append(
            dw.Circle(
                x + rng.random() * s, y + rng.random() * s, s * 0.055, fill=t.points
            )
        )


def glyph_lines(d, t, x, y, s):
    for k, (ph, amp) in enumerate([(0.0, 0.25), (1.2, 0.3), (2.4, 0.2)]):
        pts = []
        for i in range(31):
            u = i / 30
            pts += [
                x + u * s,
                y + s * (0.2 + k * 0.28 + amp * math.sin(6 * u + ph) * 0.5),
            ]
        d.append(
            dw.Lines(
                *pts,
                stroke=t.lines,
                stroke_width=s * 0.05,
                fill="none",
                stroke_linecap="round",
            )
        )


def glyph_mesh(d, t, x, y, s):
    p = [(0.05, 0.9), (0.95, 0.85), (0.55, 0.05), (0.5, 0.55)]
    for a, b, c in [(0, 1, 3), (1, 2, 3), (0, 3, 2)]:
        d.append(
            dw.Lines(
                *[v for i in (a, b, c) for v in (x + p[i][0] * s, y + p[i][1] * s)],
                close=True,
                fill=t.mesh,
                fill_opacity=0.35,
                stroke=t.mesh,
                stroke_width=s * 0.04,
                stroke_linejoin="round",
            )
        )


def glyph_volume(d, t, x, y, s):
    o = 0.28 * s
    f = s - o
    d.append(
        dw.Rectangle(
            x + o,
            y,
            f,
            f,
            fill="none",
            stroke=t.splats,
            stroke_width=s * 0.035,
            stroke_opacity=0.5,
        )
    )
    d.append(
        dw.Rectangle(
            x,
            y + o,
            f,
            f,
            fill=t.splats,
            fill_opacity=0.18,
            stroke=t.splats,
            stroke_width=s * 0.035,
        )
    )
    for dx, dy in [(0, 0), (f, 0), (0, f), (f, f)]:
        d.append(
            dw.Line(
                x + dx,
                y + o + dy,
                x + dx + o,
                y + dy,
                stroke=t.splats,
                stroke_width=s * 0.03,
                stroke_opacity=0.5,
            )
        )
    for cx, cy, r in [(0.3, 0.62, 0.13), (0.62, 0.45, 0.1), (0.45, 0.28, 0.07)]:
        d.append(
            dw.Circle(
                x + cx * f, y + o + cy * f, r * f, fill=t.splats, fill_opacity=0.85
            )
        )


def glyph_laptop(d, t, x, y, s):
    d.append(
        dw.Rectangle(
            x + 0.12 * s,
            y + 0.15 * s,
            0.76 * s,
            0.55 * s,
            rx=s * 0.04,
            fill=t.bg,
            stroke=t.text,
            stroke_width=1.2,
        )
    )
    d.append(
        dw.Lines(
            x,
            y + 0.78 * s,
            x + s,
            y + 0.78 * s,
            x + 0.9 * s,
            y + 0.70 * s,
            x + 0.1 * s,
            y + 0.70 * s,
            close=True,
            fill=t.card_edge,
            stroke=t.text,
            stroke_width=1,
        )
    )


def glyph_gpu(d, t, x, y, s):
    d.append(
        dw.Rectangle(
            x + 0.2 * s,
            y + 0.05 * s,
            0.6 * s,
            0.85 * s,
            rx=s * 0.04,
            fill=t.bg,
            stroke=t.text,
            stroke_width=1.2,
        )
    )
    d.append(
        dw.Circle(
            x + 0.5 * s,
            y + 0.6 * s,
            0.16 * s,
            fill=t.card_edge,
            stroke=t.text,
            stroke_width=1,
        )
    )
    d.append(
        dw.Line(
            x + 0.32 * s,
            y + 0.22 * s,
            x + 0.68 * s,
            y + 0.22 * s,
            stroke=t.text,
            stroke_width=1,
        )
    )


def glyph_cluster(d, t, x, y, s):
    for k in range(3):
        yy = y + 0.1 * s + k * 0.28 * s
        d.append(
            dw.Rectangle(
                x + 0.1 * s,
                yy,
                0.8 * s,
                0.2 * s,
                fill=t.bg,
                stroke=t.text,
                stroke_width=1,
            )
        )
        d.append(dw.Circle(x + 0.22 * s, yy + 0.1 * s, 0.03 * s, fill=t.accent))


def glyph_server(d, t, x, y, s):
    glyph_cluster(d, t, x, y, s)


def glyph_bucket(d, t, x, y, s):
    d.append(
        dw.Ellipse(
            x + 0.5 * s,
            y + 0.2 * s,
            0.4 * s,
            0.12 * s,
            fill=t.bg,
            stroke=t.text,
            stroke_width=1.2,
        )
    )
    d.append(
        dw.Lines(
            x + 0.1 * s,
            y + 0.2 * s,
            x + 0.2 * s,
            y + 0.88 * s,
            x + 0.8 * s,
            y + 0.88 * s,
            x + 0.9 * s,
            y + 0.2 * s,
            fill="none",
            stroke=t.text,
            stroke_width=1.2,
        )
    )
    d.append(
        dw.Ellipse(
            x + 0.5 * s,
            y + 0.88 * s,
            0.3 * s,
            0.09 * s,
            fill="none",
            stroke=t.text,
            stroke_width=1.2,
        )
    )


def glyph_page(d, t, x, y, s):
    d.append(
        dw.Lines(
            x + 0.18 * s,
            y + 0.05 * s,
            x + 0.65 * s,
            y + 0.05 * s,
            x + 0.82 * s,
            y + 0.22 * s,
            x + 0.82 * s,
            y + 0.95 * s,
            x + 0.18 * s,
            y + 0.95 * s,
            close=True,
            fill=t.bg,
            stroke=t.text,
            stroke_width=1.2,
        )
    )
    for k in range(3):
        d.append(
            dw.Line(
                x + 0.3 * s,
                y + (0.42 + 0.16 * k) * s,
                x + 0.7 * s,
                y + (0.42 + 0.16 * k) * s,
                stroke=t.faint,
                stroke_width=1,
            )
        )


def glyph_folder(d, t, x, y, s):
    d.append(
        dw.Lines(
            x + 0.08 * s,
            y + 0.2 * s,
            x + 0.4 * s,
            y + 0.2 * s,
            x + 0.48 * s,
            y + 0.3 * s,
            x + 0.92 * s,
            y + 0.3 * s,
            x + 0.92 * s,
            y + 0.85 * s,
            x + 0.08 * s,
            y + 0.85 * s,
            close=True,
            fill=t.bg,
            stroke=t.text,
            stroke_width=1.2,
            stroke_linejoin="round",
        )
    )


def glyph_hilbert(d, t, x, y, s):
    c = s * 0.42
    for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        d.append(
            dw.Rectangle(
                x + i * (c + s * 0.08),
                y + j * (c + s * 0.08),
                c,
                c,
                rx=2,
                fill=t.accent,
                fill_opacity=0.28,
            )
        )
    hc = c / 2
    p = [
        x + hc,
        y + hc,
        x + hc,
        y + hc + c + s * 0.08,
        x + hc + c + s * 0.08,
        y + hc + c + s * 0.08,
        x + hc + c + s * 0.08,
        y + hc,
    ]
    d.append(
        dw.Lines(
            *p,
            fill="none",
            stroke=t.accent,
            stroke_width=2.2,
            stroke_linecap="round",
            stroke_linejoin="round",
        )
    )


def glyph_squeeze(d, t, x, y, s):
    d.append(
        dw.Rectangle(
            x,
            y + 0.12 * s,
            s,
            0.2 * s,
            rx=2,
            fill=t.accent,
            fill_opacity=0.28,
            stroke=t.accent,
            stroke_width=1,
        )
    )
    d.append(
        dw.Rectangle(x + 0.3 * s, y + 0.68 * s, 0.4 * s, 0.2 * s, rx=2, fill=t.accent)
    )
    arrow(d, t, x + 0.5 * s, y + 0.38 * s, x + 0.5 * s, y + 0.6 * s, 1.4)


def glyph_bounds(d, t, x, y, s):
    d.append(
        dw.Rectangle(
            x,
            y,
            s,
            s,
            rx=2,
            fill="none",
            stroke=t.accent,
            stroke_width=1.2,
            stroke_dasharray="3,2",
        )
    )
    rng = random.Random(7)  # nosec B311
    for _ in range(3):
        bx, by = x + rng.random() * 0.5 * s, y + rng.random() * 0.5 * s
        bw, bh = 0.25 * s + rng.random() * 0.25 * s, 0.25 * s + rng.random() * 0.25 * s
        d.append(
            dw.Rectangle(
                bx,
                by,
                bw,
                bh,
                rx=1.5,
                fill=t.accent,
                fill_opacity=0.18,
                stroke=t.accent,
                stroke_width=1,
            )
        )


def glyph_lod(d, t, x, y, s):
    for row, n in enumerate([1, 2, 4]):
        gap = s * 0.06
        bw = (s - (n - 1) * gap) / n
        for i in range(n):
            d.append(
                dw.Rectangle(
                    x + i * (bw + gap),
                    y + row * s * 0.36,
                    bw,
                    s * 0.26,
                    rx=2,
                    fill=t.accent,
                    fill_opacity=0.32 + 0.22 * row,
                )
            )


def glyph_axes(d, t, x, y, s):
    for k, label in enumerate(["t", "ch", "…"]):
        yy = y + 0.18 * s + k * 0.32 * s
        d.append(dw.Line(x + 0.22 * s, yy, x + s, yy, stroke=t.faint, stroke_width=1.5))
        d.append(
            dw.Circle(
                x + 0.22 * s + (0.78 * s) * (0.3 + 0.25 * k),
                yy,
                s * 0.07,
                fill=t.accent,
            )
        )
        text(d, x, yy + 3, label, 9, t.muted, family=MONO)


def glyph_browser(d, t, x, y, w, h, url):
    d.append(
        dw.Rectangle(x, y, w, h, rx=8, fill=t.bg, stroke=t.card_edge, stroke_width=1.2)
    )
    bar = 26
    d.append(dw.Rectangle(x, y, w, bar, rx=8, fill=t.card))
    d.append(dw.Rectangle(x, y + bar - 8, w, 8, fill=t.card))
    d.append(dw.Line(x, y + bar, x + w, y + bar, stroke=t.card_edge, stroke_width=1))
    for k, col in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        d.append(dw.Circle(x + 14 + k * 14, y + bar / 2, 4, fill=col))
    d.append(
        dw.Rectangle(
            x + 60,
            y + 5,
            w - 72,
            bar - 10,
            rx=5,
            fill=t.bg,
            stroke=t.card_edge,
            stroke_width=0.8,
        )
    )
    text(d, x + 68, y + bar / 2 + 4, url, 9.5, t.muted, family=MONO)
    return x, y + bar, w, h - bar


def mini_scene(d, t, x, y, w, h):
    """An abstract embryo of splats plus a dimension slider, standing in for the viewer canvas."""
    d.append(dw.Rectangle(x, y, w, h, fill=t.canvas))
    cx, cy = x + w * 0.5, y + h * 0.46
    rx, ry = w * 0.36, h * 0.33
    grad = dw.RadialGradient(cx, cy, max(rx, ry))
    grad.add_stop(0, "#c084fc", 0.9)
    grad.add_stop(0.6, "#7e22ce", 0.55)
    grad.add_stop(1, "#3b0764", 0.0)
    d.append(dw.Ellipse(cx, cy, rx, ry, fill=grad))
    rng = random.Random(11)  # nosec B311
    for _ in range(140):
        a = rng.random() * 2 * math.pi
        r = math.sqrt(rng.random())
        px, py = cx + math.cos(a) * rx * 0.95 * r, cy + math.sin(a) * ry * 0.95 * r
        d.append(
            dw.Circle(
                px,
                py,
                1.6 + rng.random() * 1.6,
                fill="#f5d0fe",
                fill_opacity=0.55 + 0.4 * rng.random(),
            )
        )
    # slider
    sy = y + h - 16
    d.append(
        dw.Line(
            x + 16,
            sy,
            x + w - 16,
            sy,
            stroke="#374151",
            stroke_width=3,
            stroke_linecap="round",
        )
    )
    d.append(dw.Circle(x + 16 + (w - 32) * 0.62, sy, 5, fill="#60a5fa"))
    text(d, x + 16, sy - 8, "time", 8.5, "#9ca3af", family=MONO)


# --- code colouring -----------------------------------------------------------


def code_line(d, t, x, y, line, size=10.5):
    """Very small Python highlighter: keywords, strings, numbers, calls."""
    import re

    token = re.compile(r'"[^"]*"|\b\d+\b|[A-Za-z_][A-Za-z_0-9]*|\S|\s+')
    keywords = {"with", "as", "from", "import"}
    cx = x
    adv = size * 0.602  # Menlo advance
    for m in token.finditer(line):
        s = m.group(0)
        if s.isspace():
            cx += adv * len(s)
            continue
        if s in keywords:
            col = t.code_kw
        elif s.startswith('"'):
            col = t.code_str
        elif s.isdigit():
            col = t.code_num
        elif m.end() < len(line) and line[m.end()] == "(":
            col = t.code_fn
        else:
            col = t.text
        text(d, cx, y, s, size, col, family=MONO)
        cx += adv * len(s)


# --- the drawing --------------------------------------------------------------


def _column_describe(d, t, x, w, top, ch):
    # ---- 1. describe in Python
    card(
        d,
        t,
        x,
        top,
        w,
        ch,
        "1  Describe in Python",
        "NumPy arrays in, one scene graph out",
    )
    gs = 34
    for k, (g, lab) in enumerate(
        [
            (glyph_points, "points"),
            (glyph_lines, "lines"),
            (glyph_mesh, "meshes"),
            (glyph_volume, "nD images"),
        ]
    ):
        gx = x + 22 + k * 60
        g(d, t, gx, top + 68, gs)
        text(d, gx + gs / 2, top + 68 + gs + 14, lab, 10, t.muted, anchor="middle")
    code = [
        "dims = Dimensions([",
        '  Dimension("x", unit="um"),',
        '  Dimension("y", unit="um"),',
        '  Dimension("z", unit="um"),',
        '  Dimension("time", unit="s",',
        "            step=30, display=False)])",
        "with LuxarZarrCompiler(path) as c:",
        "  s = c.create_scene(dims)",
        '  s.add_gsplats("nuclei", mu, a, L)',
        '  s.add_points("cells", pos, rgb)',
        '  s.add_lines("tracks", verts, w)',
        '  s.add_mesh("surface", v, faces)',
    ]
    for k, ln in enumerate(code):
        code_line(d, t, x + 18, top + 150 + k * 15.5, ln, 10.2)
    text(
        d,
        x + 18,
        top + 362,
        "nD images are fitted to Gaussian splats first,",
        10.5,
        t.muted,
    )
    text(d, x + 18, top + 377, "on a laptop, a GPU or a cluster:", 10.5, t.muted)
    for k, (g, lab) in enumerate(
        [(glyph_laptop, "laptop"), (glyph_gpu, "GPU"), (glyph_cluster, "cluster")]
    ):
        gx = x + 30 + k * 78
        g(d, t, gx, top + 386, 28)
        text(d, gx + 14, top + 386 + 28 + 13, lab, 9.5, t.muted, anchor="middle")


def _column_compile(d, t, x, w, top, ch):
    # ---- 2. compile once
    card(
        d, t, x, top, w, ch, "2  Compile once", "The expensive work, done ahead of time"
    )
    steps = [
        (
            glyph_hilbert,
            "Order on a space-filling curve",
            ["Hilbert or Morton: what is near in nD", "is near on disk"],
        ),
        (
            glyph_squeeze,
            "Chunk, quantise, compress",
            ["~64 KB chunks, uint16 / uint8 codes,", "Blosc-zstd"],
        ),
        (
            glyph_bounds,
            "Index every chunk",
            ["nD bounding boxes: fetch only what", "the view intersects"],
        ),
        (
            glyph_lod,
            "Build levels of detail",
            ["streaming ladders, coarse levels,", "spatial tiles"],
        ),
        (
            glyph_axes,
            "Keep every axis navigable",
            ["time, channel, any hidden dimension"],
        ),
    ]
    for k, (g, title, sub) in enumerate(steps):
        yy = top + 74 + k * 70
        g(d, t, x + 20, yy, 30)
        text(d, x + 64, yy + 12, title, 12, t.text, "bold")
        for j, line in enumerate(sub):
            text(d, x + 64, yy + 28 + j * 13, line, 9.8, t.muted)
    text(
        d,
        x + 18,
        top + ch - 16,
        "LuxarZarrCompiler · cal → fit → lod",
        10,
        t.accent,
        family=MONO,
    )


def _column_archive(d, t, x, w, top, ch):
    # ---- 3. the archive
    card(
        d,
        t,
        x,
        top,
        w,
        ch,
        "3  A .luxar.zarr archive",
        "Self-describing, static files: no server code",
    )
    tree = [
        ("scene.luxar.zarr/", t.text, True),
        ("├─ zarr.json       scene, dimensions", t.text, False),
        ("├─ nuclei/         gsplats", t.text, False),
        ("│  ├─ centers  cholesky  amplitudes", t.muted, False),
        ("│  ├─ chunk_bounds", t.muted, False),
        ("│  └─ additive_0 … additive_3   LOD", t.muted, False),
        ("├─ cells/          points", t.text, False),
        ("├─ tracks/         lines", t.text, False),
        ("├─ surface/        mesh", t.text, False),
        ("└─ overlays/       annotations", t.text, False),
    ]
    for k, (ln, col, bold) in enumerate(tree):
        text(
            d,
            x + 18,
            top + 78 + k * 15.5,
            ln,
            10.2,
            col,
            "bold" if bold else "normal",
            family=MONO,
        )
    text(
        d,
        x + 18,
        top + 262,
        "Serve it from anywhere that serves files:",
        11,
        t.text,
        "bold",
    )
    hosts = [
        (glyph_laptop, "luxar serve"),
        (glyph_server, "lab server"),
        (glyph_bucket, "object storage"),
        (glyph_page, "GitHub Pages"),
        (glyph_folder, "export folder"),
    ]
    for k, (g, lab) in enumerate(hosts):
        gx = x + 20 + k * 53
        g(d, t, gx, top + 280, 30)
        for i, part in enumerate(lab.split(" ")):
            text(d, gx + 15, top + 326 + i * 12, part, 9.2, t.muted, anchor="middle")
    text(
        d,
        x + 18,
        top + 378,
        "One archive, every consumer: local viewer,",
        10.5,
        t.muted,
    )
    text(
        d, x + 18, top + 393, "shared link, offline export, native app.", 10.5, t.muted
    )
    text(
        d,
        x + 18,
        top + ch - 16,
        "directory or a single .zarr.zip",
        10,
        t.accent,
        family=MONO,
    )


def _column_explore(d, t, x, w, top, ch):
    # ---- 4. explore in the browser
    card(
        d,
        t,
        x,
        top,
        w,
        ch,
        "4  Explore in any browser",
        "Streams what the view needs, renders on the GPU",
    )
    bx, by, bw, bh = x + 18, top + 66, w - 36, 176
    ix, iy, iw, ih = glyph_browser(d, t, bx, by, bw, bh, "luxarviewer.dev/?src=<url>")
    mini_scene(d, t, ix + 1, iy + 1, iw - 2, ih - 2)
    feats = [
        "fetches only the chunks in view, as you orbit",
        "picks the level of detail by screen area",
        "sliders and toggles for time, channel, any axis",
        "WebGL2 or WebGPU, HDR, six blending modes",
        "layers, hover labels, overlays, sound, recording",
        "phones and tablets; embeddable; scriptable",
    ]
    for k, f in enumerate(feats):
        text(d, x + 18, top + 268 + k * 18, "•  " + f, 10.6, t.text)
    text(
        d,
        x + 18,
        top + ch - 16,
        "share the link: no install, no server",
        10,
        t.accent,
        family=MONO,
    )


def build(t: Theme) -> dw.Drawing:
    d = dw.Drawing(W, H, origin=(0, 0))
    d.append(dw.Rectangle(0, 0, W, H, fill=t.bg))

    gap = 28
    widths = [268, 262, 292, 306]
    x0 = (W - (sum(widths) + 3 * gap)) / 2
    top, ch = 22, 440
    xs = []
    x = x0
    for w in widths:
        xs.append(x)
        x += w + gap

    for column, (x, w) in zip(
        (_column_describe, _column_compile, _column_archive, _column_explore),
        zip(xs, widths, strict=True),
        strict=True,
    ):
        column(d, t, x, w, top, ch)

    # ---- arrows between stages and the closing line
    ym = top + 40
    for i in range(3):
        arrow(d, t, xs[i] + widths[i] + 4, ym, xs[i + 1] - 4, ym, 1.8)
    text(
        d,
        W / 2,
        H - 38,
        "Compile once, explore many times:",
        13,
        t.text,
        "bold",
        anchor="middle",
    )
    text(
        d,
        W / 2,
        H - 18,
        "what remains at view time is bounded by your graphics card, screen and network link, not by the size or format of the file.",
        12,
        t.muted,
        anchor="middle",
        italic=True,
    )
    return d


LAYERS_W, LAYERS_H = 1240, 760


def module_rows(d, t, x, y, rows, row_h=50, label_w=118):
    """Rows of ``module`` + title + one or two description lines; returns the next y."""
    for k, (mod, title, desc) in enumerate(rows):
        yy = y + k * row_h
        text(d, x, yy, mod, 11, t.accent, family=MONO)
        text(d, x + label_w, yy, title, 12, t.text, "bold")
        for n, line in enumerate(desc.split("\n")):
            text(d, x + label_w, yy + 15 + n * 13, line, 9.6, t.muted)
    return y + len(rows) * row_h


def section(d, t, x, y, label):
    text(d, x, y, label.upper(), 9.5, t.faint, "bold")
    return y + 20


def build_layers(t: Theme) -> dw.Drawing:
    w_all, h_all = LAYERS_W, LAYERS_H
    d = dw.Drawing(w_all, h_all, origin=(0, 0))
    d.append(dw.Rectangle(0, 0, w_all, h_all, fill=t.bg))
    top, ch, gap = 22, 660, 24
    wl, wm, wr = 440, 210, 518
    x0 = (w_all - (wl + wm + wr + 2 * gap)) / 2
    xl, xm, xr = x0, x0 + wl + gap, x0 + wl + gap + wm + gap

    card(
        d,
        t,
        xl,
        top,
        wl,
        ch,
        "luxar  ·  Python package",
        "authoring, fitting and compilation",
    )
    y = section(d, t, xl + 18, top + 78, "Author")
    y = module_rows(
        d,
        t,
        xl + 18,
        y + 8,
        [
            (
                "core/",
                "Scene graph",
                "Scene, Group, Points, Lines, GSplats, Mesh;\nDimensions, transforms, layers, appearance",
            ),
            (
                "core/",
                "Annotations",
                "text, image, video and HTML overlays;\nsound nodes; story waypoints",
            ),
        ],
    )
    y = section(d, t, xl + 18, y + 4, "Fit")
    y = module_rows(
        d,
        t,
        xl + 18,
        y + 8,
        [
            (
                "gsplats/",
                "Gaussian splats",
                "cal → fit → lod; CUDA and MPS kernels;\ntiled, batch and Slurm fitting",
            ),
            (
                "mesh/",
                "Meshes",
                "import PLY/OBJ/STL/VTP/glTF; decimation\nand mesh LOD; physical-material env maps in environment/",
            ),
        ],
    )
    y = section(d, t, xl + 18, y + 4, "Compile")
    y = module_rows(
        d,
        t,
        xl + 18,
        y + 8,
        [
            (
                "io/",
                "Compiler",
                "Hilbert or Morton ordering, chunking, nD spatial\nindex, LOD ladders, Zarr v2 and v3",
            ),
            (
                "encoding/",
                "Codes",
                "semantic types, uint16 / uint8 quantisation,\nwidth-aware Blosc-zstd",
            ),
        ],
    )
    y = section(d, t, xl + 18, y + 4, "Run and share")
    module_rows(
        d,
        t,
        xl + 18,
        y + 8,
        [
            (
                "cli/",
                "Command line",
                "luxar demo · serve · export · optimize ·\ngsplat · mesh · env",
            ),
            (
                "control/",
                "Remote control",
                "drive a running viewer from Python\nover a WebSocket",
            ),
            (
                "demos/",
                "Demos",
                "90 bundled demos; hosted data pinned by\nSHA-256 and fetched on demand",
            ),
        ],
    )

    card(d, t, xm, top, wm, ch, ".luxar.zarr", "the contract between the two")
    tree = [
        ("scene.luxar.zarr/", t.text, True),
        ("├─ zarr.json", t.text, False),
        ("├─ nuclei/", t.text, False),
        ("│  ├─ centers …", t.muted, False),
        ("│  ├─ chunk_bounds", t.muted, False),
        ("│  └─ additive_0…3", t.muted, False),
        ("├─ cells/  tracks/", t.text, False),
        ("└─ overlays/", t.text, False),
    ]
    for k, (ln, col, bold) in enumerate(tree):
        text(
            d,
            xm + 18,
            top + 84 + k * 15.5,
            ln,
            10,
            col,
            "bold" if bold else "normal",
            family=MONO,
        )
    y = top + 84 + len(tree) * 15.5 + 18
    for line in [
        "a directory of chunk files,",
        "or one .zarr.zip read by",
        "byte range",
        "",
        "served by any static host:",
        "luxar serve, a lab server,",
        "object storage, GitHub Pages,",
        "an exported folder",
        "",
        "opened by URL:",
    ]:
        text(d, xm + 18, y, line, 10.5, t.muted)
        y += 15
    text(d, xm + 18, y, "luxarviewer.dev/?src=…", 10, t.accent, family=MONO)
    for k, g in enumerate([glyph_hilbert, glyph_squeeze, glyph_bounds, glyph_lod]):
        g(d, t, xm + 22 + k * 48, top + ch - 110, 28)
    text(
        d,
        xm + wm / 2,
        top + ch - 62,
        "ordered · compressed · indexed · laddered",
        9.2,
        t.muted,
        anchor="middle",
    )
    text(
        d,
        xm + 18,
        top + ch - 16,
        "spec: LUXAR_ZARR_FORMAT.md",
        9.5,
        t.accent,
        family=MONO,
    )

    card(
        d,
        t,
        xr,
        top,
        wr,
        ch,
        "luxar-viewer  ·  TypeScript",
        "streams and renders; npm @luxar/viewer, luxarviewer.dev",
    )
    module_rows(
        d,
        t,
        xr + 18,
        top + 88,
        [
            (
                "core/",
                "LuxarApp",
                "the embeddable app and its API: camera, layers,\ndimensions, events; story tours and remote control",
            ),
            (
                "ui/ themes/",
                "Interface",
                "rail and panels: layers, dimension navigation,\nmonitor, recording, help; four themes",
            ),
            (
                "controls/ input/",
                "Navigation",
                "orbit, fly and ortho cameras;\nkeyboard, mouse and touch",
            ),
            (
                "rendering/ scene/",
                "Rendering",
                "Three.js on WebGL2 or WebGPU; HDR pipeline;\nsix blending modes; LOD selection by screen area",
            ),
            (
                "workers/ wasm/",
                "Compute",
                "background decode; Rust kernels for nD\nprojection, effective radii, Mahalanobis distance",
            ),
            (
                "data/ cache/",
                "Streaming",
                "Zarr client, spatial queries, prefetch;\nS-cache, L0, L1 and L2 (OPFS) cache tiers",
            ),
            (
                "config/",
                "Configuration",
                "URL parameters, settings sections,\ndensity guard, adaptive resolution",
            ),
            (
                "audio/",
                "Sound",
                "ambient beds, positional sources and narration\ncued by the hidden dimensions",
            ),
        ],
        row_h=54,
        label_w=132,
    )
    text(
        d,
        xr + 18,
        top + ch - 40,
        "Also shipped: luxar export writes viewer + data as an offline folder, and",
        9.8,
        t.muted,
    )
    text(
        d,
        xr + 18,
        top + ch - 26,
        "luxar-launcher (Go) wraps that folder as a native macOS or Linux app.",
        9.8,
        t.muted,
    )

    ym = top + 40
    arrow(d, t, xl + wl + 4, ym, xm - 4, ym, 1.8)
    arrow(d, t, xm + wm + 4, ym, xr - 4, ym, 1.8)
    text(
        d,
        w_all / 2,
        h_all - 36,
        "Two code bases, one file format between them.",
        13,
        t.text,
        "bold",
        anchor="middle",
    )
    text(
        d,
        w_all / 2,
        h_all - 16,
        "The Python side does the expensive work once; the viewer only ever reads chunks, so it runs anywhere a browser does.",
        12,
        t.muted,
        anchor="middle",
        italic=True,
    )
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--which", choices=["pipeline", "layers", "both"], default="both"
    )
    parser.add_argument("--theme", choices=["dark", "light", "both"], default="both")
    parser.add_argument(
        "--scale", type=float, default=2.0, help="PNG pixel density (2 = 2480 px wide)"
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("architecture"))
    args = parser.parse_args(argv)
    if shutil.which("rsvg-convert") is None:
        raise SystemExit("rsvg-convert is not on PATH (brew install librsvg)")
    args.out.mkdir(parents=True, exist_ok=True)
    themes = (
        [DARK, LIGHT]
        if args.theme == "both"
        else [DARK if args.theme == "dark" else LIGHT]
    )
    figures = {
        "pipeline": (build, W, H, "architecture"),
        "layers": (build_layers, LAYERS_W, LAYERS_H, "layers"),
    }
    for which in list(figures) if args.which == "both" else [args.which]:
        builder, width, height, stem = figures[which]
        for t in themes:
            svg = args.out / f"{stem}-{t.name}.svg"
            png = args.out / f"{stem}-{t.name}.png"
            builder(t).save_svg(str(svg))
            subprocess.run(  # nosec B603, B607: fixed argv, tool from PATH
                [
                    "rsvg-convert",
                    "-w",
                    str(int(width * args.scale)),
                    "-h",
                    str(int(height * args.scale)),
                    "-o",
                    str(png),
                    str(svg),
                ],
                check=True,
            )
            print(f"{png} ({png.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
