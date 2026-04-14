#!/usr/bin/env python3
"""Generate Figure 1: Luxar system overview (SVG version).

Improved figure using drawsvg for clean vector graphics with embedded raster elements.

Layout:
  Top row:    a) Styled Python code card  |  b) Pipeline diagram with thumbnails
  Bottom row: c) 4 viewer screenshots with metadata captions

Design principles (from Nature Methods tool paper survey):
  - Clean left-to-right pipeline flow (3DGS, Nerfstudio, SpatialData pattern)
  - Real data screenshots proving breadth (BigDataViewer, ilastik pattern)
  - Consistent color palette across all panels
  - Visual narrative flow connecting code -> pipeline -> results
"""

import io
from pathlib import Path

import drawsvg as draw
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.parent
FIGS_DIR = SCRIPT_DIR / "figs" / "overview"
RESULTS_DIR = SCRIPT_DIR.parent / "analysis" / "splat_count_vs_quality" / "results"

# ---------------------------------------------------------------------------
# Figure dimensions (Nature Methods two-column: 183mm = 7.2in)
# Using SVG user units; rsvg-convert scales to physical size via DPI flags
# ---------------------------------------------------------------------------
FIG_W = 2160
FIG_H = 1200

# Margins
MARGIN_L = 40
MARGIN_R = 40
MARGIN_T = 40
MARGIN_B = 25
CONTENT_W = FIG_W - MARGIN_L - MARGIN_R
CONTENT_H = FIG_H - MARGIN_T - MARGIN_B

# Row heights
TOP_ROW_H = 480
GAP_ROWS = 30
BOTTOM_ROW_H = CONTENT_H - TOP_ROW_H - GAP_ROWS

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
class Colors:
    BG = "#ffffff"

    # Code panel (dark editor)
    CODE_BG = "#1e1e2e"
    CODE_BORDER = "#313244"
    CODE_KEYWORD = "#cba6f7"
    CODE_STRING = "#a6e3a1"
    CODE_COMMENT = "#6c7086"
    CODE_FUNC = "#89b4fa"
    CODE_TEXT = "#cdd6f4"
    CODE_PUNCT = "#9399b2"

    # Pipeline stages
    STAGE_INPUT = "#2563eb"
    STAGE_INPUT_BG = "#eff6ff"
    STAGE_PROCESS = "#059669"
    STAGE_PROCESS_BG = "#ecfdf5"
    STAGE_STORE = "#d97706"
    STAGE_STORE_BG = "#fffbeb"
    STAGE_OUTPUT = "#dc2626"
    STAGE_OUTPUT_BG = "#fef2f2"

    # Arrows and connectors
    ARROW = "#94a3b8"

    # Labels
    LABEL_BOLD = "#1e293b"
    LABEL_SUB = "#94a3b8"
    PANEL_LABEL = "#1e293b"

    # Cards
    CARD_BORDER = "#e2e8f0"

    # Geometry type badges
    BADGE_GSPLATS = "#7c3aed"
    BADGE_POINTS = "#2563eb"
    BADGE_LINES = "#059669"


# ---------------------------------------------------------------------------
# Font settings
# ---------------------------------------------------------------------------
FONT_MONO = "Menlo"
FONT_SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def image_to_png_bytes(img_path: str | Path, max_width: int = 600,
                       target_aspect: float | None = None,
                       auto_crop_white: bool = False) -> tuple[bytes, int, int]:
    """Convert an image file to PNG bytes for SVG embedding.

    Args:
        target_aspect: If set, center-crop to this width/height ratio (cover crop).
        auto_crop_white: If True, trim white/near-white borders first.
    """
    img = Image.open(str(img_path)).convert("RGB")

    # Auto-crop: trim near-white borders
    if auto_crop_white:
        import numpy as _np
        arr = _np.array(img)
        # Find non-white pixels (threshold: any channel < 240)
        mask = _np.any(arr < 240, axis=2)
        rows = _np.any(mask, axis=1)
        cols = _np.any(mask, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = _np.where(rows)[0][[0, -1]]
            cmin, cmax = _np.where(cols)[0][[0, -1]]
            # Add small padding (2% of dimension)
            pad_r = max(int((rmax - rmin) * 0.02), 4)
            pad_c = max(int((cmax - cmin) * 0.02), 4)
            rmin = max(0, rmin - pad_r)
            rmax = min(arr.shape[0] - 1, rmax + pad_r)
            cmin = max(0, cmin - pad_c)
            cmax = min(arr.shape[1] - 1, cmax + pad_c)
            img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Center-crop to target aspect ratio, centered on content center-of-mass
    if target_aspect is not None:
        import numpy as _np
        w, h = img.size
        current_aspect = w / h

        # Find content center of mass for smarter centering
        arr = _np.array(img.convert("L"))  # grayscale
        content_mask = arr < 240  # non-white pixels
        if content_mask.any():
            ys, xs = _np.where(content_mask)
            com_x = int(xs.mean())
            com_y = int(ys.mean())
        else:
            com_x, com_y = w // 2, h // 2

        if current_aspect > target_aspect:
            # Too wide: crop width, centered on content center-of-mass
            new_w = int(h * target_aspect)
            left = max(0, min(com_x - new_w // 2, w - new_w))
            img = img.crop((left, 0, left + new_w, h))
        elif current_aspect < target_aspect:
            # Too tall: crop height, centered on content center-of-mass
            new_h = int(w / target_aspect)
            top = max(0, min(com_y - new_h // 2, h - new_h))
            img = img.crop((0, top, w, top + new_h))

    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), img.width, img.height


def numpy_to_png_bytes(arr: np.ndarray, max_width: int = 300,
                       boost_contrast: bool = True) -> tuple[bytes, int, int]:
    """Convert a numpy array (0-1 float) to PNG bytes with optional contrast boost."""
    if boost_contrast and arr.max() > 0:
        # Auto-contrast: stretch to use full range, then apply gentle gamma
        vmin, vmax = np.percentile(arr[arr > 0], [1, 99.5]) if (arr > 0).any() else (0, 1)
        if vmax > vmin:
            arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        arr = np.power(arr, 0.6)  # gamma < 1 brightens dark regions
    arr_uint8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.width, img.height


def _svg_escape(text: str) -> str:
    """Escape special XML characters for SVG text content."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def rounded_rect(d, x, y, w, h, rx=8, fill="#fff", stroke=None, stroke_width=1,
                 opacity=1.0, shadow=False, shadow_color="#00000018",
                 shadow_dx=0, shadow_dy=4, shadow_blur=10):
    """Draw a rounded rectangle, optionally with a drop shadow."""
    if shadow:
        d.append(draw.Rectangle(
            x + shadow_dx, y + shadow_dy, w, h,
            rx=rx, ry=rx, fill=shadow_color, stroke="none", opacity=0.5,
        ))
    rect_kwargs = dict(rx=rx, ry=rx, fill=fill, opacity=opacity)
    if stroke:
        rect_kwargs["stroke"] = stroke
        rect_kwargs["stroke_width"] = stroke_width
    d.append(draw.Rectangle(x, y, w, h, **rect_kwargs))


def add_panel_label(d, x, y, label):
    """Add a bold panel label (a, b, c)."""
    d.append(draw.Text(
        label, 26, x, y,
        font_weight="bold", font_family=FONT_SANS, fill=Colors.PANEL_LABEL,
    ))


def draw_arrow(d, x1, y1, x2, y2, color=Colors.ARROW, width=2.5, head_size=10):
    """Draw a straight arrow with a triangular head."""
    d.append(draw.Line(
        x1, y1, x2 - head_size, y2,
        stroke=color, stroke_width=width, stroke_linecap="round",
    ))
    d.append(draw.Lines(
        x2, y2,
        x2 - head_size * 1.2, y2 - head_size * 0.55,
        x2 - head_size * 1.2, y2 + head_size * 0.55,
        close=True, fill=color, stroke="none",
    ))


# ---------------------------------------------------------------------------
# Panel (a): Code snippet
# ---------------------------------------------------------------------------
def draw_code_panel(d, x, y, w, h):
    """Panel a: Python code in a dark editor-style card."""
    add_panel_label(d, x, y - 6, "a")

    pad = 8
    card_x, card_y = x + pad, y + 10
    card_w, card_h = w - 2 * pad, h - 14

    # Card
    rounded_rect(d, card_x, card_y, card_w, card_h, rx=12,
                 fill=Colors.CODE_BG, stroke=Colors.CODE_BORDER,
                 stroke_width=1.5, shadow=True, shadow_dy=5, shadow_blur=12)

    # macOS dots
    dot_y = card_y + 20
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        d.append(draw.Circle(card_x + 22 + i * 20, dot_y, 6, fill=c))

    # Filename
    d.append(draw.Text(
        "luxar_example.py", 12,
        card_x + 90, dot_y + 4,
        font_family=FONT_MONO, fill=Colors.CODE_COMMENT,
    ))

    # Separator line under title bar
    sep_y = card_y + 38
    d.append(draw.Line(
        card_x + 10, sep_y, card_x + card_w - 10, sep_y,
        stroke=Colors.CODE_BORDER, stroke_width=0.8,
    ))

    # Code lines as (full_line_text, [(start, end, color), ...]) for precise rendering.
    # Each line is a complete string; color_spans mark which characters get which color.
    K = Colors.CODE_KEYWORD
    F = Colors.CODE_FUNC
    S = Colors.CODE_STRING
    C = Colors.CODE_COMMENT
    T = Colors.CODE_TEXT
    P = Colors.CODE_PUNCT

    # Full lines of code with color spans: (line_string, [(start, end, color), ...])
    code_lines = [
        ("from luxar import LuxarZarrCompiler", [
            (0, 4, K), (5, 10, T), (11, 17, K), (18, 36, F)]),
        ("from luxar.gsplats import fit_gaussian_splats", [
            (0, 4, K), (5, 19, T), (20, 26, K), (27, 46, F)]),
        ("", []),
        ("# Fit Gaussian splats to a volume", [
            (0, 34, C)]),
        ("splats = fit_gaussian_splats(", [
            (0, 6, T), (7, 8, P), (9, 28, F), (28, 29, P)]),
        ("    volume, seeds=32000, device=\"cuda\")", [
            (0, 10, T), (10, 11, P), (12, 17, T), (17, 18, P),
            (18, 23, S), (23, 24, P), (25, 31, T), (31, 32, P),
            (32, 38, S), (38, 39, P)]),
        ("", []),
        ("with LuxarZarrCompiler(\"out.zarr\") as c:", [
            (0, 4, K), (5, 23, F), (23, 24, P), (24, 33, S),
            (33, 34, P), (35, 37, K), (38, 40, T)]),
        ("    s = c.create_scene()", [
            (0, 5, T), (6, 7, P), (8, 10, T), (10, 22, F), (22, 24, P)]),
        ("    s.add_gsplats(\"vol\", splats)", [
            (0, 6, T), (6, 16, F), (16, 17, P), (17, 22, S),
            (22, 24, P), (24, 30, T), (30, 31, P)]),
        ("    s.add_points(\"cells\", pos, colors)", [
            (0, 6, T), (6, 16, F), (16, 17, P), (17, 24, S),
            (24, 26, P), (26, 29, T), (29, 31, P), (31, 37, T), (37, 38, P)]),
        ("    s.add_lines(\"tracks\", verts, widths)", [
            (0, 6, T), (6, 15, F), (15, 16, P), (16, 24, S),
            (24, 26, P), (26, 31, T), (31, 33, P), (33, 39, T), (39, 40, P)]),
        ("", []),
        ("# $ luxar serve out.zarr --viewer", [
            (0, 34, C)]),
    ]

    fs = 14.5
    line_h = 26
    code_x = card_x + 24
    code_y = card_y + 56

    for line_idx, (line_text, spans) in enumerate(code_lines):
        if not line_text:
            continue
        ty = code_y + line_idx * line_h

        # Use SVG <text> with <tspan> children for correct monospace positioning.
        # drawsvg's Text with tspan support: build raw SVG text element.
        text_el = draw.Raw(f'<text x="{code_x}" y="{ty}" '
                           f'font-family="{FONT_MONO}" font-size="{fs}" '
                           f'fill="{T}" xml:space="preserve">')

        if not spans:
            # Single color for entire line
            text_el = draw.Raw(
                f'<text x="{code_x}" y="{ty}" '
                f'font-family="{FONT_MONO}" font-size="{fs}" '
                f'fill="{T}" xml:space="preserve">{_svg_escape(line_text)}</text>'
            )
            d.append(text_el)
            continue

        # Build tspan elements for each color span
        parts = []
        last_end = 0
        for start, end, color in sorted(spans, key=lambda s: s[0]):
            # Any gap between last_end and start gets default color
            if start > last_end:
                chunk = line_text[last_end:start]
                parts.append(f'<tspan fill="{T}">{_svg_escape(chunk)}</tspan>')
            chunk = line_text[start:end]
            parts.append(f'<tspan fill="{color}">{_svg_escape(chunk)}</tspan>')
            last_end = end
        # Remaining text after last span
        if last_end < len(line_text):
            chunk = line_text[last_end:]
            parts.append(f'<tspan fill="{T}">{_svg_escape(chunk)}</tspan>')

        svg_text = (f'<text x="{code_x}" y="{ty}" '
                    f'font-family="{FONT_MONO}" font-size="{fs}" '
                    f'xml:space="preserve">{"".join(parts)}</text>')
        d.append(draw.Raw(svg_text))


# ---------------------------------------------------------------------------
# Panel (b): Pipeline diagram
# ---------------------------------------------------------------------------
def draw_pipeline_panel(d, x, y, w, h):
    """Panel b: Clean pipeline with large rounded-rect nodes and thumbnails."""
    add_panel_label(d, x, y - 6, "b")

    nodes = [
        {"label": "Volume",       "sublabel": "3D / 4D / nD",      "color": Colors.STAGE_INPUT,   "bg": Colors.STAGE_INPUT_BG,   "icon": "volume"},
        {"label": "Fit splats",   "sublabel": "Per-splat Adam",     "color": Colors.STAGE_PROCESS, "bg": Colors.STAGE_PROCESS_BG, "icon": "splats"},
        {"label": "Zarr archive", "sublabel": "Chunked & indexed",  "color": Colors.STAGE_STORE,   "bg": Colors.STAGE_STORE_BG,   "icon": "zarr"},
        {"label": "Web viewer",   "sublabel": "HDR, nD navigation", "color": Colors.STAGE_OUTPUT,  "bg": Colors.STAGE_OUTPUT_BG,  "icon": "viewer"},
    ]

    # Load thumbnails
    thumb_volume = thumb_splats = None
    for dataset in ["organoid_ch0", "kidney_dapi"]:
        p = RESULTS_DIR / dataset / "slices" / "n032000_slices.npz"
        if p.exists():
            data = np.load(str(p))
            thumb_volume = data["target_slices"][1]
            thumb_splats = data["recon_slices"][1]
            break

    # Layout sizing
    n = len(nodes)
    node_w = 230
    node_h = 260
    arrow_gap = 55  # gap between nodes for arrows
    total_nodes_w = n * node_w + (n - 1) * arrow_gap
    start_x = x + (w - total_nodes_w) / 2
    node_top_y = y + (h - node_h) / 2 + 5

    for i, node in enumerate(nodes):
        nx = start_x + i * (node_w + arrow_gap)
        ny = node_top_y

        # Node card
        rounded_rect(d, nx, ny, node_w, node_h, rx=12,
                     fill=node["bg"], stroke=node["color"],
                     stroke_width=1.8, shadow=True, shadow_dy=4, shadow_blur=8)

        # Thumbnail region
        tm = 14  # margin inside card
        th = 140  # thumbnail height
        tx, ty, tw = nx + tm, ny + tm, node_w - 2 * tm

        if node["icon"] == "volume" and thumb_volume is not None:
            png_bytes, _, _ = numpy_to_png_bytes(thumb_volume, max_width=400)
            d.append(draw.Rectangle(tx, ty, tw, th, rx=8, ry=8, fill="#000"))
            d.append(draw.Image(tx, ty, tw, th,
                                data=png_bytes, mime_type="image/png"))
            # Axis annotation: small x,y,z arrows in bottom-left corner
            ax_x, ax_y = tx + 10, ty + th - 8
            ax_len = 18
            ax_col = "#ffffff"
            ax_op = 0.7
            # x-axis (right)
            d.append(draw.Line(ax_x, ax_y, ax_x + ax_len, ax_y,
                               stroke=ax_col, stroke_width=1.2, opacity=ax_op))
            d.append(draw.Text("x", 9, ax_x + ax_len + 2, ax_y + 3,
                               font_family=FONT_SANS, fill=ax_col,
                               font_style="italic", opacity=ax_op))
            # y-axis (up)
            d.append(draw.Line(ax_x, ax_y, ax_x, ax_y - ax_len,
                               stroke=ax_col, stroke_width=1.2, opacity=ax_op))
            d.append(draw.Text("y", 9, ax_x - 2, ax_y - ax_len - 2,
                               font_family=FONT_SANS, fill=ax_col,
                               font_style="italic", opacity=ax_op, text_anchor="middle"))
            # z-axis (diagonal, hinting 3D)
            d.append(draw.Line(ax_x, ax_y, ax_x - 10, ax_y - 12,
                               stroke=ax_col, stroke_width=1.0, opacity=ax_op * 0.6,
                               stroke_dasharray="2,2"))
            d.append(draw.Text("z", 9, ax_x - 14, ax_y - 14,
                               font_family=FONT_SANS, fill=ax_col,
                               font_style="italic", opacity=ax_op * 0.6, text_anchor="middle"))

        elif node["icon"] == "splats" and thumb_splats is not None:
            png_bytes, _, _ = numpy_to_png_bytes(thumb_splats, max_width=400)
            d.append(draw.Rectangle(tx, ty, tw, th, rx=8, ry=8, fill="#000"))
            d.append(draw.Image(tx, ty, tw, th,
                                data=png_bytes, mime_type="image/png"))

        elif node["icon"] == "zarr":
            # Zarr chunked-grid icon: 4x4 with varying opacity for depth feel
            cx, cy = tx + tw / 2, ty + th / 2 - 4
            grid_s = 22
            gap = 4
            n_cells = 4
            total = n_cells * grid_s + (n_cells - 1) * gap
            base_x = cx - total / 2
            base_y = cy - total / 2
            opacities = [
                [0.75, 0.55, 0.40, 0.25],
                [0.55, 0.75, 0.55, 0.40],
                [0.40, 0.55, 0.75, 0.55],
                [0.25, 0.40, 0.55, 0.75],
            ]
            for row in range(n_cells):
                for col in range(n_cells):
                    gx = base_x + col * (grid_s + gap)
                    gy = base_y + row * (grid_s + gap)
                    d.append(draw.Rectangle(
                        gx, gy, grid_s, grid_s, rx=3, ry=3,
                        fill=Colors.STAGE_STORE,
                        opacity=opacities[row][col],
                        stroke="#fff", stroke_width=1.2,
                    ))
            d.append(draw.Text(
                ".zarr", 16, cx, cy + total / 2 + 22,
                font_family=FONT_MONO, font_weight="bold",
                fill=Colors.STAGE_STORE, text_anchor="middle",
            ))

        elif node["icon"] == "viewer":
            # Browser window icon
            bm = 20
            bx, by = tx + bm, ty + 8
            bw, bh = tw - 2 * bm, th - 16

            # Chrome
            d.append(draw.Rectangle(bx, by, bw, bh, rx=6, ry=6,
                                     fill="#fff", stroke=Colors.STAGE_OUTPUT,
                                     stroke_width=1.5))
            # Title bar
            bar_h = 20
            d.append(draw.Rectangle(bx, by, bw, bar_h, rx=6, ry=0,
                                     fill=Colors.STAGE_OUTPUT_BG,
                                     stroke=Colors.STAGE_OUTPUT,
                                     stroke_width=1))
            for j, col in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
                d.append(draw.Circle(bx + 12 + j * 14, by + 10, 3.5, fill=col))

            # Content: 3D wireframe cube
            cbx = bx + bw / 2
            cby = by + bar_h + (bh - bar_h) / 2
            s = 24
            off = 8
            # Back face
            d.append(draw.Lines(
                cbx - s + off, cby - s + off,
                cbx + s + off, cby - s + off,
                cbx + s + off, cby + s + off,
                cbx - s + off, cby + s + off,
                close=True, fill="none",
                stroke=Colors.STAGE_OUTPUT, stroke_width=0.8, opacity=0.25))
            # Front face
            d.append(draw.Lines(
                cbx - s, cby - s,
                cbx + s, cby - s,
                cbx + s, cby + s,
                cbx - s, cby + s,
                close=True, fill=Colors.STAGE_OUTPUT_BG,
                stroke=Colors.STAGE_OUTPUT, stroke_width=1.0, opacity=0.5))
            # Edges
            for fx, fy in [(cbx - s, cby - s), (cbx + s, cby - s),
                           (cbx + s, cby + s), (cbx - s, cby + s)]:
                d.append(draw.Line(fx, fy, fx + off, fy + off,
                                    stroke=Colors.STAGE_OUTPUT,
                                    stroke_width=0.8, opacity=0.25))
            d.append(draw.Text(
                "WebGL", 12, cbx, cby + s + 16,
                font_family=FONT_SANS, fill=Colors.STAGE_OUTPUT,
                text_anchor="middle", opacity=0.6,
            ))

        # Main label
        d.append(draw.Text(
            node["label"], 17,
            nx + node_w / 2, ny + th + 42,
            font_family=FONT_SANS, font_weight="bold",
            fill=node["color"], text_anchor="middle",
        ))

        # Sub-label
        d.append(draw.Text(
            node["sublabel"], 13,
            nx + node_w / 2, ny + th + 62,
            font_family=FONT_SANS, fill=Colors.LABEL_SUB,
            text_anchor="middle", font_style="italic",
        ))

    # Arrows between nodes
    for i in range(n - 1):
        ax1 = start_x + i * (node_w + arrow_gap) + node_w + 4
        ax2 = start_x + (i + 1) * (node_w + arrow_gap) - 4
        ay = node_top_y + node_h / 2
        draw_arrow(d, ax1, ay, ax2, ay,
                   color=Colors.ARROW, width=2.8, head_size=12)

    # Iteration loop arrow around "Fit splats" node (index 1)
    fit_nx = start_x + 1 * (node_w + arrow_gap)
    fit_cx = fit_nx + node_w / 2
    fit_bottom = node_top_y + node_h
    loop_drop = 32  # how far below the node the arc extends
    loop_half_w = node_w * 0.38  # half-width of the loop

    # Curved path: from right side, arc below node, back up to left side
    sx = fit_cx + loop_half_w
    sy = fit_bottom - 2
    ex = fit_cx - loop_half_w
    ey = fit_bottom - 2
    ctrl_y = sy + loop_drop + 14  # control point y (below the node)

    loop_path = draw.Path(
        stroke=Colors.STAGE_PROCESS, stroke_width=2.4,
        fill="none", stroke_linecap="round",
    )
    loop_path.M(sx, sy)
    loop_path.C(sx + 8, ctrl_y,   # cp1: right side drops down and out
                ex - 8, ctrl_y,   # cp2: left side comes up and in
                ex, ey)           # end: left edge of node bottom
    d.append(loop_path)

    # Arrowhead at the end (pointing up-right)
    ah = 9
    d.append(draw.Lines(
        ex, ey,
        ex - ah * 0.5, ey + ah * 1.1,
        ex + ah * 0.7, ey + ah * 0.6,
        close=True, fill=Colors.STAGE_PROCESS, stroke="none",
    ))

    # "iterative" label centered on the arc
    d.append(draw.Text(
        "iterative", 12,
        fit_cx, sy + loop_drop + 10,
        font_family=FONT_SANS, fill=Colors.STAGE_PROCESS,
        text_anchor="middle", font_style="italic", opacity=0.8,
    ))


# ---------------------------------------------------------------------------
# Panel (c): Screenshot gallery
# ---------------------------------------------------------------------------
def draw_screenshot_gallery(d, x, y, w, h):
    """Panel c: 4 viewer screenshots with metadata."""
    add_panel_label(d, x, y - 2, "c")

    screenshots = [
        {"file": "viewer_gsplats_organoid_paper.png",
         "geom_type": "GSplats", "title": "Confocal organoid",
         "meta": "32K Gaussians", "badge_color": Colors.BADGE_GSPLATS},
        {"file": "viewer_storm_microtubules_paper.png",
         "geom_type": "Points", "title": "STORM super-resolution",
         "meta": "3.5M localizations", "badge_color": Colors.BADGE_POINTS},
        {"file": "viewer_spiral_galaxy_paper.png",
         "geom_type": "Points", "title": "Simulated galaxy",
         "meta": "500K stars", "badge_color": Colors.BADGE_POINTS},
        {"file": "viewer_lsystem_forest_paper.png",
         "geom_type": "Lines", "title": "L-system forest",
         "meta": "Procedural trees", "badge_color": Colors.BADGE_LINES},
    ]

    n = len(screenshots)
    card_gap = 22
    total_gap = card_gap * (n - 1)
    card_w = (w - total_gap) / n
    caption_space = 60
    card_img_h = h - caption_space - 16
    card_total_h = h - 8

    for i, sc in enumerate(screenshots):
        cx = x + i * (card_w + card_gap)
        cy = y + 12

        # Card
        rounded_rect(d, cx, cy, card_w, card_total_h, rx=10,
                     fill="#ffffff", stroke=Colors.CARD_BORDER,
                     stroke_width=1.0, shadow=True, shadow_dy=4, shadow_blur=10)

        # Screenshot image
        img_path = FIGS_DIR / sc["file"]
        im = 8  # image margin
        img_x, img_y = cx + im, cy + im
        img_w = card_w - 2 * im
        img_h = card_img_h - 2 * im
        target_ar = img_w / img_h  # target aspect ratio for this card

        if img_path.exists():
            # Auto-crop white borders, then center-crop to card aspect ratio
            png_bytes, iw, ih = image_to_png_bytes(
                img_path, max_width=600,
                target_aspect=target_ar, auto_crop_white=True,
            )
            clip_id = f"clip-sc-{i}"
            clip = draw.ClipPath(id=clip_id)
            clip.append(draw.Rectangle(img_x, img_y, img_w, img_h, rx=6, ry=6))
            d.append(clip)
            d.append(draw.Image(
                img_x, img_y, img_w, img_h,
                data=png_bytes, mime_type="image/png",
                clip_path=f"url(#{clip_id})",
            ))

        # Badge (top-right corner of image)
        badge_text = sc["geom_type"]
        badge_w = len(badge_text) * 8.5 + 16
        badge_h = 24
        badge_x = cx + card_w - im - badge_w - 6
        badge_y = cy + im + 8

        d.append(draw.Rectangle(
            badge_x, badge_y, badge_w, badge_h, rx=5, ry=5,
            fill=sc["badge_color"], opacity=0.90,
        ))
        d.append(draw.Text(
            badge_text, 12.5,
            badge_x + badge_w / 2, badge_y + 17,
            font_family=FONT_SANS, font_weight="bold",
            fill="#ffffff", text_anchor="middle",
        ))

        # Title caption
        cap_y = cy + card_img_h + 6
        d.append(draw.Text(
            sc["title"], 15,
            cx + card_w / 2, cap_y,
            font_family=FONT_SANS, font_weight="bold",
            fill=Colors.LABEL_BOLD, text_anchor="middle",
        ))

        # Metadata caption
        d.append(draw.Text(
            sc["meta"], 13,
            cx + card_w / 2, cap_y + 22,
            font_family=FONT_SANS, fill=Colors.LABEL_SUB,
            text_anchor="middle", font_style="italic",
        ))


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------
def main():
    d = draw.Drawing(FIG_W, FIG_H, origin=(0, 0))
    d.append(draw.Rectangle(0, 0, FIG_W, FIG_H, fill=Colors.BG))

    top_y = MARGIN_T
    code_w = CONTENT_W * 0.36
    pipe_w = CONTENT_W * 0.64

    # Panel (a)
    draw_code_panel(d, MARGIN_L, top_y, code_w, TOP_ROW_H)

    # Panel (b)
    draw_pipeline_panel(d, MARGIN_L + code_w + 20, top_y, pipe_w - 20, TOP_ROW_H)

    # Panel (c)
    gallery_y = top_y + TOP_ROW_H + GAP_ROWS
    draw_screenshot_gallery(d, MARGIN_L, gallery_y, CONTENT_W, BOTTOM_ROW_H)

    # Save SVG to subfolder
    svg_path = FIGS_DIR / "overview.svg"
    d.save_svg(str(svg_path))
    print(f"Saved SVG: {svg_path}")

    import shutil
    import subprocess

    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        # PDF in subfolder + copy to parent figs/ for LaTeX
        pdf_path = FIGS_DIR / "overview.pdf"
        subprocess.run(
            [rsvg, "-f", "pdf", "-d", "300", "-p", "300",
             str(svg_path), "-o", str(pdf_path)],
            check=True,
        )
        print(f"Saved PDF: {pdf_path}")

        png_path = FIGS_DIR / "overview.png"
        subprocess.run(
            [rsvg, "-f", "png", "-d", "200", "-p", "200",
             str(svg_path), "-o", str(png_path)],
            check=True,
        )
        print(f"Saved PNG: {png_path}")
    else:
        print("rsvg-convert not found. Install with: brew install librsvg")


if __name__ == "__main__":
    main()
