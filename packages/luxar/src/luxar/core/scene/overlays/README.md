# luxar.core.scene.overlays

2D screen-space overlay implementations for `Scene` — text, image, video, and HTML
overlays anchored to the viewport, plus auto-injection of hover overlays for
nodes that carry labels.

## Overview

Overlays are screen-space (not world-space) annotations rendered on top of the
3D scene by the viewer. They are positioned in normalized viewport coordinates
`(x, y)` in `[0, 1]` with a configurable `anchor`. This package holds the pure
function implementations that back `Scene.add_text()`, `Scene.add_image()`, and
`Scene.add_html()` — the `Scene` methods are thin delegates (see
[`scene.py`](../scene.py)) that forward to the `*_impl` functions here.

Each overlay is validated, given a unique name, persisted to the zarr store
under `overlays/{name}`, and recorded as an `Overlay` instance on
`scene._overlays`. The package also auto-injects a default hover overlay during
compiler finalization when nodes have labels but no hover overlay was defined.

## File Structure

```
overlays/
├── __init__.py        # empty package marker
├── adders.py          # add_text_impl / add_image_impl / add_video_impl / add_html_impl
├── internals.py       # next_overlay_name / write_overlay (naming + persistence)
└── hover_inject.py    # auto_inject_hover_overlay (label-driven default hover)
```

## Components

### `adders.py` — overlay adder implementations

Pure functions, each taking `scene: Scene` as the first argument. They validate
inputs, assemble the overlay's `attrs` dict, persist via `write_overlay()`, and
return the resulting `Overlay`.

| Function | Overlay type | Notable parameters |
|----------|--------------|--------------------|
| `add_text_impl` | `overlay_text` | `font_size`, `font`, `color`, `anchor`, `text_align`, `line_height`, `background`, `stroke_color`/`stroke_width`, `visible_range`, `transition`, `hover` |
| `add_image_impl` | `overlay_image` | `size` (height may be `None` = keep aspect), `anchor`, `blend_mode`, `format`, `visible_range`, `transition` |
| `add_video_impl` | `overlay_video` | `size` (height may be `None` = keep aspect), `loop`, `autoplay`, `muted`, `playback_rate`, `poster`, `anchor`, `blend_mode`, `visible_range`, `transition` |
| `add_html_impl` | `overlay_html` | `width`, `anchor`, `blend_mode`, `visible_range`, `transition`, `hover`, `hover_image_size` |

All four share a common validation pass against `luxar.validation.overlays`
(`validate_position`, `validate_anchor`, `validate_transition`,
`validate_blend_mode`, `validate_visible_range`, plus type-specific checks such
as `validate_font` / `validate_text_align` for text, `validate_image_input` for
images, and `sanitize_html` for HTML). Every overlay records a `z_index` equal
to its insertion order (`len(scene._overlays)`).

`add_image_impl` additionally normalizes the input to encoded bytes and a
matching storage format via `validate_image_input`, stores the filename as
`image_file`, and passes the raw bytes through to `write_overlay` for on-disk
persistence.

`add_video_impl` stores a WebM or MP4 **verbatim** (`validate_video_input`
sniffs the container — EBML header or `ftyp` box — and refuses anything else, so
an unplayable clip fails at authoring, not on the display) as `video_file`, plus
an optional `poster_file` still through the image path. It refuses
`autoplay=True` with `muted=False`, because browsers block un-muted autoplay
without a user gesture. The viewer renders a muted looping `<video>` and pauses
it while its `visible_range` does not match, so many clips cost one decode at a
time. With `autoplay=False`, it shows native controls and captures pointer
events regardless of `interactive`, so the controls remain usable but camera
drags do not pass through the clip. Transparency is authored as a stacked alpha
matte (`alpha_matte="stacked"`: colour on top, the alpha as a grey matte below,
one opaque frame twice as tall; `validate_video_alpha_matte` refuses any other
layout) that the viewer recombines in a shader — a VP9 alpha plane plays
transparent only in Chrome/Firefox, Safari and WKWebView drop it. Offline
capture does not synchronize the video's wall-clock playback to its synthetic
frame clock, so recorded playback speed is not preserved. Both payloads go
through `write_overlay`'s `files=` mapping.

### `internals.py` — naming and persistence

- `next_overlay_name(scene, name)` — generates `overlay_{counter}` when `name`
  is `None`, rejects names containing `/`, and raises on duplicate names.
- `write_overlay(scene, name, overlay_type, position, attrs, image_data=None,
  image_filename=None, files=None)` — writes an `overlays/{name}` group (with `attrs` as
  `.zattrs`) through `scene._writer`, optionally writes a raw image file (plus any extra `files` payloads such as a video and its poster) into
  the overlay's zarr directory, constructs an `Overlay`, appends it to
  `scene._overlays`, and returns it.

### `hover_inject.py` — label-driven hover overlay

`auto_inject_hover_overlay(scene)` is called by the compiler during
`finalize()`. It injects a default hover overlay only when:

1. at least one node has `labels` or `image_labels`
   (`scene._has_labels` / `scene._has_image_labels`), and
2. no existing overlay already has `hover=True`, and
3. `scene._suppress_hover_overlay` is `False`.

When image labels are present it injects an `overlay_html` overlay named
`__hover_image` whose `html` is the `{hover_image_label}` placeholder. When text
labels are present it injects an `overlay_text` overlay named `__hover_text`
whose `text` is the `{hover_label}` placeholder. Both anchor top-right and use a
short fade transition; image and text are kept as separate overlays so image
loading does not cause layout shift in a combined overlay.

## Usage

These functions are not called directly — use the `Scene` methods, which
delegate here:

```python
# Text overlay anchored to the top-left of the viewport
scene.add_text("Scale: 10um", position=(0.05, 0.95), font_size=0.02)

# Image overlay (logo) in the top-right corner
scene.add_image("logo.png", position=(0.9, 0.05), anchor="top-right")

# HTML overlay with a fixed width
scene.add_html("<b>Sample 3</b>", position=(0.5, 0.02), width=0.3)
```

## Dependencies

**Internal:**
- `luxar.core.overlay` — the `Overlay` dataclass returned by every adder
- `luxar.core.scene.scene` — `Scene` (type-only import; provides `_writer`,
  `_overlays`, `_overlay_counter`, label flags)
- `luxar.validation.overlays` — input validation and HTML sanitization

**External:**
- `arbol` — structured `aprint` logging

## See Also

- [core/README.md](../../README.md) — scene graph and `Scene`
- [validation/README.md](../../../validation/README.md) — overlay validation utilities
