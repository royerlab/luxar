"""Per-element image-label normalization for the compiler.

Private support module for :class:`luxar.io.compiler.LuxarZarrCompiler`. Converts
heterogeneous image-label inputs (bytes, PIL images, numpy arrays, file paths) into
encoded image bytes ready for CSR-style storage.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import zarr
from arbol import aprint

from ....encoding.compression import resolve_compressor

if TYPE_CHECKING:
    from ....encoding.compression import CompressorLike


def check_image_label_type(item: Any) -> None:
    """Type dispatch, plus the ``ndarray`` SHAPE check: is ``item`` a type (and,
    for an ``ndarray``, a shape) an image label accepts?

    Pulled out of :func:`normalize_image_label` (#1491) so the TYPE question can
    be answered without doing any of the work that question's positive
    answer would then require — no PIL import for a plain ``bytes`` blob, no
    file read for a ``str``/``Path``. That is what lets
    :func:`validate_image_labels_for_writing` run this over every entry in a
    writer's fail-fast PRE-write gate: a single mistyped entry (an ``int`` in
    an otherwise right-length list) used to be refused only from
    :func:`normalize_image_label` itself, deep inside the writer — after the
    caller's other arrays, and on a ``substitutive_lod=`` ladder, after every
    level up to the one carrying it, were already on disk. A wrong-length list
    and a right-length list with one wrong-typed entry are equally plausible
    authoring mistakes and strand the same way; only the length/index check
    moved before this fix, which closed one door and left this one open.

    The ``ndarray`` SHAPE check (``(H, W)`` / ``(H, W, 3)`` / ``(H, W, 4)``)
    is included here too, not deferred to :func:`normalize_image_label`'s PIL
    round-trip: ``item.ndim`` / ``item.shape[2]`` are pure attribute reads,
    same cost class as the ``isinstance`` checks above them, and reading them
    never touches PIL or the store. Measured: the previous version left this
    exact strand open — a ``(4, 5, 2)`` ndarray (an unsupported shape) in an
    otherwise right-length/right-typed ``image_labels`` on a
    ``substitutive_lod=`` Points wrapper reached ``normalize_image_label``
    only from inside the finest child's own write, after every coarser gsplat
    level was already on disk, for the same reason an ``int`` entry did before
    this function existed.

    Deliberately mirrors :func:`normalize_image_label`'s own dispatch order
    exactly, including its one surprising quirk: when Pillow is NOT
    installed, ``from PIL import Image`` raises ``ImportError``
    UNCONDITIONALLY at that point in the dispatch — before ``item``'s actual
    type is even inspected — so ANY item that is not ``None`` / ``bytes`` /
    ``bytearray`` / ``Path`` / ``str`` raises the *same* "Pillow is required to
    encode PIL Image objects…" ``ImportError`` when Pillow is absent, even an
    ``int`` or an ``ndarray``. That is existing behaviour this function
    preserves rather than "fixes" — changing it would make this validator
    disagree with :func:`normalize_image_label` about a type Pillow-absent
    input, defeating the one-implementation point of factoring this out.

    Args:
        item: A single ``image_labels`` entry (or ``None``, meaning "no image
            for this element").

    Raises:
        TypeError: ``item`` is not one of the accepted types.
        ValueError: ``item`` is an ``ndarray`` but not one of the accepted
            shapes.
        ImportError: Pillow is not installed and ``item`` is not one of the
            few types (``None`` / ``bytes`` / ``bytearray`` / ``Path`` /
            ``str``) that never need it.
    """
    if item is None:
        return
    if isinstance(item, (bytes, bytearray)):
        return
    if isinstance(item, Path):
        return
    if isinstance(item, str):
        return

    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError(
            "Pillow is required to encode PIL Image objects as image labels. "
            "Install it with: pip install Pillow"
        )

    if isinstance(item, PILImage.Image):
        return
    if isinstance(item, np.ndarray):
        if item.ndim == 2:
            return
        if item.ndim == 3 and item.shape[2] in (3, 4):
            return
        raise ValueError(
            f"Unsupported ndarray shape for image label: {item.shape}. "
            f"Expected (H, W), (H, W, 3), or (H, W, 4)."
        )

    raise TypeError(
        f"Unsupported image label type: {type(item).__name__}. "
        f"Expected bytes, PIL.Image, numpy.ndarray, or file path."
    )


def normalize_image_label(item: Any) -> bytes:
    """Convert a single image label input to encoded bytes.

    Accepts:
    - ``bytes`` / ``bytearray`` — used as-is (pre-encoded JPEG/WebP/PNG)
    - ``PIL.Image.Image`` — encoded to WebP (quality 85)
    - ``numpy.ndarray`` (H, W, C) uint8 — converted to PIL, then WebP
    - ``pathlib.Path`` / ``str`` — file read as raw bytes

    Runs :func:`check_image_label_type` FIRST (#1491), so the type-AND-shape
    dispatch lives in exactly one place; everything below is then free to
    assume ``item`` is one of the accepted types (and, for an ``ndarray``,
    one of the accepted shapes) and focus on the part no pure check can do —
    actually reading a file or calling PIL.

    Returns:
        Encoded image bytes, or ``b""`` for None / empty inputs.
    """
    check_image_label_type(item)

    if item is None:
        return b""
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if isinstance(item, Path):
        return item.read_bytes()
    if isinstance(item, str):
        return Path(item).read_bytes()

    # PIL Image
    from PIL import Image as PILImage

    if isinstance(item, PILImage.Image):
        import io

        buf = io.BytesIO()
        item.save(buf, format="webp", quality=85)
        return buf.getvalue()

    # numpy array (H, W, C) uint8 — check_image_label_type has already
    # confirmed Pillow is importable, the type is right, AND the shape is one
    # of (H, W) / (H, W, 3) / (H, W, 4) (raising ValueError otherwise), so the
    # only two live shapes left here are 2-D (grayscale) and 3-channel — a
    # 4-channel ndarray falls to the `else` and gets RGBA.
    import io

    if item.ndim == 2:
        pil_img = PILImage.fromarray(item, mode="L")
    elif item.shape[2] == 3:
        pil_img = PILImage.fromarray(item, mode="RGB")
    else:
        pil_img = PILImage.fromarray(item, mode="RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="webp", quality=85)
    return buf.getvalue()


def validate_image_labels_for_writing(image_labels: Any, n_elements: int) -> None:
    """Validate an ``image_labels`` input's shape/indices/types BEFORE any zarr write.

    No store and no file reads, but NOT PIL-free any more (#1491 widened it):
    a Pillow-absent item still routes through :func:`check_image_label_type`'s
    own ``ImportError``, same as :func:`normalize_image_label` would raise for
    it later. So it can still run in a writer's fail-fast pre-write gate, and
    hoisted one level further, in a ``substitutive_lod=`` wrapper's PRE-SPLIT
    gate (see ``core.group.compositing.validate_points_channels_before_split``
    / ``validate_lines_channels_before_split``). :func:`write_image_labels_csr`
    used to run only the length/index checks itself, inline, and leave every
    item's TYPE to be discovered one at a time inside
    :func:`normalize_image_label` — AFTER the caller's other arrays
    (positions/colors/radii/…) were already written, and on a ``kind=lod``
    ladder, after every coarser level was on disk too, since the finest child
    (the one carrying ``image_labels``) writes LAST. A wrong-length list and a
    right-length list with one mistyped entry are equally plausible authoring
    mistakes and stranded identically; extracting BOTH kinds of check lets
    every fail-fast gate that calls this run them before anything is written,
    without duplicating either set of rules.

    Checks run in a fixed order — structural checks on the container first
    (a dense sequence's length, or every sparse key's integrality and bounds),
    THEN a type check of EVERY entry — for both the dense sequence form and
    the sparse ``dict`` form, and this reordering changes behaviour for both,
    not just the ``dict`` one. For the sparse form specifically: ALL key
    checks run before ANY item's type is inspected. Pre-#1491,
    ``write_image_labels_csr`` interleaved the two per key (bound-check
    ``idx``, then immediately :func:`normalize_image_label` ``item``), so a
    dict like ``{0: 12345, 5: b"x"}`` against ``n_elements=3`` raised
    ``TypeError: Unsupported image label type: int`` on key 0's value — key
    5's out-of-range index was never reached. The same call now raises
    ``ValueError: Image label index 5 out of range [0, 3)`` instead: a
    different exception TYPE and message, on a call that goes straight to
    :func:`write_image_labels_csr` (bypassing the higher pre-split gates
    entirely).

    The DENSE form diverges the same way, and not only for the length check:
    the type sweep now precedes every file read and shape check too. Measured:
    ``[b"ok", "/nope/missing.png", 123]`` used to raise ``FileNotFoundError``
    (item 1's missing-file read happened, since :func:`normalize_image_label`
    walked the list in order and item 2's bad type was never reached);
    it now raises ``TypeError: Unsupported image label type: int`` instead,
    since every item's type is checked before any file is opened. Arguably an
    improvement, since ``TypeError`` IS caught by the leaf adders' ``except
    (ValueError, TypeError)`` funnel while ``FileNotFoundError`` is not (see
    the residual paragraph below). Both divergences are deliberate and new,
    not a bug — see :mod:`labels/README.md` and the pinning tests in
    ``io/tests/_compiler/test_labels.py``.

    Deliberately narrow in one remaining way: this does not normalize
    dict->list or encode any blob, and it does not READ a ``str``/``Path``
    file — those still need :func:`normalize_image_label` (which needs an
    actual file read and/or a PIL round-trip) and so remain the writer's job,
    post-write. The ``ndarray`` SHAPE check is NOT part of that residual,
    though: :func:`check_image_label_type` (called for every entry, below)
    validates ``ndim``/``shape[2]`` eagerly, since that costs nothing a type
    check does not already cost — only an unreadable path (``FileNotFoundError``)
    and the actual PIL encode remain genuinely post-write-only. See the
    writers' step-0 comments for that (smaller) residual. Note two of the
    exceptions this function can now raise are NOT caught by the leaf adders'
    ``except (ValueError, TypeError)`` funnel — an ``ImportError`` (Pillow
    absent) escapes unwrapped, exactly as it would from a post-write
    :func:`normalize_image_label` call; that gap is pre-existing, not
    introduced by widening this function to catch types earlier. A
    post-write-only ``FileNotFoundError`` (an unreadable ``str``/``Path``) has
    the same gap, but never reaches this function at all.

    Args:
        image_labels: Per-element images, either the dense sequence form (one
            entry per element, checked by length; must be RE-iterable, see
            above) or the sparse ``Dict[int, Any]`` form (keys checked for
            integer-index semantics and bounds; a missing index just means no
            image, so absence is never an error).
        n_elements: Expected element count.

    Raises:
        ValueError: If a dict key is outside ``[0, n_elements)``, a
            sequence's length does not equal ``n_elements``, or an entry is
            an ``ndarray`` with an unsupported shape (from
            :func:`check_image_label_type`).
        TypeError: If a dict key is not usable as an integer index (a
            ``float``, a ``str``, ...), the dense form is a single-pass
            iterable, or an entry's type is not one
            :func:`check_image_label_type` accepts.
        ImportError: If an entry needs Pillow (anything that is not ``None`` /
            ``bytes`` / ``bytearray`` / ``Path`` / ``str``) and Pillow is not
            installed.
    """
    if isinstance(image_labels, dict):
        for idx in image_labels:
            # An index TYPE check before the bounds one: a key is used as
            # ``normalized[idx]`` in write_image_labels_csr, so anything
            # without integer-index semantics (a float, a np.float64 from an
            # arithmetic slip, a str) raises TypeError from that subscript —
            # POST-write, the exact strand this gate exists to close. `int`,
            # `bool` and any numpy integer satisfy operator.index; `float`
            # deliberately does not, even at an integral value.
            try:
                key = operator.index(idx)
            except TypeError:
                raise TypeError(
                    f"Image label index must be an integer, got "
                    f"{type(idx).__name__}: {idx!r}"
                ) from None
            if key < 0 or key >= n_elements:
                raise ValueError(
                    f"Image label index {idx} out of range [0, {n_elements})"
                )
        for item in image_labels.values():
            check_image_label_type(item)
    else:
        if len(image_labels) != n_elements:
            raise ValueError(
                f"Image labels length ({len(image_labels)}) must match "
                f"element count ({n_elements})"
            )
        # The per-entry sweep below WALKS the sequence, and every caller of
        # this gate walks it again afterwards (the writer materialises and
        # encodes it; a substitutive_lod= wrapper forwards it to the finest
        # child, which re-runs this same gate). A single-pass iterable — one
        # whose ``__iter__`` hands back the same, already-advancing iterator —
        # therefore has nothing left for that second walk, and this gate has
        # no way to hand its own materialised copy back to the caller. Refuse
        # it HERE, before anything is written, rather than let that second walk
        # report "Image labels length (0)" mid-write with the node's other
        # arrays already on disk. Detected WITHOUT consuming an item: a
        # re-iterable sequence (list / tuple / ndarray / pandas.Series / any
        # __getitem__ sequence) hands out a FRESH iterator per iter() call, so
        # only a one-shot one compares identical.
        if iter(image_labels) is iter(image_labels):
            raise TypeError(
                "image_labels must be a re-iterable sequence (list, tuple, "
                "ndarray, ...) or a dict; got a single-pass iterable of type "
                f"{type(image_labels).__name__}, which cannot be validated "
                "before the write without consuming it. Materialise it first: "
                "image_labels=list(...)."
            )
        for item in image_labels:
            check_image_label_type(item)


def write_image_labels_csr(
    group: zarr.Group,
    image_labels: Any,
    n_elements: int,
    compressor: "CompressorLike",
    sort_order: Optional[np.ndarray] = None,
) -> None:
    """Write per-element image labels using CSR-style encoding.

    Stores two zarr arrays:
    - ``image_label_offsets``: uint64 of shape (N+1,) — byte offset of each image
    - ``image_label_bytes``: uint8 — concatenated encoded image blobs

    Image ``i`` is decoded as ``image_label_bytes[offsets[i]:offsets[i+1]]``.
    Empty entries (no image) have ``offsets[i] == offsets[i+1]``.

    The ``image_label_bytes`` array uses **no compression** (``compressor=None``)
    because the image blobs are already compressed (JPEG/WebP/PNG). The offsets
    array uses the scene's default compressor since it is small.

    Args:
        group: Zarr group to write to.
        image_labels: Per-element images. Accepted types:
            - ``List[bytes]``: pre-encoded blobs
            - ``List[PIL.Image.Image]``: auto-encoded to WebP
            - ``List[numpy.ndarray]``: (H,W,C) uint8, auto-encoded to WebP
            - ``List[Path]`` or ``List[str]``: file paths, read as bytes
            - ``Dict[int, Any]``: sparse — missing indices get empty blobs
        n_elements: Expected element count (for validation).
        compressor: Scene default compressor for the small offsets array.
        sort_order: Optional index array to reorder (from spatial ordering).

    Note:
        The dense (non-``dict``) form is materialised into a ``list`` exactly
        ONCE, up front, before either validating or encoding it — so a
        single-pass iterable (one whose ``__iter__`` keeps returning the same,
        already-advanced iterator, unlike ``list`` / ``tuple`` / ``ndarray`` /
        ``pandas.Series``, which are all re-iterable) passed directly to this
        call is walked exactly once and works.

        A single-pass iterable is only supported on a call that comes STRAIGHT
        here, though. Every gate above this function (a writer's step-0 sweep,
        a ``substitutive_lod=`` wrapper's pre-split gate) walks the value
        itself and cannot hand its own materialised copy back to its caller,
        so :func:`validate_image_labels_for_writing` refuses a one-shot dense
        iterable outright — before anything is written — rather than draining
        it and leaving this function nothing to encode. A bare generator is
        refused one step earlier still, by that gate's ``len()``:
        ``TypeError: object of type 'generator' has no len()``.

    Raises:
        ValueError: Also raised (via :func:`validate_image_labels_for_writing`)
            if ``image_labels`` is a single-pass iterable the CALLER had
            already drained before calling this — the materialisation then
            sees zero items and the length check reports ``Image labels
            length (0) must match element count (N)`` instead of writing an
            all-empty CSR.
    """
    # Length/index/type checks now shared with the callers' pre-write gates —
    # see validate_image_labels_for_writing. Materialise the dense form ONCE
    # before that call (not after) so validating and encoding walk the exact
    # same concrete list: a Sized single-pass iterable would otherwise be
    # drained by the validator's own per-entry loop, leaving the blob-encoding
    # loop below nothing to see and silently writing an all-empty CSR (#1491).
    if not isinstance(image_labels, dict):
        image_labels = list(image_labels)
    validate_image_labels_for_writing(image_labels, n_elements)
    if isinstance(image_labels, dict):
        normalized: List[bytes] = [b""] * n_elements
        for idx, item in image_labels.items():
            normalized[idx] = normalize_image_label(item)
        blob_list = normalized
    else:
        blob_list = [normalize_image_label(item) for item in image_labels]

    # Apply spatial reordering if present
    if sort_order is not None:
        blob_list = [blob_list[i] for i in sort_order]

    # Build CSR arrays
    offsets = np.zeros(n_elements + 1, dtype=np.uint64)
    for i, blob in enumerate(blob_list):
        offsets[i + 1] = offsets[i] + len(blob)

    total_bytes = int(offsets[-1])
    image_bytes = np.zeros(max(total_bytes, 1), dtype=np.uint8)
    pos = 0
    for blob in blob_list:
        if blob:
            image_bytes[pos : pos + len(blob)] = np.frombuffer(blob, dtype=np.uint8)
            pos += len(blob)

    # Write offsets (small, compressible)
    group.create_dataset(
        "image_label_offsets",
        data=offsets,
        chunks=(min(n_elements + 1, 65536),),
        compressor=resolve_compressor(compressor, offsets.dtype),
        overwrite=True,
    )
    # Write image bytes — NO compression (already compressed blobs), 1MB chunks
    group.create_dataset(
        "image_label_bytes",
        data=image_bytes,
        chunks=(min(total_bytes, 1_048_576) if total_bytes > 0 else 1,),
        compressor=None,
        overwrite=True,
    )
    group.attrs["has_image_labels"] = True
    n_nonempty = sum(1 for b in blob_list if b)
    avg_size = total_bytes / n_nonempty if n_nonempty > 0 else 0
    aprint(
        f"  ✓ Wrote image labels ({n_nonempty}/{n_elements} non-empty, "
        f"{total_bytes:,} bytes, avg {avg_size:.0f} bytes/image)"
    )
