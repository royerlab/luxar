"""``luxar env attach``: write a baked environment map into a scene store.

The map lands in a root-level ``environment/`` group that is NOT a scene node —
it carries neither a ``type`` nor a ``kind`` attr, which is exactly the shape
the viewer's node discovery skips as a metadata sidecar, and every general
Python walker that enumerates root children as nodes consults
:data:`~luxar.typing_utils.constants.RESERVED_ROOT_GROUPS` for the same reason.
Three rules make the operation safe to repeat and cheap to serve:

- **The scene digest does not move.** ``environment/`` is excluded from the
  root ``content_hash`` by the hashing walk, so attaching a map never
  invalidates a visitor's warm cache, and the ``scene_content_hash`` the header
  records can be compared EXACTLY to the root's digest — by this function (a
  stale bake is refused unless ``force=True``) and by the viewer at load.
- **The array is named by its own digest** (``faces-<xxh64[:8]>``), so a
  re-bake is a NEW path: a caching viewer cannot serve stale faces, and the
  attrs' ``faces`` pointer says which array is live. Stale siblings are removed.
- **Idempotent.** Attaching the same faces twice writes nothing the second time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import xxhash
from arbol import aprint, asection

from .._zarr_compat import consolidate, create_array, open_group
from ..encoding.compression import WIDTH_AWARE_DEFAULT, resolve_compressor
from ..typing_utils.constants import ENVIRONMENT_GROUP
from .container import SAMPLE_FORMAT, unpack

#: Attr on the environment group naming the live faces array.
FACES_ATTR = "faces"
_FACES_PREFIX = "faces-"


@dataclass
class AttachReport:
    """What :func:`attach_environment` did."""

    store: Path
    #: ``attached`` (first map), ``replaced`` (a different map was there),
    #: ``unchanged`` (the same map was already attached; nothing written).
    status: str
    array_name: str
    digest: str
    scene_content_hash: str
    resolution: int
    removed: List[str] = field(default_factory=list)


def attach_environment(
    store: Union[str, Path],
    faces: Union[str, Path, bytes],
    *,
    force: bool = False,
) -> AttachReport:
    """Attach the container at ``faces`` (a path or its bytes) to ``store``.

    Refuses a non-scene store, a malformed container (see
    :mod:`luxar.environment.container`), and — unless ``force`` — a map whose
    ``scene_content_hash`` is not the store's current root ``content_hash``: the
    viewer would ignore such a map as stale anyway, so writing it would only
    look like success.
    """
    store_path = Path(store)
    if not store_path.exists():
        raise FileNotFoundError(f"Scene not found: {store_path}")
    if not store_path.is_dir():
        raise ValueError(
            f"env attach requires an uncompressed .zarr directory; got "
            f"{store_path} (unpack a .zip/.tar.gz store first — an attrs rewrite "
            f"of a compressed archive cannot happen in place)"
        )
    blob = faces if isinstance(faces, bytes) else Path(faces).read_bytes()
    header, samples = unpack(blob)

    root = open_group(store_path, mode="r+")
    root_attrs = dict(root.attrs)
    if root_attrs.get("type") != "scene":
        raise ValueError(
            f"{store_path} is not a Luxar scene store (root type is "
            f"{root_attrs.get('type')!r}); an environment attaches to a scene."
        )
    scene_hash = root_attrs.get("content_hash")
    if not isinstance(scene_hash, str) or not scene_hash:
        raise ValueError(
            f"{store_path} carries no root content_hash; finalize the scene first."
        )
    baked_against = str(header["scene_content_hash"])
    if baked_against != scene_hash and not force:
        raise ValueError(
            f"This environment was baked against scene content_hash "
            f"{baked_against[:16]}... but {store_path} is at {scene_hash[:16]}...: "
            "the scene changed since the bake, so the viewer would ignore the map "
            "as stale. Re-run `luxar env bake`, or pass --force to attach it anyway "
            "(the header is rewritten to the current digest)."
        )

    digest = _digest(header, samples)
    array_name = f"{_FACES_PREFIX}{digest[:8]}"
    resolution = int(header["resolution"])

    with asection(f"Attaching environment map to {store_path.name}"):
        env = root.require_group(ENVIRONMENT_GROUP)
        existing = dict(env.attrs)
        if (
            existing.get(FACES_ATTR) == array_name
            and existing.get("scene_content_hash") == scene_hash
            and array_name in env
        ):
            aprint(f"unchanged: {ENVIRONMENT_GROUP}/{array_name} is already attached")
            return AttachReport(
                store=store_path,
                status="unchanged",
                array_name=array_name,
                digest=digest,
                scene_content_hash=scene_hash,
                resolution=resolution,
            )

        # Never zstd-shuffle half bits: the byte shuffle is tuned for integers
        # whose high bytes repeat, and halves do not; the width-aware policy
        # already picks the no-shuffle codec for a 2-byte unsigned dtype.
        create_array(
            env,
            array_name,
            data=samples,
            dtype=np.uint16,
            chunks=(1, resolution, resolution, 4),
            compressor=resolve_compressor(WIDTH_AWARE_DEFAULT, np.uint16),
            overwrite=True,
        )
        removed = [
            name
            for name in list(env.array_keys())
            if name.startswith(_FACES_PREFIX) and name != array_name
        ]
        for name in removed:
            del env[name]

        attrs: Dict[str, Any] = {
            k: v for k, v in header.items() if k not in ("type", "kind")
        }
        attrs["scene_content_hash"] = scene_hash
        attrs[FACES_ATTR] = array_name
        attrs["sample_format"] = SAMPLE_FORMAT
        attrs["shape"] = [6, resolution, resolution, 4]
        # The group's OWN digest, for tooling; the scene digest excludes it.
        attrs["content_hash"] = digest
        for key in list(env.attrs.keys()):
            if key not in attrs:
                del env.attrs[key]
        env.attrs.update(attrs)
        consolidate(root)

        status = "replaced" if existing else "attached"
        aprint(f"{status}: {ENVIRONMENT_GROUP}/{array_name} ({resolution}px faces)")
        for name in removed:
            aprint(f"removed stale {ENVIRONMENT_GROUP}/{name}")
        return AttachReport(
            store=store_path,
            status=status,
            array_name=array_name,
            digest=digest,
            scene_content_hash=scene_hash,
            resolution=resolution,
            removed=removed,
        )


def _digest(header: Dict[str, Any], samples: np.ndarray) -> str:
    """xxh64 over the samples and the capture parameters that define the map.

    ``baked_at`` and other bookkeeping are left out on purpose: two bakes of the
    same scene at the same probe and resolution that produce the same samples
    ARE the same map, and idempotency rests on saying so.
    """
    hasher = xxhash.xxh64()
    hasher.update(np.ascontiguousarray(samples).tobytes())
    for key in ("probe", "resolution", "coordinate_system", "face_order"):
        value = json.dumps(header.get(key), sort_keys=True, separators=(",", ":"))
        hasher.update(f"{key}={value};".encode())
    return hasher.hexdigest()
