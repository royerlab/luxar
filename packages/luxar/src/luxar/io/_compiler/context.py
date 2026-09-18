"""Narrow context objects passed to the extracted compiler helpers.

Each ``Ctx`` carries only the orchestrator state a given family of helpers actually
reads — never a back-pointer to the :class:`~luxar.io.compiler.LuxarZarrCompiler`
instance. They are built by the ``_make_*_ctx()`` methods on the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

from ...encoding import ArrayEncoder, EncodingMode

if TYPE_CHECKING:
    import zarr

    from ...encoding.compression import CompressorLike


@dataclass(frozen=True)
class DatasetCtx:
    """Encoder configuration for per-attribute zarr array writes.

    Read-set of every ``datasets/`` serializer and the gsplat array writer:
    the encoder, the active encoding mode, and the scene compressor.
    ``positive_scalar_bits`` applies to every POSITIVE_SCALAR write in the
    context. The scene compiler sets it only on its GSplats context, so Points
    radii and Lines widths retain their independent adaptive policy.
    ``deduplicate_positive_scalar`` is disabled when a per-node tier could make
    byte-identical source arrays encode differently; the registry keys on bytes.
    """

    encoder: ArrayEncoder
    encoding_mode: EncodingMode
    compressor: "CompressorLike"
    positive_scalar_bits: Optional[Literal[8, 16]] = None
    deduplicate_positive_scalar: bool = True


@dataclass(frozen=True)
class OrderingCtx:
    """Spatial-ordering configuration.

    Shared by gsplat spatial ordering and the points/lines ordering glue.
    """

    enable_spatial_index: bool
    ordering_method: Literal["morton", "hilbert"]


@dataclass(frozen=True)
class GeometryWriteCtx:
    """Narrow context for the extracted Points/Lines write pipelines.

    Carries the encoder/ordering configs + scene compressor the dataset
    serializers and ordering writers need, plus bound-method hooks for the
    orchestrator state a write must mutate: the scene-bounds accumulator, the
    warn-once colormap-LUT flag, and the per-type authoring-warning registry. The
    finalized guard and metadata-cache write stay on the orchestrator (they
    bracket the extracted body in its delegate).
    """

    store: "zarr.Group"
    dataset_ctx: DatasetCtx
    ordering_ctx: OrderingCtx
    compressor: "CompressorLike"
    update_scene_bounds: "Callable[[Dict[str, List[float]]], None]"
    write_colormap_lut: "Callable[[zarr.Group, Dict[str, Any]], None]"
    #: ``(geometry_kind, path) -> True`` exactly once, so a partition's leaves
    #: collapse to one authoring warning per logical node. Keyed by kind too, so
    #: one type's lint cannot silence another's on the same node.
    claim_authoring_warning: "Callable[[str, str], bool]"


@dataclass(frozen=True)
class GSplatsWriteCtx:
    """Narrow context for the extracted GSplats write pipelines.

    Like :class:`GeometryWriteCtx` but with the gsplat-specific group-attrs hook
    (``apply_gsplat_group_attrs`` owns the warn-once colormap-LUT flag) and the
    scene tone-mapping value the leaf-subtree writer threads into
    ``write_gsplat_leaf``. The barrier-dim inference is a pure function of the
    store (see ``scene_barrier_dims`` in geometry_writers/gsplats.py).
    """

    store: "zarr.Group"
    dataset_ctx: DatasetCtx
    ordering_ctx: OrderingCtx
    compressor: "CompressorLike"
    scene_tone_mapping: "Optional[str]"
    update_scene_bounds: "Callable[[Dict[str, List[float]]], None]"
    apply_gsplat_group_attrs: (
        "Callable[[zarr.Group, Dict[str, Any], Dict[str, Any]], None]"
    )
