"""Narrow context objects passed to the extracted compiler helpers.

Each ``Ctx`` carries only the orchestrator state a given family of helpers actually
reads — never a back-pointer to the :class:`~luxar.io.compiler.LuxarZarrCompiler`
instance. They are built by the ``_make_*_ctx()`` methods on the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ...encoding import ArrayEncoder, EncodingMode

if TYPE_CHECKING:
    from ...encoding.compression import CompressorLike


@dataclass(frozen=True)
class DatasetCtx:
    """Encoder configuration for per-attribute zarr array writes.

    Read-set of every ``datasets/`` serializer and the gsplat array writer:
    the encoder, the active encoding mode, and the scene compressor.
    """

    encoder: ArrayEncoder
    encoding_mode: EncodingMode
    compressor: "CompressorLike"


@dataclass(frozen=True)
class OrderingCtx:
    """Spatial-ordering configuration.

    Shared by gsplat spatial ordering and the points/lines ordering glue.
    """

    enable_spatial_index: bool
    ordering_method: Literal["morton", "hilbert"]
