"""Narrow context objects passed to the extracted compiler helpers.

Each ``Ctx`` carries only the orchestrator state a given family of helpers actually
reads — never a back-pointer to the :class:`~luxar.io.compiler.LuxarZarrCompiler`
instance. They are built by the ``_make_*_ctx()`` methods on the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...encoding import ArrayEncoder, EncodingMode
from ...typing_utils.protocols import CompressorProtocol


@dataclass(frozen=True)
class DatasetCtx:
    """Encoder configuration for per-attribute zarr array writes.

    Read-set of every ``datasets/`` serializer and the gsplat array writer:
    the encoder, the active encoding mode, and the scene compressor.
    """

    encoder: ArrayEncoder
    encoding_mode: EncodingMode
    compressor: CompressorProtocol


@dataclass(frozen=True)
class GsplatCtx:
    """Encoder config + spatial-ordering config for gsplat assembly."""

    encoder: ArrayEncoder
    encoding_mode: EncodingMode
    compressor: CompressorProtocol
    enable_spatial_index: bool
    ordering_method: Literal["morton", "hilbert"]


@dataclass(frozen=True)
class OrderingCtx:
    """Spatial-ordering configuration for the points/lines ordering glue."""

    enable_spatial_index: bool
    ordering_method: Literal["morton", "hilbert"]
