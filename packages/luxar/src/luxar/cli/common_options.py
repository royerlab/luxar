"""Shared typer.Option declarations for the serve-family commands.

``serve`` and ``viewer`` (and historically ``demo``) accept the same
network-simulation and CORS options. Declaring them once as
``typing.Annotated`` aliases keeps the flag spellings, short options, and
help strings from drifting between commands — each command supplies only
its own default value.

The ``--jitter`` option deliberately has no ``-j`` short flag: ``-j``
means *jobs* in ``gsplat fit`` and *output-json* in ``gsplat compare``,
so a third meaning for a niche testing flag was a muscle-memory trap.
"""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from arbol import aprint

from .network_simulation import parse_network_options

HostOption = Annotated[
    str,
    typer.Option(
        "--host",
        help="Host address to bind to (use 0.0.0.0 for all interfaces)",
    ),
]

ProfileOption = Annotated[
    Optional[str],
    typer.Option(
        "--profile",
        help=(
            "Network profile (3g, 4g, 5g, slow-broadband, broadband, "
            "fast-broadband, satellite, rural, congested)"
        ),
    ),
]

BandwidthOption = Annotated[
    Optional[str],
    typer.Option(
        "--bandwidth",
        "-b",
        help="Bandwidth limit (e.g., '1mbps', '500kbps', '10mbps')",
    ),
]

LatencyOption = Annotated[
    Optional[str],
    typer.Option(
        "--latency",
        "-l",
        help="Network latency (e.g., '100ms', '500ms', '1s')",
    ),
]

JitterOption = Annotated[
    Optional[str],
    typer.Option(
        "--jitter",
        help="Latency jitter as percentage (e.g., '10%', '0.1')",
    ),
]

PacketLossOption = Annotated[
    Optional[str],
    typer.Option(
        "--packet-loss",
        help="Packet loss rate (e.g., '1%', '0.01', '5%')",
    ),
]

CorsOriginOption = Annotated[
    str,
    typer.Option(
        "--cors-origin",
        help=(
            "Allowed CORS origin. Default 'local' allows localhost/127.0.0.1/::1. "
            "Use '*' to allow any origin without credentials."
        ),
    ),
]

AllowSensitivePathOption = Annotated[
    bool,
    typer.Option(
        "--allow-sensitive-path",
        help="Allow serving obvious system paths such as /, /etc, /proc, /sys, /dev.",
    ),
]


def parse_network_options_or_exit(
    profile: Optional[str],
    bandwidth: Optional[str],
    latency: Optional[str],
    jitter: Optional[str],
    packet_loss: Optional[str],
) -> tuple[Optional[float], Optional[float], float, float]:
    """``parse_network_options`` with the CLI's uniform error handling.

    Invalid values print a single error line and exit 1 instead of raising
    ``ValueError`` — the wrapper every serve-family command previously
    duplicated inline.
    """
    try:
        return parse_network_options(profile, bandwidth, latency, jitter, packet_loss)
    except ValueError as e:
        aprint(f"❌ [Luxar] {e}")
        raise typer.Exit(code=1) from e
