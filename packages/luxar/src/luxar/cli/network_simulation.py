"""Network simulation middleware for testing Luxar viewer under various network conditions.

This module provides ASGI middleware to simulate realistic network conditions including:
- Bandwidth throttling (limit bytes per second)
- Network latency (fixed delay + optional jitter)
- Packet loss (random request dropping)

⚠️ WARNING: For development and testing only. Never use in production.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any, Callable, Dict, Optional, TypedDict


class NetworkProfile(TypedDict):
    """Network profile configuration."""

    name: str
    bandwidth: str
    latency: str
    jitter: float
    packet_loss: float
    description: str


# Connection profiles for common network conditions
NETWORK_PROFILES: Dict[str, NetworkProfile] = {
    "3g": {
        "name": "3G Mobile",
        "bandwidth": "384kbps",
        "latency": "300ms",
        "jitter": 0.1,  # 10%
        "packet_loss": 0.01,  # 1%
        "description": "Slow 3G mobile connection",
    },
    "4g": {
        "name": "4G LTE",
        "bandwidth": "10mbps",
        "latency": "100ms",
        "jitter": 0.1,  # 10%
        "packet_loss": 0.005,  # 0.5%
        "description": "Typical 4G/LTE mobile",
    },
    "5g": {
        "name": "5G Mobile",
        "bandwidth": "100mbps",
        "latency": "30ms",
        "jitter": 0.05,  # 5%
        "packet_loss": 0.001,  # 0.1%
        "description": "Modern 5G mobile",
    },
    "slow-broadband": {
        "name": "Slow Broadband",
        "bandwidth": "5mbps",
        "latency": "50ms",
        "jitter": 0.1,  # 10%
        "packet_loss": 0.005,  # 0.5%
        "description": "Slow home broadband",
    },
    "broadband": {
        "name": "Broadband",
        "bandwidth": "50mbps",
        "latency": "20ms",
        "jitter": 0.05,  # 5%
        "packet_loss": 0.001,  # 0.1%
        "description": "Typical home broadband",
    },
    "fast-broadband": {
        "name": "Fast Broadband",
        "bandwidth": "200mbps",
        "latency": "10ms",
        "jitter": 0.02,  # 2%
        "packet_loss": 0.0005,  # 0.05%
        "description": "Fast fiber connection",
    },
    "satellite": {
        "name": "Satellite",
        "bandwidth": "25mbps",
        "latency": "600ms",
        "jitter": 0.15,  # 15%
        "packet_loss": 0.01,  # 1%
        "description": "Satellite connection (high latency)",
    },
    "rural": {
        "name": "Rural Connection",
        "bandwidth": "1mbps",
        "latency": "100ms",
        "jitter": 0.2,  # 20%
        "packet_loss": 0.02,  # 2%
        "description": "Poor rural connection",
    },
    "congested": {
        "name": "Congested Network",
        "bandwidth": "2mbps",
        "latency": "200ms",
        "jitter": 0.25,  # 25%
        "packet_loss": 0.03,  # 3%
        "description": "Congested/overloaded network",
    },
}


def parse_bandwidth(bandwidth_str: str) -> float:
    """Parse bandwidth string to Mbps.

    Args:
        bandwidth_str: Bandwidth with unit (e.g., "1mbps", "500kbps", "2.5gbps")

    Returns:
        Bandwidth in megabits per second

    Raises:
        ValueError: If format is invalid

    Examples:
        >>> parse_bandwidth("1mbps")
        1.0
        >>> parse_bandwidth("500kbps")
        0.5
        >>> parse_bandwidth("2.5gbps")
        2500.0
    """
    bandwidth_str = bandwidth_str.lower().strip()

    if bandwidth_str.endswith("mbps"):
        value = float(bandwidth_str[:-4])
    elif bandwidth_str.endswith("kbps"):
        value = float(bandwidth_str[:-4]) / 1000.0
    elif bandwidth_str.endswith("gbps"):
        value = float(bandwidth_str[:-4]) * 1000.0
    else:
        raise ValueError(
            f"Invalid bandwidth format: '{bandwidth_str}'. "
            "Expected format: <number><unit> where unit is kbps, mbps, or gbps. "
            "Examples: '1mbps', '500kbps', '2.5gbps'"
        )

    if value <= 0:
        raise ValueError(f"Bandwidth must be positive, got: {value}")

    return value


def parse_latency(latency_str: str) -> float:
    """Parse latency string to milliseconds.

    Args:
        latency_str: Latency with unit (e.g., "100ms", "1s", "0.5s")

    Returns:
        Latency in milliseconds

    Raises:
        ValueError: If format is invalid

    Examples:
        >>> parse_latency("100ms")
        100.0
        >>> parse_latency("1s")
        1000.0
        >>> parse_latency("0.5s")
        500.0
    """
    latency_str = latency_str.lower().strip()

    if latency_str.endswith("ms"):
        value = float(latency_str[:-2])
    elif latency_str.endswith("s") and not latency_str.endswith("ms"):
        value = float(latency_str[:-1]) * 1000.0
    else:
        raise ValueError(
            f"Invalid latency format: '{latency_str}'. "
            "Expected format: <number><unit> where unit is ms or s. "
            "Examples: '100ms', '1s', '0.5s'"
        )

    if value < 0:
        raise ValueError(f"Latency cannot be negative, got: {value}")

    return value


def parse_jitter(jitter_str: str) -> float:
    """Parse jitter string to decimal (0.0-1.0).

    Args:
        jitter_str: Jitter as percentage or decimal (e.g., "10%", "0.1")

    Returns:
        Jitter as decimal (0.0-1.0)

    Raises:
        ValueError: If format is invalid or out of range

    Examples:
        >>> parse_jitter("10%")
        0.1
        >>> parse_jitter("0.15")
        0.15
        >>> parse_jitter("25%")
        0.25
    """
    jitter_str = jitter_str.strip()

    if jitter_str.endswith("%"):
        value = float(jitter_str[:-1]) / 100.0
    else:
        value = float(jitter_str)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Jitter must be between 0% and 100% (or 0.0 and 1.0), got: {jitter_str}"
        )

    return value


def parse_packet_loss(loss_str: str) -> float:
    """Parse packet loss string to decimal (0.0-1.0).

    Args:
        loss_str: Packet loss as percentage or decimal (e.g., "1%", "0.01")

    Returns:
        Packet loss rate as decimal (0.0-1.0)

    Raises:
        ValueError: If format is invalid or out of range

    Examples:
        >>> parse_packet_loss("1%")
        0.01
        >>> parse_packet_loss("0.05")
        0.05
        >>> parse_packet_loss("10%")
        0.1
    """
    loss_str = loss_str.strip()

    if loss_str.endswith("%"):
        value = float(loss_str[:-1]) / 100.0
    else:
        value = float(loss_str)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Packet loss must be between 0% and 100% (or 0.0 and 1.0), got: {loss_str}"
        )

    return value


def parse_network_options(
    profile: Optional[str] = None,
    bandwidth: Optional[str] = None,
    latency: Optional[str] = None,
    jitter: Optional[str] = None,
    packet_loss: Optional[str] = None,
) -> tuple[Optional[float], Optional[float], float, float]:
    """Parse and merge network simulation CLI parameters.

    Loads a profile first (if given), then overrides with individual params.

    Args:
        profile: Profile name (e.g., "3g", "satellite").
        bandwidth: Bandwidth string (e.g., "1mbps", "500kbps").
        latency: Latency string (e.g., "100ms", "1s").
        jitter: Jitter string (e.g., "10%", "0.1").
        packet_loss: Packet loss string (e.g., "1%", "0.01").

    Returns:
        Tuple of (bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate).

    Raises:
        ValueError: If any parameter has invalid format.
    """
    bandwidth_mbps: Optional[float] = None
    latency_ms: Optional[float] = None
    jitter_percent: float = 0.0
    packet_loss_rate: float = 0.0

    # 1. Load profile defaults (if specified)
    if profile:
        profile_data = load_network_profile(profile)
        bandwidth_mbps = parse_bandwidth(profile_data["bandwidth"])
        latency_ms = parse_latency(profile_data["latency"])
        jitter_percent = profile_data["jitter"]
        packet_loss_rate = profile_data["packet_loss"]

    # 2. Override with individual parameters
    if bandwidth:
        bandwidth_mbps = parse_bandwidth(bandwidth)
    if latency:
        latency_ms = parse_latency(latency)
    if jitter:
        jitter_percent = parse_jitter(jitter)
    if packet_loss:
        packet_loss_rate = parse_packet_loss(packet_loss)

    return bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate


def _truthy_env(value: Optional[str]) -> bool:
    """Return True for common truthy environment variable values."""
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _production_guard_enabled() -> bool:
    """Whether network simulation should be refused for production envs."""
    return _truthy_env(os.getenv("LUXAR_PRODUCTION")) or os.getenv(
        "LUXAR_ENV", ""
    ).strip().lower() in {"prod", "production"}


def has_network_simulation(
    bandwidth_mbps: Optional[float],
    latency_ms: Optional[float],
    jitter_percent: float,
    packet_loss_rate: float,
) -> bool:
    """Check if any network simulation parameters are active.

    Returns True when at least one of bandwidth, latency, jitter, or packet
    loss is set to a non-zero/non-None value.
    """
    return any([bandwidth_mbps, latency_ms, jitter_percent > 0, packet_loss_rate > 0])


def print_network_params(
    bandwidth_mbps: Optional[float],
    latency_ms: Optional[float],
    jitter_percent: float,
    packet_loss_rate: float,
    qualifier: str = "",
) -> None:
    """Print active network simulation parameters.

    Args:
        bandwidth_mbps: Bandwidth in Mbps.
        latency_ms: Latency in milliseconds.
        jitter_percent: Jitter as decimal (0.0-1.0).
        packet_loss_rate: Packet loss as decimal (0.0-1.0).
        qualifier: Optional qualifier for the message (e.g., "data server only").
    """
    from arbol import aprint

    suffix = f" ({qualifier})" if qualifier else ""
    aprint(f"🌐 [Luxar] Network simulation enabled{suffix}:")
    if bandwidth_mbps:
        aprint(f"   • Bandwidth: {bandwidth_mbps:.2f} Mbps")
    if latency_ms:
        aprint(f"   • Latency: {latency_ms:.0f} ms")
    if jitter_percent > 0:
        aprint(f"   • Jitter: {jitter_percent * 100:.0f}%")
    if packet_loss_rate > 0:
        aprint(f"   • Packet loss: {packet_loss_rate * 100:.1f}%")


def load_network_profile(profile_name: str) -> NetworkProfile:
    """Load network profile by name.

    Args:
        profile_name: Profile identifier (e.g., "3g", "4g", "satellite")

    Returns:
        NetworkProfile dictionary with all parameters

    Raises:
        ValueError: If profile name is not recognized
    """
    profile_name = profile_name.lower().strip()

    if profile_name not in NETWORK_PROFILES:
        available = ", ".join(sorted(NETWORK_PROFILES.keys()))
        raise ValueError(
            f"Unknown network profile: '{profile_name}'. "
            f"Available profiles: {available}"
        )

    return NETWORK_PROFILES[profile_name].copy()


class NetworkSimulationMiddleware:
    """ASGI middleware to simulate realistic network conditions.

    This middleware simulates:
    - Bandwidth throttling (limits bytes per second)
    - Network latency (fixed delay + optional jitter)
    - Packet loss (random request dropping)

    The middleware intercepts HTTP requests and responses, applying
    the configured simulation parameters.

    ⚠️ WARNING: For development and testing only. Never use in production.
    """

    def __init__(
        self,
        app: Any,
        bandwidth_limit_mbps: Optional[float] = None,
        latency_ms: Optional[float] = None,
        jitter_percent: float = 0.0,
        packet_loss_rate: float = 0.0,
    ):
        """Initialize network simulation middleware.

        Args:
            app: ASGI application to wrap
            bandwidth_limit_mbps: Maximum bandwidth in megabits per second
            latency_ms: Fixed latency to add to each response (milliseconds)
            jitter_percent: Latency variation as percentage (0.0-1.0)
            packet_loss_rate: Probability of dropping a request (0.0-1.0)
        """
        self.app = app
        self.bandwidth_limit_mbps = bandwidth_limit_mbps
        self.latency_ms = latency_ms
        self.jitter_percent = jitter_percent
        self.packet_loss_rate = packet_loss_rate

        if _production_guard_enabled() and has_network_simulation(
            bandwidth_limit_mbps, latency_ms, jitter_percent, packet_loss_rate
        ):
            raise RuntimeError(
                "Network simulation is for development/testing only and is disabled "
                "when LUXAR_ENV=production/prod or LUXAR_PRODUCTION is set. "
                "Remove --profile/--bandwidth/--latency/--jitter/--packet-loss "
                "or unset the production environment flag."
            )

        # Convert bandwidth to bytes per second
        self.bytes_per_second = (
            (bandwidth_limit_mbps * 1_000_000 / 8) if bandwidth_limit_mbps else None
        )

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable,
        send: Callable,
    ) -> None:
        """ASGI middleware entry point."""
        # Only intercept HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Packet loss simulation - randomly drop requests
        if self.packet_loss_rate > 0:
            if random.random() < self.packet_loss_rate:
                # Drop this request - client will timeout
                return

        # Latency simulation - delay before processing request
        if self.latency_ms:
            # Calculate actual latency with jitter
            base_latency = self.latency_ms / 1000.0  # Convert to seconds

            if self.jitter_percent > 0:
                # Jitter is ±(jitter_percent * base_latency)
                jitter_amount = base_latency * self.jitter_percent
                # Random value in range [-jitter_amount, +jitter_amount]
                actual_latency = base_latency + random.uniform(
                    -jitter_amount, jitter_amount
                )
                # Ensure latency never goes negative
                actual_latency = max(0, actual_latency)
            else:
                actual_latency = base_latency

            # Sleep to simulate request propagation time
            await asyncio.sleep(actual_latency)

        # Per-request state for bandwidth throttling
        bytes_sent = 0
        start_time = time.time()

        async def send_wrapper(message: Dict[str, Any]) -> None:
            """Wrap send() to throttle response body."""
            nonlocal bytes_sent, start_time

            if message["type"] == "http.response.body":
                body = message.get("body", b"")

                # Throttle bandwidth if enabled and body is not empty
                if self.bytes_per_second and body:
                    # Track total bytes sent for this request
                    bytes_sent += len(body)

                    # Calculate expected elapsed time based on bytes sent
                    elapsed = time.time() - start_time
                    expected_time = bytes_sent / self.bytes_per_second

                    # If we're sending too fast, sleep to match target bandwidth
                    if expected_time > elapsed:
                        sleep_time = expected_time - elapsed
                        await asyncio.sleep(sleep_time)

            # Send the message (possibly after throttling delay)
            await send(message)

        # Call wrapped application with our send wrapper
        await self.app(scope, receive, send_wrapper)
