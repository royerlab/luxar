# Network Simulation for Luxar CLI - Technical Specification

**Version**: 1.1.0
**Status**: ✅ Implemented — server-side ASGI middleware (see `packages/luxar/src/luxar/cli/network_simulation.py`). Network conditions are simulated transparently; no viewer-side implementation is needed.
**Created**: 2025-01-06
**Last Updated**: 2025-12-15
**Author**: Claude Code

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Requirements](#requirements)
4. [Architecture](#architecture)
5. [Component Specifications](#component-specifications)
6. [CLI Interface](#cli-interface)
7. [Connection Profiles](#connection-profiles)
8. [Implementation Details](#implementation-details)
9. [Error Handling](#error-handling)
10. [Testing Strategy](#testing-strategy)
11. [Examples](#examples)
12. [Future Extensions](#future-extensions)

---

## Overview

This specification defines a network simulation system for the Luxar CLI that allows developers to test the viewer's performance under realistic network conditions. The system simulates bandwidth throttling, latency, jitter, and packet loss through ASGI middleware applied to the FastAPI data server.

### Goals

- Enable realistic testing of viewer performance under various network conditions
- Provide simple CLI interface for common network scenarios
- Support both individual parameter control and preset connection profiles
- Maintain backward compatibility (all features optional)
- Zero performance impact when simulation is disabled

### Non-Goals

- Simulating browser-side network conditions (only server-side)
- Complex network topology simulation
- Per-route or per-file type different throttling rules
- Real-time adjustment of parameters during serving

### ⚠️ Security Warning

**CRITICAL**: Network simulation is for **DEVELOPMENT AND TESTING ONLY**.

**DO NOT use in production environments.**

When network simulation is enabled:
- Server responses are intentionally delayed and throttled
- Requests may be randomly dropped (packet loss simulation)
- The server may appear unresponsive or unreliable to clients
- This can degrade user experience and appear as server failures

**Use cases**:
- ✅ Local development testing
- ✅ Performance benchmarking
- ✅ Integration testing in CI/CD (with known controlled conditions)
- ✅ User experience research

**NOT for**:
- ❌ Production deployments
- ❌ Public-facing servers
- ❌ User-facing staging environments
- ❌ Load testing (use real network conditions instead)

**This tool simulates poor network conditions to help you test your application's resilience and user experience under various network scenarios.**

---

## Motivation

### Problem Statement

The Luxar viewer is designed to work with datasets served over HTTP, potentially from remote servers with varying network characteristics. During development, testing only occurs on localhost with ideal network conditions (low latency, high bandwidth, no packet loss). This masks potential issues:

1. **Progressive loading problems**: Slow chunk loading may reveal UI blocking issues
2. **Cache strategy validation**: Cache effectiveness is only apparent under bandwidth constraints
3. **Spatial index tuning**: Query range optimization requires realistic network conditions
4. **User experience baseline**: Minimum viable network requirements are unknown
5. **Error handling**: Network failures (packet loss, timeouts) may not be tested

### Use Cases

1. **Performance testing**: "How does the viewer perform on a 3G mobile connection?"
2. **Cache validation**: "Does our caching system effectively reduce redundant requests?"
3. **UX research**: "What's the minimum bandwidth for acceptable user experience?"
4. **Regression testing**: "Did this change make loading slower on poor connections?"
5. **Documentation**: "What network requirements should we document for users?"

---

## Requirements

### Functional Requirements

**FR-1**: Simulate bandwidth throttling (limit bytes per second)
**FR-2**: Simulate fixed network latency (round-trip delay)
**FR-3**: Simulate latency jitter (variable latency as percentage of fixed latency)
**FR-4**: Simulate packet loss (probability of dropping responses)
**FR-5**: Provide preset connection profiles (3G, 4G, 5G, broadband, satellite, etc.)
**FR-6**: Support combining individual parameters with profiles
**FR-7**: Apply simulation to `serve`, `viewer`, and `demo` commands (see note below about viewer semantics)
**FR-8**: Display clear indication when simulation is active
**FR-9**: Validate all input parameters with helpful error messages
**FR-10**: Support standard units (kbps/mbps/gbps for bandwidth, ms/s for latency)

### Non-Functional Requirements

**NFR-1**: Zero overhead when simulation is disabled
**NFR-2**: Backward compatible (existing commands work unchanged)
**NFR-3**: Thread-safe (works with concurrent data + viewer serving)
**NFR-4**: Accurate simulation (within 5% of specified values)
**NFR-5**: Clear documentation with examples
**NFR-6**: Comprehensive test coverage (>80%)

### Important: Viewer Command Semantics

**FR-7 Clarification**: When network simulation is applied to the `viewer` command, it **only affects the data server**, not the viewer HTML/JS/CSS files.

**Background**: The `luxar viewer` command can serve:
1. **Viewer files** (HTML/JS/CSS) - the web application itself
2. **Data files** (zarr datasets) - if `--data` option is provided

**Simulation Behavior**:
- ✅ **Data server** (serves .zarr files) → Simulation APPLIED
- ❌ **Viewer server** (serves HTML/JS/CSS) → Simulation NOT APPLIED

**Rationale**:
- Users want to test how the viewer performs loading slow data, not how slowly the viewer itself loads
- Throttling viewer HTML/JS would make the app unusable and provide no useful insights
- Consistent with `serve --viewer` behavior (data is throttled, viewer is not)

**Example**:
```bash
# This throttles the ZARR DATA, not the viewer HTML
luxar viewer --data foo.luxar.zarr --bandwidth 1mbps --latency 200ms

# User expects:
# - Viewer HTML loads quickly (normal speed)
# - Zarr data loads slowly (throttled to 1mbps with 200ms latency)
```

**Implementation Note**: The `viewer` command runs two servers (viewer server + data server in background thread). Only the data server should have the NetworkSimulationMiddleware applied.

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│ CLI Command (serve/viewer/demo)                             │
│  - Parse network simulation flags                           │
│  - Select connection profile (if specified)                 │
│  - Validate parameters                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ FastAPI Application                                         │
│  - CORS Middleware                                          │
│  - NetworkSimulationMiddleware ◄──── (conditionally added) │
│  - StaticFiles Handler                                      │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ NetworkSimulationMiddleware (ASGI Middleware)               │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Request Received                                      │  │
│  └────────┬─────────────────────────────────────────────┘  │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Packet Loss Check                                     │  │
│  │  - Generate random number                             │  │
│  │  - If < loss_rate: Drop request (return immediately) │  │
│  │  - Else: Continue                                     │  │
│  └────────┬─────────────────────────────────────────────┘  │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Latency Simulation                                    │  │
│  │  - Calculate: base_latency ± (jitter% * base_latency)│  │
│  │  - await asyncio.sleep(actual_latency)                │  │
│  └────────┬─────────────────────────────────────────────┘  │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Call Wrapped Application                              │  │
│  │  - Pass through to next middleware/handler            │  │
│  └────────┬─────────────────────────────────────────────┘  │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Response Interception                                 │  │
│  │  - Wrap send() function                               │  │
│  │  - Detect http.response.body messages                 │  │
│  └────────┬─────────────────────────────────────────────┘  │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Bandwidth Throttling                                  │  │
│  │  - Split body into time-based chunks                  │  │
│  │  - Send chunk, sleep, send chunk, sleep, ...          │  │
│  │  - Maintain specified bytes/second rate               │  │
│  └────────┬─────────────────────────────────────────────┘  │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Response Sent to Client                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Component Overview

1. **CLI Parameter Parsing**: Parse and validate user input
2. **Connection Profiles**: Preset configurations for common network types
3. **NetworkSimulationMiddleware**: ASGI middleware implementing simulation
4. **Parameter Validators**: Parse and validate bandwidth/latency strings
5. **Simulation Logic**: Core algorithms for throttling, latency, jitter, packet loss

---

## Component Specifications

### 1. Connection Profiles

**Purpose**: Provide preset configurations for common network conditions.

**Data Structure**:
```python
NetworkProfile = TypedDict('NetworkProfile', {
    'name': str,              # Human-readable name
    'bandwidth': str,         # Bandwidth string (e.g., "1mbps")
    'latency': str,           # Latency string (e.g., "100ms")
    'jitter': float,          # Jitter as percentage (0.0-1.0)
    'packet_loss': float,     # Packet loss rate (0.0-1.0)
    'description': str,       # Description for help text
})
```

**Profiles**:

| Profile | Bandwidth | Latency | Jitter | Packet Loss | Description |
|---------|-----------|---------|--------|-------------|-------------|
| `3g` | 384 kbps | 300 ms | 10% | 1% | Slow 3G mobile connection |
| `4g` | 10 mbps | 100 ms | 10% | 0.5% | Typical 4G/LTE mobile |
| `5g` | 100 mbps | 30 ms | 5% | 0.1% | Modern 5G mobile |
| `slow-broadband` | 5 mbps | 50 ms | 10% | 0.5% | Slow home broadband |
| `broadband` | 50 mbps | 20 ms | 5% | 0.1% | Typical home broadband |
| `fast-broadband` | 200 mbps | 10 ms | 2% | 0.05% | Fast fiber connection |
| `satellite` | 25 mbps | 600 ms | 15% | 1% | Satellite connection (high latency) |
| `rural` | 1 mbps | 100 ms | 20% | 2% | Poor rural connection |
| `congested` | 2 mbps | 200 ms | 25% | 3% | Congested/overloaded network |

**Profile Selection Rules**:
1. If `--profile` is specified, load that profile's defaults
2. Individual flags (`--bandwidth`, `--latency`, etc.) override profile values
3. If no profile and no individual flags, no simulation is applied
4. Invalid profile names should show available profiles and exit

**Example**:
```bash
# Use 3G profile as-is
luxar serve data.luxar.zarr --profile 3g

# Use 4G profile but increase latency
luxar serve data.luxar.zarr --profile 4g --latency 200ms

# Use satellite profile but disable packet loss
luxar serve data.luxar.zarr --profile satellite --packet-loss 0
```

---

### 2. Parameter Parsing and Validation

#### Bandwidth Parser

**Function**: `parse_bandwidth(bandwidth_str: str) -> float`

**Input**: String like `"1mbps"`, `"500kbps"`, `"0.5gbps"`
**Output**: Float representing megabits per second
**Supported Units**: `kbps`, `mbps`, `gbps`

**Algorithm**:
```python
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
```

**Test Cases**:
- Valid: `"1mbps"` → `1.0`, `"500kbps"` → `0.5`, `"2.5gbps"` → `2500.0`
- Valid: `"  10MBPS  "` → `10.0` (whitespace and case insensitive)
- Invalid: `"500"` (no unit), `"abc"` (not a number), `"-1mbps"` (negative)

---

#### Latency Parser

**Function**: `parse_latency(latency_str: str) -> float`

**Input**: String like `"100ms"`, `"1s"`, `"0.5s"`
**Output**: Float representing milliseconds
**Supported Units**: `ms`, `s`

**Algorithm**:
```python
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
```

**Test Cases**:
- Valid: `"100ms"` → `100.0`, `"1s"` → `1000.0`, `"0.5s"` → `500.0`
- Valid: `"  200MS  "` → `200.0` (whitespace and case insensitive)
- Invalid: `"100"` (no unit), `"abc"` (not a number), `"-10ms"` (negative)

---

#### Jitter Parser

**Function**: `parse_jitter(jitter_str: str) -> float`

**Input**: String like `"10%"`, `"0.15"`, `"15%"`
**Output**: Float between 0.0 and 1.0 representing percentage
**Supported Formats**: `"<number>%"` or plain float

**Algorithm**:
```python
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
```

**Test Cases**:
- Valid: `"10%"` → `0.1`, `"0.15"` → `0.15`, `"100%"` → `1.0`, `"0"` → `0.0`
- Invalid: `"150%"` (out of range), `"-10%"` (negative), `"abc"` (not a number)

---

#### Packet Loss Parser

**Function**: `parse_packet_loss(loss_str: str) -> float`

**Input**: String like `"1%"`, `"0.01"`, `"5%"`
**Output**: Float between 0.0 and 1.0 representing probability
**Supported Formats**: `"<number>%"` or plain float

**Algorithm**:
```python
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
```

**Test Cases**:
- Valid: `"1%"` → `0.01`, `"0.05"` → `0.05`, `"100%"` → `1.0`, `"0"` → `0.0`
- Invalid: `"150%"` (out of range), `"-5%"` (negative), `"abc"` (not a number)

---

### 3. NetworkSimulationMiddleware

**Purpose**: ASGI middleware that intercepts HTTP requests/responses and applies network simulation.

**Class Definition**:
```python
class NetworkSimulationMiddleware:
    """ASGI middleware to simulate realistic network conditions.

    Simulates:
    - Bandwidth throttling (limits bytes per second)
    - Network latency (fixed delay + optional jitter)
    - Packet loss (random request dropping)

    The middleware intercepts HTTP requests and responses, applying
    the configured simulation parameters.
    """

    def __init__(
        self,
        app,
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

        # Convert bandwidth to bytes per second
        self.bytes_per_second = (
            (bandwidth_limit_mbps * 1_000_000 / 8)
            if bandwidth_limit_mbps else None
        )

    async def __call__(self, scope, receive, send):
        """ASGI middleware entry point."""
        # Implementation details below...
```

---

#### Packet Loss Simulation

**Algorithm**:
```python
# At the start of __call__, before processing request
if self.packet_loss_rate > 0:
    if random.random() < self.packet_loss_rate:
        # Drop this request - don't call app, don't send response
        # Client will timeout waiting for response
        return
```

**Behavior**:
- Randomly drops entire requests based on `packet_loss_rate`
- Client experiences timeout (no response received)
- Simulates real network packet loss
- Applied before latency simulation (packet dropped = no delay)

**Edge Cases**:
- `packet_loss_rate = 0.0`: No packets dropped (disabled)
- `packet_loss_rate = 1.0`: All packets dropped (server appears down)
- Non-HTTP requests (websockets, etc.): Should not drop

---

#### Latency Simulation (with Jitter)

**Algorithm**:
```python
if self.latency_ms:
    # Calculate actual latency with jitter
    base_latency = self.latency_ms / 1000.0  # Convert to seconds

    if self.jitter_percent > 0:
        # Jitter is ±(jitter_percent * base_latency)
        jitter_amount = base_latency * self.jitter_percent
        # Random value in range [-jitter_amount, +jitter_amount]
        actual_latency = base_latency + random.uniform(-jitter_amount, jitter_amount)
        # Ensure latency never goes negative
        actual_latency = max(0, actual_latency)
    else:
        actual_latency = base_latency

    # Sleep to simulate round-trip time
    await asyncio.sleep(actual_latency)
```

**Jitter Calculation Details**:
- Jitter is specified as percentage of base latency (typically 10%)
- Applied as uniform distribution: `base ± (jitter% * base)`
- Example: 100ms latency with 10% jitter → random latency in [90ms, 110ms]
- Latency is clamped to non-negative values (jitter cannot make latency negative)

**Timing and Semantics**:
- Applied AFTER packet loss check (dropped packets have no delay)
- Applied BEFORE calling wrapped application (simulates request propagation time)
- **Represents request propagation latency** - the time it takes for the HTTP request to travel from client to server
- Does NOT add latency to response transmission (response sent immediately after processing)
- This matches real-world behavior: in HTTP, you notice latency when clicking/requesting, not when receiving the response stream

**Total Perceived Delay**:
```
User Experience Timeline:
1. User clicks/requests data      (t=0)
2. Request travels to server      (t=latency)     ← Simulated here
3. Server processes request       (t=latency+processing)
4. Response travels to client     (t=latency+processing+rtt/2)  ← NOT simulated
5. User sees data                 (t=end)

Our simulation:
- Adds delay at step 2 (request propagation)
- Does not add delay at step 4 (response propagation)
- In practice, this is sufficient for testing because:
  * The main UX impact is initial response delay (steps 2-3)
  * Bandwidth throttling already simulates slow data transfer
  * Adding response latency would be redundant with throttling
```

**Future Enhancement**:
- Could add separate `--response-latency` option for asymmetric networks
- Could split latency 50/50 between request and response
- For v1.0, request-only latency is sufficient for testing purposes

---

#### Bandwidth Throttling

**Algorithm** (Updated to handle streaming responses):
```python
async def __call__(self, scope, receive, send):
    """ASGI middleware entry point."""
    if scope["type"] != "http":
        await self.app(scope, receive, send)
        return

    # Packet loss and latency checks here (see sections above)
    # ...

    # Per-request state for bandwidth throttling
    bytes_sent = 0
    start_time = time.time()

    async def send_wrapper(message):
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
```

**Throttling Strategy**:
1. Track total bytes sent per request (across all response chunks)
2. Track elapsed time since request started
3. Calculate expected time: `total_bytes / bytes_per_second`
4. If ahead of schedule, sleep to maintain target bandwidth
5. Works correctly for both single-chunk and streaming responses

**How It Works**:
- **Single-chunk responses**: Body arrives at once, throttled as expected
- **Streaming responses**: Each chunk adds to `bytes_sent`, maintains consistent rate across all chunks
- **Large files**: Chunks sent by ASGI server are throttled continuously

**Example**:
```
1MB file at 1 Mbps (125 KB/s)
ASGI sends 10 chunks of 100KB each

Chunk 1 (100KB): bytes_sent=100KB, expected=0.8s, elapsed=0.01s → sleep 0.79s
Chunk 2 (100KB): bytes_sent=200KB, expected=1.6s, elapsed=0.81s → sleep 0.79s
...
Chunk 10 (100KB): bytes_sent=1000KB, expected=8.0s, elapsed=7.21s → sleep 0.79s

Total time: ~8 seconds (matches 1MB / 125KB/s)
```

**Edge Cases**:
- Empty body (`body = b""`): No throttling, pass through immediately
- Small responses: Still throttled proportionally (no special fast path)
- Very small responses (<1KB): May have slight overhead, but acceptable
- Streaming responses: Correctly throttled across all chunks
- Multiple `http.response.body` messages: Handled correctly via per-request state

**Accuracy**:
- Target: Average bandwidth matches specified limit (±5%)
- Short responses may have slightly higher variance due to sleep granularity
- Long responses (>10s) should closely match limit
- More accurate than chunk-based approach for all response sizes

**Advantages Over Chunk-Based Approach**:
- ✅ Works correctly with streaming responses (ASGI may pre-chunk large files)
- ✅ Maintains consistent bandwidth across entire request
- ✅ Simpler algorithm (no manual chunking)
- ✅ No assumptions about response body structure

---

### 4. Profile Loading Logic

**Function**: `load_network_profile(profile_name: str) -> NetworkProfile`

**Algorithm**:
```python
NETWORK_PROFILES: Dict[str, NetworkProfile] = {
    "3g": {
        "name": "3G Mobile",
        "bandwidth": "384kbps",
        "latency": "300ms",
        "jitter": 0.1,  # 10%
        "packet_loss": 0.01,  # 1%
        "description": "Slow 3G mobile connection",
    },
    # ... other profiles ...
}

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
```

---

## CLI Interface

### New CLI Options

Add the following options to `serve`, `viewer`, and `demo` commands:

```python
@app.command()
def serve(
    # ... existing parameters ...

    # Network simulation parameters
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Network profile (3g, 4g, 5g, slow-broadband, broadband, fast-broadband, satellite, rural, congested)",
    ),
    bandwidth: Optional[str] = typer.Option(
        None,
        "--bandwidth",
        "-b",
        help="Bandwidth limit (e.g., '1mbps', '500kbps', '10mbps')",
    ),
    latency: Optional[str] = typer.Option(
        None,
        "--latency",
        "-l",
        help="Network latency (e.g., '100ms', '500ms', '1s')",
    ),
    jitter: Optional[str] = typer.Option(
        None,
        "--jitter",
        "-j",
        help="Latency jitter as percentage (e.g., '10%', '0.1')",
    ),
    packet_loss: Optional[str] = typer.Option(
        None,
        "--packet-loss",
        help="Packet loss rate (e.g., '1%', '0.01', '5%')",
    ),
) -> None:
    """Serve a directory, Zarr dataset, or viewer via HTTP.

    Network Simulation:
        Use network simulation options to test viewer performance under
        realistic network conditions. You can use a preset profile or
        specify individual parameters.

        Profiles: 3g, 4g, 5g, slow-broadband, broadband, fast-broadband,
        satellite, rural, congested

        Individual parameters override profile defaults.

    Examples:
        # Simulate 3G mobile connection
        luxar serve data.luxar.zarr --profile 3g --viewer

        # Simulate custom slow connection
        luxar serve data.luxar.zarr --bandwidth 500kbps --latency 200ms

        # Use 4G profile with custom latency
        luxar serve data.luxar.zarr --profile 4g --latency 300ms

        # Test packet loss
        luxar serve data.luxar.zarr --bandwidth 10mbps --packet-loss 5%
    """
```

### Parameter Resolution Logic

When both profile and individual parameters are specified:

```python
# 1. Load profile (if specified)
bandwidth_mbps = None
latency_ms = None
jitter_percent = 0.0
packet_loss_rate = 0.0

if profile:
    try:
        profile_data = load_network_profile(profile)
        bandwidth_mbps = parse_bandwidth(profile_data["bandwidth"])
        latency_ms = parse_latency(profile_data["latency"])
        jitter_percent = profile_data["jitter"]
        packet_loss_rate = profile_data["packet_loss"]
        aprint(f"📊 [Luxar] Using network profile: {profile_data['name']}")
    except ValueError as e:
        aprint(f"❌ [Luxar] {e}")
        raise typer.Exit(code=1)

# 2. Override with individual parameters
if bandwidth:
    try:
        bandwidth_mbps = parse_bandwidth(bandwidth)
    except ValueError as e:
        aprint(f"❌ [Luxar] {e}")
        raise typer.Exit(code=1)

if latency:
    try:
        latency_ms = parse_latency(latency)
    except ValueError as e:
        aprint(f"❌ [Luxar] {e}")
        raise typer.Exit(code=1)

if jitter:
    try:
        jitter_percent = parse_jitter(jitter)
    except ValueError as e:
        aprint(f"❌ [Luxar] {e}")
        raise typer.Exit(code=1)

if packet_loss:
    try:
        packet_loss_rate = parse_packet_loss(packet_loss)
    except ValueError as e:
        aprint(f"❌ [Luxar] {e}")
        raise typer.Exit(code=1)

# 3. Display simulation parameters (if any enabled)
if any([bandwidth_mbps, latency_ms, jitter_percent > 0, packet_loss_rate > 0]):
    aprint("🌐 [Luxar] Network simulation enabled:")
    if bandwidth_mbps:
        aprint(f"   • Bandwidth: {bandwidth_mbps:.2f} Mbps")
    if latency_ms:
        aprint(f"   • Latency: {latency_ms:.0f} ms")
    if jitter_percent > 0:
        aprint(f"   • Jitter: {jitter_percent*100:.0f}%")
    if packet_loss_rate > 0:
        aprint(f"   • Packet loss: {packet_loss_rate*100:.1f}%")
    aprint("⚠️  [Luxar] Responses will be throttled - this is intentional for testing")
```

### Help Text Enhancement

Add a new command to list available profiles:

```python
@app.command()
def profiles() -> None:
    """List available network simulation profiles.

    Display all preset network profiles with their parameters.
    """
    aprint("📊 [Luxar] Available Network Profiles\n")

    for profile_name in sorted(NETWORK_PROFILES.keys()):
        profile = NETWORK_PROFILES[profile_name]
        aprint(f"  {profile_name}")
        aprint(f"    Name: {profile['name']}")
        aprint(f"    Bandwidth: {profile['bandwidth']}")
        aprint(f"    Latency: {profile['latency']}")
        aprint(f"    Jitter: {profile['jitter']*100:.0f}%")
        aprint(f"    Packet Loss: {profile['packet_loss']*100:.1f}%")
        aprint(f"    Description: {profile['description']}")
        aprint("")

    aprint("Usage: luxar serve data.luxar.zarr --profile <profile-name>")
```

---

## Implementation Details

### File Structure

```
packages/luxar/src/luxar/cli/
├── __init__.py                    # Exports app
├── main.py                        # Commands (serve, viewer, demo, profiles)
├── utils.py                       # Existing utilities
├── network_simulation.py          # NEW: Network simulation components
│   ├── NetworkSimulationMiddleware
│   ├── NETWORK_PROFILES
│   ├── parse_bandwidth()
│   ├── parse_latency()
│   ├── parse_jitter()
│   ├── parse_packet_loss()
│   └── load_network_profile()
└── tests/
    ├── test_cli.py                # Existing tests
    ├── test_cli_utils.py          # Existing tests
    └── test_network_simulation.py # NEW: Network simulation tests
```

### Integration Points

**In `main.py`**:
```python
from .network_simulation import (
    NetworkSimulationMiddleware,
    NETWORK_PROFILES,
    parse_bandwidth,
    parse_latency,
    parse_jitter,
    parse_packet_loss,
    load_network_profile,
)

# In serve() function, CORRECT integration:
# (Based on existing code in the `serve()` function in main.py)

# 1. Create FastAPI app
api = FastAPI(title="Luxar static server", docs_url=None, redoc_url=None)

# 2. Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount static files handler
api.mount("/", DirectoryListingStaticFiles(directory=serve_path, html=True))

# 4. Wrap the complete ASGI app with network simulation (if enabled)
asgi_app = api
if any([bandwidth_mbps, latency_ms, jitter_percent > 0, packet_loss_rate > 0]):
    asgi_app = NetworkSimulationMiddleware(
        api,  # Wrap the complete FastAPI app
        bandwidth_limit_mbps=bandwidth_mbps,
        latency_ms=latency_ms,
        jitter_percent=jitter_percent,
        packet_loss_rate=packet_loss_rate,
    )

# 5. Run the (possibly wrapped) ASGI app
uvicorn.run(asgi_app, host=host, port=actual_port, reload=False, log_level="warning")
```

**Critical**: The middleware must wrap the complete FastAPI application AFTER all routes and middleware are added, but BEFORE calling `uvicorn.run()`. This is because:
- Routes and middleware are already mounted on the `api` object
- We cannot "reassign" the FastAPI app - we must wrap it
- `uvicorn.run()` receives the wrapped ASGI application
- The NetworkSimulationMiddleware is a pure ASGI middleware (not Starlette middleware)

### Dependencies

All required dependencies are already in the project:
- `asyncio`: Standard library (async/await support)
- `random`: Standard library (packet loss, jitter)
- `typer`: Already used for CLI
- `fastapi`: Already used for serving
- `uvicorn`: Already used for ASGI server

**No new dependencies required.**

---

## Error Handling

### Validation Errors

All parsing functions should raise `ValueError` with helpful messages:

```python
# Example error messages:
"Invalid bandwidth format: 'abc'. Expected format: <number><unit> where unit is kbps, mbps, or gbps. Examples: '1mbps', '500kbps', '2.5gbps'"

"Bandwidth must be positive, got: -1.0"

"Invalid latency format: '100'. Expected format: <number><unit> where unit is ms or s. Examples: '100ms', '1s', '0.5s'"

"Jitter must be between 0% and 100% (or 0.0 and 1.0), got: 150%"

"Unknown network profile: 'fast-5g'. Available profiles: 3g, 4g, 5g, broadband, satellite, rural, congested"
```

### CLI Error Display

Catch `ValueError` exceptions and display with arbol:

```python
try:
    bandwidth_mbps = parse_bandwidth(bandwidth)
except ValueError as e:
    aprint(f"❌ [Luxar] {e}")
    raise typer.Exit(code=1)
```

### Runtime Errors

- **Middleware errors**: Log and continue serving (don't crash server)
- **Negative latency after jitter**: Clamp to 0 (already handled in algorithm)
- **Division by zero**: Prevented by validation (bandwidth > 0)

---

## Testing Strategy

### Unit Tests

**File**: `packages/luxar/src/luxar/cli/tests/test_network_simulation.py`

**Test Coverage**:

1. **Bandwidth Parsing**:
   - Valid inputs (kbps, mbps, gbps)
   - Case insensitivity
   - Whitespace handling
   - Invalid inputs (no unit, negative, non-numeric)

2. **Latency Parsing**:
   - Valid inputs (ms, s)
   - Case insensitivity
   - Whitespace handling
   - Invalid inputs (no unit, negative, non-numeric)

3. **Jitter Parsing**:
   - Valid inputs (percentage, decimal)
   - Boundary values (0%, 100%)
   - Invalid inputs (negative, > 100%, non-numeric)

4. **Packet Loss Parsing**:
   - Valid inputs (percentage, decimal)
   - Boundary values (0%, 100%)
   - Invalid inputs (negative, > 100%, non-numeric)

5. **Profile Loading**:
   - Valid profile names
   - Invalid profile names
   - Case insensitivity

6. **Middleware Logic** (mock-based):
   - Packet loss rate verification (statistical test with many requests)
   - Latency addition (timing test)
   - Jitter range verification (statistical test)
   - Bandwidth throttling (timing test with known payload size)

**Important: Testing Tolerances**:
- Timing-based tests (bandwidth, latency) should use **generous tolerances** (±15-20%)
- OS scheduling, Python interpreter overhead, and network stack all add variability
- Statistical tests (packet loss, jitter) need large sample sizes (1000+ requests)
- Use `pytest.approx()` with appropriate `rel` tolerance for floating-point comparisons
- Timing tests may be flaky in CI environments - consider using `@pytest.mark.flaky` decorator

**Example Test**:
```python
def test_bandwidth_parsing():
    """Test bandwidth string parsing."""
    assert parse_bandwidth("1mbps") == 1.0
    assert parse_bandwidth("500kbps") == 0.5
    assert parse_bandwidth("2.5gbps") == 2500.0
    assert parse_bandwidth("  10MBPS  ") == 10.0

    with pytest.raises(ValueError, match="Invalid bandwidth format"):
        parse_bandwidth("100")  # No unit

    with pytest.raises(ValueError, match="must be positive"):
        parse_bandwidth("-1mbps")  # Negative

def test_packet_loss_simulation():
    """Test packet loss drops requests at expected rate."""
    # Statistical test: Run 1000 requests, expect ~1% dropped
    drops = 0
    for _ in range(1000):
        if random.random() < 0.01:  # 1% loss rate
            drops += 1

    # Allow ±50% deviation (0.5% to 1.5% is acceptable)
    assert 5 <= drops <= 15
```

### Integration Tests

**File**: `packages/luxar/src/luxar/cli/tests/test_cli_integration.py`

**Test Coverage**:

1. **CLI Flag Parsing**:
   - `luxar serve --profile 3g` loads correct parameters
   - Individual flags override profile defaults
   - Invalid profile name shows error and exits

2. **Middleware Application**:
   - Middleware is added when simulation enabled
   - Middleware is NOT added when simulation disabled
   - Multiple parameters work together

3. **End-to-End Timing** (requires actual server):
   - Start server with known bandwidth/latency
   - Make HTTP request for known file size
   - Verify response time matches expected (±10%)

**Example Test**:
```python
def test_serve_with_profile(cli_runner, tmp_path):
    """Test serve command with network profile."""
    # Create test zarr store
    zarr_path = tmp_path / "test.zarr"
    # ... create zarr ...

    # Run serve command with profile (non-blocking mode for testing)
    result = cli_runner.invoke(app, [
        "serve", str(zarr_path),
        "--profile", "3g",
        # ... additional test flags ...
    ])

    assert result.exit_code == 0
    assert "Network simulation enabled" in result.output
    assert "Bandwidth: 0.38 Mbps" in result.output  # 384 kbps = 0.384 Mbps
    assert "Latency: 300 ms" in result.output
```

### Manual Testing Checklist

- [ ] `luxar serve data.luxar.zarr --profile 3g --viewer` - loads slowly, UI responsive
- [ ] `luxar serve data.luxar.zarr --bandwidth 100kbps` - very slow loading
- [ ] `luxar serve data.luxar.zarr --latency 1s` - noticeable delay before each response
- [ ] `luxar serve data.luxar.zarr --packet-loss 10%` - occasional failed requests
- [ ] `luxar serve data.luxar.zarr --profile satellite --latency 1s` - override works
- [ ] `luxar profiles` - shows all profiles with parameters
- [ ] `luxar serve --profile invalid` - shows error with available profiles
- [ ] Browser DevTools Network tab shows throttled speeds matching settings

---

## Examples

### Example 1: Test 3G Mobile Performance
```bash
luxar serve examples/lorenz_attractor.luxar.zarr --profile 3g --viewer --open
```

**Expected Behavior**:
- Bandwidth limited to 384 kbps (~48 KB/s)
- 300ms latency before each response
- ±30ms jitter (10% of 300ms)
- ~1% of requests fail (packet loss)
- Viewer loads slowly but remains interactive

### Example 2: Test Satellite Connection
```bash
luxar demo --profile satellite --open
```

**Expected Behavior**:
- High bandwidth (25 Mbps) but very high latency (600ms)
- Noticeable delay between clicking and response
- Once chunks load, streaming is fast
- Tests if UI handles high-latency scenarios

### Example 3: Custom Worst-Case Scenario
```bash
luxar serve data.luxar.zarr --bandwidth 100kbps --latency 500ms --jitter 50% --packet-loss 5% --viewer
```

**Expected Behavior**:
- Very slow loading (100 kbps = 12.5 KB/s)
- High latency with extreme variation (250ms-750ms)
- Frequent request failures (5%)
- Tests viewer's resilience to poor conditions

### Example 4: Test Only Latency
```bash
luxar serve data.luxar.zarr --latency 200ms --jitter 10% --viewer
```

**Expected Behavior**:
- Full bandwidth available (no throttling)
- 200ms ±20ms delay before each response
- Tests if progressive loading handles latency well

### Example 5: Broadband Override
```bash
luxar serve data.luxar.zarr --profile broadband --packet-loss 2% --viewer
```

**Expected Behavior**:
- Uses broadband bandwidth (50 Mbps) and latency (20ms)
- Overrides packet loss to 2% (instead of 0.1%)
- Tests packet loss handling in otherwise good conditions

---

## Future Extensions

**Not in v1.0, but possible future additions:**

### 1. Per-Route Configuration
Allow different simulation parameters for different routes:
```python
network_rules = {
    "/data.zarr/.zmetadata": {"bandwidth": "10mbps"},  # Metadata loads fast
    "/data.zarr/points/0": {"bandwidth": "1mbps"},     # Point data loads slow
}
```

### 2. Dynamic Parameter Adjustment
Allow changing parameters without restarting server:
```python
# HTTP endpoint to update simulation
POST /api/network-simulation
{"bandwidth": "500kbps", "latency": "300ms"}
```

### 3. Time-Based Variation
Simulate network conditions that change over time:
```python
# Bandwidth varies between 1-5 Mbps over 60-second cycle
bandwidth_schedule = [(0, "1mbps"), (30, "5mbps"), (60, "1mbps")]
```

### 4. Response Size-Based Rules
Different throttling for small vs large responses:
```python
# Small responses (<1KB) bypass throttling
# Large responses (>100KB) get full throttling
```

### 5. Connection Type Detection
Suggest profile based on detected dataset characteristics:
```python
# If dataset is large (>1GB), suggest testing with slow profiles
luxar serve huge.luxar.zarr
# Suggestion: "Large dataset detected. Consider testing with --profile 3g"
```

### 6. Recording and Playback
Record real network conditions and replay them:
```bash
# Record network performance
luxar record-network --output network-trace.json

# Replay recorded trace
luxar serve data.luxar.zarr --replay network-trace.json
```

### 7. Statistics and Reporting
Generate report of simulated network performance:
```bash
luxar serve data.luxar.zarr --profile 3g --report network-stats.json
# After session, generates report:
# - Total bytes served
# - Number of requests
# - Packets dropped
# - Average response time
# - etc.
```

---

## Implementation Checklist

When implementing this specification:

- [ ] Create `network_simulation.py` module
- [ ] Implement `NetworkSimulationMiddleware` class
- [ ] Implement `parse_bandwidth()` function with tests
- [ ] Implement `parse_latency()` function with tests
- [ ] Implement `parse_jitter()` function with tests
- [ ] Implement `parse_packet_loss()` function with tests
- [ ] Define `NETWORK_PROFILES` constant
- [ ] Implement `load_network_profile()` function with tests
- [ ] Add CLI options to `serve` command
- [ ] Add CLI options to `viewer` command
- [ ] Add CLI options to `demo` command
- [ ] Implement `profiles` command
- [ ] Add parameter resolution logic
- [ ] Add simulation status display (arbol output)
- [ ] Write unit tests for all parsing functions
- [ ] Write unit tests for middleware logic
- [ ] Write integration tests for CLI
- [ ] Manual testing with browser DevTools
- [ ] Update `README.md` with network simulation examples
- [ ] Update `CLAUDE.md` with testing guidance
- [ ] Update CLI help text and docstrings

---

## Revision History

**v1.1.0** (2025-01-06):
- **CRITICAL FIX**: Corrected middleware integration method (wrap ASGI app before uvicorn.run)
- **CRITICAL FIX**: Updated bandwidth throttling to handle streaming responses correctly
- **MAJOR**: Clarified viewer command semantics (simulation only applies to data server)
- **MAJOR**: Clarified latency semantics (represents request propagation latency)
- Added prominent security warning (development/testing only)
- Added testing tolerance guidance (±15-20% for timing tests)
- Improved documentation with detailed rationale and examples
- Status changed to "Ready for Implementation"

**v1.0.0** (2025-01-06):
- Initial specification
- Core features: bandwidth throttling, latency, jitter, packet loss
- Connection profiles (9 presets)
- CLI integration for serve/viewer/demo commands
- Complete testing strategy
