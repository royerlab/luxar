"""Tests for network simulation middleware and utilities."""

from __future__ import annotations

import asyncio
import time

import pytest

from luxar.cli.network_simulation import (
    NETWORK_PROFILES,
    NetworkSimulationMiddleware,
    load_network_profile,
    parse_bandwidth,
    parse_jitter,
    parse_latency,
    parse_packet_loss,
)


class TestBandwidthParsing:
    """Test bandwidth string parsing."""

    def test_parse_mbps(self):
        """Test parsing megabits per second."""
        assert parse_bandwidth("1mbps") == 1.0
        assert parse_bandwidth("10mbps") == 10.0
        assert parse_bandwidth("2.5mbps") == 2.5

    def test_parse_kbps(self):
        """Test parsing kilobits per second."""
        assert parse_bandwidth("500kbps") == 0.5
        assert parse_bandwidth("1000kbps") == 1.0
        assert parse_bandwidth("384kbps") == 0.384

    def test_parse_gbps(self):
        """Test parsing gigabits per second."""
        assert parse_bandwidth("1gbps") == 1000.0
        assert parse_bandwidth("2.5gbps") == 2500.0

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert parse_bandwidth("1MBPS") == 1.0
        assert parse_bandwidth("500KBPS") == 0.5
        assert parse_bandwidth("1Gbps") == 1000.0

    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert parse_bandwidth("  1mbps  ") == 1.0
        assert parse_bandwidth("\t500kbps\n") == 0.5

    def test_invalid_format(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid bandwidth format"):
            parse_bandwidth("500")  # No unit

        with pytest.raises(ValueError, match="Invalid bandwidth format"):
            parse_bandwidth("abc")  # Not a number

        with pytest.raises(ValueError, match="Invalid bandwidth format"):
            parse_bandwidth("500mb")  # Wrong unit

    def test_negative_value(self):
        """Test negative value raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            parse_bandwidth("-1mbps")

    def test_zero_value(self):
        """Test zero value raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            parse_bandwidth("0mbps")


class TestLatencyParsing:
    """Test latency string parsing."""

    def test_parse_milliseconds(self):
        """Test parsing milliseconds."""
        assert parse_latency("100ms") == 100.0
        assert parse_latency("500ms") == 500.0
        assert parse_latency("1.5ms") == 1.5

    def test_parse_seconds(self):
        """Test parsing seconds."""
        assert parse_latency("1s") == 1000.0
        assert parse_latency("0.5s") == 500.0
        assert parse_latency("2.5s") == 2500.0

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert parse_latency("100MS") == 100.0
        assert parse_latency("1S") == 1000.0

    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert parse_latency("  100ms  ") == 100.0
        assert parse_latency("\t1s\n") == 1000.0

    def test_invalid_format(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid latency format"):
            parse_latency("100")  # No unit

        with pytest.raises(ValueError, match="Invalid latency format"):
            parse_latency("abc")  # Not a number

    def test_negative_value(self):
        """Test negative value raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            parse_latency("-100ms")

    def test_zero_value(self):
        """Test zero value is allowed."""
        assert parse_latency("0ms") == 0.0


class TestJitterParsing:
    """Test jitter string parsing."""

    def test_parse_percentage(self):
        """Test parsing percentage format."""
        assert parse_jitter("10%") == 0.1
        assert parse_jitter("25%") == 0.25
        assert parse_jitter("100%") == 1.0
        assert parse_jitter("0%") == 0.0

    def test_parse_decimal(self):
        """Test parsing decimal format."""
        assert parse_jitter("0.1") == 0.1
        assert parse_jitter("0.25") == 0.25
        assert parse_jitter("1.0") == 1.0
        assert parse_jitter("0") == 0.0

    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert parse_jitter("  10%  ") == 0.1
        assert parse_jitter("\t0.1\n") == 0.1

    def test_invalid_format(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="could not convert string to float"):
            parse_jitter("abc")

    def test_out_of_range(self):
        """Test out of range values raise ValueError."""
        with pytest.raises(ValueError, match="must be between"):
            parse_jitter("150%")  # > 100%

        with pytest.raises(ValueError, match="must be between"):
            parse_jitter("-10%")  # < 0%

        with pytest.raises(ValueError, match="must be between"):
            parse_jitter("1.5")  # > 1.0

        with pytest.raises(ValueError, match="must be between"):
            parse_jitter("-0.1")  # < 0.0


class TestPacketLossParsing:
    """Test packet loss string parsing."""

    def test_parse_percentage(self):
        """Test parsing percentage format."""
        assert parse_packet_loss("1%") == 0.01
        assert parse_packet_loss("5%") == 0.05
        assert parse_packet_loss("10%") == 0.1
        assert parse_packet_loss("0%") == 0.0

    def test_parse_decimal(self):
        """Test parsing decimal format."""
        assert parse_packet_loss("0.01") == 0.01
        assert parse_packet_loss("0.05") == 0.05
        assert parse_packet_loss("0.1") == 0.1
        assert parse_packet_loss("0") == 0.0

    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert parse_packet_loss("  1%  ") == 0.01
        assert parse_packet_loss("\t0.01\n") == 0.01

    def test_invalid_format(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="could not convert string to float"):
            parse_packet_loss("abc")

    def test_out_of_range(self):
        """Test out of range values raise ValueError."""
        with pytest.raises(ValueError, match="must be between"):
            parse_packet_loss("150%")  # > 100%

        with pytest.raises(ValueError, match="must be between"):
            parse_packet_loss("-5%")  # < 0%

        with pytest.raises(ValueError, match="must be between"):
            parse_packet_loss("1.5")  # > 1.0

        with pytest.raises(ValueError, match="must be between"):
            parse_packet_loss("-0.1")  # < 0.0


class TestProfileLoading:
    """Test network profile loading."""

    def test_load_valid_profile(self):
        """Test loading a valid profile."""
        profile = load_network_profile("3g")
        assert profile["name"] == "3G Mobile"
        assert profile["bandwidth"] == "384kbps"
        assert profile["latency"] == "300ms"
        assert profile["jitter"] == 0.1
        assert profile["packet_loss"] == 0.01

    def test_all_profiles_exist(self):
        """Test all documented profiles exist."""
        expected_profiles = [
            "3g",
            "4g",
            "5g",
            "slow-broadband",
            "broadband",
            "fast-broadband",
            "satellite",
            "rural",
            "congested",
        ]
        for profile_name in expected_profiles:
            profile = load_network_profile(profile_name)
            assert "name" in profile
            assert "bandwidth" in profile
            assert "latency" in profile
            assert "jitter" in profile
            assert "packet_loss" in profile
            assert "description" in profile

    def test_case_insensitive(self):
        """Test profile names are case insensitive."""
        profile1 = load_network_profile("3g")
        profile2 = load_network_profile("3G")
        profile3 = load_network_profile("3G")
        assert profile1 == profile2 == profile3

    def test_whitespace_handling(self):
        """Test whitespace is stripped from profile names."""
        profile1 = load_network_profile("3g")
        profile2 = load_network_profile("  3g  ")
        assert profile1 == profile2

    def test_invalid_profile(self):
        """Test invalid profile name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown network profile"):
            load_network_profile("invalid-profile")

        with pytest.raises(ValueError, match="Available profiles"):
            load_network_profile("fast-5g")

    def test_profile_copy(self):
        """Test profile returns a copy, not reference."""
        profile1 = load_network_profile("3g")
        profile2 = load_network_profile("3g")
        profile1["bandwidth"] = "999mbps"  # Modify copy
        assert profile2["bandwidth"] == "384kbps"  # Original unchanged


class TestProfileValues:
    """Test network profile values are realistic."""

    def test_3g_profile(self):
        """Test 3G profile has realistic values."""
        profile = load_network_profile("3g")
        assert parse_bandwidth(profile["bandwidth"]) == 0.384  # 384 kbps
        assert parse_latency(profile["latency"]) == 300.0  # 300 ms
        assert 0.0 <= profile["jitter"] <= 1.0
        assert 0.0 <= profile["packet_loss"] <= 1.0

    def test_5g_profile(self):
        """Test 5G profile has realistic values."""
        profile = load_network_profile("5g")
        bandwidth = parse_bandwidth(profile["bandwidth"])
        latency = parse_latency(profile["latency"])
        assert bandwidth >= 50.0  # 5G should be fast
        assert latency <= 50.0  # 5G should have low latency

    def test_satellite_profile(self):
        """Test satellite profile has high latency."""
        profile = load_network_profile("satellite")
        latency = parse_latency(profile["latency"])
        assert latency >= 500.0  # Satellite should have very high latency


class TestNetworkProfilesConstant:
    """Test the NETWORK_PROFILES constant."""

    def test_profiles_dict_exists(self):
        """Test NETWORK_PROFILES dictionary exists."""
        assert isinstance(NETWORK_PROFILES, dict)
        assert len(NETWORK_PROFILES) > 0

    def test_all_profiles_have_required_keys(self):
        """Test all profiles have required keys."""
        required_keys = {
            "name",
            "bandwidth",
            "latency",
            "jitter",
            "packet_loss",
            "description",
        }
        for profile_name, profile in NETWORK_PROFILES.items():
            assert required_keys.issubset(profile.keys()), (
                f"Profile {profile_name} missing keys"
            )

    def test_profile_values_are_valid(self):
        """Test all profile values can be parsed."""
        for profile_name, profile in NETWORK_PROFILES.items():
            # Test bandwidth can be parsed
            bandwidth = parse_bandwidth(profile["bandwidth"])
            assert bandwidth > 0, f"Profile {profile_name} has invalid bandwidth"

            # Test latency can be parsed
            latency = parse_latency(profile["latency"])
            assert latency >= 0, f"Profile {profile_name} has invalid latency"

            # Test jitter is in range
            assert 0.0 <= profile["jitter"] <= 1.0, (
                f"Profile {profile_name} has invalid jitter"
            )

            # Test packet loss is in range
            assert 0.0 <= profile["packet_loss"] <= 1.0, (
                f"Profile {profile_name} has invalid packet_loss"
            )


class TestNetworkSimulationMiddleware:
    """Test the NetworkSimulationMiddleware class."""

    def test_middleware_initialization(self):
        """Test middleware initializes correctly."""

        async def dummy_app(scope, receive, send):
            """Dummy ASGI app."""
            pass

        middleware = NetworkSimulationMiddleware(
            dummy_app,
            bandwidth_limit_mbps=1.0,
            latency_ms=100.0,
            jitter_percent=0.1,
            packet_loss_rate=0.01,
        )

        assert middleware.bandwidth_limit_mbps == 1.0
        assert middleware.latency_ms == 100.0
        assert middleware.jitter_percent == 0.1
        assert middleware.packet_loss_rate == 0.01
        assert middleware.bytes_per_second == 1_000_000 / 8  # 1 Mbps = 125 KB/s

    def test_middleware_passes_through_non_http(self):
        """Test middleware passes through non-HTTP requests."""
        called = False

        async def dummy_app(scope, receive, send):
            nonlocal called
            called = True

        async def test_async():
            middleware = NetworkSimulationMiddleware(dummy_app, latency_ms=100.0)
            await middleware({"type": "websocket"}, None, None)
            return called

        result = asyncio.run(test_async())
        assert result  # Should pass through to app

    def test_latency_adds_delay(self):
        """Test that latency actually delays requests."""

        async def dummy_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"test"})

        async def test_async():
            middleware = NetworkSimulationMiddleware(dummy_app, latency_ms=100.0)

            scope = {"type": "http"}
            messages = []

            async def send(message):
                messages.append(message)

            async def receive():
                return {}

            start = time.time()
            await middleware(scope, receive, send)
            elapsed = time.time() - start
            return elapsed, len(messages)

        elapsed, message_count = asyncio.run(test_async())

        # Should have at least 100ms delay (allow some overhead)
        assert elapsed >= 0.09  # 90ms (allow 10% tolerance)
        assert message_count == 2  # Should have sent response

    @pytest.mark.skip(reason="Timing-sensitive test flaky on CI runners")
    def test_bandwidth_throttling(self):
        """Test that bandwidth throttling works."""

        async def dummy_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            # Send 10KB of data
            await send({"type": "http.response.body", "body": b"x" * 10000})

        async def test_async():
            # 1 Mbps = 125 KB/s, so 10KB should take ~0.08s
            middleware = NetworkSimulationMiddleware(
                dummy_app, bandwidth_limit_mbps=1.0
            )

            scope = {"type": "http"}
            messages = []

            async def send(message):
                messages.append(message)

            async def receive():
                return {}

            start = time.time()
            await middleware(scope, receive, send)
            elapsed = time.time() - start
            return elapsed, len(messages)

        elapsed, message_count = asyncio.run(test_async())

        # 10KB at 125 KB/s = 0.08s, allow generous tolerance
        assert elapsed >= 0.06  # At least 60ms (75% of expected)
        assert elapsed <= 0.15  # At most 150ms (allow overhead)
        assert message_count == 2

    def test_packet_loss_drops_requests(self):
        """Test packet loss actually drops requests (statistical test)."""

        async def test_async():
            call_count = 0

            async def dummy_app(scope, receive, send):
                nonlocal call_count
                call_count += 1
                await send(
                    {"type": "http.response.start", "status": 200, "headers": []}
                )
                await send({"type": "http.response.body", "body": b"test"})

            # 50% packet loss for easier statistical testing
            middleware = NetworkSimulationMiddleware(dummy_app, packet_loss_rate=0.5)

            scope = {"type": "http"}

            async def send(message):
                pass

            async def receive():
                return {}

            # Run many times to test statistically
            trials = 100
            for _ in range(trials):
                await middleware(scope, receive, send)

            return call_count

        call_count = asyncio.run(test_async())

        # Should have dropped approximately 50% (allow 30-70%)
        assert 30 <= call_count <= 70, f"Expected ~50 calls, got {call_count}"


class TestParseNetworkOptions:
    """Tests for the parse_network_options shared helper (#10)."""

    def test_no_params_returns_defaults(self):
        """All None inputs return zeros/None."""
        from luxar.cli.network_simulation import parse_network_options

        bw, lat, jit, pl = parse_network_options()
        assert bw is None
        assert lat is None
        assert jit == 0.0
        assert pl == 0.0

    def test_profile_loads_all_params(self):
        """Loading a profile populates all four values."""
        from luxar.cli.network_simulation import parse_network_options

        bw, lat, jit, pl = parse_network_options(profile="3g")
        assert bw is not None and bw > 0
        assert lat is not None and lat > 0
        assert jit > 0
        assert pl > 0

    def test_individual_overrides_profile(self):
        """Individual params override profile values."""
        from luxar.cli.network_simulation import parse_network_options

        bw, lat, jit, pl = parse_network_options(
            profile="3g", bandwidth="100mbps", latency="5ms"
        )
        assert bw == 100.0
        assert lat == 5.0
        # jitter and packet_loss come from 3g profile
        assert jit > 0
        assert pl > 0

    def test_invalid_bandwidth_raises(self):
        """Invalid bandwidth string raises ValueError."""
        from luxar.cli.network_simulation import parse_network_options

        with pytest.raises(ValueError):
            parse_network_options(bandwidth="invalid")

    def test_invalid_profile_raises(self):
        """Invalid profile name raises ValueError."""
        from luxar.cli.network_simulation import parse_network_options

        with pytest.raises(ValueError):
            parse_network_options(profile="nonexistent")


class TestHasNetworkSimulation:
    """Tests for has_network_simulation helper."""

    def test_all_defaults_is_false(self):
        from luxar.cli.network_simulation import has_network_simulation

        assert has_network_simulation(None, None, 0.0, 0.0) is False

    def test_bandwidth_alone_is_true(self):
        from luxar.cli.network_simulation import has_network_simulation

        assert has_network_simulation(10.0, None, 0.0, 0.0) is True

    def test_latency_alone_is_true(self):
        from luxar.cli.network_simulation import has_network_simulation

        assert has_network_simulation(None, 100.0, 0.0, 0.0) is True
