"""A LAN bind must serve its own viewer — and nothing it was not asked to.

`luxar serve --viewer` runs the data server and the viewer on two ports of one
host and derives the viewer's ``?src=`` from the address it bound. So on a
non-loopback bind the browser fetches ``http://<lan>:8000`` from a page on
``http://<lan>:5173`` — a cross-origin pair the loopback-only default refuses,
which broke the whole LAN kiosk story (blank viewer, console full of CORS
errors, data perfectly readable by ``curl``).

These tests pin the fix in both directions: the pairing works on any address
the request actually arrived on, a loopback bind keeps its historical
strictness, and a malformed header fails closed rather than crashing or
widening anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ..serving import _build_data_app
from ..utils import (
    ALL_INTERFACES_HOSTS,
    advertised_host,
    authority_hostname,
    is_loopback_host,
    origin_authority,
    origin_hostname,
    primary_lan_address,
)


@pytest.fixture
def served(tmp_path: Path) -> Path:
    """A directory with one readable file in it."""
    (tmp_path / "probe.txt").write_text("payload")
    return tmp_path


def _allowed(served: Path, *, bind: str, host: str, origin: str) -> bool:
    """True when the data server would let ``origin`` read it over ``host``."""
    client = TestClient(_build_data_app(served, bind_host=bind))
    response = client.get("/probe.txt", headers={"Host": host, "Origin": origin})
    return response.headers.get("access-control-allow-origin") == origin


def _preflight_allowed(served: Path, *, bind: str, host: str, origin: str) -> bool:
    """The same question asked as a preflight, which must always agree."""
    client = TestClient(_build_data_app(served, bind_host=bind))
    response = client.options(
        "/probe.txt",
        headers={
            "Host": host,
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    return response.headers.get("access-control-allow-origin") == origin


class TestTheKioskPairing:
    """The case the feature exists for: a tablet and a display on one LAN."""

    def test_the_viewer_may_read_the_data_server_on_a_lan_bind(
        self, served: Path
    ) -> None:
        assert _allowed(
            served,
            bind="0.0.0.0",  # nosec B104 - a test argument, not a bind
            host="10.0.0.55:8000",
            origin="http://10.0.0.55:5173",
        )

    def test_a_sibling_local_address_is_allowed_too(self, served: Path) -> None:
        """A machine can answer a wildcard bind on several addresses.

        This is the case that killed the first attempt at this fix, which
        resolved the bind to ONE address and baked it into the allow-regex: a
        development Mac with Wi-Fi on 10.0.0.55 and Ethernet on 10.0.0.146
        would then refuse whichever one the route probe did not name. The
        check compares against the ``Host`` that actually arrived, so both
        work and no address has to be guessed.
        """
        for address in ("10.0.0.55", "10.0.0.146"):
            assert _allowed(
                served,
                bind="0.0.0.0",  # nosec B104 - a test argument, not a bind
                host=f"{address}:8000",
                origin=f"http://{address}:5173",
            ), address

    def test_a_hostname_bind_works_the_same_way(self, served: Path) -> None:
        assert _allowed(
            served,
            bind="kiosk.local",
            host="kiosk.local:8000",
            origin="http://kiosk.local:5173",
        )

    @pytest.mark.parametrize(
        ("host", "origin"),
        [
            ("10.0.0.55:8000", "http://10.0.0.55:5173"),
            ("[::1]:8000", "http://[::1]:5173"),
        ],
    )
    def test_preflight_agrees_with_the_simple_response(
        self, served: Path, host: str, origin: str
    ) -> None:
        """A disagreement here would break POST/HEAD while GET looked fine."""
        bind = "0.0.0.0"  # nosec B104 - a test argument, not a bind
        assert _allowed(served, bind=bind, host=host, origin=origin) is True
        assert _preflight_allowed(served, bind=bind, host=host, origin=origin) is True


class TestTheAllowanceIsNarrow:
    """Same host, any port — and nothing beyond that."""

    @pytest.mark.parametrize(
        "origin",
        [
            "http://10.0.0.146:5173",  # a different local address
            "http://evil.example:5173",
            "http://10-0-0-55.example:5173",  # host merely CONTAINS the address
            "null",  # a sandboxed iframe
            "http://10.0.0.55:5173/path",  # not an origin: carries a path
            "http://10.0.0.55:5173#frag",
            "http://user@10.0.0.55:5173",  # userinfo
            "ws://10.0.0.55:5173",  # not an http(s) origin
            "http://10.0.0.55:5173\r\nX-Injected: 1",
        ],
    )
    def test_a_foreign_or_malformed_origin_is_refused(
        self, served: Path, origin: str
    ) -> None:
        """Each of these would be REFLECTED in the response header if allowed.

        The last one is why the shape is matched rather than loosely parsed:
        Starlette echoes the raw ``Origin`` back, so accepting a value with an
        embedded CRLF would hand an attacker a response-header injection.
        """
        assert not _allowed(
            served,
            bind="0.0.0.0",  # nosec B104 - a test argument, not a bind
            host="10.0.0.55:8000",
            origin=origin,
        )

    def test_a_loopback_bind_keeps_its_historical_strictness(
        self, served: Path
    ) -> None:
        """Binding loopback exposes nothing, so it gains no new allowance."""
        assert not _allowed(
            served,
            bind="127.0.0.1",
            host="10.0.0.55:8000",
            origin="http://10.0.0.55:5173",
        )
        assert _allowed(
            served,
            bind="127.0.0.1",
            host="127.0.0.1:8000",
            origin="http://localhost:5173",
        )

    def test_an_unspecified_bind_is_treated_as_loopback(self, served: Path) -> None:
        """A caller that never says where it bound gets the strict default."""
        assert not _allowed(
            served,
            bind=None,  # type: ignore[arg-type]
            host="10.0.0.55:8000",
            origin="http://10.0.0.55:5173",
        )

    @pytest.mark.parametrize(
        "host",
        [
            "[::1",  # unterminated bracket — raised ValueError before
            "host:notaport",
            "10.0.0.55:8000, evil.example",  # a smuggled second host
            "",
            " ",
        ],
    )
    def test_a_malformed_host_header_fails_closed(
        self, served: Path, host: str
    ) -> None:
        """Never a guess, never a crash — no match at all."""
        client = TestClient(
            _build_data_app(served, bind_host="0.0.0.0")  # nosec B104
        )
        response = client.get(
            "/probe.txt",
            headers={"Host": host, "Origin": "http://10.0.0.55:5173"},
        )
        assert response.status_code < 500
        assert "access-control-allow-origin" not in response.headers


class TestOriginParsing:
    """The strict parsers both the hub and the CORS layer are built on."""

    def test_an_unterminated_bracket_does_not_raise(self) -> None:
        """`urlparse(...).hostname` AND `.netloc` raise ValueError on this.

        An unauthenticated client can send it, and in the control hub the
        parse runs on the handshake *before* the token check — so a throw
        there is a remote crash of the socket handler.
        """
        assert origin_hostname("http://[::1") == ""
        assert origin_authority("http://[::1") == ""
        assert authority_hostname("[::1") == ""

    @pytest.mark.parametrize(
        ("origin", "hostname", "authority"),
        [
            ("http://kiosk.local:5173", "kiosk.local", "kiosk.local:5173"),
            ("HTTPS://Kiosk.Local:5173", "kiosk.local", "kiosk.local:5173"),
            ("http://kiosk.local", "kiosk.local", "kiosk.local"),
            ("http://[::1]:5173", "::1", "[::1]:5173"),
            ("http://127.0.0.1:8000", "127.0.0.1", "127.0.0.1:8000"),
            ("null", "", ""),
            ("", "", ""),
            ("not a url", "", ""),
        ],
    )
    def test_the_two_readings_of_one_origin(
        self, origin: str, hostname: str, authority: str
    ) -> None:
        """CORS wants the host alone; the control hub wants the port as well.

        The hub's socket is served from the very port that served the page, so
        a differing port there is a real mismatch. The data server is a
        different port by construction, so including it would reject every
        legitimate pairing. Two functions, deliberately.
        """
        assert origin_hostname(origin) == hostname
        assert origin_authority(origin) == authority


class TestAdvertisedHost:
    """A wildcard bind is not a destination, so a printed URL must resolve it."""

    def test_a_concrete_bind_is_returned_unchanged(self) -> None:
        for host in ("127.0.0.1", "localhost", "10.0.0.55", "kiosk.local"):
            assert advertised_host(host) == host

    def test_a_wildcard_bind_resolves_to_something_dialable(self) -> None:
        """Whatever it picks, it must not be the sentinel itself.

        The exact address is the machine's, so it is deliberately not asserted
        — only that `http://<result>:5173` is a URL a tablet could try.
        """
        for sentinel in ("0.0.0.0", "::"):  # nosec B104 - test arguments
            resolved = advertised_host(sentinel)
            assert resolved not in ALL_INTERFACES_HOSTS
            assert resolved

    def test_it_falls_back_to_loopback_with_no_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a machine with no route, loopback is the only thing that works."""
        monkeypatch.setattr(
            "luxar.cli.utils.primary_lan_address", lambda: None, raising=True
        )
        assert advertised_host("0.0.0.0") == "127.0.0.1"  # nosec B104

    def test_the_probe_sends_nothing_and_cannot_raise(self) -> None:
        """`connect()` on a datagram socket only fixes a route.

        The peer is an RFC 5737 documentation address that is never routed, so
        this is free; a machine without a route returns None rather than
        throwing.
        """
        result = primary_lan_address()
        assert result is None or result.count(".") == 3


class TestLoopbackHostPredicate:
    @pytest.mark.parametrize(
        ("bind", "expected"),
        [
            (None, True),
            ("127.0.0.1", True),
            ("localhost", True),
            ("::1", True),
            ("[::1]", True),
            ("0.0.0.0", False),  # nosec B104 - a test argument, not a bind
            ("::", False),
            ("10.0.0.55", False),
            ("kiosk.local", False),
        ],
    )
    def test_only_genuine_loopback_counts(self, bind: str, expected: bool) -> None:
        assert is_loopback_host(bind) is expected
