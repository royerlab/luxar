"""Byte ranges in the exported folder's server (`_export_serve_template`).

A media element needs ranges to let go of a connection: without them a paused
`<video>` keeps its download open and the browser holds the socket, and an
HTTP/1.1 origin has six. These tests drive the real handler over a socket with
`http.client`, an independent client, so a parser and its test cannot agree on
a wrong answer.
"""

from __future__ import annotations

import http.client
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import pytest

from luxar.cli import _export_serve_template as template

PAYLOAD = bytes(range(256)) * 40  # 10,240 bytes, every offset distinguishable


@contextmanager
def serving(root: Path) -> Iterator[int]:
    handler = partial(template.LuxarHandler, directory=str(root))
    server = template.ControlServer(("127.0.0.1", 0), handler, None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def port(tmp_path: Path) -> Iterator[int]:
    (tmp_path / "clip.webm").write_bytes(PAYLOAD)
    (tmp_path / "sub").mkdir()
    with serving(tmp_path) as p:
        yield p


def get(
    port: int, path: str, headers: dict[str, str] | None = None, method: str = "GET"
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, headers=headers or {})
        r = conn.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("spec", "start", "end"),
    [
        ("bytes=0-99", 0, 99),
        ("bytes=100-", 100, len(PAYLOAD) - 1),
        ("bytes=-500", len(PAYLOAD) - 500, len(PAYLOAD) - 1),
        ("bytes=10000-99999", 10000, len(PAYLOAD) - 1),  # end clamped to the file
        ("bytes=-999999", 0, len(PAYLOAD) - 1),  # suffix longer than the file
        (" bytes = 7-7 ", 7, 7),
    ],
)
def test_a_range_is_served_as_206_with_exactly_those_bytes(
    port: int, spec: str, start: int, end: int
) -> None:
    status, headers, body = get(port, "/clip.webm", {"Range": spec})
    assert status == 206
    assert body == PAYLOAD[start : end + 1]
    assert headers["content-range"] == f"bytes {start}-{end}/{len(PAYLOAD)}"
    assert headers["content-length"] == str(end - start + 1)
    assert headers["accept-ranges"] == "bytes"
    assert headers["content-type"] == "video/webm"


def test_a_plain_request_advertises_ranges_and_gets_the_whole_file(port: int) -> None:
    status, headers, body = get(port, "/clip.webm")
    assert status == 200
    assert body == PAYLOAD
    assert headers["accept-ranges"] == "bytes"


def test_a_range_past_the_end_is_416_with_the_size(port: int) -> None:
    status, headers, body = get(port, "/clip.webm", {"Range": f"bytes={len(PAYLOAD)}-"})
    assert status == 416
    assert headers["content-range"] == f"bytes */{len(PAYLOAD)}"
    assert body == b""
    assert get(port, "/clip.webm", {"Range": "bytes=-0"})[0] == 416


@pytest.mark.parametrize(
    "spec", ["bytes=0-1,5-9", "items=0-9", "bytes=9-3", "bytes=abc", "bytes=5"]
)
def test_what_a_server_may_ignore_gets_the_whole_file(port: int, spec: str) -> None:
    status, _, body = get(port, "/clip.webm", {"Range": spec})
    assert status == 200
    assert body == PAYLOAD


def test_if_range_serves_the_range_only_for_an_unchanged_file(
    port: int, tmp_path: Path
) -> None:
    _, headers, _ = get(port, "/clip.webm", method="HEAD")
    current = headers["last-modified"]
    status, _, body = get(
        port, "/clip.webm", {"Range": "bytes=0-9", "If-Range": current}
    )
    assert (status, body) == (206, PAYLOAD[:10])
    stale = "Mon, 01 Jan 2001 00:00:00 GMT"
    status, _, body = get(port, "/clip.webm", {"Range": "bytes=0-9", "If-Range": stale})
    assert (status, body) == (200, PAYLOAD)


def test_head_with_a_range_sends_the_headers_and_no_body(port: int) -> None:
    status, headers, body = get(
        port, "/clip.webm", {"Range": "bytes=0-9"}, method="HEAD"
    )
    assert status == 206
    assert headers["content-length"] == "10"
    assert body == b""


def test_a_keepalive_connection_survives_a_ranged_response(port: int) -> None:
    """HTTP/1.1 reuse: the next request on the socket must start at a boundary."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/clip.webm", headers={"Range": "bytes=0-9"})
        first = conn.getresponse().read()
        conn.request("GET", "/clip.webm", headers={"Range": "bytes=10-19"})
        second = conn.getresponse().read()
    finally:
        conn.close()
    assert first + second == PAYLOAD[:20]


def test_directories_and_missing_files_behave_as_before(port: int) -> None:
    assert get(port, "/missing.webm", {"Range": "bytes=0-9"})[0] == 404
    status, headers, _ = get(port, "/sub/")
    assert status == 200
    assert "accept-ranges" not in headers


@pytest.mark.parametrize("headers", [{}, {"Range": "bytes=0-9"}])
def test_file_path_with_trailing_slash_is_not_a_file(
    port: int, headers: dict[str, str]
) -> None:
    status, response_headers, _ = get(port, "/clip.webm/", headers)
    assert status == 404
    assert "accept-ranges" not in response_headers


def test_parse_byte_range_on_an_empty_file() -> None:
    assert template.parse_byte_range("bytes=0-", 0) == template.UNSATISFIABLE
    assert template.parse_byte_range("bytes=-5", 0) == template.UNSATISFIABLE
