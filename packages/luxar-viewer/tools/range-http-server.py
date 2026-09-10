#!/usr/bin/env python3
"""Static file server that actually honours HTTP ``Range`` requests.

Python's ``http.server`` does NOT: it ignores ``Range`` entirely and answers
``200`` with the whole body. That is fine for a directory-backed zarr store,
where every chunk is its own file, and fatal for a zipped one, where the reader
asks for byte windows inside a single archive and would silently receive the
whole file in place of each window.

The E2E and zip-benchmark harnesses use this to serve ``.zarr.zip`` fixtures to
a browser. It speaks exactly as much of RFC 9110 §14 as a zip reader needs: a
single ``bytes=`` range, answered ``206`` with ``Content-Range``; ``416`` when
unsatisfiable; plain ``200`` otherwise.

The private ``/__luxar_slow_wave__`` endpoint supports the hermetic slow-link
Playwright regression: it sends every response's headers immediately, streams
the first body for 8.5 seconds, then releases the remaining bodies.

CORS is wide open and ``Accept-Ranges``/``Content-Range`` are exposed, because
the viewer runs on the Vite port and the data on this one.

Usage::

    python3 tools/range-http-server.py <port> [--bind 127.0.0.1] [--directory .]
"""

from __future__ import annotations

import argparse
import os
import re
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import BinaryIO
from urllib.parse import parse_qs, urlsplit

#: ``bytes=<start>-<end>`` / ``bytes=<start>-`` / ``bytes=-<suffix>``
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_SLOW_WAVE_PATH = "/__luxar_slow_wave__"
_SLOW_WAVE_CHUNK = b"x" * 256
_SLOW_WAVE_CHUNKS = 17
_SLOW_WAVE_INTERVAL_SECONDS = 0.5


def parse_byte_range(header: str, size: int) -> tuple[int, int] | None:
    """Resolve a ``Range`` header against a known file ``size``.

    Returns an inclusive ``(start, end)`` pair, or ``None`` when the header is
    malformed, asks for multiple ranges, or cannot be satisfied.
    """
    match = _RANGE_RE.match(header.strip())
    if not match:
        return None
    raw_start, raw_end = match.group(1), match.group(2)

    if not raw_start:
        # Suffix range: the LAST n bytes. This is the one a zip reader issues
        # first, to find the end-of-central-directory record.
        if not raw_end:
            return None
        suffix = int(raw_end)
        if suffix == 0 or size == 0:
            return None
        return max(0, size - suffix), size - 1

    start = int(raw_start)
    if start >= size:
        return None
    end = int(raw_end) if raw_end else size - 1
    return start, min(end, size - 1)


class _RangeFile:
    """Reader that stops after ``remaining`` bytes, for ``copyfile`` to drain."""

    def __init__(self, handle: BinaryIO, remaining: int) -> None:
        self._handle = handle
        self._remaining = remaining

    def read(self, amount: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        want = (
            self._remaining
            if amount is None or amount < 0
            else min(amount, self._remaining)
        )
        chunk = self._handle.read(want)
        self._remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self._handle.close()


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """``SimpleHTTPRequestHandler`` plus single-range support and CORS."""

    protocol_version = "HTTP/1.1"
    _slow_wave_lock = Lock()
    _slow_wave_leaders: set[str] = set()

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Expose-Headers",
            "Accept-Ranges, Content-Range, Content-Length, ETag, Last-Modified",
        )
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib naming
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlsplit(self.path)
        if parsed.path != _SLOW_WAVE_PATH:
            super().do_GET()
            return

        wave_id = parse_qs(parsed.query).get("slow-link-wave", ["default"])[0]
        with self._slow_wave_lock:
            is_leader = wave_id not in self._slow_wave_leaders
            if is_leader:
                self._slow_wave_leaders.add(wave_id)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header(
            "Content-Length", str(len(_SLOW_WAVE_CHUNK) * _SLOW_WAVE_CHUNKS)
        )
        self.end_headers()
        self.wfile.flush()
        try:
            if is_leader:
                for _ in range(_SLOW_WAVE_CHUNKS):
                    self.wfile.write(_SLOW_WAVE_CHUNK)
                    self.wfile.flush()
                    time.sleep(_SLOW_WAVE_INTERVAL_SECONDS)
            else:
                time.sleep(_SLOW_WAVE_CHUNKS * _SLOW_WAVE_INTERVAL_SECONDS)
                self.wfile.write(_SLOW_WAVE_CHUNK * _SLOW_WAVE_CHUNKS)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_head(self):  # type: ignore[override]
        header = self.headers.get("Range")
        if header is None:
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            handle = open(path, "rb")  # noqa: SIM115 - closed by copyfile/_RangeFile
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(handle.fileno()).st_size
        resolved = parse_byte_range(header, size)
        if resolved is None:
            handle.close()
            self.send_response(416, "Requested Range Not Satisfiable")
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        start, end = resolved
        length = end - start + 1
        self.send_response(206, "Partial Content")
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        handle.seek(start)
        return _RangeFile(handle, length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--directory", default=os.getcwd())
    args = parser.parse_args()

    handler = partial(RangeHTTPRequestHandler, directory=args.directory)
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(
            f"range-http-server on http://{args.bind}:{args.port} rooted at {args.directory}"
        )
        httpd.serve_forever()


if __name__ == "__main__":
    main()
