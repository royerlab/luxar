"""The exported serve.py's panel address: a QR that can be printed.

A panel scene's QR is only worth printing if it keeps working, so three things
must hold from one start to the next: the TOKEN it carries (saved beside the
script, generated once), the PORT it names (a busy port is an error, never a
silent move), and the ADDRESS it names (`--advertise-host` puts a fixed name
in it while the server listens on every interface). The end-to-end test at the
bottom runs the generated script twice and compares the QR bytes.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import zlib
from collections.abc import Iterator
from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from luxar.cli import _export_serve_template as template
from luxar.cli import export as export_module
from luxar.cli.export import SceneFacts, _get_serve_script_content


def free_port() -> int:
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextmanager
def listener(host: str = "127.0.0.1") -> Iterator[int]:
    """Another server holding a port, as a second serve.py would."""
    with closing(socket.socket()) as held:
        held.bind((host, 0))
        held.listen()
        yield int(held.getsockname()[1])


def parse(*argv: str) -> Any:
    return template._parse_args(list(argv))


# ── the token ───────────────────────────────────────────────────────────────


def test_the_first_start_saves_a_token_and_later_starts_reuse_it(
    tmp_path: Path,
) -> None:
    args = parse("--control", "--host", "0.0.0.0")
    first = template.resolve_control_token(args, tmp_path)
    saved = tmp_path / template.CONTROL_TOKEN_FILE
    assert first.value is not None and template._TOKEN_PATTERN.fullmatch(first.value)
    assert first.printable
    assert saved.read_text(encoding="utf-8").strip() == first.value
    assert first.note is not None and "saved" in first.note
    if os.name != "nt":
        assert saved.stat().st_mode & 0o777 == 0o600
    again = template.resolve_control_token(args, tmp_path)
    assert again.value == first.value and again.printable
    assert again.note is not None and "the same every start" in again.note


def test_a_first_start_that_loses_the_race_adopts_the_winners_token(
    tmp_path: Path,
) -> None:
    path = tmp_path / template.CONTROL_TOKEN_FILE
    assert template.create_token_file(path, "winnertoken1") == "winnertoken1"
    assert template.create_token_file(path, "losertoken22") == "winnertoken1"


def test_a_saved_token_with_a_byte_order_mark_still_reads(tmp_path: Path) -> None:
    path = tmp_path / template.CONTROL_TOKEN_FILE
    path.write_bytes(b"\xef\xbb\xbfsavedtoken123\r\n")
    assert template.read_saved_token(path) == "savedtoken123"


def test_an_explicit_token_is_for_this_run_only(tmp_path: Path) -> None:
    saved = tmp_path / template.CONTROL_TOKEN_FILE
    args = parse("--control", "--host", "0.0.0.0", "--control-token", "one-off-secret")
    token = template.resolve_control_token(args, tmp_path)
    assert token.value == "one-off-secret"
    assert not token.printable  # its QR must not land in the file people print
    assert not saved.exists()
    saved.write_text("savedtoken123\n", encoding="utf-8")
    assert template.resolve_control_token(args, tmp_path).value == "one-off-secret"
    assert saved.read_text(encoding="utf-8") == "savedtoken123\n"


@pytest.mark.parametrize("value", ["", "   ", " padded", "padded "])
def test_an_empty_or_padded_control_token_is_refused(value: str) -> None:
    """Empty is not "no token" (that is --no-control-token), and the viewer
    trims the URL's token, so a padded one could never match."""
    with pytest.raises(SystemExit):
        parse("--control-token", value)


@pytest.mark.parametrize("value", ["-1", "65536", "http", "8000.5"])
def test_the_port_is_a_port_number(value: str) -> None:
    with pytest.raises(SystemExit):
        parse("--port", value)


@pytest.mark.parametrize(
    "argv",
    [
        ("--control", "--host", "0.0.0.0", "--no-control-token"),
        ("--control", "--host", "127.0.0.1"),  # nothing off this machine connects
        ("--no-control", "--host", "0.0.0.0"),  # no relay, nothing to protect
    ],
    ids=["opted-out", "loopback", "no-relay"],
)
def test_no_token_and_no_file_when_none_is_needed(
    tmp_path: Path, argv: tuple[str, ...]
) -> None:
    token = template.resolve_control_token(parse(*argv), tmp_path)
    assert token.value is None and token.note is None
    assert not (tmp_path / template.CONTROL_TOKEN_FILE).exists()


@pytest.mark.parametrize("content", ["short", "has space in it", "slash/ed/token", ""])
def test_an_unusable_saved_token_stops_the_start_instead_of_being_replaced(
    tmp_path: Path, content: str
) -> None:
    """A printed QR may carry what that file was meant to hold."""
    (tmp_path / template.CONTROL_TOKEN_FILE).write_text(content, encoding="utf-8")
    with pytest.raises(template.StartupError, match=template.CONTROL_TOKEN_FILE):
        template.resolve_control_token(
            parse("--control", "--host", "0.0.0.0"), tmp_path
        )
    assert (tmp_path / template.CONTROL_TOKEN_FILE).read_text() == content


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs a folder the test user cannot write",
)
def test_a_read_only_folder_still_gets_a_token_and_a_warning(tmp_path: Path) -> None:
    tmp_path.chmod(0o500)
    try:
        token = template.resolve_control_token(
            parse("--control", "--host", "0.0.0.0"), tmp_path
        )
    finally:
        tmp_path.chmod(0o700)
    assert token.value is not None and not token.printable
    assert token.note is not None and "do not print this QR" in token.note
    assert not (tmp_path / template.CONTROL_TOKEN_FILE).exists()


def test_a_token_and_no_token_cannot_both_be_asked_for() -> None:
    with pytest.raises(SystemExit):
        parse("--control-token", "x" * 10, "--no-control-token")


# ── the advertised address ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "parsed"),
    [
        ("lobby-pc.local", "lobby-pc.local"),
        ("Lobby-PC.local.", "Lobby-PC.local"),
        ("10.1.2.3", "10.1.2.3"),
        ("fe80::1", "fe80::1"),
        ("[fe80::1]", "fe80::1"),
        ("localhost", "localhost"),
    ],
)
def test_advertise_host_takes_a_name_or_an_ip(value: str, parsed: str) -> None:
    assert parse("--advertise-host", value).advertise_host == parsed


@pytest.mark.parametrize(
    "value",
    [
        "",
        "http://lobby.local",
        "lobby.local:8000",
        "lobby.local/x",
        "a b",
        "-x.local",
        "0.0.0.0",  # names no machine
        "::",
        "[::]",
        "224.0.0.1",  # multicast
        "fe80::1%en0",  # a zone is not valid inside a URL
        "1.2.3",  # a mistyped address, not a name
        "999.1.1.1",
    ],
)
def test_advertise_host_refuses_anything_that_is_not_a_bare_host(value: str) -> None:
    with pytest.raises(SystemExit):
        parse("--advertise-host", value)


def test_the_advertised_name_is_in_every_url_and_the_data_stays_same_origin() -> None:
    display, panel = template.urls(
        "0.0.0.0", 8123, True, "tok12345", None, advertise="lobby-pc.local"
    )
    assert display.startswith("http://lobby-pc.local:8123/viewer/?src=")
    assert "src=http://lobby-pc.local:8123/" in display
    assert display.endswith("&control&controlToken=tok12345")
    assert panel == (
        "http://lobby-pc.local:8123/viewer/control.html?control&controlToken=tok12345"
    )


def test_an_advertised_ipv6_address_is_bracketed() -> None:
    panel = template.urls("::", 8123, True, None, None, advertise="fe80::1")[1]
    assert panel.startswith("http://[fe80::1]:8123/viewer/control.html")


def test_the_redirect_falls_back_to_the_advertised_name(tmp_path: Path) -> None:
    handler = partial(template.LuxarHandler, directory=str(tmp_path))
    server = template.ControlServer(("127.0.0.1", 0), handler, None)
    server.advertise = "lobby-pc.local"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            client.sendall(
                b"GET / HTTP/1.1\r\nHost: not a host!\r\nConnection: close\r\n\r\n"
            )
            reply = client.recv(4096).decode("latin-1")
    finally:
        server.shutdown()
        server.server_close()
    location = re.search(r"Location: (\S+)", reply)
    assert location is not None
    assert f"src=http://lobby-pc.local:{port}/" in location.group(1)


def test_an_unresolvable_name_is_flagged_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_such_host(*_args: object) -> None:
        raise socket.gaierror("nodename nor servname provided")

    monkeypatch.setattr(template.socket, "getaddrinfo", no_such_host)
    warning = template.unresolved_name_warning("lobby-pc.local")
    assert warning is not None and "lobby-pc.local does not resolve" in warning
    assert template.unresolved_name_warning("10.1.2.3") is None  # never looked up


@pytest.mark.parametrize(
    ("bind", "local"),
    [
        ("0.0.0.0", "127.0.0.1"),
        ("", "127.0.0.1"),
        ("::", "::1"),
        ("127.0.0.1", "127.0.0.1"),
        ("localhost", "localhost"),
        ("10.1.2.3", None),
    ],
)
def test_this_machine_opens_the_display_on_loopback(
    bind: str, local: str | None
) -> None:
    """A name or LAN address meant for the tablets need not resolve here."""
    assert template.local_address(bind) == local


# ── the port ────────────────────────────────────────────────────────────────


def test_a_listening_server_is_seen_through_a_wildcard_bind() -> None:
    with listener("127.0.0.1") as port:
        assert template.someone_listening("127.0.0.1", port)
        assert template.someone_listening("0.0.0.0", port)
    assert not template.someone_listening("127.0.0.1", free_port())


@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0"])
def test_with_the_relay_a_busy_port_is_an_error_not_a_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bind: str,
) -> None:
    monkeypatch.setattr(template, "SCRIPT_DIR", tmp_path)

    def started(self: object) -> None:  # a regression must fail, not hang
        raise AssertionError("serve.py started beside the other server")

    monkeypatch.setattr(template.ControlServer, "serve_forever", started)
    with listener("127.0.0.1") as port:
        argv = ["--control", "--host", bind, "--port", str(port), "--no-open"]
        assert template.main(argv) == 1
    out = capsys.readouterr().out
    assert f"Error: port {port} is already in use" in out
    assert "will\nnot move to another one" in out
    assert "Traceback" not in out


def test_without_the_relay_a_busy_port_still_moves() -> None:
    """Nothing printed names the port, and a double-clicked folder must start."""
    with listener("127.0.0.1") as port:
        args = parse("--no-control", "--host", "127.0.0.1", "--port", str(port))
        server = template.open_server(args, None)
        try:
            assert server.server_address[1] != port
        finally:
            server.server_close()


def test_a_restart_rebinds_the_port_its_last_run_used(tmp_path: Path) -> None:
    """Right after Ctrl+C the port's old connections are draining (TIME_WAIT);
    nothing listens there and the restart must get the SAME port."""
    port = free_port()
    args = parse("--control", "--host", "127.0.0.1", "--port", str(port))
    for _ in range(2):
        server = template.open_server(args, template.Relay(None))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                client.sendall(
                    b"GET /x HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n"
                )
                assert client.recv(64).startswith(b"HTTP/1.1")
        finally:
            server.shutdown()
            server.server_close()
        assert server.server_address[1] == port


def test_a_second_server_cannot_share_the_port() -> None:
    """The bind itself refuses a port another server listens on -- the probe is
    not the only guard (on Windows SO_REUSEADDR would let it share; the server
    keeps the default bind there, see `ControlServer`)."""
    assert template.ControlServer.allow_reuse_address is not template._IS_WINDOWS
    handler = partial(template.LuxarHandler, directory=".")
    first = template.ControlServer(("127.0.0.1", 0), handler, None)
    try:
        port = first.server_address[1]
        with pytest.raises(OSError) as refused:
            template.ControlServer(("127.0.0.1", port), handler, None)
        assert refused.value.errno in template._ADDRESS_IN_USE
    finally:
        first.server_close()


def test_a_wildcard_bind_also_probes_the_address_the_url_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server bound to the LAN address alone would answer the tablets."""
    dialled: list[str] = []

    def dial(address: tuple[str, int], timeout: float) -> object:
        dialled.append(address[0])
        raise OSError("refused")

    monkeypatch.setattr(template.socket, "create_connection", dial)
    monkeypatch.setattr(template, "advertised_host", lambda bind: "10.1.2.3")
    args = parse("--control", "--host", "0.0.0.0", "--port", "8123")
    template.someone_listening(args.host, args.port, template._probe_targets(args))
    assert dialled == ["127.0.0.1", "::1", "10.1.2.3"]
    named = parse("--control", "--advertise-host", "lobby-pc.local")
    assert template._probe_targets(named) == ()  # a name is the tablets' to resolve


def test_a_bind_refused_by_the_os_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window between the listener probe and the bind surfaces as an
    OSError from the server's constructor."""

    def refuse(*_args: object) -> None:
        raise OSError(errno.EADDRINUSE, "Address already in use")

    monkeypatch.setattr(template, "ControlServer", refuse)
    monkeypatch.setattr(template, "someone_listening", lambda *args: False)
    args = parse("--control", "--host", "127.0.0.1", "--port", "8123")
    with pytest.raises(template.StartupError, match="port 8123 is already in use"):
        template.open_server(args, None)
    args = parse("--no-control", "--host", "127.0.0.1", "--port", str(free_port()))
    with pytest.raises(template.StartupError) as error:
        template.open_server(args, None)
    assert "was taken while starting" in str(error.value)
    assert "panel" not in str(error.value)  # there is none without the relay


def test_the_local_display_url_is_loopback_with_the_same_token() -> None:
    args = parse("--control", "--host", "0.0.0.0", "--advertise-host", "lobby.local")
    local = template._local_display_url(args, 8123, "tok12345")
    assert local is not None
    assert local.startswith("http://127.0.0.1:8123/viewer/?src=http://127.0.0.1:8123/")
    assert local.endswith("&control&controlToken=tok12345")
    assert template._local_display_url(parse("--host", "10.1.2.3"), 8123, None) is None


def test_advertising_a_name_from_a_loopback_bind_says_it_cannot_work() -> None:
    args = parse("--control", "--host", "127.0.0.1", "--advertise-host", "lobby.local")
    notes = template._control_notes(args, template.ControlToken(None))
    assert any("listens on this machine only" in note for note in notes)


def test_a_one_off_token_never_lands_in_the_printable_qr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copy(
        Path(export_module.__file__).parent / "_qr.py", tmp_path / "luxar_qr.py"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import luxar_qr  # type: ignore[import-not-found]

    monkeypatch.setattr(template, "qr_matrix", luxar_qr.qr_matrix, raising=False)
    monkeypatch.setattr(template, "qr_ascii", luxar_qr.qr_ascii, raising=False)
    monkeypatch.setattr(template, "qr_png_bytes", luxar_qr.qr_png_bytes, raising=False)
    monkeypatch.setattr(template, "qr_pdf_bytes", luxar_qr.qr_pdf_bytes, raising=False)
    monkeypatch.setattr(template, "SCRIPT_DIR", tmp_path)
    url = "http://h:1/viewer/control.html?control"
    template.print_control_qr(url, printable=False)
    assert not (tmp_path / template.QR_FILE).exists()
    assert not (tmp_path / template.SHEET_FILE).exists()
    assert "for this run only" in capsys.readouterr().out
    template.print_control_qr(url, has_token=True)
    png, sheet = tmp_path / template.QR_FILE, tmp_path / template.SHEET_FILE
    assert png.read_bytes().startswith(b"\x89PNG")
    assert sheet.read_bytes().startswith(b"%PDF-1.4")
    assert b"access key" in zlib.decompress(_pdf_stream(sheet.read_bytes()))
    if os.name != "nt":  # both encode the token
        assert png.stat().st_mode & 0o777 == 0o600
        assert sheet.stat().st_mode & 0o777 == 0o600


def _pdf_stream(pdf: bytes) -> bytes:
    start = pdf.index(b"stream\n") + len(b"stream\n")
    return pdf[start : pdf.index(b"\nendstream", start)]


def test_a_re_export_keeps_the_token_behind_printed_qrs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / template.CONTROL_TOKEN_FILE).write_text("keptsecret12\n", encoding="utf-8")
    kept = export_module._saved_control_token(out)
    assert kept == b"keptsecret12\n"
    shutil.rmtree(out)
    out.mkdir()
    export_module._restore_control_token(out, kept)
    restored = out / template.CONTROL_TOKEN_FILE
    assert template.read_saved_token(restored) == "keptsecret12"
    if os.name != "nt":
        assert restored.stat().st_mode & 0o777 == 0o600
    assert export_module._saved_control_token(tmp_path / "missing") is None


# ── end to end: the generated script, started twice ─────────────────────────


def _start(script: Path, port: int) -> tuple[subprocess.Popen[str], str]:
    """Run the generated panel script on its defaults; return it and its banner.

    Its default bind is every interface -- the token is only minted off
    loopback, which is what is under test -- so the relay is briefly reachable
    from the LAN while this runs, guarded by that token. The banner is read on
    a thread so a script that stalls before printing fails the test instead of
    hanging it.
    """
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--port",
            str(port),
            "--no-open",
            "--advertise-host",
            "127.0.0.1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    started = threading.Event()

    def read_banner() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if line.startswith("Press Ctrl+C"):
                started.set()
                return

    threading.Thread(target=read_banner, daemon=True).start()
    if not started.wait(timeout=20):
        process.kill()
        pytest.fail("serve.py never finished starting:\n" + "".join(lines))
    return process, "".join(lines)


def _stop(process: subprocess.Popen[str]) -> None:
    process.terminate()
    process.communicate(timeout=20)


def _relay_answers(port: int, token: str | None) -> bool:
    """Whether a display and a panel presenting ``token`` can talk."""
    query = "" if token is None else f"&token={token}"
    base = f"ws://127.0.0.1:{port}/control"
    try:
        with connect(f"{base}?role=viewer{query}", open_timeout=5) as viewer:
            with connect(f"{base}?role=controller{query}", open_timeout=5) as panel:
                panel.send(json.dumps({"id": 1, "method": "ping"}))
                return json.loads(viewer.recv(timeout=5))["method"] == "ping"
    except ConnectionClosed:
        return False


def test_two_starts_print_the_same_panel_url_and_qr(tmp_path: Path) -> None:
    """The invariant a printed QR depends on, on the generated script itself."""
    (tmp_path / "viewer").mkdir()
    (tmp_path / "viewer" / "index.html").write_text("<html></html>")
    script = tmp_path / "serve.py"
    script.write_text(
        _get_serve_script_content("data", None, SceneFacts(has_control_panel=True))
    )
    shutil.copy(
        Path(export_module.__file__).parent / "_qr.py", tmp_path / "luxar_qr.py"
    )
    port = free_port()

    banners, qrs, sheets = [], [], []
    for _ in range(2):
        process, banner = _start(script, port)
        try:
            panel_url = re.search(r"http://\S+/viewer/control\.html\S*", banner)
            assert panel_url is not None, banner
            token = re.search(r"controlToken=([A-Za-z0-9_-]+)", panel_url.group(0))
            assert token is not None, panel_url.group(0)
            assert _relay_answers(port, token.group(1))
            assert not _relay_answers(port, None)  # the QR is the credential
        finally:
            _stop(process)
        banners.append(panel_url.group(0))
        qrs.append((tmp_path / "control-qr.png").read_bytes())
        sheets.append((tmp_path / "control-panel.pdf").read_bytes())

    assert banners[0] == banners[1]
    assert banners[0].startswith(f"http://127.0.0.1:{port}/viewer/control.html?control")
    assert qrs[0] == qrs[1]
    assert sheets[0] == sheets[1]  # the printed sheet stays valid too
    saved = (tmp_path / template.CONTROL_TOKEN_FILE).read_text(encoding="utf-8").strip()
    assert banners[0].endswith(f"controlToken={saved}")


# ── --auto-advance: retime the scene's tour on site ─────────────────────────


def test_auto_advance_reaches_the_display_url_and_the_redirect(tmp_path: Path) -> None:
    display, panel = template.urls(
        "0.0.0.0", 8123, True, "tok12345", None, playback_period=30.0
    )
    assert "&playbackPeriod=30&" in display + "&"
    assert "playbackPeriod" not in panel  # the panel is not the one that plays
    handler = partial(template.LuxarHandler, directory=str(tmp_path))
    server = template.ControlServer(("127.0.0.1", 0), handler, None)
    server.playback_period = 0.0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            client.sendall(b"GET / HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n")
            reply = client.recv(4096).decode("latin-1")
    finally:
        server.shutdown()
        server.server_close()
    assert "&playbackPeriod=0" in reply


@pytest.mark.parametrize("value", ["-1", "inf", "nan", "soon"])
def test_auto_advance_must_be_a_period_or_zero(value: str) -> None:
    with pytest.raises(SystemExit):
        parse("--auto-advance", value)


def test_auto_advance_absent_leaves_the_scenes_own_setting() -> None:
    assert parse().auto_advance is None
    display = template.urls("127.0.0.1", 8123, False, None, None)[0]
    assert "playbackPeriod" not in display


# ── --chrome: the display in Chrome, and the check of what it got ───────────


GOOD = {
    "api": "webgpu",
    "backend": "webgpu-native",
    "webgpuAvailable": True,
    "adapter": {"vendor": "nvidia", "architecture": "blackwell", "description": "",
                "isFallbackAdapter": False},
    "screen": {"width": 7200, "height": 1360},
    "window": {"width": 7200, "height": 1360},
    "userAgent": "Mozilla/5.0 ... Chrome/142.0.0.0 Safari/537.36",
}  # fmt: skip


def test_a_good_display_is_one_ok_line() -> None:
    assert template.assess_display(GOOD) == [
        ("OK", "WebGPU on the GPU (nvidia blackwell)")
    ]


@pytest.mark.parametrize(
    ("change", "level", "words"),
    [
        ({"backend": "webgpu-gl"}, "WARNING", "WebGL2 backend"),
        ({"backend": "webgl", "webgpuAvailable": False}, "WARNING", "has no WebGPU"),
        ({"backend": "webgl"}, "WARNING", "WebGPU did not come up"),
        (
            {"adapter": {"vendor": "google", "description": "SwiftShader"}},
            "WARNING",
            "SOFTWARE",
        ),
        (
            {"adapter": {"vendor": "x", "isFallbackAdapter": True}},
            "WARNING",
            "SOFTWARE",
        ),
        ({"window": {"width": 3600, "height": 1360}}, "NOTE", "not full screen"),
        ({"userAgent": "Firefox/140"}, "NOTE", "not Chrome"),
    ],
)
def test_each_sub_optimal_display_is_called_out(
    change: dict, level: str, words: str
) -> None:
    lines = template.assess_display({**GOOD, **change})
    assert any(lvl == level and words in msg for lvl, msg in lines), lines


def test_a_report_with_missing_fields_still_assesses() -> None:
    assert template.assess_display({})[0][0] == "WARNING"


def test_chrome_gets_its_own_profile_and_webgpu_on_linux(tmp_path: Path) -> None:
    linux = template.chrome_command("chrome", "http://x/", tmp_path, platform="linux")
    mac = template.chrome_command("chrome", "http://x/", tmp_path, platform="darwin")
    assert linux[0] == "chrome" and linux[-1] == "http://x/"
    assert f"--user-data-dir={tmp_path}" in linux and "--kiosk" in linux
    assert "--enable-unsafe-webgpu" in linux and "--enable-features=Vulkan" in linux
    assert "--enable-unsafe-webgpu" not in mac  # on by default there
    assert "--autoplay-policy=no-user-gesture-required" in mac


def test_find_chrome_per_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        template.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == "chromium" else None,
    )
    assert template.find_chrome("linux") == "/usr/bin/chromium"
    monkeypatch.setattr(template.shutil, "which", lambda name: None)
    assert template.find_chrome("linux") is None
    exe = tmp_path / "Google/Chrome/Application/chrome.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("PROGRAMFILES", raising=False)
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
    assert template.find_chrome("win32") == str(exe)


def _post(port: int, body: bytes, path: str = "/display-report") -> int:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        client.sendall(
            f"POST {path} HTTP/1.1\r\nHost: h\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        return int(client.recv(64).split(b" ")[1])


def test_the_report_is_taken_once_and_only_while_awaited(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    handler = partial(template.LuxarHandler, directory=str(tmp_path))
    server = template.ControlServer(("127.0.0.1", 0), handler, None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        body = json.dumps(GOOD).encode()
        assert _post(port, body) == 404  # nobody asked
        server.awaiting_display_report.set()
        assert _post(port, b"{not json") == 400
        assert _post(port, body, "/elsewhere") == 404
        assert _post(port, body) == 204
        assert _post(port, body) == 404  # once
    finally:
        server.shutdown()
        server.server_close()
    out = capsys.readouterr().out
    assert out.count("Display check") == 1
    assert "OK       WebGPU on the GPU (nvidia blackwell)" in out


def test_chrome_opens_on_the_local_url_asking_for_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(template, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(
        template.subprocess, "Popen", lambda cmd, **_: launched.append(cmd)
    )
    monkeypatch.setattr(template, "_DISPLAY_REPORT_WAIT_S", 3600.0)
    fake = tmp_path / "chrome"
    fake.write_text("")
    server = object.__new__(template.ControlServer)
    server.awaiting_display_report = threading.Event()
    args = parse("--chrome", str(fake))
    assert template.open_display_in_chrome(
        args, server, "http://127.0.0.1:8000/viewer/?src=x"
    )
    assert launched[0][0] == str(fake)
    assert launched[0][-1] == "http://127.0.0.1:8000/viewer/?src=x&displayReport"
    assert f"--user-data-dir={tmp_path / template.CHROME_PROFILE_DIR}" in launched[0]
    assert server.awaiting_display_report.is_set()


def test_no_chrome_falls_back_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(template, "find_chrome", lambda: None)
    server = object.__new__(template.ControlServer)
    server.awaiting_display_report = threading.Event()
    assert not template.open_display_in_chrome(parse("--chrome"), server, "http://x/")
    assert "Chrome not found" in capsys.readouterr().out
    assert not server.awaiting_display_report.is_set()
