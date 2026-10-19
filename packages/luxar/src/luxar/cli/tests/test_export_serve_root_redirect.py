"""The exported folder's root redirects to the viewer (`_export_serve_template`).

`python3 serve.py` used to answer `/` with a bare directory listing (`data/`,
`viewer/`, `serve.py`, ...), so anyone who typed just the host and port saw a
file browser instead of the scene.
"""

from __future__ import annotations

import threading
from functools import partial
from pathlib import Path

import pytest

from luxar.cli import _export_serve_template as template

from .test_export_serve_ranges import get, serving


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "viewer").mkdir()
    (tmp_path / "viewer" / "index.html").write_text("<html></html>")
    (tmp_path / template.DATA_DIR_NAME).mkdir()
    return tmp_path


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_the_root_redirects_to_the_viewer_with_the_scene(
    folder: Path, method: str
) -> None:
    with serving(folder) as port:
        status, headers, body = get(port, "/", method=method)
    assert status == 302
    assert body == b""
    assert headers["location"] == template.viewer_target(
        f"http://127.0.0.1:{port}/{template.DATA_DIR_NAME}"
    )
    assert headers["location"].startswith("/viewer/?src=http://127.0.0.1:")


def test_the_redirect_follows_the_host_the_client_used(folder: Path) -> None:
    """A port forward reaches us on another host:port; the data URL must too."""
    with serving(folder) as port:
        _, headers, _ = get(port, "/?x=1", {"Host": "localhost:18080"})
    assert "src=http://localhost:18080/data" in headers["location"]


def test_an_unsafe_host_header_falls_back_to_the_bind(folder: Path) -> None:
    with serving(folder) as port:
        _, headers, _ = get(port, "/", {"Host": "evil.example/x?y=<z>"})
    assert f"src=http://127.0.0.1:{port}/data" in headers["location"]
    assert "evil" not in headers["location"]


def test_the_redirect_never_carries_a_control_token(folder: Path) -> None:
    """Anyone reaching `/` would otherwise be handed the secret.

    Run with the relay ON and a token set — the configuration where leaking it
    would matter — so the assertion is about something.
    """
    handler = partial(template.LuxarHandler, directory=str(folder))
    server = template.ControlServer(("127.0.0.1", 0), handler, template.Relay("secret"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, headers, _ = get(server.server_address[1], "/")
    finally:
        server.shutdown()
        server.server_close()
    assert status == 302
    location = headers["location"]
    assert "secret" not in location
    assert "control" not in location.lower()
    assert location.startswith("/viewer/?src=")


def test_a_control_run_says_the_bare_address_is_view_only(
    folder: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The redirect drops control, so the operator must be told which URL drives."""
    monkeypatch.setattr(template, "SCRIPT_DIR", folder)

    class _NoServer:  # binds nothing: only the startup narration is under test
        def __init__(self, *_args: object) -> None:
            self.text_scale = None

        def serve_forever(self) -> None:
            return None

    monkeypatch.setattr(template, "ControlServer", _NoServer)
    monkeypatch.setattr(template, "print_control_qr", lambda url: None)
    assert template.main(["--control", "--control-token", "s", "--no-open"]) == 0
    out = capsys.readouterr().out
    assert out.count("VIEW-ONLY") == 1
    assert template.main(["--no-open", "--host", "127.0.0.1"]) == 0
    assert "VIEW-ONLY" not in capsys.readouterr().out


def test_other_paths_are_still_served(folder: Path) -> None:
    with serving(folder) as port:
        status, _, body = get(port, "/viewer/index.html")
        listing, _, _ = get(port, "/data/")
    assert status == 200 and body == b"<html></html>"
    assert listing == 200


def test_the_printed_url_and_the_redirect_agree() -> None:
    printed = template.urls("127.0.0.1", 8123, False, None, 0.8)[0]
    data = f"http://127.0.0.1:8123/{template.DATA_DIR_NAME}"
    assert printed == "http://127.0.0.1:8123" + template.viewer_target(data, 0.8)
    assert printed.endswith("&textScale=0.8")
