"""Tests for the Zenodo draft uploader's safety properties.

No network: ``_get`` or ``urllib.request.urlopen`` is stubbed. What matters here
is the REFUSALS and the credential's route onto the wire — this is the only tool in
the repo that mutates a Zenodo record, and the guarantees it claims (draft only,
never publish) have to be enforced rather than merely documented.
"""

from __future__ import annotations

import email.message
import http.client
import importlib.util
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "zenodo_upload_draft.py"


def _load():
    spec = importlib.util.spec_from_file_location("_zen_upload", _SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_zen_upload"] = module
    spec.loader.exec_module(module)
    return module


_up = _load()


def _draft(**over) -> dict:
    base = {
        "submitted": False,
        "state": "unsubmitted",
        "metadata": {"title": "Luxar demo datasets"},
        "links": {"bucket": "https://zenodo.org/api/files/abc"},
        "files": [],
    }
    base.update(over)
    return base


class TestDraftOnlyGuard:
    def test_an_open_draft_is_accepted(self, monkeypatch) -> None:
        monkeypatch.setattr(_up, "_get", lambda url, token: _draft())
        dep = _up.fetch_deposition("1", "tok")
        assert dep["links"]["bucket"].endswith("abc")

    def test_a_submitted_deposition_is_refused(self, monkeypatch) -> None:
        """A published record's files are immutable — never try to mutate one."""
        monkeypatch.setattr(
            _up, "_get", lambda url, token: _draft(submitted=True, state="done")
        )
        with pytest.raises(SystemExit, match="REFUSING"):
            _up.fetch_deposition("1", "tok")

    @pytest.mark.parametrize("state", ["done", "inprogress", "error", "published"])
    def test_any_state_other_than_unsubmitted_is_refused(
        self, monkeypatch, state: str
    ) -> None:
        monkeypatch.setattr(_up, "_get", lambda url, token: _draft(state=state))
        with pytest.raises(SystemExit, match="REFUSING"):
            _up.fetch_deposition("1", "tok")

    def test_a_deposition_without_a_bucket_is_refused(self, monkeypatch) -> None:
        monkeypatch.setattr(_up, "_get", lambda url, token: _draft(links={}))
        with pytest.raises(SystemExit, match="no bucket link"):
            _up.fetch_deposition("1", "tok")


class TestNoPublishPath:
    def test_the_source_contains_no_publish_call(self) -> None:
        """The strongest form of "cannot publish": the call is not in the file.

        Zenodo publishes via POST to `<deposition>/actions/publish`; if that path
        ever appears here, this tool stopped being safe by construction.
        """
        source = _SCRIPT.read_text()
        assert "actions/publish" not in source
        assert "/publish" not in source

    def test_only_get_and_put_are_used(self) -> None:
        """PUT adds a file to the bucket; POST would be an action like publish."""
        source = _SCRIPT.read_text()
        assert 'method="PUT"' in source
        assert 'method="POST"' not in source


class TestCredentialNeverInTheUrl:
    """The token is a `deposit:write` credential, so it must stay in a header.

    A query string is the one part of an HTTPS request that gets written down by
    things that are not the endpoint — server and proxy access logs, TLS-terminating
    middleboxes, and any tracing or error-reporting layer that formats
    ``HTTPError.url``. A header is not, and being *unredirected* it is not replayed
    to a redirect target either.
    """

    TOKEN = "s3cr3t-t0k3n-value"

    @staticmethod
    def _capture(monkeypatch, body: dict) -> list[urllib.request.Request]:
        """Collect the ``Request`` objects the script hands to ``urlopen``."""
        seen: list[urllib.request.Request] = []

        def fake_urlopen(req, timeout=None):  # test stub
            seen.append(req)
            return io.BytesIO(json.dumps(body).encode())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return seen

    @staticmethod
    def _redirect(req: urllib.request.Request) -> urllib.request.Request | None:
        """What urllib would send to another host after a 302 on ``req``."""
        headers = email.message.Message()
        headers["Location"] = "https://elsewhere.example/x"
        return urllib.request.HTTPRedirectHandler().redirect_request(
            req, io.BytesIO(), 302, "Found", headers, "https://elsewhere.example/x"
        )

    def test_fetch_deposition_sends_a_bearer_header_and_a_clean_url(
        self, monkeypatch
    ) -> None:
        seen = self._capture(monkeypatch, _draft())
        _up.fetch_deposition("21912280", self.TOKEN)

        (req,) = seen
        assert "access_token" not in req.full_url
        assert self.TOKEN not in req.full_url
        assert req.get_header("Authorization") == f"Bearer {self.TOKEN}"

    def test_the_bucket_put_sends_a_bearer_header_and_a_clean_url(
        self, monkeypatch, tmp_path
    ) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello luxar")
        seen = self._capture(monkeypatch, {"checksum": "md5:abc"})
        _up.upload("https://zenodo.org/api/files/abc", f, self.TOKEN)

        (req,) = seen
        assert "access_token" not in req.full_url
        assert self.TOKEN not in req.full_url
        assert req.get_header("Authorization") == f"Bearer {self.TOKEN}"

    def test_a_crlf_terminated_token_is_stripped_before_it_becomes_a_header(
        self, monkeypatch, tmp_path
    ) -> None:
        r"""``main()`` strips the credential it reads, restoring an old tolerance.

        A token file with CRLF line endings read as ``ZENODO_TOKEN=$(cat
        token.txt)`` leaves a trailing ``\r``: harmless in the old
        ``?access_token=`` form, because ``urlsplit`` drops ASCII ``\t\r\n`` from a
        URL, but as a header value ``http.client.putheader`` rejects it with
        ``ValueError: Invalid header value b'Bearer <token>'`` — which no handler
        here catches, so CPython prints the whole credential in a traceback. Pinned
        twice: the header the request carries is the bare token, and
        ``http.client``'s own validation — what used to raise, exercised here
        without a socket — accepts it.
        """
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello luxar")
        monkeypatch.setenv("ZENODO_TOKEN", f"{self.TOKEN}\r\n")
        monkeypatch.setattr(
            sys,
            "argv",
            ["zenodo_upload_draft.py", "--deposition", "21912280", "--files", str(f)],
        )
        seen = self._capture(monkeypatch, _draft())

        assert _up.main() == 0  # dry run: one GET, no --yes

        (req,) = seen
        header = req.get_header("Authorization")
        assert header == f"Bearer {self.TOKEN}"
        conn = http.client.HTTPConnection("localhost")
        conn.putrequest("GET", "/x", skip_host=True, skip_accept_encoding=True)
        conn.putheader("Authorization", header)

    def test_a_token_with_an_embedded_newline_is_refused_before_any_request(
        self, monkeypatch, tmp_path
    ) -> None:
        r"""Stripping cannot save a ``\r``/``\n`` in the MIDDLE of the value.

        Such a token reaches ``http.client.putheader``, which raises ``ValueError:
        Invalid header value b'Bearer <token>...'`` — neither ``OSError`` nor
        ``HTTPException``, so it escapes every handler in the script and CPython
        prints the credential (plus whatever the newline appended) in a traceback.
        ``main()`` must refuse it up front instead: a message naming the problem and
        no part of the value, and not one request attempted.
        """
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello luxar")
        injected = "X-Injected: 1"
        monkeypatch.setenv("ZENODO_TOKEN", f"{self.TOKEN}\n{injected}")
        monkeypatch.setattr(
            sys,
            "argv",
            ["zenodo_upload_draft.py", "--deposition", "21912280", "--files", str(f)],
        )
        seen = self._capture(monkeypatch, _draft())

        with pytest.raises(SystemExit) as excinfo:
            _up.main()

        message = str(excinfo.value)
        assert self.TOKEN not in message
        assert injected not in message
        assert "ZENODO_TOKEN" in message and "line break" in message
        assert seen == []

    def test_a_redirect_is_not_handed_the_bearer_header(self, monkeypatch) -> None:
        """A header is only safer than a query string if it is *unredirected*.

        ``HTTPRedirectHandler.redirect_request`` copies everything in
        ``Request.headers`` (it strips only content-length/type) onto the follow-up
        request, whatever host ``Location`` names — whereas a ``?access_token=``
        was never carried, because a redirect is built from the ``Location`` URL.
        So an ordinary ``add_header`` would have opened the very exposure path the
        change closes. Unredirected, the redirect goes out bare and fails closed
        with a 401 that ``fetch_deposition`` reports without a URL.
        """
        seen = self._capture(monkeypatch, _draft())
        _up.fetch_deposition("21912280", self.TOKEN)
        (req,) = seen

        followup = self._redirect(req)
        assert followup is not None
        assert followup.get_header("Authorization") is None
        assert self.TOKEN not in str(followup.header_items())
        assert self.TOKEN not in followup.full_url
        # ...and the header was not merely deleted: the real request carries it.
        assert req.get_header("Authorization") == f"Bearer {self.TOKEN}"

    def test_the_bucket_put_refuses_to_redirect_at_all(
        self, monkeypatch, tmp_path
    ) -> None:
        """The PUT's own honest behaviour: urllib will not redirect a PUT, so it
        raises instead of building a follow-up request and there is no second
        request to leak to. The header is unredirected there too, so the two call
        sites cannot drift apart if that ever changes.
        """
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello luxar")
        seen = self._capture(monkeypatch, {"checksum": "md5:abc"})
        _up.upload("https://zenodo.org/api/files/abc", f, self.TOKEN)
        (req,) = seen

        with pytest.raises(urllib.error.HTTPError):
            self._redirect(req)
        assert "Authorization" not in req.headers
        assert req.get_header("Authorization") == f"Bearer {self.TOKEN}"

    def test_the_source_never_puts_a_token_in_a_query_string(self) -> None:
        """A grep guard, in the spirit of the no-publish ones: it cannot come back.

        Scoped to the defect's shape rather than the bare word, so a docstring or
        comment stays free to name the hazard it is warning about.
        """
        source = _SCRIPT.read_text()
        assert "access_token=" not in source
        assert "?access_token" not in source

    def test_a_rejected_bucket_put_exits_without_leaking_the_credential(
        self, monkeypatch, tmp_path
    ) -> None:
        """A 403 must terminate the run loudly — but the message is printed, so it
        must carry neither the token nor the URL that would otherwise hold it."""
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello luxar")
        leaky_url = f"https://zenodo.org/api/files/abc/a.bin?access_token={self.TOKEN}"

        def fake_urlopen(req, timeout=None):  # test stub
            raise urllib.error.HTTPError(leaky_url, 403, "FORBIDDEN", {}, None)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(SystemExit) as excinfo:
            _up.upload("https://zenodo.org/api/files/abc", f, self.TOKEN)

        message = str(excinfo.value)
        assert self.TOKEN not in message
        assert "access_token" not in message
        assert "https://" not in message
        assert "403" in message and "a.bin" in message
        assert "deposit:write" in message

    @pytest.mark.parametrize(
        ("failure", "expected_reason"),
        [
            pytest.param(
                http.client.RemoteDisconnected(
                    "Remote end closed connection without response"
                ),
                "Remote end closed connection",
                id="remote-disconnected",
            ),
            pytest.param(
                TimeoutError("The read operation timed out"),
                "timed out",
                id="response-read-timeout",
            ),
            pytest.param(
                http.client.BadStatusLine("garbage-status-line"),
                "garbage-status-line",
                id="bad-status-line",
            ),
            pytest.param(
                http.client.IncompleteRead(b'{"chec', 42),
                "IncompleteRead",
                id="body-shorter-than-content-length",
            ),
            pytest.param(
                urllib.error.URLError(
                    ConnectionResetError(104, "Connection reset by peer")
                ),
                "Connection reset by peer",
                id="send-phase-urlerror",
            ),
        ],
    )
    def test_a_bucket_put_that_never_gets_a_status_still_exits_cleanly(
        self, monkeypatch, tmp_path, failure: Exception, expected_reason: str
    ) -> None:
        """The negative control for the exception set ``upload``'s except-block
        comment enumerates: each must exit as loudly and as credential-free as a
        403 rather than escaping as a bare traceback out of the upload loop. A
        clause narrowed back to ``urllib.error.URLError`` fails all but the last
        case here.
        """
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello luxar")

        def fake_urlopen(req, timeout=None):  # test stub
            raise failure

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(SystemExit) as excinfo:
            _up.upload("https://zenodo.org/api/files/abc", f, self.TOKEN)

        message = str(excinfo.value)
        assert "a.bin" in message
        assert expected_reason in message
        assert self.TOKEN not in message
        assert "access_token" not in message
        assert "https://" not in message

    def test_a_rejected_deposition_get_exits_without_leaking_the_credential(
        self, monkeypatch
    ) -> None:
        """The other half of the invariant, and the branch a wrong-scope token hits
        first. Provably clean today — the message interpolates only the status and
        the deposition id — so this is a regression guard."""
        leaky_url = f"{_up.API}/21912280?access_token={self.TOKEN}"

        def fake_urlopen(req, timeout=None):  # test stub
            raise urllib.error.HTTPError(leaky_url, 403, "FORBIDDEN", {}, None)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(SystemExit) as excinfo:
            _up.fetch_deposition("21912280", self.TOKEN)

        message = str(excinfo.value)
        assert self.TOKEN not in message
        assert "access_token" not in message
        assert "https://" not in message
        assert "403" in message and "21912280" in message


class TestIdempotence:
    def test_identical_file_is_recognised_by_md5_and_size(self, tmp_path) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello luxar")
        digest = _up._md5(f)

        dep = _draft(
            files=[
                {
                    "filename": "a.bin",
                    "checksum": f"md5:{digest}",
                    "filesize": f.stat().st_size,
                }
            ]
        )
        have = _up.existing_files(dep)
        assert have["a.bin"] == (digest, f.stat().st_size)

    def test_checksum_prefix_is_optional(self) -> None:
        """Zenodo has returned both `md5:<hex>` and bare `<hex>` over time."""
        dep = _draft(files=[{"filename": "a", "checksum": "deadbeef", "filesize": 3}])
        assert _up.existing_files(dep)["a"] == ("deadbeef", 3)

    def test_key_and_size_spellings_are_accepted(self) -> None:
        """The bucket API spells them `key`/`size`, the deposition API differently."""
        dep = _draft(files=[{"key": "b", "checksum": "md5:abc", "size": 7}])
        assert _up.existing_files(dep)["b"] == ("abc", 7)

    def test_the_plan_skips_identical_and_flags_changed_bytes(self, tmp_path) -> None:
        """The resume policy: same bytes = skip, same name + new bytes = REPLACE."""
        same = tmp_path / "same.bin"
        same.write_bytes(b"identical")
        changed = tmp_path / "changed.bin"
        changed.write_bytes(b"new bytes")
        fresh = tmp_path / "fresh.bin"
        fresh.write_bytes(b"never seen")

        have = {
            "same.bin": (_up._md5(same), same.stat().st_size),
            "changed.bin": ("0" * 32, 1),
        }
        actions = {
            path.name: action
            for path, _, action in _up.plan_uploads([fresh, changed, same], have)
        }
        assert actions == {
            "same.bin": "skip (identical)",
            "changed.bin": "REPLACE (same name, different bytes)",
            "fresh.bin": "upload",
        }

    def test_a_matching_md5_with_a_different_size_is_not_skipped(
        self, tmp_path
    ) -> None:
        """Both halves are checked — a truncated remote copy must be re-sent."""
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello luxar")
        have = {"a.bin": (_up._md5(f), f.stat().st_size + 1)}
        ((_, _, action),) = _up.plan_uploads([f], have)
        assert action == "REPLACE (same name, different bytes)"

    def test_changed_bytes_under_the_same_name_are_not_silently_skipped(
        self, tmp_path
    ) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"new content")
        dep = _draft(
            files=[{"filename": "a.bin", "checksum": "md5:0" * 1, "filesize": 999}]
        )
        have = _up.existing_files(dep)
        assert have["a.bin"] != (_up._md5(f), f.stat().st_size)
