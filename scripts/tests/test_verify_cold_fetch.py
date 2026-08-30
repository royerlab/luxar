"""Tests for ``scripts/verify_cold_fetch.py``.

Every test here asserts the harness goes **red**. That is the point of it: it is
the only gate standing between "the hosting is broken" and "sixteen demos
catch the failure, most silently starting a multi-minute GPU refit; Census
instead prints regeneration guidance". A gate that cannot fail would be worse
than no gate, because it would be believed.

The origins are real local HTTP servers rather than mocks, because two of the
three failure modes are properties of an HTTP response — a soft-404 that
answers 200 with HTML, and a body whose bytes are not what was pinned — and a
mocked transport would let us assert them into existence rather than observe
them.
"""

from __future__ import annotations

import hashlib
import http.server
import importlib.util
import json
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/verify_cold_fetch.py"

PAYLOAD = b"a plausible little archive" * 64


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_cold_fetch", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SoftNotFound(http.server.SimpleHTTPRequestHandler):
    """Answers a miss with 200 + text/html, exactly as Cloudflare Pages does.

    This is the behaviour that makes a status-code check unsound: the response
    is indistinguishable from success to anything that does not look at the
    body.
    """

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = Path(self.directory) / Path(self.path.split("?")[0]).name
        if path.is_file():
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b"<!doctype html><html><body>SPA index</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # noqa: D102 - silence the server
        pass


@pytest.fixture
def origin(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """A local origin that soft-404s, serving files from a directory."""
    served = tmp_path / "origin"
    served.mkdir()

    def handler(*args: Any, **kwargs: Any) -> _SoftNotFound:
        return _SoftNotFound(*args, directory=str(served), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", served
    finally:
        server.shutdown()
        server.server_close()


def _manifest(base_url: str | None, *, sha: str, hosted_sha: str | None = None) -> dict:
    entry: dict[str, Any] = {
        "name": "thing.zarr.zip",
        "sha256": sha,
        "bytes": len(PAYLOAD),
    }
    if hosted_sha is not None:
        entry["hosted_sha256"] = hosted_sha
        entry["hosted_bytes"] = len(PAYLOAD)
    record: dict[str, Any] = {"published": base_url is not None}
    if base_url:
        record["base_url"] = base_url
    return {
        "records": {"test-record": record},
        "datasets": {
            "thing": {
                "bucket": "zenodo",
                "record": "test-record",
                "dir": "thing",
                "files": [entry],
            }
        },
    }


def _variant_manifest(base_url: str) -> dict[str, Any]:
    def entry(name: str) -> dict[str, Any]:
        return {
            "name": name,
            "sha256": _digest(PAYLOAD),
            "hosted_sha256": _digest(PAYLOAD),
            "bytes": len(PAYLOAD),
            "hosted_bytes": len(PAYLOAD),
        }

    return {
        "records": {
            "test-record": {"published": True, "base_url": base_url},
        },
        "datasets": {
            "thing": {
                "bucket": "zenodo",
                "record": "test-record",
                "dir": "thing",
                "variants": {
                    "light": {"default": True, "files": [entry("light.zip")]},
                    "full": {"default": False, "files": [entry("full.zip")]},
                },
            }
        },
    }


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #


def test_a_healthy_origin_passes(harness: ModuleType, origin, tmp_path: Path) -> None:
    base_url, served = origin
    (served / "thing.zarr.zip").write_bytes(PAYLOAD)
    manifest = _manifest(base_url, sha=_digest(PAYLOAD), hosted_sha=_digest(PAYLOAD))

    ok, detail, _ = harness.verify("thing", manifest, keep=False)
    assert ok, detail
    assert detail.startswith("OK")
    assert "hosted" in detail


def test_a_soft_404_is_caught(harness: ModuleType, origin, tmp_path: Path) -> None:
    """The headline failure: 200 + text/html where an archive should be.

    Nothing is written to the origin, so the server answers the SPA index. A
    status-code check would call this a success.
    """
    base_url, _served = origin
    manifest = _manifest(base_url, sha=_digest(PAYLOAD), hosted_sha=_digest(PAYLOAD))

    ok, detail, _ = harness.verify("thing", manifest, keep=False)
    assert not ok, f"a soft-404 must not pass, got: {detail}"
    assert detail.startswith("FAIL")


def test_wrong_bytes_are_caught(harness: ModuleType, origin, tmp_path: Path) -> None:
    """A real archive, served successfully, that is not the pinned one."""
    base_url, served = origin
    (served / "thing.zarr.zip").write_bytes(PAYLOAD + b"drift")
    manifest = _manifest(base_url, sha=_digest(PAYLOAD), hosted_sha=_digest(PAYLOAD))

    ok, detail, _ = harness.verify("thing", manifest, keep=False)
    assert not ok, f"a digest mismatch must not pass, got: {detail}"
    assert detail.startswith("FAIL")


def test_the_harness_digest_backstop_rejects_wrong_resolver_output(
    harness: ModuleType,
    origin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url, _served = origin
    manifest = _manifest(base_url, sha=_digest(PAYLOAD), hosted_sha=_digest(PAYLOAD))
    wrong_path = tmp_path / "thing.zarr.zip"
    wrong_path.write_bytes(b"resolver returned unchecked bytes")
    monkeypatch.setattr(
        harness.data_fetch, "ensure_dataset", lambda *args, **kwargs: [wrong_path]
    )

    ok, detail, _ = harness.verify("thing", manifest, keep=False)
    assert not ok
    assert "sha256" in detail


def test_a_record_serving_the_repo_copy_is_caught(
    harness: ModuleType, origin, tmp_path: Path
) -> None:
    """A record accidentally filled with the in-repo bytes must not pass.

    Note where the catch comes from, because it is not where I first assumed:
    ``_accepted_contract`` accepts EITHER the hosted or the local digest (so a
    mid-migration *cache* stays usable), but the DOWNLOAD leg validates
    strictly against the hosted pin and deletes the file. So this is caught by
    ``download_with_checksum``, and the harness's own re-check is a backstop
    rather than the primary detector. The behaviour is what matters and is
    asserted here; the attribution is recorded so nobody removes the wrong half.
    """
    base_url, served = origin
    repo_bytes = PAYLOAD
    hosted_bytes = PAYLOAD + b"the published build differs"
    served_file = served / "thing.zarr.zip"
    served_file.write_bytes(repo_bytes)  # the WRONG generation on the record

    manifest = _manifest(
        base_url, sha=_digest(repo_bytes), hosted_sha=_digest(hosted_bytes)
    )

    ok, detail, _ = harness.verify("thing", manifest, keep=False)
    assert not ok, f"a record serving the repo copy must not pass: {detail}"
    assert detail.startswith("FAIL")


def test_a_superseded_generation_on_the_record_is_caught(
    harness: ModuleType, origin, tmp_path: Path
) -> None:
    """Same shape: acceptable to the resolver, wrong for a published record."""
    base_url, served = origin
    old_bytes = PAYLOAD
    current_bytes = PAYLOAD + b"regenerated"
    (served / "thing.zarr.zip").write_bytes(old_bytes)

    manifest = _manifest(
        base_url, sha=_digest(current_bytes), hosted_sha=_digest(current_bytes)
    )
    manifest["datasets"]["thing"]["files"][0]["superseded_sha256"] = [
        _digest(old_bytes)
    ]

    ok, detail, _ = harness.verify("thing", manifest, keep=False)
    assert not ok, f"a superseded build on the record must not pass: {detail}"


def test_an_in_repo_copy_cannot_rescue_a_broken_origin(
    harness: ModuleType, origin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requirement that makes this harness worth having.

    A correct in-repo payload exists and the origin is broken. If the harness
    let the in-repo leg answer, it would report OK and we would delete the only
    working copy on the strength of it.
    """
    base_url, _served = origin  # deliberately empty -> soft-404
    manifest = _manifest(base_url, sha=_digest(PAYLOAD), hosted_sha=_digest(PAYLOAD))

    in_repo = tmp_path / "demos-data"
    (in_repo / "thing").mkdir(parents=True)
    (in_repo / "thing" / "thing.zarr.zip").write_bytes(PAYLOAD)
    monkeypatch.setattr(harness.data_fetch, "_DEMOS_DATA_DIR", in_repo)

    ok, detail, _ = harness.verify("thing", manifest, keep=False)
    assert not ok, f"the in-repo copy must not satisfy the check, got: {detail}"


def test_the_in_repo_dir_is_restored_afterwards(
    harness: ModuleType, origin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leaking the redirect would silently break every later caller in-process."""
    base_url, served = origin
    (served / "thing.zarr.zip").write_bytes(PAYLOAD)
    manifest = _manifest(base_url, sha=_digest(PAYLOAD), hosted_sha=_digest(PAYLOAD))

    sentinel = tmp_path / "sentinel-data"
    monkeypatch.setattr(harness.data_fetch, "_DEMOS_DATA_DIR", sentinel)
    harness.verify("thing", manifest, keep=False)
    assert harness.data_fetch._DEMOS_DATA_DIR == sentinel


def test_a_dormant_record_skips_rather_than_failing(harness: ModuleType) -> None:
    """Before publication every dataset is dormant; that is not a failure."""
    manifest = _manifest(None, sha=_digest(PAYLOAD))
    manifest["records"]["test-record"] = {"published": False, "zenodo_record": "1"}

    ok, detail, _ = harness.verify("thing", manifest, keep=False)
    assert ok
    assert detail.startswith("SKIP")


def test_the_hosted_digest_wins_over_the_repo_digest(harness: ModuleType) -> None:
    """Two different contracts. Only the hosted one describes the record."""
    entry = {"name": "x", "sha256": "repo-digest", "hosted_sha256": "hosted-digest"}
    assert harness.expected_digest(entry)[0] == "hosted-digest"

    repo_only = {"name": "x", "sha256": "repo-digest"}
    wanted, contract = harness.expected_digest(repo_only)
    assert wanted == "repo-digest"
    assert "no hosted_sha256" in contract, "falling back must be reported, not silent"

    assert harness.expected_digest({"name": "x"})[0] is None


def test_a_file_with_no_declared_digest_fails(
    harness: ModuleType, origin, tmp_path: Path
) -> None:
    """Nothing to verify against is a failure, not a pass."""
    base_url, served = origin
    (served / "thing.zarr.zip").write_bytes(PAYLOAD)
    manifest = _manifest(base_url, sha=_digest(PAYLOAD))
    del manifest["datasets"]["thing"]["files"][0]["sha256"]

    ok, detail, _ = harness.verify("thing", manifest, keep=False)
    assert not ok
    assert "no digest" in detail


# --------------------------------------------------------------------------- #
# The real manifest
# --------------------------------------------------------------------------- #


def test_every_shipped_dataset_is_currently_dormant(harness: ModuleType) -> None:
    """Documents today's state, and fails loudly the day it stops being true.

    No record is published yet, so nothing should be reachable. When the first
    record goes live this test must be updated *and* a real cold fetch run —
    which is exactly the moment someone should be forced to think about it.
    """
    manifest = harness.data_fetch.load_manifest()
    targets = harness.verification_targets(manifest, harness.hosted_datasets(manifest))
    reachable = [
        harness.target_label(name, variant)
        for name, variant in targets
        if harness.is_reachable(manifest, name, variant)
    ]
    assert reachable == [], (
        "a dataset became reachable: run `hatch run python "
        "scripts/verify_cold_fetch.py` against it and update this test. "
        f"Reachable: {reachable}"
    )


def test_the_manifest_parses_and_declares_hosted_datasets(harness: ModuleType) -> None:
    manifest = harness.data_fetch.load_manifest()
    names = harness.hosted_datasets(manifest)
    assert len(names) > 10
    assert "gsplats_kidney" in names
    # Every hosted dataset must be resolvable end to end by the harness.
    for name, variant in harness.verification_targets(manifest, names):
        assert isinstance(harness.is_reachable(manifest, name, variant), bool)


def test_main_lists_without_fetching(harness: ModuleType, capsys) -> None:
    assert harness.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "gsplats_kidney" in out
    assert "dormant" in out


def test_help_documents_skip_and_teardown_guards(harness: ModuleType, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        harness.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--allow-skip" in out
    assert "--require-verified" in out
    assert harness.__doc__ is not None
    assert "explicitly named" in harness.__doc__
    assert "unless ``--allow-skip``" in harness.__doc__
    assert "pre-removal teardown" in out


def test_an_explicitly_named_skip_is_a_failure(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Asking for a dataset by name and getting no answer is not a pass.

    This is the fails-open shape a tolerant skip branch creates: at teardown
    someone runs the harness on one dataset, reads exit 0, and deletes the only
    copy — having verified nothing. Every record is dormant today, so this is
    also the path they would actually take.
    """
    monkeypatch.setattr(
        harness.data_fetch,
        "load_manifest",
        lambda: _manifest(None, sha=_digest(PAYLOAD)),
    )
    assert harness.main(["thing"]) == 1
    err = capsys.readouterr().err
    assert "NOT verified" in err
    assert "target(s)" in err
    assert "asked for them by name" in err


def test_allow_skip_is_the_only_way_to_tolerate_it(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        harness.data_fetch,
        "load_manifest",
        lambda: _manifest(None, sha=_digest(PAYLOAD)),
    )
    assert harness.main(["thing", "--allow-skip"]) == 0


def test_require_verified_fails_when_nothing_was_verified(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The teardown gate: demand a count rather than trusting a green exit."""
    monkeypatch.setattr(
        harness.data_fetch,
        "load_manifest",
        lambda: _manifest(None, sha=_digest(PAYLOAD)),
    )
    assert harness.main(["--require-verified", "1"]) == 1
    assert "0 target(s) verified" in capsys.readouterr().err


def test_a_bare_all_skipped_run_says_so_rather_than_implying_success(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Exit 0 is correct while nothing is published — silence would not be."""
    monkeypatch.setattr(
        harness.data_fetch,
        "load_manifest",
        lambda: _manifest(None, sha=_digest(PAYLOAD)),
    )
    assert harness.main([]) == 0
    out = capsys.readouterr().out
    assert "nothing was actually verified" in out
    assert "NOT evidence any payload is safe to remove" in out


def test_main_rejects_an_unknown_dataset(harness: ModuleType, capsys) -> None:
    assert harness.main(["no_such_dataset"]) == 2
    assert "unknown dataset" in capsys.readouterr().err


def test_main_rejects_a_cache_root_that_is_a_file(
    harness: ModuleType, tmp_path: Path, capsys
) -> None:
    cache_root = tmp_path / "not-a-directory"
    cache_root.write_text("occupied")

    assert harness.main(["--list", "--cache-root", str(cache_root)]) == 2
    stderr = capsys.readouterr().err
    assert "--cache-root" in stderr
    assert "not a directory" in stderr


@pytest.mark.parametrize("bucket", ["local-compute", "regenerate"])
def test_non_hosted_datasets_are_labelled_as_not_hosted(
    harness: ModuleType,
    bucket: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    manifest = {"records": {}, "datasets": {"thing": {"bucket": bucket}}}

    ok, detail, _ = harness.verify("thing", manifest, keep=False)
    assert ok
    assert detail == f"SKIP  not hosted ({bucket} dataset)"

    monkeypatch.setattr(harness.data_fetch, "load_manifest", lambda: manifest)
    assert harness.main(["--list", "thing"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{'thing':<40} not hosted"]

    assert harness.main(["thing"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "1 not hosted" in captured.out
    assert "nothing was actually verified" not in captured.out


def test_main_checks_every_variant_and_labels_each_row(
    harness: ModuleType,
    origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    base_url, served = origin
    (served / "light.zip").write_bytes(PAYLOAD)
    manifest = _variant_manifest(base_url)
    monkeypatch.setattr(harness.data_fetch, "load_manifest", lambda: manifest)

    assert harness.main(["thing"]) == 1
    captured = capsys.readouterr()
    assert "thing:light" in captured.out
    assert "thing:full" in captured.out
    assert "1 failed" in captured.out


def test_main_returns_one_and_prints_the_payload_warning(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    manifest = _manifest(None, sha=_digest(PAYLOAD))
    monkeypatch.setattr(harness.data_fetch, "load_manifest", lambda: manifest)
    monkeypatch.setattr(
        harness, "verify", lambda *args, **kwargs: (False, "FAIL  test failure", None)
    )

    assert harness.main(["thing"]) == 1
    stderr = capsys.readouterr().err
    assert "Do NOT" in stderr
    assert "remove its in-repo payload" in stderr


def test_cache_root_and_kept_path_are_reported_after_the_result(
    harness: ModuleType,
    origin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    base_url, served = origin
    (served / "thing.zarr.zip").write_bytes(PAYLOAD)
    manifest = _manifest(base_url, sha=_digest(PAYLOAD), hosted_sha=_digest(PAYLOAD))
    monkeypatch.setattr(harness.data_fetch, "load_manifest", lambda: manifest)
    cache_root = tmp_path / "large-disk"

    assert harness.main(["thing", "--keep", "--cache-root", str(cache_root)]) == 0
    lines = capsys.readouterr().out.splitlines()
    result_index = next(i for i, line in enumerate(lines) if line.startswith("thing"))
    kept_index = next(i for i, line in enumerate(lines) if "kept:" in line)
    kept_path = Path(lines[kept_index].split("kept:", 1)[1].strip())
    assert result_index < kept_index
    assert kept_path.parent == cache_root
    assert kept_path.exists()
    harness.shutil.rmtree(kept_path)

    assert harness.main(["thing", "--require-verified", "1"]) == 0
    assert harness.main(["thing", "--require-verified", "2"]) == 1


def test_json_manifest_is_valid(harness: ModuleType) -> None:
    raw = (REPO / "packages/luxar/src/luxar/demos/data_manifest.json").read_text()
    assert json.loads(raw)["datasets"]
