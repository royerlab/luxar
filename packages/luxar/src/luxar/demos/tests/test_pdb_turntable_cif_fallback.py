"""The structure fetch falls back to mmCIF for entries RCSB serves only as CIF."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from luxar.demos import _pdb_turntable as tt

_PDB = (
    b"ATOM      1  N   MET A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
)
_CIF = (
    b"data_6N2Y\nloop_\n_atom_site.group_PDB\n"
    b"ATOM 1 N N . MET A 1 1 ? 11.1 6.1 -6.5 1.0 0.0 ? 1 MET A N 1\n"
    b"HETATM 2 MG MG . MG B 1 1 ? 1.0 2.0 3.0 1.0 0.0 ? 1 MG B MG 1\n"
)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _opener(served: dict[str, bytes]):
    def urlopen(url: str, timeout: float = 0):  # noqa: ARG001
        if url.endswith(".pdb") and url not in served:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]
        if url in served:
            return _Response(served[url])
        return _Response(json.dumps({"struct": {"title": "Whole machine"}}).encode())

    return urlopen


def test_fetch_falls_back_to_cif_when_the_pdb_is_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cif_url = tt.RCSB_CIF_URL.format(pdb_id="6N2Y")
    monkeypatch.setattr(tt.urllib.request, "urlopen", _opener({cif_url: _CIF}))
    path, title = tt.fetch_pdb("6n2y", tmp_path)
    assert path == tmp_path / "6N2Y.cif" and path.read_bytes() == _CIF
    assert title == "Whole machine"
    assert not (tmp_path / "6N2Y.pdb").exists()
    # Atom records are counted the same way in either format.
    assert tt.structure_atoms("6N2Y", tmp_path) == 2
    # Idempotent: a second call reuses the cached CIF without touching the net.
    monkeypatch.setattr(
        tt.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network hit")
    )
    assert tt.fetch_pdb("6N2Y", tmp_path)[0] == path


def test_fetch_keeps_the_pdb_when_rcsb_serves_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdb_url = tt.RCSB_FILE_URL.format(pdb_id="1BMF")
    monkeypatch.setattr(tt.urllib.request, "urlopen", _opener({pdb_url: _PDB}))
    path, _ = tt.fetch_pdb("1BMF", tmp_path)
    assert path == tmp_path / "1BMF.pdb"
    assert tt.structure_atoms("1BMF", tmp_path) == 1


def test_other_http_errors_still_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def urlopen(url: str, timeout: float = 0):  # noqa: ARG001
        raise urllib.error.HTTPError(url, 503, "Unavailable", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(tt.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        tt.fetch_pdb("1BMF", tmp_path)
