"""The structure fetch prefers the biological assembly, then PDB, then mmCIF."""

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
_ASSEMBLY = (
    b"data_6N2Y_assembly1\nloop_\n_atom_site.group_PDB\n"
    b"ATOM 1 N N . MET A 1 1 ? 11.1 6.1 -6.5 1.0 0.0 ? 1 MET A N 1\n"
    b"ATOM 2 N N . MET B 1 1 ? 21.1 6.1 -6.5 1.0 0.0 ? 1 MET B N 1\n"
    b"HETATM 3 MG MG . MG C 1 1 ? 1.0 2.0 3.0 1.0 0.0 ? 1 MG C MG 1\n"
)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _opener(served: dict[str, bytes]):
    """Serve only what ``served`` names; 404 every other structure URL.

    The entry JSON endpoint always answers, since the title is fetched
    separately from the structure.
    """

    def urlopen(url: str, timeout: float = 0):  # noqa: ARG001
        if url in served:
            return _Response(served[url])
        if url.startswith("https://data.rcsb.org/"):
            return _Response(
                json.dumps({"struct": {"title": "Whole machine"}}).encode()
            )
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]

    return urlopen


def test_fetch_prefers_the_biological_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assembly 1 is the molecule; the asymmetric unit is bookkeeping.

    Measured over the protein-universe tour: 4OO8's asymmetric unit packs two
    whole Cas9 complexes, 8RUC is half a RuBisCO and 1U94 one protomer of a
    six-subunit RecA filament — so a turntable fed the asymmetric unit showed
    a pair, a half and a monomer.
    """
    asm_url = tt.RCSB_ASSEMBLY_URL.format(pdb_id="6N2Y")
    pdb_url = tt.RCSB_FILE_URL.format(pdb_id="6N2Y")
    monkeypatch.setattr(
        tt.urllib.request, "urlopen", _opener({asm_url: _ASSEMBLY, pdb_url: _PDB})
    )
    path, title = tt.fetch_pdb("6n2y", tmp_path)
    assert path == tmp_path / "6N2Y-assembly1.cif"
    assert path.read_bytes() == _ASSEMBLY
    assert title == "Whole machine"
    # The asymmetric unit is not even downloaded when the assembly answers.
    assert not (tmp_path / "6N2Y.pdb").exists()
    assert tt.structure_atoms("6N2Y", tmp_path) == 3
    # Idempotent: a cached assembly short-circuits without touching the net.
    monkeypatch.setattr(
        tt.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network hit")
    )
    assert tt.fetch_pdb("6N2Y", tmp_path)[0] == path


def test_fetch_falls_back_to_the_pdb_when_no_assembly_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdb_url = tt.RCSB_FILE_URL.format(pdb_id="1BMF")
    monkeypatch.setattr(tt.urllib.request, "urlopen", _opener({pdb_url: _PDB}))
    path, _ = tt.fetch_pdb("1BMF", tmp_path)
    assert path == tmp_path / "1BMF.pdb"
    assert tt.structure_atoms("1BMF", tmp_path) == 1


def test_fetch_falls_back_to_cif_when_assembly_and_pdb_are_both_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy format cannot hold the large cryo-EM assemblies."""
    cif_url = tt.RCSB_CIF_URL.format(pdb_id="6N2Y")
    monkeypatch.setattr(tt.urllib.request, "urlopen", _opener({cif_url: _CIF}))
    path, title = tt.fetch_pdb("6n2y", tmp_path)
    assert path == tmp_path / "6N2Y.cif" and path.read_bytes() == _CIF
    assert title == "Whole machine"
    assert not (tmp_path / "6N2Y.pdb").exists()
    # Atom records are counted the same way in either format.
    assert tt.structure_atoms("6N2Y", tmp_path) == 2


def test_a_cache_written_before_assembly_support_upgrades_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this nearly shipped with.

    A `elif not pdb_path.exists()` would see the legacy file already on disk
    and never look for the assembly, so every machine with a warm cache would
    keep rendering asymmetric units forever — silently, since the only visible
    difference is the picture.
    """
    (tmp_path / "1BMF.pdb").write_bytes(_PDB)
    asm_url = tt.RCSB_ASSEMBLY_URL.format(pdb_id="1BMF")
    monkeypatch.setattr(tt.urllib.request, "urlopen", _opener({asm_url: _ASSEMBLY}))
    path, _ = tt.fetch_pdb("1BMF", tmp_path)
    assert path == tmp_path / "1BMF-assembly1.cif"
    assert tt.structure_atoms("1BMF", tmp_path) == 3


def test_a_legacy_cache_is_reused_when_rcsb_has_no_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upgrading must not mean re-downloading what is already there."""
    (tmp_path / "1BMF.pdb").write_bytes(_PDB)
    monkeypatch.setattr(tt.urllib.request, "urlopen", _opener({}))
    path, _ = tt.fetch_pdb("1BMF", tmp_path)
    assert path == tmp_path / "1BMF.pdb"


def test_other_http_errors_still_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def urlopen(url: str, timeout: float = 0):  # noqa: ARG001
        raise urllib.error.HTTPError(url, 503, "Unavailable", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(tt.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        tt.fetch_pdb("1BMF", tmp_path)
