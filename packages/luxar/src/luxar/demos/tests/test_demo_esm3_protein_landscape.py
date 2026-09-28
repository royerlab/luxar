"""Tests for the ESM-3 demo's cache/dependency gate.

The demo's embeddings cache is quarantined to ``<name>.corrupt`` when it fails
validation. A quarantine left behind by an *earlier* run used to be invisible:
the ``.npy`` is already gone, so the validation block never fires and the demo
silently restarted a multi-gigabyte compute/download. The gate must name the
quarantined path, its size, and what to do about it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from luxar._zarr_compat import read_node_attrs
from luxar.demos import demo_esm3_protein_landscape as demo
from luxar.demos import require_module
from luxar.demos._dependencies import MissingDependencyError
from luxar.demos._support.downloads.download import QUARANTINE_SUFFIX
from luxar.demos.demo_esm3_protein_landscape import (
    _compute_esm3_embeddings,
    generate_esm3_landscape,
)

SEQUENCES = ["MKV", "MTL", "MGG"]
HEAVY_DEPS = ("torch", "esm", "umap")


def _decode_strings(node, channel: str) -> list[str]:
    offsets = np.asarray(node[f"{channel}_offsets"][:]).astype(int)
    data = bytes(np.asarray(node[f"{channel}_bytes"][:]).tobytes())
    return [
        data[offsets[i] : offsets[i + 1]].decode("utf-8")
        for i in range(len(offsets) - 1)
    ]


@pytest.fixture
def without_heavy_deps(monkeypatch):
    """Make torch / esm / umap-learn look uninstalled, machine-independently.

    A ``None`` entry in ``sys.modules`` makes any import of that name raise
    ``ImportError``, which is exactly the condition under test. Preferred over
    patching ``importlib.import_module``: that attribute lives on the shared
    stdlib module object, so replacing it intercepts *every* import in the
    process for the duration of the test, whereas these three keys are scoped
    to the names that matter (and monkeypatch restores any real entry).
    """
    for name in HEAVY_DEPS:
        monkeypatch.setitem(sys.modules, name, None)


class TestQuarantineReporting:
    def test_pre_existing_quarantine_is_reported(self, tmp_path, capsys) -> None:
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            pytest.skip("CUDA available: the demo would compute instead of failing")

        corrupt = tmp_path / f"embeddings_esmc_300m.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 3072)

        with pytest.raises(RuntimeError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        message = str(excinfo.value)
        assert str(corrupt) in message
        assert "3.00 KB" in message
        assert "QUARANTINED" in message
        # This layer carries the notice in the EXCEPTION, not on the console:
        # `main()` owns the console report (see TestMainReportsQuarantine), and
        # printing here too showed the user the same file twice.
        assert corrupt.name not in capsys.readouterr().out

    def test_clean_cache_message_has_no_quarantine_noise(self, tmp_path) -> None:
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            pytest.skip("CUDA available: the demo would compute instead of failing")

        with pytest.raises(RuntimeError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        assert "QUARANTINED" not in str(excinfo.value)

    def test_invalid_cache_is_quarantined_and_reported(self, tmp_path, capsys) -> None:
        """A wrong-shaped cache is renamed, then reported by the same gate."""
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            pytest.skip("CUDA available: the demo would compute instead of failing")

        cache = tmp_path / "embeddings_esmc_300m.npy"
        np.save(cache, np.zeros((2, 7), dtype=np.float32))  # wrong shape

        with pytest.raises(RuntimeError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        quarantined = Path(f"{cache}{QUARANTINE_SUFFIX}")
        assert quarantined.is_file(), "invalid cache was not quarantined"
        assert not cache.exists()
        assert str(quarantined) in str(excinfo.value)
        assert "quarantined" in capsys.readouterr().out.lower()


class TestDependencyGatesAreDeferred:
    """The heavy deps must be demanded where used, never at the entry point.

    The demo tells the user that a complete cached embeddings file "skips the
    model entirely". An entry-point preflight that imports torch/esm/umap-learn
    and exits makes that advice a lie: a machine holding every artifact it needs
    would still be refused.
    """

    def test_complete_cache_needs_none_of_them(
        self, tmp_path, without_heavy_deps
    ) -> None:
        cache = tmp_path / "embeddings_esmc_300m.npy"
        np.save(cache, np.zeros((len(SEQUENCES), 960), dtype=np.float32))

        got = _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        assert got.shape == (len(SEQUENCES), 960)

    def test_main_reaches_the_generator_without_them(
        self, tmp_path, monkeypatch, without_heavy_deps
    ) -> None:
        """main() must hand off to the generator, not sys.exit on a preflight."""
        called: list[Path] = []

        def fake_generate(output_path, **_kwargs):
            called.append(output_path)
            return 0  # 0 => main() returns without serving

        monkeypatch.setattr(demo, "generate_esm3_landscape", fake_generate)
        monkeypatch.setattr(demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(demo.sys, "argv", ["demo", "--no-serve"])

        demo.main()

        assert called, "main() exited before reaching generate_esm3_landscape"

    def test_missing_dep_message_is_actionable_at_point_of_use(
        self, without_heavy_deps
    ) -> None:
        with pytest.raises(ImportError) as excinfo:
            require_module("esm")

        message = str(excinfo.value)
        assert "pip install 'esm>=3.0.0'" in message
        assert "luxar[demos]" in message
        # Names the cache escape hatch, which is the whole point of deferring.
        assert "cached embeddings" in message

    def test_compute_path_gates_torch_actionably(
        self, tmp_path, without_heavy_deps
    ) -> None:
        """The PRODUCTION path must gate torch, not just the helper in isolation.

        Mutation-checked: replacing the `require_module("torch")` call with a
        bare `import torch` must fail this test. Without it, a cache miss on a
        torch-less machine raises a raw ModuleNotFoundError instead of naming
        the pinned spec.
        """
        with pytest.raises(ImportError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        message = str(excinfo.value)
        assert "torch>=2.2,<3.0" in message, f"torch gate not actionable: {message}"
        assert "luxar[gsplats]" in message

    def test_compute_path_gates_esm_after_the_cuda_check(
        self, tmp_path, monkeypatch
    ) -> None:
        """`esm` must be demanded on the model-load path, and only after CUDA.

        Mutation-checked twice: deleting the `require_module("esm")` call fails
        this test, and moving it above the CUDA probe fails
        ``TestQuarantineReporting`` (a GPU-less machine must get the
        supply-a-cache message, which carries the quarantine notice).
        """
        torch = pytest.importorskip("torch")
        # Pretend this machine can compute, so the CUDA guard passes and the
        # model-load path — the one place `esm` is genuinely needed — is reached.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setitem(sys.modules, "esm", None)

        with pytest.raises(ImportError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        message = str(excinfo.value)
        assert "pip install 'esm>=3.0.0'" in message, (
            f"esm gate missing or not actionable on the compute path: {message}"
        )


class TestQuarantineEnrichesDependencyErrors:
    """A missing torch/esm on the compute path must still name a quarantined copy.

    A direct/programmatic caller (bypassing ``main()``) that hits a MISSING
    ``torch`` used to get a plain ``MissingDependencyError`` with no mention of a
    multi-gigabyte rejected cache sitting on disk. The dependency error now
    carries the quarantine notice — but only for paths ``main()`` has not already
    reported (``already_reported_quarantine``), so the CLI never double-reports
    while a file freshly quarantined THIS run is still named.
    """

    def test_direct_caller_sees_quarantine_when_torch_missing(
        self, tmp_path, monkeypatch
    ) -> None:
        corrupt = tmp_path / f"embeddings_esmc_300m.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 1234)

        # Make `import torch` fail, so the compute path's require_module raises.
        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(MissingDependencyError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        message = str(excinfo.value)
        assert corrupt.name in message, "quarantine notice missing from torch error"
        assert "torch" in message, "original dependency message was lost"

    def test_dependency_remedy_names_the_destination_and_the_install_route(
        self, tmp_path, monkeypatch
    ) -> None:
        """The remedy must be actionable on its OWN, unlike the no-CUDA one.

        The no-CUDA RuntimeError lists the destination `.npy` in a bullet above
        the notice, so it can say "that path". A dependency error has no such
        surrounding text: a bare "that path" points at the `.corrupt` file the
        notice just listed. It must also offer the install route — with the
        package missing, deleting the quarantined copy does not let you proceed.
        """
        corrupt = tmp_path / f"embeddings_esmc_300m.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 1234)
        destination = tmp_path / "embeddings_esmc_300m.npy"

        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(MissingDependencyError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        remedy = str(excinfo.value).rsplit("To proceed you must", 1)[-1]
        assert f"{destination} " in remedy, (
            f"remedy does not name the destination .npy: {remedy}"
        )
        assert "install the missing dependency" in remedy, (
            f"remedy omits the install route: {remedy}"
        )

    def test_already_reported_does_not_repeat_the_notice(
        self, tmp_path, monkeypatch
    ) -> None:
        corrupt = tmp_path / f"embeddings_esmc_300m.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 1234)

        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(MissingDependencyError) as excinfo:
            _compute_esm3_embeddings(
                SEQUENCES,
                tmp_path,
                model_name="esmc-300m",
                already_reported_quarantine=frozenset({corrupt}),
            )

        message = str(excinfo.value)
        assert corrupt.name not in message, "notice repeated after main() reported it"
        assert "torch" in message, "original dependency message was lost"

    def test_freshly_quarantined_file_is_reported_even_if_another_was_reported(
        self, tmp_path, monkeypatch
    ) -> None:
        """Defect-1 regression: the set is per-PATH, not a dir-wide flag.

        `main()` may have reported an UNRELATED leftover (a different --model),
        but a file quarantined THIS run must still be named — otherwise the
        multi-GB copy rejected this run is invisible, worse than the baseline.
        """
        other = tmp_path / f"embeddings_esm3_open.npy{QUARANTINE_SUFFIX}"
        other.write_bytes(b"x" * 1234)
        current = tmp_path / f"embeddings_esmc_300m.npy{QUARANTINE_SUFFIX}"
        current.write_bytes(b"x" * 1234)

        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(MissingDependencyError) as excinfo:
            _compute_esm3_embeddings(
                SEQUENCES,
                tmp_path,
                model_name="esmc-300m",
                already_reported_quarantine=frozenset({other}),
            )

        message = str(excinfo.value)
        assert current.name in message, "freshly quarantined file was suppressed"

    def test_direct_caller_sees_quarantine_when_esm_missing(
        self, tmp_path, monkeypatch
    ) -> None:
        """The esm gate (reached only when CUDA is available) enriches too."""
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setitem(sys.modules, "esm", None)

        corrupt = tmp_path / f"embeddings_esmc_300m.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 1234)

        with pytest.raises(MissingDependencyError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        message = str(excinfo.value)
        assert corrupt.name in message, "quarantine notice missing from esm error"

    def test_compute_path_stays_silent_on_the_console(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The compute layer must NOT print the notice — `main()` owns the console.

        Mutation guard: the notice reaches a direct caller via the raised
        exception, never stdout. Hermetic (torch-less) so it runs on a GPU box
        too, unlike the CUDA-skipped `test_pre_existing_quarantine_is_reported`.
        Flipping `warn_if_quarantined(..., verbose=False)` to `verbose=True`
        prints here and fails this test.
        """
        corrupt = tmp_path / f"embeddings_esmc_300m.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 1234)

        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(MissingDependencyError):
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        out = capsys.readouterr().out
        assert corrupt.name not in out, "compute path printed the notice to stdout"
        assert "QUARANTINED" not in out, "compute path printed the notice to stdout"


class TestMainReportsQuarantine:
    """`main()` owns the CONSOLE report — exactly once, before anything costly.

    The compute path deliberately stays silent (`verbose=False`) so the same
    quarantined file is not announced twice per run; this test is what keeps the
    console coverage that move gave up.
    """

    def test_notice_is_printed_once(self, tmp_path, monkeypatch, capsys) -> None:
        # Path.home() honours $HOME on posix, so this redirects the demo's
        # hard-coded cache dir without patching pathlib globally.
        monkeypatch.setenv("HOME", str(tmp_path))
        cache_dir = tmp_path / ".cache" / "luxar" / "esm3_swissprot"
        cache_dir.mkdir(parents=True)
        corrupt = cache_dir / f"embeddings_esmc_300m.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 2048)

        # Record the kwargs main() forwards, so the exactly-once contract is
        # protected end to end: main() must tell the compute path which paths it
        # already reported, or a regression to the call sites passes silently.
        recorded_kwargs: list[dict] = []

        def fake_generate(output_path, **kwargs):
            recorded_kwargs.append(kwargs)
            return 0

        monkeypatch.setattr(demo, "generate_esm3_landscape", fake_generate)
        monkeypatch.setattr(demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(demo.sys, "argv", ["demo", "--no-serve"])

        demo.main()

        out = capsys.readouterr().out
        assert str(corrupt) in out, "main() did not report the quarantined cache"
        assert out.count("QUARANTINED") == 1, (
            f"quarantine notice printed {out.count('QUARANTINED')} times; "
            "exactly one report per run is the contract"
        )
        assert recorded_kwargs, "main() did not reach the generator"
        reported = recorded_kwargs[0]["already_reported_quarantine"]
        assert isinstance(reported, frozenset)
        assert corrupt in reported, "main() did not forward the reported path set"

    def test_serve_path_forwards_the_reported_set(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The DEFAULT (serve) call site must forward the set too, not just --no-serve.

        The other tests hardcode `--no-serve`, covering only that call site; a
        regression that blanked `already_reported_quarantine` on the serve-path
        call would pass silently. The fake returns 0, so `main()` hits
        `if n == 0: return` before `launch_viewer` — no viewer stub needed.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        cache_dir = tmp_path / ".cache" / "luxar" / "esm3_swissprot"
        cache_dir.mkdir(parents=True)
        corrupt = cache_dir / f"embeddings_esmc_300m.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 2048)

        recorded_kwargs: list[dict] = []

        def fake_generate(output_path, **kwargs):
            recorded_kwargs.append(kwargs)
            return 0

        monkeypatch.setattr(demo, "generate_esm3_landscape", fake_generate)
        monkeypatch.setattr(demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(demo.sys, "argv", ["demo"])  # serve path (no --no-serve)

        demo.main()

        capsys.readouterr()  # drain
        assert recorded_kwargs, "main() did not reach the generator"
        reported = recorded_kwargs[0]["already_reported_quarantine"]
        assert isinstance(reported, frozenset)
        assert corrupt in reported, "serve-path call did not forward the reported set"


def _write_instant_cache(
    cache_dir: Path, n: int = 40, model_name: str = "esmc-300m"
) -> int:
    """Fabricate the two artifacts the instant path reads (sample_size=0).

    Returns ``n`` so callers can assert the generator's protein count (the scene
    itself carries ``2 * n`` rows — one block per coloring). The UMAP cache is
    keyed on the model, so ``model_name`` must match the generator's.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((n, 3)).astype(np.float32)
    tag = model_name.replace("-", "_")
    np.savez(cache_dir / f"umap3d_{tag}_all.npz", positions=positions)

    # Realistic kingdoms present in TAXON_COLORS/_DOMAIN_OF, plus an unknown one
    # ("Slime Mold") to exercise the "Other" fallback.
    kingdoms = np.array(
        [
            ["Human", "Proteobacteria", "Archaea", "Viruses", "Slime Mold"][i % 5]
            for i in range(n)
        ],
        dtype=object,
    )
    names = np.array([f"PROT_{i}" for i in range(n)], dtype=object)
    organisms = np.array([f"Organism {i}" for i in range(n)], dtype=object)
    np.savez(
        cache_dir / "metadata_all.npz",
        names=names,
        organisms=organisms,
        kingdoms=kingdoms,
    )
    return n


class TestCompleteCacheRunsWithoutLodDeps:
    """The instant (full-cache) path must build a scene without the LOD deps.

    The demo documents a "complete cache runs anywhere" contract, but the scene
    build used to request substitutive Points LOD unconditionally. That path
    imports ``luxar.gsplats.lod``, which needs torch (coarsening kernels) and
    scipy (``additive.py`` imports ``scipy.sparse`` at module level) — neither is
    a core dependency — so scene generation died with a ModuleNotFoundError even
    when every expensive cache was supplied.

    The first fix gated LOD on both and DEGRADED to flat, ladderless Points. The
    scene now carries an additive ladder rather than substitutive levels, and the
    additive write path imports neither module, so the contract holds with the
    ladder intact and there is no degraded mode left to announce. These tests
    assert that stronger property: the structure is IDENTICAL with either module
    blocked.
    """

    @pytest.mark.parametrize("blocked", ["torch", "scipy"])
    def test_full_cache_path_builds_scene_without_lod_dep(
        self, tmp_path, monkeypatch, capsys, blocked
    ) -> None:
        cache_dir = tmp_path / "cache"
        n = _write_instant_cache(cache_dir)

        # Simulate a machine missing the dependency EXACTLY as the issue's
        # reproduction does: a None entry makes is_installed() return False AND
        # makes any `import <blocked>` raise instead of quietly succeeding.
        monkeypatch.setitem(sys.modules, blocked, None)

        out_path = tmp_path / "esm3.luxar.zarr"
        got = generate_esm3_landscape(
            out_path, sample_size=0, model_name="esmc-300m", cache_dir=cache_dir
        )

        assert got == n
        assert out_path.exists(), f"scene was not written with {blocked} blocked"

        # Nothing degrades any more, so nothing should be announced.
        out = capsys.readouterr().out
        assert "skipping Points LOD" not in out, (
            "a degradation notice was printed, but the additive ladder needs "
            f"neither torch nor scipy — so {blocked} being blocked is a no-op"
        )

        # Structural assertion, independent of import ordering. A None entry in
        # sys.modules only blocks a *fresh* import; in a warm suite where some
        # module already imported the real package, its namespace keeps the
        # binding — so asserting on stdout alone would not catch a regression to
        # the substitutive path. Assert the on-disk shape: a laddered Points leaf
        # writes `proteins/positions` and has no `kind: lod`, whereas a
        # substitutive group has `kind == "lod"` and child_0..N with no top-level
        # positions.
        proteins = out_path / "proteins"
        assert (proteins / "positions").exists(), (
            "Points leaf missing positions — a substitutive LOD group was "
            "written instead"
        )
        attrs = read_node_attrs(proteins) or {}
        assert attrs.get("kind") != "lod", (
            f"expected a laddered Points leaf, got a substitutive-LOD group: "
            f"{attrs.get('kind')!r}"
        )
        assert not (proteins / "child_0").exists(), (
            "substitutive levels were written; this scene should carry only an "
            "additive ladder"
        )

    def test_legacy_cache_writes_protein_name_search_keys(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        cache_dir = tmp_path / "cache"
        n = _write_instant_cache(cache_dir)
        monkeypatch.setitem(sys.modules, "torch", None)

        out_path = tmp_path / "esm3.luxar.zarr"
        generate_esm3_landscape(
            out_path, sample_size=0, model_name="esmc-300m", cache_dir=cache_dir
        )

        import zarr

        proteins = zarr.open_group(str(out_path), mode="r")["proteins"]
        attrs = dict(proteins.attrs)
        assert attrs["link"] == "https://www.uniprot.org/uniprotkb?query={hover_key}"
        assert attrs["copy"] == "{hover_key}"
        assert attrs["has_keys"] is True

        keys = _decode_strings(proteins, "key")
        labels = _decode_strings(proteins, "label")
        assert len(keys) == len(labels) == 2 * n
        expected = {f"PROT_{i}" for i in range(n)}
        assert set(keys) == expected
        assert all(keys.count(key) == 2 for key in expected)
        assert all(label.startswith(f"{key} — ") for key, label in zip(keys, labels))
        assert all(label.endswith(")") for label in labels)
        assert "using protein-name search" in capsys.readouterr().out

    def test_deps_present_path_builds_the_same_ladder(self, tmp_path, capsys) -> None:
        """Deps-PRESENT branch, which must now be INDISTINGUISHABLE from blocked.

        The point of the change is that torch/scipy no longer influence this
        scene's structure at all. Asserting "same shape either way" is what
        catches a regression to a dependency-conditional build — including the
        subtle one the old pairing was written against, a typo like
        ``is_installed("torchvision")`` that satisfies only one branch.
        """
        pytest.importorskip("torch")
        pytest.importorskip("scipy")

        cache_dir = tmp_path / "cache"
        n = _write_instant_cache(cache_dir)

        out_path = tmp_path / "esm3.luxar.zarr"
        got = generate_esm3_landscape(
            out_path, sample_size=0, model_name="esmc-300m", cache_dir=cache_dir
        )

        assert got == n
        assert "skipping Points LOD" not in capsys.readouterr().out
        proteins = out_path / "proteins"
        attrs = read_node_attrs(proteins) or {}
        assert attrs.get("kind") != "lod", (
            f"expected a laddered Points leaf, not a substitutive group: "
            f"{attrs.get('kind')!r}"
        )
        assert (proteins / "positions").exists()
        assert not (proteins / "child_0").exists()


class TestCitationNamesTheModelThatRan:
    """The stored credit must name the model that produced these embeddings.

    The demo picks its model at runtime (``--model=``), so a static credit taken
    straight from ``DEMO_META`` would write "embeddings by ESM-3" into a store
    whose coordinates came from ESM C — contradicting the scene's own footer and
    crediting the wrong party on whichever path was not the default.
    """

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("esmc-300m", "EvolutionaryScale ESM C, 2024"),
            ("esm3-open", "Hayes et al. 2025"),
        ],
    )
    def test_root_citation_follows_the_model(
        self, tmp_path, model_name, expected
    ) -> None:
        cache_dir = tmp_path / "cache"
        _write_instant_cache(cache_dir, model_name=model_name)

        out_path = tmp_path / f"esm3_{model_name}.luxar.zarr"
        generate_esm3_landscape(
            out_path, sample_size=0, model_name=model_name, cache_dir=cache_dir
        )

        citation = (read_node_attrs(out_path) or {}).get("citation")
        assert citation is not None, "scene wrote no citation"
        assert citation["short"] == f"UniProt/Swiss-Prot; embeddings by {expected}", (
            f"credit does not name the model that ran: {citation['short']!r}"
        )
        # The dataset half is not the model's to claim, and the licence rides
        # along from DEMO_META rather than being dropped by the override.
        assert citation.get("license") == "CC BY 4.0"

    def test_default_model_matches_the_static_credit(self, tmp_path) -> None:
        """A caller omitting ``model_name`` gets the model DEMO_META credits."""
        cache_dir = tmp_path / "cache"
        _write_instant_cache(cache_dir)

        out_path = tmp_path / "esm3_default.luxar.zarr"
        generate_esm3_landscape(out_path, sample_size=0, cache_dir=cache_dir)

        citation = (read_node_attrs(out_path) or {}).get("citation")
        assert citation["short"] == demo.DEMO_META["citation"]["short"], (
            "the default run's credit disagrees with DEMO_META's"
        )


# ---------------------------------------------------------------------------
# Taxon categories from the UniProt lineage
# ---------------------------------------------------------------------------

_LINEAGES = {
    # The organism-name rules got each of these wrong ("influenzae" read as a
    # virus; an unknown genus defaulted to a eukaryote).
    "P43747": "cellular organisms (no rank), Bacteria (domain), Pseudomonadati "
    "(kingdom), Pseudomonadota (phylum), Gammaproteobacteria (class), "
    "Pasteurellales (order), Haemophilus (genus)",
    "Q8YXR5": "cellular organisms (no rank), Bacteria (domain), "
    "Bacillati (kingdom), Cyanobacteriota (phylum), Nostoc (genus)",
    "P0DMV8": "cellular organisms (no rank), Eukaryota (domain), Metazoa "
    "(kingdom), Chordata (phylum), Vertebrata (clade), Mammalia (class), Homo "
    "(genus)",
    "Q9LXX0": "cellular organisms (no rank), Eukaryota (domain), Viridiplantae "
    "(kingdom), Streptophyta (phylum), Arabidopsis (genus)",
    "P0DTC2": "Viruses (no rank), Riboviria (realm), Coronaviridae (family)",
    "Q57XX0": "cellular organisms (no rank), Archaea (domain), "
    "Methanobacteriota (phylum)",
}


@pytest.mark.parametrize(
    ("acc", "expected"),
    [
        ("P43747", "Proteobacteria"),
        ("Q8YXR5", "Other Bacteria"),
        ("P0DMV8", "Human"),
        ("Q9LXX0", "Plants"),
        ("P0DTC2", "Viruses"),
        ("Q57XX0", "Archaea"),
    ],
)
def test_lineage_categories(acc: str, expected: str) -> None:
    assert demo._classify_lineage(_LINEAGES[acc]) == expected


def test_lineage_without_a_domain_defers_to_the_name_rules() -> None:
    assert demo._classify_lineage("") is None


def test_lineage_download_cleans_partial_file_after_truncated_read(
    tmp_path: Path, monkeypatch
) -> None:
    import http.client
    import urllib.request

    class BrokenResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            if not hasattr(self, "started"):
                self.started = True
                return b"partial"
            raise http.client.IncompleteRead(b"", 100)

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_args, **_kwargs: BrokenResponse()
    )
    assert demo._lineage_table(tmp_path) is None
    assert not (tmp_path / demo.LINEAGE_CACHE_NAME).exists()
    assert not (tmp_path / (demo.LINEAGE_CACHE_NAME + ".part")).exists()


def test_refresh_kingdoms_upgrades_an_old_cache_once(tmp_path: Path) -> None:
    import gzip

    with gzip.open(tmp_path / demo.LINEAGE_CACHE_NAME, "wt") as f:
        f.write("Entry\tTaxonomic lineage\n")
        for acc, lineage in _LINEAGES.items():
            f.write(f"{acc}\t{lineage}\n")
    accs = np.array(["P43747", "Q8YXR5", "NOT_IN_TABLE"], dtype=object)
    meta = {
        "names": np.array(["a", "b", "c"], dtype=object),
        "organisms": np.array(
            ["Haemophilus influenzae", "Nostoc sp.", "Homo sapiens"], dtype=object
        ),
        "kingdoms": np.array(["Viruses", "Other Eukaryotes", "Human"], dtype=object),
        "accessions": accs,
    }
    cache = tmp_path / "metadata_all.npz"
    np.savez(cache, **meta)
    fresh = demo.refresh_kingdoms(tmp_path, cache, meta)
    # The last one is not in the table, so the name rules still decide it.
    assert list(fresh["kingdoms"]) == ["Proteobacteria", "Other Bacteria", "Human"]
    stored = np.load(cache, allow_pickle=True)
    assert str(stored["kingdom_source"]) == demo.KINGDOM_SOURCE
    assert list(stored["kingdoms"]) == list(fresh["kingdoms"])
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == [cache.name]
    # Stamped: a second call touches nothing, even with the table gone.
    (tmp_path / demo.LINEAGE_CACHE_NAME).unlink()
    again = demo.refresh_kingdoms(tmp_path, cache, {k: stored[k] for k in stored.files})
    assert list(again["kingdoms"]) == list(fresh["kingdoms"])
    assert not (tmp_path / demo.LINEAGE_CACHE_NAME).exists()


def test_refresh_kingdoms_leaves_a_cache_without_accessions(tmp_path: Path) -> None:
    meta = {
        "organisms": np.array(["x"], dtype=object),
        "kingdoms": np.array(["Plants"], dtype=object),
    }
    assert list(
        demo.refresh_kingdoms(tmp_path, tmp_path / "m.npz", meta)["kingdoms"]
    ) == ["Plants"]
    assert not (tmp_path / "m.npz").exists()


def test_accessions_are_recovered_from_the_matching_cached_fasta(
    tmp_path: Path,
) -> None:
    import gzip

    with gzip.open(tmp_path / "uniprot_sprot.fasta.gz", "wt") as f:
        f.write(">sp|P43747|X_HAEIN Protein one OS=Haemophilus influenzae OX=71421\n")
        f.write("MKV\n")
        f.write(">sp|Q8YXR5|Y_NOSS1 Protein two OS=Nostoc sp. OX=103690\n")
        f.write("MAL\n")
    names = np.array(["Protein one", "Protein two"], dtype=object)
    got = demo._recover_accessions(tmp_path, {"names": names})
    assert got is not None and list(got) == ["P43747", "Q8YXR5"]
    # A cache the FASTA does not describe row for row is left alone.
    other = np.array(["Protein two", "Protein one"], dtype=object)
    assert demo._recover_accessions(tmp_path, {"names": other}) is None


def test_refresh_preserves_existing_cache_if_write_fails(
    tmp_path: Path, monkeypatch
) -> None:
    import gzip

    with gzip.open(tmp_path / demo.LINEAGE_CACHE_NAME, "wt") as f:
        f.write("Entry\tTaxonomic lineage\nP43747\t" + _LINEAGES["P43747"] + "\n")
    cache = tmp_path / "metadata_all.npz"
    meta = {
        "names": np.array(["a"], dtype=object),
        "organisms": np.array(["Haemophilus influenzae"], dtype=object),
        "kingdoms": np.array(["Viruses"], dtype=object),
        "accessions": np.array(["P43747"], dtype=object),
    }
    np.savez(cache, **meta)
    original = cache.read_bytes()

    def interrupted_save(path, **_fields):
        Path(path).write_bytes(b"truncated zip")
        raise OSError("disk full")

    monkeypatch.setattr(demo.np, "savez", interrupted_save)
    with pytest.raises(OSError, match="disk full"):
        demo.refresh_kingdoms(tmp_path, cache, meta)
    assert cache.read_bytes() == original
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == [cache.name]


def test_both_demos_use_recovered_accessions_on_first_load(tmp_path: Path) -> None:
    import gzip

    from luxar.demos.demo_esm3_protein_stories import load_landscape_cache

    with gzip.open(tmp_path / "uniprot_sprot.fasta.gz", "wt") as f:
        f.write(
            ">sp|P43747|X_HAEIN Protein one OS=Haemophilus influenzae OX=71421\nMKV\n"
        )
        f.write(">sp|Q8YXR5|Y_NOSS1 Protein two OS=Nostoc sp. OX=103690\nMAL\n")
    with gzip.open(tmp_path / demo.LINEAGE_CACHE_NAME, "wt") as f:
        f.write("Entry\tTaxonomic lineage\n")
        for acc in ("P43747", "Q8YXR5"):
            f.write(f"{acc}\t{_LINEAGES[acc]}\n")
    np.savez(tmp_path / "umap3d_esmc_300m_all.npz", positions=np.zeros((2, 3)))
    metadata_cache = tmp_path / "metadata_all.npz"

    def write_old_metadata() -> None:
        np.savez(
            metadata_cache,
            names=np.array(["Protein one", "Protein two"], dtype=object),
            organisms=np.array(["Haemophilus influenzae", "Nostoc sp."], dtype=object),
            kingdoms=np.array(["Viruses", "Other Eukaryotes"], dtype=object),
        )

    write_old_metadata()
    scene = tmp_path / "landscape.luxar.zarr"
    generate_esm3_landscape(scene, sample_size=0, cache_dir=tmp_path)
    import zarr

    proteins = zarr.open_group(str(scene), mode="r")["proteins"]
    assert (
        dict(proteins.attrs)["link"]
        == "https://www.uniprot.org/uniprotkb/{hover_key}/entry"
    )
    assert set(_decode_strings(proteins, "key")) == {"P43747", "Q8YXR5"}

    write_old_metadata()
    _, fields = load_landscape_cache(tmp_path)
    assert list(fields["kingdoms"]) == ["Proteobacteria", "Other Bacteria"]
    assert list(fields["accessions"]) == ["P43747", "Q8YXR5"]
