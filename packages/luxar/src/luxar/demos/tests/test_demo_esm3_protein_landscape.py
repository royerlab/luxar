"""Tests for the ESM-3 demo's cache/dependency gate.

The demo's embeddings cache is quarantined to ``<name>.corrupt`` when it fails
validation. A quarantine left behind by an *earlier* run used to be invisible:
the ``.npy`` is already gone, so the validation block never fires and the demo
silently restarted a multi-gigabyte compute/download. The gate must name the
quarantined path, its size, and what to do about it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from luxar.demos import demo_esm3_protein_landscape as demo
from luxar.demos import require_module
from luxar.demos._dependencies import MissingDependencyError
from luxar.demos.demo_esm3_protein_landscape import (
    _compute_esm3_embeddings,
    generate_esm3_landscape,
)
from luxar.utils.download import QUARANTINE_SUFFIX

SEQUENCES = ["MKV", "MTL", "MGG"]
HEAVY_DEPS = ("torch", "esm", "umap")


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


def _write_instant_cache(cache_dir: Path, n: int = 40) -> int:
    """Fabricate the two artifacts the instant path reads (sample_size=0/esmc-300m).

    Returns ``n`` so callers can assert the generator's protein count (the scene
    itself carries ``2 * n`` rows — one block per coloring).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((n, 3)).astype(np.float32)
    np.savez(cache_dir / "umap3d_esmc_300m_all.npz", positions=positions)

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
    when every expensive cache was supplied. The fix gates LOD on both and falls
    back to flat Points, so the whole demo runs with either one blocked.
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

        # The fallback must ANNOUNCE the degradation (arbol aprint → stdout, which
        # capsys captures — the same channel the quarantine-notice tests assert on).
        out = capsys.readouterr().out
        assert "skipping Points LOD" in out, (
            "fallback did not print its degradation notice"
        )
        assert blocked in out, f"degradation notice did not name {blocked}"

        # Structural proof of the FALLBACK, independent of import ordering. A
        # None entry in sys.modules only blocks a *fresh* import; in a warm suite
        # where a module already imported the real package, its namespace keeps
        # the binding, so the old (unconditional-LOD) code would build a
        # substitutive-LOD group and NOT crash. Assert the on-disk shape instead:
        # a flat Points leaf writes `proteins/positions` and has no `kind: lod`,
        # whereas a substitutive-LOD group has `kind == "lod"` and child_0..N
        # (with no top-level positions) — so this fails against the old code in
        # both cold and warm orderings.
        proteins = out_path / "proteins"
        assert (proteins / "positions").exists(), (
            "flat Points leaf missing positions — LOD group written instead"
        )
        attrs = json.loads((proteins / ".zattrs").read_text())
        assert attrs.get("kind") != "lod", (
            f"expected a flat Points leaf, got a substitutive-LOD group: {attrs.get('kind')!r}"
        )

    def test_full_cache_path_builds_lod_when_deps_present(
        self, tmp_path, capsys
    ) -> None:
        """Positive branch: with torch and scipy installed, the LOD ladder IS built.

        Pins the deps-PRESENT path so a mutant that drops LOD entirely — or a
        typo like ``is_installed("torchvision")`` — cannot pass silently by only
        satisfying the fallback test above. Substitutive-LOD writes a
        ``kind == "lod"`` group with ``child_0..N`` and NO top-level positions,
        and the degradation notice must stay quiet.
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
        assert "skipping Points LOD" not in capsys.readouterr().out, (
            "degradation notice printed even though torch and scipy are installed"
        )
        proteins = out_path / "proteins"
        attrs = json.loads((proteins / ".zattrs").read_text())
        assert attrs.get("kind") == "lod", (
            f"expected a substitutive-LOD group with the deps present: {attrs.get('kind')!r}"
        )
        assert (proteins / "child_0").exists(), "LOD group missing child_0"
        assert not (proteins / "positions").exists(), (
            "LOD group must not write top-level positions"
        )
