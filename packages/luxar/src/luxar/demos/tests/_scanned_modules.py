"""The set of demo-package modules the demo guards must scan.

Seventeen guards across ten modules read this set: four in
``test_demos_dependencies.py``, one in
``test_no_entrypoint_dependency_preflight.py``, one in
``test_substitutive_lod_gated.py``, one in ``test_demo_layers.py`` (the
Layers-panel lint — the first consumer that is not about optional dependencies;
it reuses the set for the same reason, that a shared helper authors scene nodes
too), one in ``test_demo_import_spelling.py`` (the import-spelling lint —
two invariants, the single ``luxar.demos`` spelling and the barrel actually
re-exporting what the demos ask of it; it widens the set with the three
``luxar/gsplats/**/demos`` trees), one in
``test_demos_tone_mapping_policy.py`` (the tone-mapping lint — ACES unless the
module carries a justified exception; it too covers shared helpers, since
``_interop_common.build_interop_scene`` sets a scene's tone mapping by
default), one in ``test_demo_fit_provenance.py`` (the fit-provenance lint —
``_interop_common.py`` saves gsplat stores, so an ``include_fitting_info=False``
save could hide in a helper), one in ``test_demo_caption_coverage.py`` (the
standard-caption lint, including shared helpers that can author overlays), one
in ``test_demo_link_templates.py`` (the canonical click-through registry and
its unclaimed-literal backstop, including links authored in shared helpers,
mappings and constants; unlike the collection-time consumers below, it calls
this function inside its test), and
five in ``test_demos_cinematic_mode.py`` (each scene passes a non-None viewer
config, every config enables the preset, authored cameras leave the FOV unpinned
and compose for 63°, scientific-fidelity overrides stay explicit, and the
Python FOV constants match the viewer contract). The dependency guards
all used to enumerate ``demo_*.py`` only, which left the package's SHARED helper
modules unscanned. That became a real blind spot when ``_roundtrip_common.py``
moved a ``require_module("matplotlib.pyplot")`` gate out of five ``demo_*.py``
files into one shared module: the invariants those guards advertise — every
gated module is in ``INSTALL_SPECS``, no runtime ``pip install``, no unbounded
install hint, no entry-point preflight, no ungated ``substitutive_lod`` —
silently stopped covering it. Shared helpers build scene nodes as well as gate
imports, so the LOD guard needs the wider set for the same reason
(``_interop_common.build_interop_scene`` calls ``add_gsplats_from_file``).

Widening the file set is necessary but NOT sufficient for the last of those
invariants: the preflight guard walks outward from ``main()``, so a module with
no entry point of its own had every function unreachable and passed vacuously.
``test_no_entrypoint_dependency_preflight._entry_reachable`` seeds all
module-level ``def``s for such modules; the two changes only work together.

The set is a DENYLIST on purpose: every ``*.py`` directly under ``demos/``
*except* :data:`EXCLUDED`, plus every helper recursively below
``demos/_support/`` except package barrels. An allowlist keyed on a filename
pattern would be opt-in, so a future
``demos/_plot_helpers.py`` or ``demos/_support/_plot_helpers.py`` would escape
all seventeen guards and reopen the very blind spot this module exists to close.

The flip side of a denylist: ANY ``*.py`` dropped into ``demos/`` or
``demos/_support/`` joins the guarded set, tracked by git or not. A scratch file
such as ``demos/tmp_probe.py`` with a top-level ``import umap`` therefore turns
the suite red (with a message that calls it a demo module) where the old
``demo_*.py`` glob would have ignored it. That is the intended trade — an
untracked module in the package directory is importable and can carry the same
defects — but put throwaway scripts in ``delme/`` rather than here.

Two modules directly under ``demos/`` are excluded because they are
infrastructure rather than demo code, and one would produce a *false* positive:

``_dependencies.py``
    IS the gate and the table. Its module docstring quotes a bare
    ``pip install anndata`` as the anti-example that rule 2 forbids, so the
    unbounded-install-hint guard (which reads docstrings too) would flag the
    very text that documents the rule.
``__init__.py``
    A pure re-export barrel; it holds no demo code and no gates.

Every ``__init__.py`` below ``_support/`` is excluded for the same barrel-only
reason. Every other Python module in that subtree is scanned recursively.

``registry.py`` is deliberately NOT excluded — it passes all seventeen guards, so
there is no reason to carve it out.

Not a test module and not a demo (no ``test_`` / ``demo_`` prefix), so neither
pytest collection nor the demo import smoke test picks it up.

One operational note: several guards call :func:`scanned_demo_modules` before any
test runs — four ``pytest.mark.parametrize`` decorators, plus module-level
constants in ``test_demo_layers.py``, ``test_substitutive_lod_gated.py`` and
``test_demo_fit_provenance.py`` — so its assertions run at COLLECTION time. A trip
therefore aborts the whole session with ``Interrupted: 1 error during collection``
and reports zero test results, rather than failing one test — an alarming-looking
symptom for a deliberate tripwire, so recognise it as this.
"""

from __future__ import annotations

from pathlib import Path

#: ``.../luxar/demos`` — this file lives in ``.../luxar/demos/tests``.
DEMOS_DIR = Path(__file__).resolve().parent.parent

#: Modules directly under ``demos/`` that the guards must NOT scan. See the module
#: docstring for the justification of each; do not extend this without one.
EXCLUDED = frozenset({"__init__.py", "_dependencies.py"})

#: Shared helpers that must always be scanned. Pinned so that excluding one
#: (or breaking the discovery below) fails loudly instead of quietly shrinking
#: the guarded set — the failure mode this module was written to prevent.
REQUIRED_SHARED_HELPERS = frozenset(
    {
        "_caption.py",
        "_graph_common.py",
        "_interop_common.py",
        "_roundtrip_common.py",
        "_support/_fields.py",
        "_support/_umap_utils.py",
        "_support/datasets/bundles.py",
        "_support/datasets/cache.py",
        "_support/datasets/data_fetch.py",
        "_support/datasets/lfs.py",
        "_support/datasets/payload_agreement.py",
        "_support/downloads/download.py",
        "_support/downloads/remote_zip.py",
        "_support/downloads/zip_safety.py",
        "_support/runtime/device.py",
        "_support/runtime/flags.py",
        "_support/runtime/provenance.py",
        "_support/runtime/viewer.py",
    }
)

#: A floor, not a count: the registry holds ~80 demos, so anything near zero
#: means discovery broke rather than that demos were deleted.
MIN_DEMO_MODULES = 50


def scanned_demo_modules(demos_dir: Path | None = None) -> list[Path]:
    """Every demo module, plus the shared helpers the demos delegate to.

    Raises:
        AssertionError: If discovery looks broken — too few ``demo_*.py``
            modules, or a shared helper missing from the result.
    """
    root = DEMOS_DIR if demos_dir is None else demos_dir
    paths = sorted(
        [p for p in root.glob("*.py") if p.name not in EXCLUDED]
        + [p for p in (root / "_support").rglob("*.py") if p.name != "__init__.py"]
    )
    names = {p.name for p in paths}
    relative_paths = {p.relative_to(root).as_posix() for p in paths}

    n_demos = sum(1 for name in names if name.startswith("demo_"))
    assert n_demos >= MIN_DEMO_MODULES, (
        f"found only {n_demos} demo_*.py modules under {root} (expected at "
        f"least {MIN_DEMO_MODULES}) — discovery or the package layout changed"
    )
    missing = REQUIRED_SHARED_HELPERS - relative_paths
    assert not missing, (
        f"shared helper module(s) {sorted(missing)} are not in the scanned set "
        f"— the dependency guards would no longer cover them"
    )
    return paths
