"""A demo that preprocesses must tell the fit what its source actually was.

Every fit entry point casts its volume to float32 internally, so a demo that
ALSO casts before calling one has put a second cast in front of the first: the
fitter then measures a float32 working copy and the stamped ``source_bytes`` --
the denominator of the published compression ratio -- is inflated. Measured on
the DAPI demo, the working copy said 83:1 where the acquisition says 349:1.

The fix per demo is to pass ``source_dtype`` (and ``source_shape`` when the grid
was changed too). This gate makes the remaining work enumerable instead of
remembered: a demo either declares, or appears below with a reason.
"""

from __future__ import annotations

import ast

from luxar.demos import registry

FIT_CALLS = {
    "fit_gaussian_splats",
    "fit_progressive_gaussian_splats",
    "fit_tiled",
}

#: Demos that fit but legitimately need no declaration, and why.
#:
#: The bar is that the array handed to the fitter genuinely IS the source --
#: not that declaring would be inconvenient.
_IS_ITS_OWN_SOURCE = {
    "demo_gsplats_4d_nexrad_supercell.py": (
        "there IS a download (82 Level II scans, 800 MB gzipped) but it is not "
        "an array of this volume: it is polar sweeps carrying several moments "
        "and elevation angles, of which the demo re-grids ONE moment inside one "
        "box. Quoting 800 MB against the splats would credit the fit for "
        "dropping the other moments and for the polar-to-Cartesian resampling "
        "-- a format-conversion ratio, not a compression one. The Cartesian "
        "grid it builds is float32 by construction (not a cast) and its size is "
        "set by --grid-m, so measuring the array handed to the fitter is both "
        "the honest denominator and what the default already records"
    ),
    "demo_quantum_orbitals.py": (
        "the volume is evaluated from a closed-form wavefunction, so it has no "
        "acquisition at all"
    ),
    "demo_storm_3d_microtubules.py": (
        "the fitted volume is synthesized directly from localization-event photon "
        "counts and a PSF, so that float32 raster is the source; treating the CSV "
        "event table as an image acquisition would measure format conversion rather "
        "than volume compression"
    ),
}

#: Demos still to be wired. SHRINKS to empty; entries are work, not exemptions.
#:
#: These fit AND preprocess first, so a dataset regenerated from one today would
#: publish a compression ratio against the working copy — the same defect the
#: wired demos had. None is shipped in ``demos/data`` and none is assigned to a
#: Zenodo record (all ``record: None``), which is why they are not urgent.
#:
#: WATCH ``gsplats_flylight_mcfo``: it is a manifest entry with ``files: []``,
#: i.e. awaiting upload. Wire ``demo_gsplats_3d_flylight_mcfo_neurons.py``
#: BEFORE that dataset is uploaded, or it goes to Zenodo with an inflated ratio.
_NOT_YET_DECLARING = {
    "demo_gsplats_3d_acto3d_heart.py",
    "demo_gsplats_3d_tng_cosmic_web.py",
    "demo_gsplats_3d_tribolium_embryo.py",
    "demo_gsplats_2d_codex_pancreas.py",
}


def _fitting_demos() -> dict[str, bool]:
    """Demo file name -> whether it declares a source to at least one fit call."""
    out: dict[str, bool] = {}
    for path in sorted(registry._DEMOS_DIR.glob("demo_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        fits = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FIT_CALLS
        ]
        if not fits:
            continue
        out[path.name] = any(
            kw.arg in ("source_dtype", "source_shape")
            for node in fits
            for kw in node.keywords
        )
    return out


def test_every_fitting_demo_declares_its_source_or_is_listed() -> None:
    fitting = _fitting_demos()
    undeclared = {name for name, declares in fitting.items() if not declares}
    accounted = set(_IS_ITS_OWN_SOURCE) | _NOT_YET_DECLARING
    assert undeclared <= accounted, (
        "a demo fits without declaring its source and is not listed:\n  "
        + "\n  ".join(sorted(undeclared - accounted))
        + "\nPass source_dtype (and source_shape if the grid changed), or add it "
        "to _IS_ITS_OWN_SOURCE with the reason its input IS the acquisition."
    )


def test_the_pending_list_shrinks_and_does_not_go_stale() -> None:
    """A demo that has been wired must be removed from the pending list.

    Without this the list would quietly become a permanent exemption set, which
    is precisely what it must not be.
    """
    fitting = _fitting_demos()
    for name in sorted(_NOT_YET_DECLARING):
        assert name in fitting, f"{name} no longer fits — drop it from the list"
        assert not fitting[name], f"{name} now declares its source — remove it"
    for name in sorted(_IS_ITS_OWN_SOURCE):
        assert name in fitting, f"{name} no longer fits — drop the exemption"


def test_every_exemption_says_why_the_input_is_the_acquisition() -> None:
    for name, reason in _IS_ITS_OWN_SOURCE.items():
        assert len(reason) > 40, f"{name}: state why there is no separate source"


def test_the_detector_actually_finds_fit_calls() -> None:
    """A scan that matched nothing would pass every assertion above."""
    fitting = _fitting_demos()
    assert len(fitting) >= 10, f"only found {len(fitting)} fitting demos"
    assert any(fitting.values()), "no demo detected as declaring — detector is blind"
