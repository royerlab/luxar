"""``ZarrWriterProtocol`` must actually describe ``LuxarZarrCompiler``.

A ``Protocol`` that no implementation is checked against enforces nothing, and
this one had reached that state: all four geometry write methods carried
``# type: ignore[override]``, so mypy skipped them entirely. Underneath the
silence were two real disagreements.

**The annotations.** ``io/writer.py`` defined its own ``PositionArray`` and
``ColorArray``, wider than the same names in ``typing_utils.aliases``, in a
module that also imports from it. The protocol used the wide pair and the
implementation the narrow one, so they could not agree by construction. Measured
against the running code, the WIDE pair is correct: the write path accepts
float16 positions, uint8 and uint16 colors, and float16/uint8 scalar attributes
— see ``test_the_declared_input_dtypes_are_actually_accepted`` below, which
writes a real store per dtype rather than trusting either annotation.

**The signature.** The protocol's ``write_mesh`` omitted the eleven texture
parameters the implementation takes between ``scalars`` and ``shading``. Those
are positional-or-keyword, so the omission renumbered everything after them: a
caller typed against the protocol passing ``shading`` positionally would have
landed it in ``uvs``.

mypy now checks all four methods (zero errors, no ignores). These tests add what
mypy cannot see: parameter ORDER, which ``**attrs`` and ``Any`` annotations would
otherwise let drift silently, and the absence of the ignores that would switch
the checking back off.
"""

from __future__ import annotations

import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.encoding.decoder import ArrayDecoder
from luxar.io.compiler import LuxarZarrCompiler
from luxar.io.writer import ZarrWriterProtocol

COMPILER_SOURCE = Path(inspect.getfile(LuxarZarrCompiler))


def _declared_methods() -> list[str]:
    """Every public method the protocol declares, read off the class.

    Derived, not listed. A first draft of this file hardcoded the four geometry
    writers and its own completeness check immediately failed: the protocol
    declares SIXTEEN methods, including `transaction`, the rollback pair and the
    three `*_multi_lod` ladder writers. All sixteen now agree, and a
    seventeenth is covered the moment it is added.
    """
    return sorted(
        name
        for name, value in vars(ZarrWriterProtocol).items()
        if not name.startswith("_") and inspect.isfunction(value)
    )


PROTOCOL_METHODS = tuple(_declared_methods())


def test_the_scan_found_the_protocol_methods() -> None:
    """Fail closed: an empty or truncated scan would make every test vacuous."""
    assert len(PROTOCOL_METHODS) >= 16, PROTOCOL_METHODS
    for expected in ("write_points", "write_lines", "write_mesh", "write_gsplats"):
        assert expected in PROTOCOL_METHODS


@pytest.mark.parametrize("method", PROTOCOL_METHODS)
def test_the_compiler_implements_it(method: str) -> None:
    """Every declared method must exist on the one implementation.

    `LuxarZarrCompiler` subclasses the protocol explicitly, so a missing method
    is a runtime `AttributeError` waiting on whichever caller reaches for it
    first — not something the `Protocol` base catches for us.
    """
    assert callable(getattr(LuxarZarrCompiler, method, None)), (
        f"ZarrWriterProtocol declares {method}() but LuxarZarrCompiler has no "
        f"such method."
    )


@pytest.mark.parametrize("method", PROTOCOL_METHODS)
def test_the_parameter_names_and_order_agree(method: str) -> None:
    """Same parameters, same ORDER, in the protocol and the implementation.

    Order is the half mypy cannot be relied on for: a parameter inserted in the
    middle of one signature and not the other is only an *error* when the
    shifted types happen to disagree. That is precisely how ``write_mesh``'s
    eleven texture parameters went missing from the protocol, and how an earlier
    insertion into ``_write_mesh_impl`` slipped through — mypy caught the second
    one, but by luck.
    """
    protocol_params = list(
        inspect.signature(getattr(ZarrWriterProtocol, method)).parameters
    )
    impl_params = list(inspect.signature(getattr(LuxarZarrCompiler, method)).parameters)
    assert protocol_params, f"{method} has no parameters — signature read failed"
    assert impl_params == protocol_params, (
        f"{method} signatures diverge.\n"
        f"  protocol: {protocol_params}\n"
        f"  compiler: {impl_params}\n"
        f"These are positional-or-keyword parameters: a difference in ORDER "
        f"silently renumbers every argument after it for any caller typed "
        f"against the protocol."
    )


@pytest.mark.parametrize("method", PROTOCOL_METHODS)
def test_the_parameter_defaults_agree(method: str) -> None:
    """A default that differs makes the protocol describe a different API.

    Checked separately from order so a failure says which of the two went
    wrong. Only parameters that HAVE a default are compared; requiredness is
    already covered by the name/order test above.
    """
    protocol_sig = inspect.signature(getattr(ZarrWriterProtocol, method))
    impl_sig = inspect.signature(getattr(LuxarZarrCompiler, method))
    for name, protocol_param in protocol_sig.parameters.items():
        if protocol_param.default is inspect.Parameter.empty:
            continue
        impl_default = impl_sig.parameters[name].default
        assert impl_default == protocol_param.default, (
            f"{method}({name}=...): protocol says {protocol_param.default!r}, "
            f"compiler says {impl_default!r}"
        )


def test_the_override_ignores_are_gone() -> None:
    """No ``# type: ignore[override]`` on the compiler's write methods.

    This is the switch that turned the checking off, and all four were removed
    by reconciling the signatures rather than by widening the protocol into
    vagueness.

    ``warn_unused_ignores = true`` in ``pyproject.toml`` means mypy does flag a
    *gratuitous* ignore on its own — verified, not assumed. What it cannot flag
    is the case this test is for: someone reintroduces a real mismatch and
    silences it in the same edit. mypy then considers the ignore used and says
    nothing further about that method's whole signature, exactly as before. The
    test also fails with a message naming the mechanism, which "unused
    type: ignore comment" does not.
    """
    source = COMPILER_SOURCE.read_text()
    assert "class LuxarZarrCompiler" in source, (
        f"{COMPILER_SOURCE} does not look like the compiler module — the scan "
        f"would pass vacuously"
    )
    offenders = re.findall(r"type:\s*ignore\[override\]", source)
    assert not offenders, (
        "these write methods have opted out of override checking:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    ("label", "kwargs_factory"),
    [
        ("positions float32", lambda p: {"positions": p}),
        ("positions float16", lambda p: {"positions": p.astype(np.float16)}),
        (
            "colors float32",
            lambda p: {
                "positions": p,
                "colors": np.linspace(0.1, 0.9, len(p) * 3, dtype=np.float32).reshape(
                    len(p), 3
                ),
            },
        ),
        (
            "colors uint8",
            lambda p: {
                "positions": p,
                "colors": np.arange(len(p) * 3, dtype=np.uint8).reshape(len(p), 3),
            },
        ),
        (
            "colors uint16",
            lambda p: {
                "positions": p,
                "colors": np.arange(len(p) * 3, dtype=np.uint16).reshape(len(p), 3),
            },
        ),
        (
            "radii float16",
            lambda p: {
                "positions": p,
                "radii": np.linspace(1, 2, len(p), dtype=np.float16),
            },
        ),
        (
            "radii uint8",
            lambda p: {
                "positions": p,
                "radii": np.arange(1, len(p) + 1, dtype=np.uint8),
            },
        ),
    ],
)
def test_the_declared_input_dtypes_are_actually_accepted(
    label: str, kwargs_factory, tmp_path: Path
) -> None:
    """Every dtype the aliases declare must survive a real write.

    This is the evidence the alias unification rests on. Both candidate
    annotations were self-consistent and only one could be true, so the question
    was settled by writing a store per dtype instead of by reading either
    signature. Keeping it as a test means narrowing an alias back — or a
    validator quietly rejecting a dtype — fails here rather than in a user's
    notebook.
    """
    positions = np.linspace(0, 1, 96, dtype=np.float32).reshape(32, 3)
    kwargs = kwargs_factory(positions)
    store = tmp_path / f"{label.replace(' ', '_')}.luxar.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with LuxarZarrCompiler(store) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            compiler.write_points("P", **kwargs)

    root = zarr.open_group(store, mode="r")
    point_group = root["P"]
    decoder = ArrayDecoder()
    tolerances = {"positions": 5e-4, "colors": 5e-3}
    for name, expected in kwargs.items():
        assert name in point_group, f"write_points did not write P/{name}"
        array = point_group[name]
        tolerance = tolerances.get(name)
        if name == "radii":
            encoding = dict(array.attrs["encoding"])
            assert encoding["name"] == "bounded_scalar_uint8"
            levels = 2 ** int(encoding["bits"]) - 1
            minimum = float(encoding["min"])
            value_range = float(encoding["max"]) - minimum
            tolerance = value_range / levels
            if expected.dtype == np.uint8:
                decoded = (
                    np.asarray(array[:], dtype=np.float64) / levels * value_range
                    + minimum
                )
                # #2554 tracks the decoder truncating instead of rounding integers.
                actual = np.rint(decoded)
            else:
                actual = decoder.decode(array, root)
        else:
            actual = decoder.decode(array, root)
        assert tolerance is not None
        if expected.ndim == 1:
            actual = np.sort(actual)
            expected = np.sort(expected)
        else:
            actual = actual[np.lexsort(actual.T[::-1])]
            expected = expected[np.lexsort(expected.T[::-1])]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0,
            atol=tolerance,
            err_msg=f"P/{name} did not decode back to the written {label} values",
        )
