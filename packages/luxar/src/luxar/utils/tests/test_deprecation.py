"""Tests for the post-release deprecation helpers.

These helpers have no production caller yet — pre-release renames are hard
cuts — so the tests are the only thing pinning their contract until the first
post-release alias uses them. Each assertion is about what a user of a
deprecated name would see: the warning category (so ``-W error`` catches it),
the sentence (so the release and the window are in it), and the attribution
(so the traceback points at *their* code, not at Luxar).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

import luxar.utils
from luxar.utils.deprecation import (
    deprecated_kwarg_alias,
    deprecation_message,
    warn_deprecated,
)

WINDOW: dict[str, str] = {"since": "2026.10.01", "remove_after": "2027.04.01"}


def test_message_names_old_new_release_and_window() -> None:
    message = deprecation_message("optimise", "optimize", **WINDOW)
    assert message == (
        "optimise is deprecated since Luxar 2026.10.01 and will be removed "
        "after 2027.04.01; use optimize instead."
    )


def test_message_without_a_replacement_drops_the_use_clause() -> None:
    message = deprecation_message("--legacy", None, **WINDOW)
    assert message == (
        "--legacy is deprecated since Luxar 2026.10.01 and will be removed "
        "after 2027.04.01."
    )
    assert "instead" not in message


def test_warn_deprecated_emits_a_deprecation_warning_with_the_message() -> None:
    with pytest.warns(DeprecationWarning, match=r"optimise is deprecated") as rec:
        warn_deprecated("optimise", "optimize", **WINDOW)
    assert len(rec) == 1
    assert str(rec[0].message) == deprecation_message("optimise", "optimize", **WINDOW)


def _deprecated_function() -> None:
    """Stand-in for a deprecated public function that warns on entry."""
    warn_deprecated("_deprecated_function", "_new_function", **WINDOW)


def test_default_stacklevel_attributes_the_warning_to_the_users_call_site() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _deprecated_function()  # <- this line is what the warning must point at
    assert len(rec) == 1
    assert rec[0].filename == __file__
    this_test = test_default_stacklevel_attributes_the_warning_to_the_users_call_site
    assert rec[0].lineno == this_test.__code__.co_firstlineno + 3


def test_kwarg_alias_forwards_the_value_and_warns() -> None:
    kwargs: dict[str, Any] = {"reveal_centre": (1, 2, 3), "other": 1}
    with pytest.warns(DeprecationWarning, match=r"reveal_centre= is deprecated"):
        returned = deprecated_kwarg_alias(
            kwargs, "reveal_centre", "reveal_center", **WINDOW
        )
    assert returned is kwargs
    assert kwargs == {"reveal_center": (1, 2, 3), "other": 1}


def test_kwarg_alias_is_silent_when_the_old_name_is_absent() -> None:
    kwargs: dict[str, Any] = {"reveal_center": (1, 2, 3)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        deprecated_kwarg_alias(kwargs, "reveal_centre", "reveal_center", **WINDOW)
    assert kwargs == {"reveal_center": (1, 2, 3)}


def test_kwarg_alias_rejects_both_spellings_at_once() -> None:
    kwargs: dict[str, Any] = {"reveal_centre": 1, "reveal_center": 2}
    with pytest.raises(TypeError, match=r"both 'reveal_centre'.*'reveal_center'"):
        deprecated_kwarg_alias(kwargs, "reveal_centre", "reveal_center", **WINDOW)
    # Nothing was moved: the caller's mistake is left intact for them to fix.
    assert kwargs == {"reveal_centre": 1, "reveal_center": 2}


@pytest.mark.parametrize(
    "name", ["warn_deprecated", "deprecated_kwarg_alias", "deprecation_message"]
)
def test_helpers_are_public_on_the_utils_barrel(name: str) -> None:
    assert name in luxar.utils.__all__
    assert getattr(luxar.utils, name) is getattr(luxar.utils.deprecation, name)
