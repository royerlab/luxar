"""Property-based tests for transform composition algebra.

Complement the example-based tests in ``test_transforms.py`` with the group
axioms checked over many machine-generated transforms (Hypothesis): identity is
a two-sided neutral element, composition is associative, and ``inverse``
genuinely undoes a transform. Ranges are kept moderate so the float32 matrix
products stay well inside the assertion tolerance.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from luxar.core.transforms import (
    compose,
    identity,
    inverse,
    rotate_x,
    rotate_y,
    rotate_z,
    scale,
    translate,
)

_coord = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_angle = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
_scale = st.floats(min_value=0.2, max_value=5.0, allow_nan=False, allow_infinity=False)


@st.composite
def _transform(draw: st.DrawFn) -> np.ndarray:
    kind = draw(st.sampled_from(["translate", "scale", "rot_x", "rot_y", "rot_z"]))
    if kind == "translate":
        return translate(draw(_coord), draw(_coord), draw(_coord))
    if kind == "scale":
        return scale(draw(_scale), draw(_scale), draw(_scale))
    if kind == "rot_x":
        return rotate_x(draw(_angle))
    if kind == "rot_y":
        return rotate_y(draw(_angle))
    return rotate_z(draw(_angle))


@given(t=_transform())
def test_identity_is_two_sided_neutral(t: np.ndarray) -> None:
    np.testing.assert_allclose(compose(identity(), t), t, atol=1e-5)
    np.testing.assert_allclose(compose(t, identity()), t, atol=1e-5)


@given(a=_transform(), b=_transform(), c=_transform())
def test_composition_is_associative(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> None:
    left = compose(compose(a, b), c)
    right = compose(a, compose(b, c))
    np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-4)


@given(t=_transform())
def test_inverse_undoes_transform(t: np.ndarray) -> None:
    np.testing.assert_allclose(compose(t, inverse(t)), identity(), atol=1e-4)
    np.testing.assert_allclose(compose(inverse(t), t), identity(), atol=1e-4)
