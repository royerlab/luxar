"""JSON-attr coercion for values headed into zarr ``attrs``.

Lives in the foundation package because all three of ``core``, ``io`` and
``gsplats`` need it: the node-meta projection in the compiler, the per-adder
attribute pass in ``core.group.adders``, and the root ``pipeline/`` stats bucket
in :mod:`luxar.gsplats.io.save_gsplats`. It was in ``luxar.io._compiler``, which
made three of those call sites a ``core`` -> ``io`` back-edge (audit A1-03).

Depends on nothing but numpy, which is what makes the foundation the right home
rather than any one of its three callers.
"""

import math
from typing import Any

import numpy as np


def json_safe_value(value: Any) -> tuple[bool, Any]:
    """``(ok, converted)`` — recursively coerce ``value`` to a JSON-attr-safe
    form (numpy scalars → Python scalars; tuples → lists; nested dicts/lists
    filtered element-wise). ``ok`` is False for values with no *strictly*-JSON
    form. Shared by the node-meta projection here and the root ``pipeline/``
    stats bucket in :mod:`luxar.gsplats.io.save_gsplats`.

    Two subtleties this guards:

    * **numpy floats are checked BEFORE the Python-scalar branch.** ``np.float64``
      is a subclass of ``float``, so a ``(bool, int, float, str)`` check would
      accept it *un-coerced* and leak a numpy scalar into ``.zattrs``. numpy /
      bool checks come first so every numpy scalar is coerced to a Python one.
    * **non-finite floats are rejected** (``ok=False``). ``NaN`` / ``±Inf`` are
      not valid JSON; zarr writes them as bare ``NaN`` / ``Infinity`` tokens
      that a strict parser — notably the TypeScript viewer's ``JSON.parse`` —
      refuses, so a non-finite stat must be dropped, not persisted.
    """
    # numpy scalars first (np.float64 is a subclass of float; np.bool_ of int).
    if isinstance(value, np.integer):
        return True, int(value)
    if isinstance(value, np.floating):
        fv = float(value)
        return (True, fv) if math.isfinite(fv) else (False, None)
    if isinstance(value, np.bool_):
        return True, bool(value)
    if value is None or isinstance(value, (bool, int, str)):
        return True, value
    if isinstance(value, float):
        return (True, value) if math.isfinite(value) else (False, None)
    if isinstance(value, (list, tuple)):
        out_list = []
        for item in value:
            ok, conv = json_safe_value(item)
            if not ok:
                return False, None
            out_list.append(conv)
        return True, out_list
    if isinstance(value, dict):
        out_dict = {}
        for k, v in value.items():
            ok, conv = json_safe_value(v)
            if ok:
                out_dict[str(k)] = conv
        return True, out_dict
    return False, None
