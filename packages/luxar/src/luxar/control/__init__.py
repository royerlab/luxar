"""Driving a running Luxar viewer from Python.

A viewer launched with ``?control`` attaches to the hub that
``luxar serve --control`` exposes; this package is the other end of that hub.
The method names are the viewer's own embedder API, so anything the browser's
``LuxarApp`` can do, a script here can ask for:

```python
from luxar.control import Viewer

with Viewer("ws://kiosk.local:5173/control") as viewer:
    state = viewer.get_viewer_state()
    story = state["dimensions"]["metadata"].index(
        next(d for d in state["dimensions"]["metadata"] if d["name"] == "story")
    )
    viewer.set_dimension_value(story, 3)      # fly to the fourth chapter
```

What it is for: scripted demos, reproducible screenshots, driving a kiosk from
a cron job, and prototyping an agent. What it is not: a second API. If a call
is not in the viewer's embedder surface it is not here either — see
``packages/luxar-viewer/src/core/app/control/method-policy.ts`` for the list
and ``docs/guides/specs/REMOTE_CONTROL_SPEC.md`` for the contract.
"""

from .viewer import ControlError, Viewer

__all__ = ["ControlError", "Viewer"]
