#### Sound no longer piles up behind a blocked audio context

A browser will not start an `AudioContext` without a gesture, and the viewer
deferred each `on_arrive` narration until it could. One was held per node, so a
listener who walked several stops before sound started heard all of them at
once the moment it did. A deferred waypoint trigger is now superseded by the
next waypoint event, so at most one is ever waiting.

The gate that enables sound is also no longer a single attempt. It is decided
from `context.state` rather than the outcome of `resume()`, whose promise a
blocked context may leave unsettled, and it reopens from the context's own
state change, from a one-shot pointer or key listener, or from `enableSound()`.

The rail's Sound button gains a third state for this. "Blocked by the browser"
is not the same as "muted": nothing is audible either way, but only one of them
is the listener's choice, and clicking a blocked button resumes the context
instead of toggling a mute they never set.
