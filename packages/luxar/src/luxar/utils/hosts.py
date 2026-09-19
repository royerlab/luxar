"""Bind-address predicates shared across layers.

Small on purpose, and here rather than in ``luxar.cli`` because two different
layers need the same answer: the CLI decides a CORS policy from it, and the
demo runner decides whether to warn that a display has been opened to the
network. ``luxar.demos`` sits ABOVE ``luxar.cli`` in the layering contract, so
a demo importing the CLI's copy is a layering violation — and duplicating the
predicate would let the warning and the policy drift apart, which is the worse
outcome of the two.
"""

from __future__ import annotations

from typing import Optional

# The wildcard bind sentinels. Binding one of these listens on every
# interface, so the address a client actually used is not knowable from the
# bind string alone.
ALL_INTERFACES_HOSTS = frozenset({"0.0.0.0", "::", ""})  # nosec B104

# The loopback spellings, bare (an IPv6 literal may also arrive bracketed).
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback_host(host: Optional[str]) -> bool:
    """True when ``host`` reaches only this machine.

    A wildcard bind is **not** loopback: it listens on every interface, which
    is the whole case the callers exist to notice. An unset host is treated as
    loopback, so a caller that never says where it bound gets the safe answer
    rather than a warning it cannot act on.
    """
    if host is None:
        return True
    normalised = host.strip().lower()
    if normalised in ALL_INTERFACES_HOSTS:
        return False
    # An IPv6 literal is bracketed in a URL authority but bare as a bind
    # address, and both spellings reach this.
    return normalised.strip("[]") in LOOPBACK_HOSTS
