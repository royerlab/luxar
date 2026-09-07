"""Command bodies for the ``luxar env`` group (registered by ``cli/env_commands.py``).

Registration order is ``--help`` order: ``bake`` (the whole loop) before ``attach``
(its manual half). Each body is a thin error funnel over :mod:`luxar.environment`, so the
CLI-to-library boundary test keeps the real work importable from Python.
"""

from .attach_commands import register_attach_commands
from .bake_commands import register_bake_commands

__all__ = ["register_attach_commands", "register_bake_commands"]
