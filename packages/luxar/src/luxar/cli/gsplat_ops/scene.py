"""``luxar gsplat scene`` command registration + compatibility wrappers."""

from __future__ import annotations

from typing import Any

from .scene_commands import convert_to_scene as _convert_to_scene_cmd
from .scene_commands import migrate_format_command as _migrate_format_command_cmd
from .scene_commands import reencode_command as _reencode_command_cmd
from .scene_commands import (
    register_scene_commands as register_scene_commands,
)


def convert_to_scene(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the convert command function."""
    return _convert_to_scene_cmd(*args, **kwargs)


def migrate_format_command(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the migrate-format command function."""
    return _migrate_format_command_cmd(*args, **kwargs)


def reencode_command(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the reencode command function."""
    return _reencode_command_cmd(*args, **kwargs)
