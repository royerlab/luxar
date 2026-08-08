"""Regression tests for per-invocation parallel-fit staging directories.

The single-GPU parallel tiled fit (``fit --tiling uniform -j N``) and the
content-box path (``fit --tiling content -j N``) each hand a staging
``tmp_dir`` to a helper that clean-slates it (``rmtree`` + ``mkdir``). If that
dir were derived only from the output path, two concurrent ``fit -j`` runs
targeting the SAME output would delete each other's in-progress tiles mid-run
(issue #1040). The staging dir is now made per-invocation by appending a
unique host+pid+random token; these tests pin that isolation.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

from luxar.cli.gsplat_ops.fitting.fit_utils import (
    _invocation_token,
)
from luxar.cli.gsplat_ops.fitting.fit_utils import (
    _parallel_staging_dir as _tiles_staging_dir,
)
from luxar.cli.gsplat_ops.planner import (
    _internal_plan_json,
)
from luxar.cli.gsplat_ops.planner import (
    _parallel_staging_dir as _boxes_staging_dir,
)


def test_tiles_staging_dir_includes_token() -> None:
    """The uniform-tile staging dir carries the token, not the bare name."""
    output = Path("/tmp/scene.gsplats.zarr")
    token = "host42-12345"
    staging = _tiles_staging_dir(output, token)

    assert staging != output.parent / f".{output.name}.tiles"
    assert staging.name == f".{output.name}.tiles.{token}"
    assert staging.parent == output.parent


def test_boxes_staging_dir_includes_token() -> None:
    """The content-box staging dir carries the token, not the bare name."""
    output = Path("/tmp/scene.gsplats.zarr")
    token = "host42-12345"
    staging = _boxes_staging_dir(output, token)

    assert staging != output.parent / f".{output.name}.boxes"
    assert staging.name == f".{output.name}.boxes.{token}"
    assert staging.parent == output.parent


def test_tiles_distinct_tokens_give_distinct_dirs() -> None:
    """Two invocations (distinct tokens) never share a staging dir."""
    output = Path("/tmp/scene.gsplats.zarr")
    a = _tiles_staging_dir(output, "hostA-111")
    b = _tiles_staging_dir(output, "hostB-222")

    assert a != b


def test_boxes_distinct_tokens_give_distinct_dirs() -> None:
    """Two invocations (distinct tokens) never share a staging dir."""
    output = Path("/tmp/scene.gsplats.zarr")
    a = _boxes_staging_dir(output, "hostA-111")
    b = _boxes_staging_dir(output, "hostB-222")

    assert a != b


def test_internal_plan_json_includes_token() -> None:
    """The internal plan JSON carries the token, not the bare name."""
    output = Path("/tmp/scene.gsplats.zarr")
    token = "host42-12345"
    plan = _internal_plan_json(output, token)

    assert plan != output.parent / f".{output.name}.plan.json"
    assert plan.name == f".{output.name}.plan.{token}.json"
    assert plan.parent == output.parent


def test_internal_plan_json_distinct_tokens_give_distinct_paths() -> None:
    """Two invocations (distinct tokens) never share a plan JSON path."""
    output = Path("/tmp/scene.gsplats.zarr")
    a = _internal_plan_json(output, "hostA-111")
    b = _internal_plan_json(output, "hostB-222")

    assert a != b


def test_invocation_token_unique_per_call() -> None:
    """The production token is unique even for same-host, same-pid calls.

    host+pid alone can collide (threads in one process; containers with
    identical hostnames and PID-namespaced pids sharing a mount), so the
    token carries a random component.
    """
    tokens = {_invocation_token() for _ in range(50)}

    assert len(tokens) == 50


def test_invocation_token_shape() -> None:
    """The token names its run (host, pid) and is a single path component."""
    token = _invocation_token()

    assert token.startswith(f"{socket.gethostname()}-{os.getpid()}-")
    assert "/" not in token and os.sep not in token
