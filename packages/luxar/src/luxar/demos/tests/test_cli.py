"""Tests for the shared demo CLI runner."""

import pytest

from luxar.demos._support.runtime.cli import run_luxar_cli


def test_run_luxar_cli_accepts_success_exit(monkeypatch) -> None:
    from luxar.cli import main

    monkeypatch.setattr(main, "app", lambda _args: (_ for _ in ()).throw(SystemExit(0)))

    run_luxar_cli("info", "scene.luxar.zarr")


def test_run_luxar_cli_translates_failure_exit(monkeypatch) -> None:
    from luxar.cli import main

    monkeypatch.setattr(main, "app", lambda _args: (_ for _ in ()).throw(SystemExit(7)))

    with pytest.raises(RuntimeError, match=r"`luxar info broken.zarr` failed: exit 7"):
        run_luxar_cli("info", "broken.zarr")
