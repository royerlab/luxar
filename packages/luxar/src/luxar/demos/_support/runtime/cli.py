"""Helpers for demos that rebuild data through the shipped CLI."""

from importlib import import_module

from arbol import aprint


def run_luxar_cli(*args: str) -> None:
    """Run the Luxar CLI in-process and translate a failing exit."""
    app = import_module("luxar.cli.main").app

    aprint(f"$ luxar {' '.join(args)}")
    try:
        app(list(args))
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise RuntimeError(f"`luxar {' '.join(args)}` failed: exit {exc.code}")
