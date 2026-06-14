"""Command-group modules for the ``luxar gsplat`` CLI.

Each module owns one thematic group of subcommands and exposes a
``register_<group>(app)`` that wires its commands onto the shared ``app_gsplat``
Typer (defined in ``cli/gsplat_commands.py``). The aggregator imports and calls
them — keeping ``gsplat_commands.py`` a thin registration surface rather than a
4.7k-line god-file.
"""
