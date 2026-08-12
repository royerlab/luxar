#### Cap numpy below 2.5 to keep the mypy gate green

numpy 2.5.0 rewrote its type stubs to use PEP 695 `type` statements (62 of them in
`numpy/__init__.pyi`). mypy checks third-party stubs against the configured target
(`python_version = "3.10"`), where a `type` statement is a syntax error — so a fresh
CI environment that resolved numpy 2.5.x aborted the type-check step ("Type statement
is only supported in Python 3.12 and greater") and turned `main` and every open PR red.
numpy is capped to `>=2.0,<2.5` (the newest clean stubs are 2.4.6) until the mypy target
moves to 3.12 or numpy and mypy reconcile.
