#### Gate Python 3.14 typing compatibility

CI and `make type-check-python` now run an additional host-platform mypy pass
pinned to Python 3.14, with an isolated cache. Together with the existing Python
3.12 baseline, this brackets the supported interpreter range for stdlib typeshed
and Luxar's own version-conditional typing.
