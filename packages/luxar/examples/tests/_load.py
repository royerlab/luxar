"""Import an example module by file path (the examples dir is not a package)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

EXAMPLES_DIR = Path(__file__).resolve().parent.parent


def load_example(stem: str) -> ModuleType:
    """Load ``packages/luxar/examples/<stem>.py`` as a fresh module object.

    Examples import the shared explainer helper with
    ``from _overlay_style import add_explainer``; when run as scripts the examples
    dir is ``sys.path[0]``, so the same is arranged here.
    """
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    path = EXAMPLES_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_example_{stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
