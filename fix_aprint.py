#!/usr/bin/env python3
"""Fix all broken aprint statements in demo files."""

import re
from pathlib import Path

files_to_fix = [
    "packages/luxar/src/luxar/gsplats/demos/demo_splats_3d_dapi_napari.py",
    "packages/luxar/src/luxar/gsplats/multiscale/demos/demo_decompose_2d.py",
    "packages/luxar/src/luxar/gsplats/multiscale/demos/demo_decompose_2d_mitosis.py",
    "packages/luxar/src/luxar/gsplats/multiscale/demos/demo_decompose_3d_dapi_nuclei.py",
    "packages/luxar/src/luxar/gsplats/candidates/demos/demo_decomp_candidates_mitosis.py",
]

for filepath in files_to_fix:
    path = Path(filepath)
    print(f"Fixing {path.name}...")

    content = path.read_text()

    # Fix pattern: aprint("Running all computations...") followed by content on next lines
    # This handles both single-line and multi-line cases

    # Pattern 1: Single line with ")string"
    content = re.sub(
        r'aprint\("Running all computations without napari visualization\.\.\."\)(")',
        r'aprint(\1',
        content
    )

    # Pattern 2: Single line with ")f"string""
    content = re.sub(
        r'aprint\("Running all computations without napari visualization\.\.\."\)(f")',
        r'aprint(\1',
        content
    )

    # Pattern 3: Multi-line with newline and content
    content = re.sub(
        r'aprint\("Running all computations without napari visualization\.\.\."\)\n\s+(f?")',
        r'aprint(\n    \1',
        content
    )

    path.write_text(content)
    print(f"  ✓ {path.name}")

print("\n✅ All files fixed!")
