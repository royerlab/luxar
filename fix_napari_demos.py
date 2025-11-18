#!/usr/bin/env python3
"""Script to fix --no-napari flag behavior in all demo files."""

import re
import sys
from pathlib import Path


def fix_demo_file(filepath: Path) -> bool:
    """
    Fix a demo file to properly handle --no-napari flag.
    Returns True if changes were made, False otherwise.
    """
    print(f"Processing: {filepath}")

    content = filepath.read_text()
    original_content = content

    # Fix 1: Remove early exit, just print message
    old_pattern1 = r'''# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys\.argv
if NO_NAPARI and len\(sys\.argv\) > 1:
    aprint\(".*?"\)
    aprint\("Note: This demo is designed for interactive napari visualization\."\)
    aprint\(
        "✅ Demo structure verified - would run with full napari functionality when enabled"
    \)
    sys\.exit\(0\)'''

    new_pattern1 = '''# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🔬 Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")'''

    content = re.sub(old_pattern1, new_pattern1, content, flags=re.DOTALL)

    # Fix 2: Change napari_movie=True to napari_movie=(not NO_NAPARI)
    content = re.sub(
        r'napari_movie=True,',
        'napari_movie=(not NO_NAPARI),',
        content
    )

    # Fix 3: This is trickier - need to wrap napari viewer creation and napari.run()
    # We'll look for the pattern of viewer = napari.Viewer() followed by napari.run()
    # and wrap it with if not NO_NAPARI:

    # Find the last occurrence of "napari.run()"
    if "napari.run()" in content and "if not NO_NAPARI:" not in content:
        # Find where napari viewer setup starts (usually after the computation loop)
        # Look for the pattern: viewer = napari.Viewer
        viewer_pattern = r'\n(# .*?Napari viewer.*?\n)?viewer = napari\.Viewer\('
        match = re.search(viewer_pattern, content)

        if match:
            start_pos = match.start()

            # Find the napari.run() call
            run_pattern = r'\nnapari\.run\(\)'
            run_match = re.search(run_pattern, content[start_pos:])

            if run_match:
                end_pos = start_pos + run_match.end()

                # Extract the napari section
                napari_section = content[start_pos:end_pos]

                # Indent the napari section
                napari_lines = napari_section.split('\n')
                indented_napari = '\n'.join('    ' + line if line.strip() else line
                                           for line in napari_lines)

                # Find where console summary starts (usually has aprint before napari viewer)
                # Look backwards from viewer creation to find aprint statements
                before_viewer = content[:start_pos]

                # Find the last aprint block before viewer
                summary_pattern = r'\n(# Console summary.*?)(\nviewer = napari\.Viewer)'
                summary_match = re.search(summary_pattern, content, re.DOTALL)

                if summary_match:
                    summary_text = summary_match.group(1)
                    # Keep summary outside the if block
                    new_content = (
                        content[:summary_match.start()] +
                        '\n' + summary_text + '\n\n' +
                        'if not NO_NAPARI:' +
                        indented_napari +
                        '\nelse:\n    aprint("\\n✅ Demo completed successfully (napari visualization disabled)")' +
                        content[end_pos:]
                    )
                    content = new_content

    if content != original_content:
        filepath.write_text(content)
        print(f"  ✓ Fixed {filepath.name}")
        return True
    else:
        print(f"  - No changes needed for {filepath.name}")
        return False


def main():
    """Find and fix all demo files with NO_NAPARI issues."""

    # Find all demo files in gsplats package
    gsplats_dir = Path("packages/luxar/src/luxar/gsplats")

    demo_files = list(gsplats_dir.rglob("demo*.py"))

    # Filter to only files that have the problematic pattern
    files_to_fix = []
    for demo_file in demo_files:
        content = demo_file.read_text()
        if "NO_NAPARI" in content and "sys.exit(0)" in content:
            files_to_fix.append(demo_file)

    print(f"Found {len(files_to_fix)} demo files to fix:")
    for f in files_to_fix:
        print(f"  - {f.relative_to('packages/luxar/src/luxar/gsplats')}")

    print()

    fixed_count = 0
    for demo_file in files_to_fix:
        if fix_demo_file(demo_file):
            fixed_count += 1

    print()
    print(f"✅ Fixed {fixed_count} out of {len(files_to_fix)} files")


if __name__ == "__main__":
    main()
