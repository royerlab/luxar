#!/usr/bin/env python3
"""Properly fix all demo files to handle --no-napari correctly."""

import re
from pathlib import Path


def fix_demo_early_exit(filepath: Path) -> bool:
    """Fix the early exit pattern in demo files."""
    content = filepath.read_text()
    original = content

    # Pattern 1: Match the problematic early exit block
    pattern = re.compile(
        r'(# Check for --no-napari flag\n'
        r'NO_NAPARI = "--no-napari" in sys\.argv\n)'
        r'if NO_NAPARI and len\(sys\.argv\) > 1:\n'
        r'    aprint\("([^"]+)"\)\n'
        r'    aprint\("Note: This demo is designed for interactive napari visualization\."\)\n'
        r'    aprint\(\n'
        r'        "✅ Demo structure verified - would run with full napari functionality when enabled"\n'
        r'    \)\n'
        r'    sys\.exit\(0\)\n',
        re.MULTILINE
    )

    def replacement(match):
        header = match.group(1)
        title = match.group(2)
        return (
            f'{header}'
            f'if NO_NAPARI:\n'
            f'    aprint("{title}")\n'
            f'    aprint("Running all computations without napari visualization...")\n'
        )

    content = pattern.sub(replacement, content)

    # Pattern 2: Fix napari_movie parameter
    content = re.sub(
        r'(\s+)napari_movie=True,',
        r'\1napari_movie=(not NO_NAPARI),',
        content
    )

    # Pattern 3: Wrap napari viewer sections (only if not already wrapped)
    if 'napari.run()' in content and 'if not NO_NAPARI:' not in content:
        # Find where viewer creation starts
        viewer_match = re.search(r'\n(# .*?[Nn]apari viewer.*?\n)?viewer = napari\.Viewer', content)
        if viewer_match:
            start_idx = viewer_match.start()

            # Find console summary before viewer (to keep it outside)
            before_viewer = content[:start_idx]
            summary_match = re.search(r'\n# Console summary\n.*?(?=\n(?:# |viewer =))', before_viewer, re.DOTALL)

            if summary_match:
                summary_end = summary_match.end()
                summary_text = content[summary_match.start():summary_end]

                # Find napari.run() at the end
                run_match = re.search(r'\nnapari\.run\(\)\n', content[start_idx:])
                if run_match:
                    run_end = start_idx + run_match.end()

                    # Extract napari section
                    napari_section = content[start_idx:run_end].rstrip()

                    # Indent napari section
                    lines = napari_section.split('\n')
                    indented = '\n'.join('    ' + line if line.strip() else '' for line in lines)

                    # Rebuild content
                    content = (
                        content[:summary_match.start()] +
                        summary_text +
                        '\n\nif not NO_NAPARI:' +
                        indented +
                        '\nelse:\n    aprint("\\n✅ Demo completed successfully (napari visualization disabled)")\n' +
                        content[run_end:]
                    )

    if content != original:
        filepath.write_text(content)
        return True
    return False


def main():
    """Fix all demos with NO_NAPARI issues."""
    gsplats_dir = Path("packages/luxar/src/luxar/gsplats")
    demo_files = list(gsplats_dir.rglob("demo*.py"))

    # Only process files with NO_NAPARI and sys.exit(0)
    to_fix = []
    for f in demo_files:
        content = f.read_text()
        if 'NO_NAPARI' in content and 'sys.exit(0)' in content:
            to_fix.append(f)

    print(f"Files to fix: {len(to_fix)}")
    for f in to_fix:
        print(f"  {f.relative_to(gsplats_dir)}")

    fixed = 0
    for f in to_fix:
        print(f"Processing {f.name}...", end=' ')
        if fix_demo_early_exit(f):
            print("✓")
            fixed += 1
        else:
            print("(no changes)")

    print(f"\n✅ Fixed {fixed}/{len(to_fix)} files")


if __name__ == '__main__':
    main()
