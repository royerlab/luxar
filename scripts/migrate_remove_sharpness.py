#!/usr/bin/env python3
"""Migration script: Remove sharpness arrays from .gsplats.zarr.zip demo data files.

Strips `*/sharpnesses/*` entries and updates `splats/.zattrs` to set
`has_sharpness: false` and remove `sharpness_bounds`.
"""

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def migrate_zip(zip_path: Path, *, dry_run: bool = False) -> bool:
    """Remove sharpness data from a single .gsplats.zarr.zip file.

    Returns True if the file was modified, False if no sharpness data found.
    """
    had_sharpness = False

    with zipfile.ZipFile(zip_path, "r") as zin:
        names = zin.namelist()
        sharpness_entries = [
            n for n in names if "/sharpnesses/" in n or n.endswith("/sharpnesses")
        ]
        if not sharpness_entries:
            print(f"  SKIP (no sharpness entries): {zip_path.name}")
            return False

        had_sharpness = True
        print(
            f"  Removing {len(sharpness_entries)} sharpness entries from {zip_path.name}"
        )

        if dry_run:
            for entry in sharpness_entries:
                print(f"    - {entry}")
            return True

        # Write to temp file, then replace
        fd, tmp_path = tempfile.mkstemp(suffix=".zip", dir=zip_path.parent)
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zout:
                for item in zin.infolist():
                    if item.filename in sharpness_entries:
                        continue

                    data = zin.read(item.filename)

                    # Update .zattrs to remove sharpness metadata
                    if item.filename.endswith(".zattrs"):
                        try:
                            attrs = json.loads(data)
                            modified = False
                            if "has_sharpness" in attrs:
                                attrs["has_sharpness"] = False
                                modified = True
                            if "sharpness_bounds" in attrs:
                                del attrs["sharpness_bounds"]
                                modified = True
                            if "max_sharpness" in attrs:
                                del attrs["max_sharpness"]
                                modified = True
                            if modified:
                                data = json.dumps(attrs, indent=2).encode("utf-8")
                                print(f"    Updated attrs: {item.filename}")
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass

                    zout.writestr(item, data)

            # Replace original
            shutil.move(tmp_path, zip_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    return had_sharpness


def verify_zip(zip_path: Path) -> bool:
    """Verify a migrated zip has no sharpness entries."""
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        sharpness_entries = [
            n for n in names if "/sharpnesses/" in n or n.endswith("/sharpnesses")
        ]
        if sharpness_entries:
            print(
                f"  FAIL: {zip_path.name} still has sharpness entries: {sharpness_entries}"
            )
            return False

        # Check .zattrs
        for name in names:
            if name.endswith(".zattrs"):
                try:
                    attrs = json.loads(z.read(name))
                    if attrs.get("has_sharpness", False):
                        print(f"  FAIL: {name} still has has_sharpness=True")
                        return False
                    if "sharpness_bounds" in attrs:
                        print(f"  FAIL: {name} still has sharpness_bounds")
                        return False
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

    print(f"  OK: {zip_path.name}")
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    verify_only = "--verify" in sys.argv

    demo_data = (
        Path(__file__).parent.parent
        / "packages"
        / "luxar"
        / "src"
        / "luxar"
        / "demos"
        / "data"
    )
    zip_files = sorted(demo_data.glob("**/*.gsplats.zarr.zip"))

    if not zip_files:
        print(f"No .gsplats.zarr.zip files found in {demo_data}")
        sys.exit(1)

    print(f"Found {len(zip_files)} gsplats zip files")

    if verify_only:
        print("\n--- Verification ---")
        all_ok = True
        for zf in zip_files:
            if not verify_zip(zf):
                all_ok = False
        sys.exit(0 if all_ok else 1)

    if dry_run:
        print("\n--- Dry Run ---")

    print("\n--- Migration ---")
    modified = 0
    for zf in zip_files:
        if migrate_zip(zf, dry_run=dry_run):
            modified += 1

    print(f"\nModified {modified}/{len(zip_files)} files")

    if not dry_run and modified > 0:
        print("\n--- Verification ---")
        all_ok = True
        for zf in zip_files:
            if not verify_zip(zf):
                all_ok = False
        if all_ok:
            print("\nAll files verified successfully!")
        else:
            print("\nSome files failed verification!")
            sys.exit(1)


if __name__ == "__main__":
    main()
