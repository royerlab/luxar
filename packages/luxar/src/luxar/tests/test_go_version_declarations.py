"""Guard the launcher Go floor, CI toolchain, and local bootstrap pin."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


@pytest.fixture(scope="module")
def declarations() -> tuple[str, str, str]:
    workflow = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    go_mod = (REPO / "packages/luxar-launcher/go.mod").read_text(encoding="utf-8")
    ci = re.search(r"go-version: '(\d+\.\d+)'", workflow)
    bootstrap = re.search(r"^GO_VERSION \?= (\d+\.\d+\.\d+)$", makefile, re.MULTILINE)
    floor = re.search(r"^go (\d+\.\d+)$", go_mod, re.MULTILINE)
    assert ci and bootstrap and floor
    return ci.group(1), bootstrap.group(1), floor.group(1)


def test_ci_toolchain_satisfies_module_floor(
    declarations: tuple[str, str, str],
) -> None:
    ci, _, floor = declarations
    assert _version_tuple(ci) >= _version_tuple(floor)


def test_bootstrap_pin_matches_ci_minor(
    declarations: tuple[str, str, str],
) -> None:
    ci, bootstrap, _ = declarations
    assert _version_tuple(bootstrap)[:2] == _version_tuple(ci)


def test_go_launcher_disables_automatic_toolchain_downloads() -> None:
    workflow = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = re.search(r"(?ms)^  go-launcher:\n(.*?)(?=^  \S|\Z)", workflow)
    assert job
    assert re.search(
        r"^    env:\n      GOTOOLCHAIN: local$", job.group(1), re.MULTILINE
    )


@pytest.mark.parametrize("target", ["install-go", "build-launchers"])
def test_old_go_is_rejected_before_use(
    tmp_path: Path, target: str, declarations: tuple[str, str, str]
) -> None:
    _, bootstrap, _ = declarations
    go = tmp_path / "go"
    go.write_text(
        "#!/bin/sh\necho 'go version go1.22.10 linux/amd64'\n", encoding="utf-8"
    )
    go.chmod(0o755)
    env = os.environ | {"PATH": f"{tmp_path}:{os.defpath}"}
    result = subprocess.run(
        ["make", "--silent", target],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"Go 1.22.10 is older than the pinned {bootstrap}" in result.stdout
