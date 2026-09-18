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


def _parse_declarations(
    workflow: str, makefile: str, go_mod: str
) -> tuple[str, str, str]:
    job = re.search(r"(?ms)^  go-launcher:\n(.*?)(?=^  \S|\Z)", workflow)
    assert job, "ci.yml must define the go-launcher job"
    ci = re.search(r"go-version: '(\d+\.\d+)'", job.group(1))
    assert ci, "the go-launcher job must declare a major.minor go-version"
    bootstrap = re.search(r"^GO_VERSION \?= (\d+\.\d+\.\d+)$", makefile, re.MULTILINE)
    assert bootstrap, "Makefile must declare a three-part GO_VERSION pin"
    floor = re.search(r"^go (\d+\.\d+(?:\.\d+)?)$", go_mod, re.MULTILINE)
    assert floor, "go.mod must declare a two- or three-part Go language floor"
    return ci.group(1), bootstrap.group(1), floor.group(1)


@pytest.fixture(scope="module")
def declarations() -> tuple[str, str, str]:
    workflow = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    go_mod = (REPO / "packages/luxar-launcher/go.mod").read_text(encoding="utf-8")
    return _parse_declarations(workflow, makefile, go_mod)


def test_declaration_parser_scopes_ci_version_to_launcher_job() -> None:
    workflow = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    go_mod = (REPO / "packages/luxar-launcher/go.mod").read_text(encoding="utf-8")
    unrelated_job = "  unrelated:\n    with:\n      go-version: '9.99'\n"
    ci, _, _ = _parse_declarations(unrelated_job + workflow, makefile, go_mod)
    assert ci == "1.27"


def test_declaration_parser_accepts_three_part_module_floor() -> None:
    workflow = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    go_mod = (REPO / "packages/luxar-launcher/go.mod").read_text(encoding="utf-8")
    ci, _, _ = _parse_declarations(workflow, makefile, go_mod)
    patched_go_mod = re.sub(
        r"^go \d+\.\d+(?:\.\d+)?$",
        f"go {ci}.0",
        go_mod,
        count=1,
        flags=re.MULTILINE,
    )
    _, _, floor = _parse_declarations(workflow, makefile, patched_go_mod)
    assert floor == f"{ci}.0"


def test_ci_toolchain_satisfies_module_floor(
    declarations: tuple[str, str, str],
) -> None:
    ci, _, floor = declarations
    assert _version_tuple(ci) >= _version_tuple(floor)[:2]


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
    assert f"Go binary: {go}" in result.stdout


@pytest.mark.parametrize(
    ("os_name", "remedy"),
    [
        ("linux", "Upgrade or remove that Go binary from PATH"),
        ("macos", "Upgrade Homebrew Go: brew upgrade go"),
    ],
)
def test_old_go_remedy_matches_platform(
    tmp_path: Path,
    os_name: str,
    remedy: str,
) -> None:
    go = tmp_path / "go"
    go.write_text(
        "#!/bin/sh\necho 'go version go1.22.10 linux/amd64'\n", encoding="utf-8"
    )
    go.chmod(0o755)
    env = os.environ | {"PATH": f"{tmp_path}:{os.defpath}"}
    result = subprocess.run(
        ["make", "--silent", "install-go", f"OS={os_name}"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"Go binary: {go}" in result.stdout
    assert remedy in result.stdout


def test_old_local_go_reports_reinstall_remedy(tmp_path: Path) -> None:
    home = tmp_path / "home"
    go = home / ".local/go/bin/go"
    go.parent.mkdir(parents=True)
    go.write_text(
        "#!/bin/sh\necho 'go version go1.22.10 linux/amd64'\n", encoding="utf-8"
    )
    go.chmod(0o755)
    env = os.environ | {"HOME": str(home), "PATH": os.defpath}
    result = subprocess.run(
        ["make", "--silent", "install-go", "OS=linux"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"Go binary: {go}" in result.stdout
    assert "Reinstall the local toolchain: rm -rf ~/.local/go" in result.stdout
