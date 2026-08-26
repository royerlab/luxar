from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/sync_demo_counts.py"


@pytest.fixture
def sync_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_demo_counts", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    readme = tmp_path / "README.md"
    readme.write_text(
        "luxar demo              # Browse the 2 bundled demos\n"
        "🎬 2 Luxar demos  ·  1 built  ·  0 cached  ·  1 not generated yet\n"
        "Historical note: all 80 demos were available then.\n"
    )
    claude = tmp_path / "CLAUDE.md"
    claude.write_text("luxar demo  # List the 2 bundled demos (table)\n")
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "- `packages/luxar/src/luxar/demos/demo_*.py` (2 complete demos)\n"
        "- `packages/luxar/examples/*_example.py` (1 focused examples)\n"
    )
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "one_example.py").touch()
    (examples / "two_example.py").touch()
    (examples / "helper.py").touch()
    return readme, claude, skill, examples


def _configure(module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    readme, claude, skill, examples = _checkout(tmp_path)
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "README", readme)
    monkeypatch.setattr(module, "CLAUDE", claude)
    monkeypatch.setattr(module, "VISUALIZATION_SKILL", skill)
    monkeypatch.setattr(module, "EXAMPLES_DIR", examples)
    monkeypatch.setattr(module, "iter_demos", lambda **_kwargs: [object()] * 3)
    return readme, claude, skill


def test_sync_updates_only_live_sites_and_absorbs_delta(
    sync_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readme, claude, skill = _configure(sync_module, tmp_path, monkeypatch)

    assert sync_module.main([]) == 0

    readme_text = readme.read_text()
    assert "Browse the 3 bundled demos" in readme_text
    assert (
        "🎬 3 Luxar demos  ·  1 built  ·  0 cached  ·  2 not generated yet"
        in readme_text
    )
    assert "Historical note: all 80 demos were available then." in readme_text
    assert "List the 3 bundled demos (table)" in claude.read_text()
    assert "(3 complete demos)" in skill.read_text()
    assert "(2 focused examples)" in skill.read_text()


def test_check_reports_drift_without_writing(
    sync_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readme, claude, skill = _configure(sync_module, tmp_path, monkeypatch)
    before = {path: path.read_text() for path in (readme, claude, skill)}

    assert sync_module.main(["--check"]) == 1

    assert {path: path.read_text() for path in before} == before
    output = capsys.readouterr().out
    assert "README.md (synchronized)" in output
    assert "CLAUDE.md (synchronized)" in output
    assert "SKILL.md (synchronized)" in output


def test_check_passes_after_sync(
    sync_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(sync_module, tmp_path, monkeypatch)

    assert sync_module.main([]) == 0
    assert sync_module.main(["--check"]) == 0


def test_invalid_banner_aborts_without_partial_writes(
    sync_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readme, claude, skill = _configure(sync_module, tmp_path, monkeypatch)
    readme.write_text(
        readme.read_text().replace(
            "1 built  ·  0 cached  ·  1 not generated yet",
            "1 built  ·  0 cached  ·  0 not generated yet",
        )
    )
    before = {path: path.read_text() for path in (readme, claude, skill)}

    assert sync_module.main([]) == 2

    assert {path: path.read_text() for path in before} == before
    assert "sum to 1, not 2" in capsys.readouterr().err


def test_committed_counts_are_synchronized(sync_module: ModuleType) -> None:
    assert sync_module.main(["--check"]) == 0, (
        "demo documentation counts are stale — run `hatch run sync-demo-counts`"
    )
