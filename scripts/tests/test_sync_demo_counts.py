from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/sync_demo_counts.py"
EXAMPLE_SMOKE_TEST = REPO / "packages/luxar/examples/tests/test_examples_smoke.py"


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
    (examples / "directory_example.py").mkdir()
    (examples / "helper.py").touch()
    return readme, claude, skill, examples


def _synchronize(module: ModuleType, tmp_path: Path, *, check: bool):
    readme, claude, skill, examples = _checkout(tmp_path)
    result = module.synchronize(
        3,
        module._example_count(examples),
        check=check,
        repo=tmp_path,
        readme=readme,
        claude=claude,
        visualization_skill=skill,
    )
    return result, readme, claude, skill


def test_sync_updates_only_live_sites_and_absorbs_delta(
    sync_module: ModuleType, tmp_path: Path
) -> None:
    result, readme, claude, skill = _synchronize(sync_module, tmp_path, check=False)

    assert result == 0

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


@pytest.mark.parametrize(
    ("quick_start", "match_count"),
    [
        ("luxar demo              # Browse all 2 bundled demos\n", 0),
        (
            "luxar demo              # Browse the 2 bundled demos\n"
            "luxar demo              # Browse the 2 bundled demos\n",
            2,
        ),
    ],
)
def test_sync_rejects_missing_or_duplicate_live_site(
    sync_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    quick_start: str,
    match_count: int,
) -> None:
    readme, claude, skill, examples = _checkout(tmp_path)
    readme.write_text(
        readme.read_text().replace(
            "luxar demo              # Browse the 2 bundled demos\n",
            quick_start,
        )
    )
    before = {path: path.read_bytes() for path in (readme, claude, skill)}

    assert (
        sync_module.synchronize(
            3,
            sync_module._example_count(examples),
            check=False,
            repo=tmp_path,
            readme=readme,
            claude=claude,
            visualization_skill=skill,
        )
        == 2
    )

    assert {path: path.read_bytes() for path in before} == before
    assert (
        f"README quick-start count: expected exactly one match, found {match_count}"
    ) in capsys.readouterr().err


def test_check_reports_drift_without_writing(
    sync_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readme, claude, skill, examples = _checkout(tmp_path)
    before = {path: path.read_text() for path in (readme, claude, skill)}

    assert (
        sync_module.synchronize(
            3,
            sync_module._example_count(examples),
            check=True,
            repo=tmp_path,
            readme=readme,
            claude=claude,
            visualization_skill=skill,
        )
        == 1
    )

    assert {path: path.read_text() for path in before} == before
    output = capsys.readouterr().out
    assert "README.md (synchronized)" in output
    assert "CLAUDE.md (synchronized)" in output
    assert "SKILL.md (synchronized)" in output


def test_main_check_uses_current_document_paths(
    sync_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readme, claude, skill, examples = _checkout(tmp_path)
    demo_count = len(sync_module.iter_demos(refresh=True))
    example_count = sync_module._example_count(sync_module.EXAMPLES_DIR)
    for index in range(2, example_count):
        (examples / f"extra_{index}_example.py").touch()
    before = {path: path.read_bytes() for path in (readme, claude, skill)}
    monkeypatch.setattr(sync_module, "REPO", tmp_path)
    monkeypatch.setattr(sync_module, "README", readme)
    monkeypatch.setattr(sync_module, "CLAUDE", claude)
    monkeypatch.setattr(sync_module, "VISUALIZATION_SKILL", skill)
    monkeypatch.setattr(sync_module, "EXAMPLES_DIR", examples)
    monkeypatch.setattr(
        sync_module,
        "iter_demos",
        lambda *, refresh: [None] * demo_count,
    )

    assert sync_module.main(["--check"]) == 1
    assert {path: path.read_bytes() for path in before} == before


def test_check_passes_after_sync(sync_module: ModuleType, tmp_path: Path) -> None:
    result, readme, claude, skill = _synchronize(sync_module, tmp_path, check=False)

    assert result == 0
    assert (
        sync_module.synchronize(
            3,
            2,
            check=True,
            repo=tmp_path,
            readme=readme,
            claude=claude,
            visualization_skill=skill,
        )
        == 0
    )


def test_example_count_matches_smoke_test_discovery(sync_module: ModuleType) -> None:
    spec = importlib.util.spec_from_file_location(
        "test_examples_smoke", EXAMPLE_SMOKE_TEST
    )
    assert spec is not None and spec.loader is not None
    smoke_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke_module)

    assert sync_module._example_count(sync_module.EXAMPLES_DIR) == len(
        smoke_module._discover_example_stems()
    )


def test_invalid_banner_aborts_without_partial_writes(
    sync_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readme, claude, skill, examples = _checkout(tmp_path)
    readme.write_text(
        readme.read_text().replace(
            "1 built  ·  0 cached  ·  1 not generated yet",
            "1 built  ·  0 cached  ·  0 not generated yet",
        )
    )
    before = {path: path.read_text() for path in (readme, claude, skill)}

    assert (
        sync_module.synchronize(
            3,
            sync_module._example_count(examples),
            check=False,
            repo=tmp_path,
            readme=readme,
            claude=claude,
            visualization_skill=skill,
        )
        == 2
    )

    assert {path: path.read_text() for path in before} == before
    assert "sum to 1, not 2" in capsys.readouterr().err


def test_demo_count_decrease_cannot_make_missing_bucket_negative(
    sync_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readme, claude, skill, examples = _checkout(tmp_path)
    before = {path: path.read_bytes() for path in (readme, claude, skill)}

    assert (
        sync_module.synchronize(
            0,
            sync_module._example_count(examples),
            check=False,
            repo=tmp_path,
            readme=readme,
            claude=claude,
            visualization_skill=skill,
        )
        == 2
    )

    assert {path: path.read_bytes() for path in before} == before
    assert "cannot absorb the demo-count decrease" in capsys.readouterr().err


def test_committed_counts_are_synchronized(sync_module: ModuleType) -> None:
    assert sync_module.main(["--check"]) == 0, (
        "demo documentation counts are stale — run `hatch run sync-demo-counts`"
    )
