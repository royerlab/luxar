from __future__ import annotations

import importlib.util
import json
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


class Checkout:
    """A miniature checkout carrying every count site the script owns."""

    def __init__(self, root: Path) -> None:
        self.readme = root / "README.md"
        self.readme.write_text(
            "**[Try it](https://demos.example)** — 5 live demos, no\n"
            "install.\n"
            "luxar demo              # Browse the 2 bundled demos\n"
            "🎬 2 Luxar demos  ·  1 built  ·  0 cached  ·  1 not generated yet\n"
            "Historical note: all 80 demos were available then.\n"
            "| **[Live demo gallery](https://demos.example)** | 5 demos as "
            "interactive scenes in the browser |\n"
        )
        self.claude = root / "CLAUDE.md"
        self.claude.write_text("luxar demo  # List the 2 bundled demos (table)\n")
        self.skill = root / "SKILL.md"
        self.skill.write_text(
            "- `packages/luxar/src/luxar/demos/demo_*.py` (2 complete demos)\n"
            "- `packages/luxar/examples/*_example.py` (1 focused examples)\n"
        )
        self.docs_index = root / "index.rst"
        self.docs_index.write_text(
            "   `demos.example <https://demos.example>`_ hosts 5 of the\n"
            "   bundled demos as live, interactive scenes.\n"
        )
        self.examples = root / "examples"
        self.examples.mkdir()
        (self.examples / "one_example.py").touch()
        (self.examples / "two_example.py").touch()
        (self.examples / "directory_example.py").mkdir()
        (self.examples / "helper.py").touch()
        self.manifest = root / "manifest.json"
        self.manifest.write_text(
            json.dumps({"$comment": "x", "demos": [{"id": f"d{i}"} for i in range(7)]})
        )

    @property
    def documents(self) -> tuple[Path, ...]:
        return (self.readme, self.claude, self.skill, self.docs_index)

    def snapshot(self) -> dict[Path, str]:
        return {path: path.read_text() for path in self.documents}

    def synchronize(
        self,
        module: ModuleType,
        *,
        check: bool,
        demo_count: int = 3,
        hosted_count: int = 7,
    ) -> int:
        return module.synchronize(
            demo_count,
            module._example_count(self.examples),
            hosted_count,
            check=check,
            repo=self.readme.parent,
            readme=self.readme,
            claude=self.claude,
            visualization_skill=self.skill,
            docs_index=self.docs_index,
        )


def test_sync_updates_only_live_sites_and_absorbs_delta(
    sync_module: ModuleType, tmp_path: Path
) -> None:
    checkout = Checkout(tmp_path)

    assert checkout.synchronize(sync_module, check=False) == 0

    readme_text = checkout.readme.read_text()
    assert "Browse the 3 bundled demos" in readme_text
    assert (
        "🎬 3 Luxar demos  ·  1 built  ·  0 cached  ·  2 not generated yet"
        in readme_text
    )
    assert "Historical note: all 80 demos were available then." in readme_text
    assert "List the 3 bundled demos (table)" in checkout.claude.read_text()
    assert "(3 complete demos)" in checkout.skill.read_text()
    assert "(2 focused examples)" in checkout.skill.read_text()


def test_sync_projects_the_hosted_count_onto_all_three_sites(
    sync_module: ModuleType, tmp_path: Path
) -> None:
    checkout = Checkout(tmp_path)

    assert checkout.synchronize(sync_module, check=False, hosted_count=7) == 0

    readme_text = checkout.readme.read_text()
    assert "— 7 live demos, no\ninstall." in readme_text
    assert "| 7 demos as interactive scenes in the browser |" in readme_text
    # The bundled count is a different fact and must not have been touched by it.
    assert "Browse the 3 bundled demos" in readme_text
    assert checkout.docs_index.read_text() == (
        "   `demos.example <https://demos.example>`_ hosts 7 of the\n"
        "   bundled demos as live, interactive scenes.\n"
    )
    assert sync_module.find_hosted_claims(readme_text) == [
        (7, "intro banner"),
        (7, "docs table row"),
    ]


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
    checkout = Checkout(tmp_path)
    checkout.readme.write_text(
        checkout.readme.read_text().replace(
            "luxar demo              # Browse the 2 bundled demos\n",
            quick_start,
        )
    )
    before = checkout.snapshot()

    assert checkout.synchronize(sync_module, check=False) == 2

    assert checkout.snapshot() == before
    assert (
        f"README quick-start count: expected exactly one match, found {match_count}"
    ) in capsys.readouterr().err


@pytest.mark.parametrize(
    ("original", "replacement", "site"),
    [
        ("5 live demos", "five live demos", "README hosted count (intro banner)"),
        (
            "5 demos as interactive scenes",
            "5 interactive scenes",
            "README hosted count (docs table row)",
        ),
    ],
)
def test_sync_rejects_a_reworded_hosted_site_instead_of_skipping_it(
    sync_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    original: str,
    replacement: str,
    site: str,
) -> None:
    """A claim that stops matching must FAIL, not silently stop being owned."""
    checkout = Checkout(tmp_path)
    checkout.readme.write_text(
        checkout.readme.read_text().replace(original, replacement)
    )
    before = checkout.snapshot()

    assert checkout.synchronize(sync_module, check=False) == 2

    assert checkout.snapshot() == before
    assert f"{site}: expected exactly one match, found 0" in capsys.readouterr().err


def test_sync_rejects_a_reworded_docs_index_site(
    sync_module: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = Checkout(tmp_path)
    checkout.docs_index.write_text("hosts many of the bundled demos\n")
    before = checkout.snapshot()

    assert checkout.synchronize(sync_module, check=False) == 2

    assert checkout.snapshot() == before
    assert (
        "docs/index.rst hosted count: expected exactly one match, found 0"
        in capsys.readouterr().err
    )


def test_check_reports_drift_without_writing(
    sync_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = Checkout(tmp_path)
    before = checkout.snapshot()

    assert checkout.synchronize(sync_module, check=True) == 1

    assert checkout.snapshot() == before
    captured = capsys.readouterr()
    assert "README.md (synchronized)" in captured.out
    assert "CLAUDE.md (synchronized)" in captured.out
    assert "SKILL.md (synchronized)" in captured.out
    assert "index.rst (synchronized)" in captured.out
    assert (
        "demo documentation counts are stale — run `hatch run sync-demo-counts` "
        "and commit the result."
    ) in captured.err


def test_check_flags_a_hosted_count_off_by_one(
    sync_module: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The drift that actually happened: README said 86, the manifest had 87."""
    checkout = Checkout(tmp_path)
    assert checkout.synchronize(sync_module, check=False, hosted_count=7) == 0
    checkout.readme.write_text(
        checkout.readme.read_text().replace("7 live demos", "6 live demos")
    )

    assert checkout.synchronize(sync_module, check=True, hosted_count=7) == 1
    assert "-**[Try it](https://demos.example)** — 6 live demos" in (
        capsys.readouterr().out
    )


def test_main_check_uses_current_document_paths(
    sync_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Checkout(tmp_path)
    example_count = sync_module._example_count(sync_module.EXAMPLES_DIR)
    for index in range(2, example_count):
        (checkout.examples / f"extra_{index}_example.py").touch()
    before = checkout.snapshot()
    monkeypatch.setattr(sync_module, "REPO", tmp_path)
    monkeypatch.setattr(sync_module, "README", checkout.readme)
    monkeypatch.setattr(sync_module, "CLAUDE", checkout.claude)
    monkeypatch.setattr(sync_module, "VISUALIZATION_SKILL", checkout.skill)
    monkeypatch.setattr(sync_module, "DOCS_INDEX", checkout.docs_index)
    monkeypatch.setattr(sync_module, "EXAMPLES_DIR", checkout.examples)
    monkeypatch.setattr(sync_module, "MANIFEST", checkout.manifest)
    monkeypatch.setattr(sync_module, "_bundled_count", lambda: 3)

    assert sync_module.main(["--check"]) == 1
    assert checkout.snapshot() == before


@pytest.mark.parametrize(
    ("manifest_text", "complaint"),
    [
        ("{not json", "not valid JSON"),
        (json.dumps(["d0", "d1"]), "expected an object with a 'demos' list"),
        (json.dumps({"demos": {"d0": 1}}), "expected an object with a 'demos' list"),
    ],
)
def test_malformed_manifest_fails_closed(
    sync_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    manifest_text: str,
    complaint: str,
) -> None:
    """An unreadable manifest is a failure (exit 2), never a hosted count of 0."""
    checkout = Checkout(tmp_path)
    checkout.manifest.write_text(manifest_text)
    before = checkout.snapshot()
    monkeypatch.setattr(sync_module, "REPO", tmp_path)
    monkeypatch.setattr(sync_module, "README", checkout.readme)
    monkeypatch.setattr(sync_module, "CLAUDE", checkout.claude)
    monkeypatch.setattr(sync_module, "VISUALIZATION_SKILL", checkout.skill)
    monkeypatch.setattr(sync_module, "DOCS_INDEX", checkout.docs_index)
    monkeypatch.setattr(sync_module, "EXAMPLES_DIR", checkout.examples)
    monkeypatch.setattr(sync_module, "MANIFEST", checkout.manifest)
    monkeypatch.setattr(sync_module, "_bundled_count", lambda: 3)

    assert sync_module.main([]) == 2
    assert checkout.snapshot() == before
    assert complaint in capsys.readouterr().err


def test_hosted_count_is_the_manifest_length(sync_module: ModuleType) -> None:
    manifest = json.loads(sync_module.MANIFEST.read_text())
    assert sync_module._hosted_count(sync_module.MANIFEST) == len(manifest["demos"])
    assert sync_module._hosted_count(sync_module.MANIFEST) > 0


def test_check_passes_after_sync(sync_module: ModuleType, tmp_path: Path) -> None:
    checkout = Checkout(tmp_path)

    assert checkout.synchronize(sync_module, check=False) == 0
    assert checkout.synchronize(sync_module, check=True) == 0


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
    checkout = Checkout(tmp_path)
    checkout.readme.write_text(
        checkout.readme.read_text().replace(
            "1 built  ·  0 cached  ·  1 not generated yet",
            "1 built  ·  0 cached  ·  0 not generated yet",
        )
    )
    before = checkout.snapshot()

    assert checkout.synchronize(sync_module, check=False) == 2

    assert checkout.snapshot() == before
    assert "sum to 1, not 2" in capsys.readouterr().err


def test_demo_count_decrease_cannot_make_missing_bucket_negative(
    sync_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = Checkout(tmp_path)
    before = checkout.snapshot()

    assert checkout.synchronize(sync_module, check=False, demo_count=0) == 2

    assert checkout.snapshot() == before
    assert "cannot absorb the demo-count decrease" in capsys.readouterr().err


def test_module_import_needs_no_luxar(sync_module: ModuleType) -> None:
    """The deploy-time gallery audit loads this file without the package installed."""
    import ast

    tree = ast.parse(SCRIPT.read_text())
    top_level_luxar_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(
            (alias.name if isinstance(node, ast.Import) else node.module or "").split(
                "."
            )[0]
            == "luxar"
            for alias in node.names
        )
    ]
    assert not top_level_luxar_imports


def test_committed_counts_are_synchronized(sync_module: ModuleType) -> None:
    assert sync_module.main(["--check"]) == 0, (
        "demo documentation counts are stale — run `hatch run sync-demo-counts`"
    )
