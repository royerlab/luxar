"""Guard: every input of a gated CI check must be named by the diff classifier.

The ``changes`` job in ``.github/workflows/ci.yml`` sorts a PR diff into four
language domains and each downstream job runs only when its domain is set. A gate
input the classifier does not name is therefore a gate that skips for exactly the
change it exists to catch — and it skips *green*, because the job still publishes
its required context.

That is not hypothetical. ``scripts/complexity_baseline.json`` is the only input of
the C901 ratchet that carries no Python extension, and that ratchet is enforced by a
pytest test inside ``hatch run test-cov``. It matched no pattern, so PR #1678 — a
baseline-only diff — reported ``python-tests (3.12)`` green in about seven seconds
without running the test whose whole job is to assert the tree still MATCHES that
baseline. What that test catches is a baseline that has come loose from the tree: keys
dropped while still over the limit come back as ``report.new``, values below what the
tree measures come back as ``report.worsened``, and an emptied or deleted file trips a
fail-closed ``assert``. It does not police a deliberately RAISED entry —
``evaluate_ratchet`` reads that as debt paid down elsewhere — so the harm the hole
enabled is a baseline that no longer describes the code, merging unexamined. (#1678's
own baseline was a legitimate tightening; the hole, not the harm, was what was real.)
The classifier now explicitly owns the baseline, the documentation inputs guarded
by pytest, and the other non-Python inputs those tests consume.

Matches are decided by invoking real ``grep -E`` rather than Python's ``re``. CI's
verdict comes from POSIX ERE under GNU grep, and the two dialects differ enough
(escapes, intervals, backreferences) that a ``re``-based test could pass while the
workflow still skipped.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[5]
WORKFLOW = REPO / ".github/workflows/ci.yml"

#: One row per gate input whose required domain is not guaranteed by its ordinary
#: source extension or package path, so an explicit pattern alternative is required.
#: ``(path, domain, why)`` — the reason is quoted back in the failure message.
#: Viewer-source readers name their paths through ``viewer_source()``; the static
#: scan below checks those literal calls against this declaration.
#:
#: Not every row is load-bearing to the same degree: some are matched by a broad
#: alternative that could not plausibly be removed (``Cargo.lock`` via the whole
#: ``^packages/luxar-viewer/`` prefix, say), so they are belt-and-braces rather
#: than the single thing keeping their gate alive. Do not read the table as a list
#: of narrow escapes.
GATE_INPUTS: list[tuple[str, str, str]] = [
    # The native sources. `.cu`/`.cuh` already routed; the C++/ObjC++/shader
    # ones did NOT, so a change to any of these three reached no gate at all
    # until `check-native` existed to be reached (A15-03).
    (
        "packages/luxar/src/luxar/gsplats/models/gsplats/cuda/src/bindings.cpp",
        "py",
        "hatch run check-native -fsyntax-only's it against torch's headers",
    ),
    (
        "packages/luxar/src/luxar/gsplats/models/gsplats/metal/src/bindings.mm",
        "py",
        "hatch run check-native -fsyntax-only's it on a macOS runner",
    ),
    (
        "packages/luxar/src/luxar/gsplats/models/gsplats/metal/src/kernels.metal",
        "py",
        "hatch run check-native compiles it with `metal -c -Werror`",
    ),
    (
        "packages/luxar/src/luxar/gsplats/models/gsplats/cuda/src/cuda_splatting.cu",
        "py",
        "hatch run check-native runs `nvcc -cuda` over it where CUDA exists",
    ),
    (
        ".pre-commit-config.yaml",
        "py",
        "test_mypy_gate_targets_stay_synchronized parses the mypy hook entry",
    ),
    (
        ".gitattributes",
        "py",
        "test_docs_workflow.py derives the published LFS candidate set from it",
    ),
    (
        ".github/workflows/docs.yml",
        "py",
        "test_docs_workflow.py guards the Pages workflow itself",
    ),
    (
        ".github/workflows/external-reference-audits.yml",
        "py",
        "test_run_external_reference_audits.py pins its schedule and token wiring",
    ),
    (
        "scripts/complexity_baseline.json",
        "py",
        "the C901 ratchet's only non-.py input; test_check_complexity.py is what "
        "proves the baseline still matches the tree",
    ),
    (
        "scripts/lint_baseline.json",
        "py",
        "the bugbear/RUF012 ratchet's only non-.py input; editing it is precisely "
        "how real debt would be silenced, so the edit must run "
        "test_check_lint_ratchet.py, which re-derives it from the tree",
    ),
    (
        "Makefile",
        "py",
        "test_python_version_declarations.py greps it for a sub-floor interpreter, "
        "and test_demo_commands.py grades the `clean-*` recipes and cache root",
    ),
    (
        "scripts/gallery/manifest.json",
        "py",
        "test_demo_meta.py cross-validates it against the demo registry",
    ),
    (
        "scripts/gallery/manifest.json",
        "ts",
        "gallery-selection.test.ts validates README capture ids against it",
    ),
    (
        "scripts/benchmarks/benchmark_bisect.sh",
        "py",
        "test_benchmark_bisect.py executes the committed launcher directly",
    ),
    (
        "scripts/demo_archive_characteristics.json",
        "py",
        "test_demo_gsplats_3d_flylight_mcfo_63x_brain.py checks its measured "
        "splat count against the hosted demo recipe",
    ),
    (
        "scripts/release.sh",
        "py",
        "test_set_version.py asserts its `make set-version` remedy string, and a "
        "shell file matches no other domain, so a release.sh-only diff would "
        "otherwise run zero tests",
    ),
    (
        "docs/guides/user/CLI_REFERENCE.md",
        "py",
        "test_docs_command_coverage.py drift-guards it against the live Typer app; "
        "its .md only selects docs-quality, which runs no pytest",
    ),
    (
        ".agents/skills/luxar-gsplat-pipeline/SKILL.md",
        "py",
        "test_cholesky_documentation.py executes its documented Cholesky packing",
    ),
    (
        ".agents/skills/luxar-visualization/SKILL.md",
        "py",
        "test_cholesky_documentation.py executes its documented Cholesky packing, "
        "and check-demo-counts synchronizes its demo and focused-example counts",
    ),
    (
        ".agents/skills/luxar-visualization/references/scene-api.md",
        "py",
        "test_cholesky_documentation.py executes its documented Cholesky packing",
    ),
    (
        "CLAUDE.md",
        "py",
        "test_cholesky_documentation.py guards its contributor-facing convention, "
        "and check-demo-counts synchronizes its bundled-demo count with the registry",
    ),
    (
        "docs/specs/GSPLATS_DIMENSION_MAPPING.md",
        "py",
        "test_cholesky_documentation.py guards its hand-authoring contract",
    ),
    (
        "README.md",
        "py",
        "test_readme_demo_docs.py drift-guards the root demo documentation",
    ),
    (
        "README.md",
        "ts",
        "gallery-selection.test.ts derives the README capture set from it",
    ),
    (
        "packages/luxar/src/luxar/demos/README.md",
        "py",
        "test_demo_import_spelling.py validates its shared-helper inventory",
    ),
    (
        "pyproject.toml",
        "py",
        "hatch envs, ruff/mypy targets, the import-linter contracts, and the "
        "coverage floor (`[tool.coverage.report] fail_under`) the python-tests "
        "gate enforces — a floor-only diff must run the gate it configures",
    ),
    (
        "format-contract/contract.yaml",
        "py",
        "the source of truth `hatch run check-contract` compares both halves to",
    ),
    (
        "packages/luxar/src/luxar/demos/data_manifest.json",
        "py",
        "`hatch run check-data-manifest` compares it against the demos/data tree",
    ),
    (
        "packages/luxar-viewer/package.json",
        "py",
        "scripts/check_version_consistency.py pins it to the Python version",
    ),
    (
        "CITATION.cff",
        "py",
        "scripts/check_version_consistency.py pins it to the Python version",
    ),
    (
        "packages/luxar-viewer/src/config/sections/camera/data.ts",
        "py",
        "test_viewer_config.py and test_demos_cinematic_mode.py parse its FOV "
        "presets and defaults",
    ),
    (
        "packages/luxar-viewer/src/config/sections/rendering-controls/data.ts",
        "py",
        "test_demos_cinematic_mode.py parses its default FOV",
    ),
    (
        "packages/luxar-viewer/src/data/attrs-composer.ts",
        "py",
        "test_blending_warnings.py parses the scene-root exclusion from the "
        "attrs-inheritance chain",
    ),
    (
        "packages/luxar-viewer/src/data/loaders/spatial-query/tolerance-computer.ts",
        "py",
        "test_ordering_gsplats.py parses the continuous-dimension tolerance",
    ),
    (
        "packages/luxar-viewer/src/rendering/blending-state.ts",
        "py",
        "test_blending_warnings.py parses the normal-mode depth-write threshold",
    ),
    (
        "packages/luxar-viewer/src/rendering/element-texture-layout.ts",
        "py",
        "test_element_cap_warning.py parses the element-texture capacity inputs",
    ),
    (
        "packages/luxar-viewer/src/rendering/materials/mesh/appearance.ts",
        "py",
        "test_blending_warnings.py parses MESH_SUPPORTED_BLENDING_MODES",
    ),
    (
        "packages/luxar-viewer/src/rendering/node-factory/create-points-node.ts",
        "py",
        "test_blending_warnings.py parses the points blending default",
    ),
    (
        "packages/luxar-viewer/src/rendering/node-factory/create-lines-node.ts",
        "py",
        "test_blending_warnings.py parses the lines blending default",
    ),
    (
        "packages/luxar-viewer/src/rendering/node-factory/create-gsplats-node.ts",
        "py",
        "test_blending_warnings.py parses the gsplats blending default",
    ),
    (
        "packages/luxar-viewer/src/rendering/node-factory/create-mesh-node.ts",
        "py",
        "test_blending_warnings.py parses the mesh-node blending default",
    ),
    (
        "packages/luxar-viewer/src/scene/lod-group-registry.ts",
        "py",
        "LOD and biodiversity contract tests parse the live screen-coverage constants",
    ),
    (
        "packages/luxar-viewer/src/types/geometry-capabilities.ts",
        "py",
        "test_geometry_capabilities.py compares the viewer capability table",
    ),
    (
        "packages/luxar-viewer/src/types/line-join.ts",
        "py",
        "test_constants.py compares the line-join vocabulary",
    ),
    (
        "packages/luxar-viewer/src/types/lod-group.ts",
        "py",
        "test_constants.py compares the LOD selector and display-type vocabularies",
    ),
    (
        "packages/luxar-viewer/src/types/partition-group.ts",
        "py",
        "test_constants.py compares the partition display-type vocabulary",
    ),
    (
        "packages/luxar-viewer/src/types/data-monitor-types.ts",
        "py",
        "test_constants.py compares the monitor display-type vocabulary",
    ),
    (
        "packages/luxar-viewer/src/tests/screenshots/exposure-policy.ts",
        "py",
        "test_score_exposure.py parses the capture harness thresholds",
    ),
    (
        "packages/luxar-viewer/tools/example-fixture-freshness.ts",
        "py",
        "test_run_examples.py pins the shared stale-fixture exit code",
    ),
    (
        "packages/luxar-viewer/src/types/format-contract.ts",
        "py",
        "the generated TypeScript half `check-contract` judges",
    ),
    (
        "packages/luxar-viewer/src/tests/global-setup.ts",
        "py",
        "test_fixture_environment.py checks its fixture-generator invocation",
    ),
    (
        "packages/luxar-viewer/src/tests/README.md",
        "py",
        "test_fixture_environment.py checks its fixture-generator invocation",
    ),
    (
        "packages/luxar-viewer/src/tests/screenshots/generate-gallery.spec.ts",
        "py",
        "test_generate_gallery_datasets.py derives the manifest field contract from it",
    ),
    (
        "packages/luxar-viewer/tests/fixtures/README.md",
        "py",
        "test_fixture_environment.py checks its fixture-generator invocation",
    ),
    (
        "packages/luxar-viewer/src/wasm/rust/Cargo.lock",
        "rust",
        "the pinned crate graph every cargo check/clippy/test build resolves",
    ),
    (
        "packages/luxar-launcher/go.mod",
        "go",
        "the launcher module graph `go build`/`go test` resolve",
    ),
    (
        "packages/luxar-viewer/coverage-thresholds.mjs",
        "ts",
        "the viewer coverage floors, read by both vitest.config.ts and "
        "check-coverage-slack.mjs; the counterpart of pyproject.toml's "
        "`fail_under`, and likewise must trigger the gate it configures",
    ),
]

#: Paths that belong to no LANGUAGE domain (they do select the docs gate, which is
#: a separate axis — ``docs/index.rst`` matches ``docs_pattern`` by design). Without
#: these the table above proves nothing: a pattern that matched everything would
#: satisfy every positive row.
NON_DOMAIN_PATHS: list[str] = ["CHANGELOG.md", "docs/index.rst"]

#: Inputs with no Python consumer. These must remain outside ``dom_py`` so adding
#: a few contract inputs cannot silently widen ownership to whole subtrees and
#: make unrelated PRs pay for the Python matrix.
NON_PYTHON_DOMAIN_PATHS: list[str] = [
    ".github/workflows/publish.yml",
    "packages/luxar-viewer/src/config/sections/adaptive-dpr/data.ts",
    "packages/luxar-viewer/src/data/loaders/spatial-query/spatial-query-builder.ts",
    "packages/luxar-viewer/src/rendering/display-range.ts",
    "packages/luxar-viewer/src/scene/lod-fade.ts",
    "packages/luxar-viewer/src/tests/screenshots/crop-policy.ts",
    "packages/luxar-viewer/src/types/blending.ts",
    "packages/luxar-viewer/tools/example-smoke-inventory.ts",
]

#: The docs gate's own negative control: real tracked files that must NOT set
#: ``docs_relevant``. Kept separate from ``NON_DOMAIN_PATHS`` because the two
#: controls test opposite gates — these DO belong to a language domain.
NON_DOCS_PATHS: list[str] = [
    "packages/luxar-launcher/go.mod",
    "packages/luxar-viewer/src/wasm/rust/src/lib.rs",
]

_PYTHON_SOURCE_ROOTS = (
    REPO / "packages/luxar/src/luxar",
    REPO / "packages/luxar/examples/tests",
    REPO / "scripts",
    REPO / "stats",
)

_NON_SCANNED_PYTHON_VIEWER_INPUTS = {
    "packages/luxar-viewer/package.json": "read by check_version_consistency.py",
    "packages/luxar-viewer/src/types/format-contract.ts": (
        "generated and checked by scripts/gen_format_contract.py"
    ),
    "packages/luxar-viewer/src/tests/global-setup.ts": (
        "matched by test_fixture_environment.py through git grep"
    ),
    "packages/luxar-viewer/src/tests/README.md": (
        "matched by test_fixture_environment.py through git grep"
    ),
    "packages/luxar-viewer/tests/fixtures/README.md": (
        "matched by test_fixture_environment.py through git grep"
    ),
}

#: The rule that puts the workflow itself in every domain. Extracted as text so a
#: reword breaks this file rather than silently dropping the only classification
#: ``.github/workflows/ci.yml`` has (none of the four domain patterns match it).
_ALL_DOMAINS_BLOCK_RE = re.compile(
    r"if printf '%s\\n' \"\$changed\" \| grep -qE '([^']*)'; then\n([^\n]*)\n"
)


@pytest.fixture(scope="module")
def workflow() -> str:
    """Raw CI workflow text.

    Read as TEXT on purpose: the patterns live inside single-quoted shell strings
    in a ``run: |`` block, so YAML gives back one opaque script and the literal
    spelling a maintainer edits is the thing under test. Job-graph assertions
    parse this text as YAML where structure, rather than shell spelling, matters.
    """
    return WORKFLOW.read_text(encoding="utf-8")


def _domain_patterns(workflow: str) -> dict[str, str]:
    """Map ``py``/``ts``/``rust``/``go`` to the ERE each is classified by.

    Extracted generically from the ``grep -qE '<PATTERN>' && dom_<name>=true``
    lines so a rename or restructure of those lines fails loudly here rather than
    leaving the assertions below matching nothing.

    A domain assigned by more than one line is rejected instead of last-wins:
    splitting the (long) ``dom_py`` grep across two lines is a perfectly good CI
    edit, but it would silently halve what every assertion below actually checks.
    """
    patterns: dict[str, str] = {}
    for pattern, name in re.findall(r"grep -qE '(.*)' && dom_(\w+)=true", workflow):
        assert name not in patterns, (
            f"dom_{name} is set by more than one grep line in ci.yml. That is a "
            "valid workflow, but this test would silently check only the last "
            "one — update it to union the alternatives before landing the split."
        )
        patterns[name] = pattern
    return patterns


def _docs_pattern(workflow: str) -> str:
    """The ``docs_pattern='...'`` assignment that gates the documentation job."""
    match = re.search(r"docs_pattern='(.*)'\n", workflow)
    assert match, "could not find the docs_pattern assignment in ci.yml"
    return match.group(1)


def _classifies(pattern: str, path: str) -> bool:
    """Whether GNU ``grep -E`` accepts ``path`` for ``pattern``, as CI would.

    Exit status 2 means grep rejected the pattern itself; that must fail loudly
    instead of being read as an ordinary "no match".
    """
    proc = subprocess.run(
        ["grep", "-qE", pattern],
        input=path + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 1), (
        f"grep rejected the pattern (exit {proc.returncode}): "
        f"{proc.stderr.strip()}\npattern: {pattern}"
    )
    return proc.returncode == 0


@pytest.fixture(scope="module", autouse=True)
def _require_cli_tools() -> None:
    """The real CLI tools used by the workflow must be on PATH.

    For ``grep``, only the executable's presence is checked — which flavour it is
    (GNU here and in CI, BSD on macOS) is not, and does not need to be: what matters
    is that the verdict comes from a POSIX ``grep -E`` rather than from ``re``, whose
    ERE dialect differs in escapes, intervals and backreferences.
    """
    assert shutil.which("grep"), (
        "a POSIX `grep -E` is required to evaluate the CI patterns the way the "
        "workflow does; Python's `re` is a different dialect"
    )
    assert shutil.which("jq"), (
        "`jq` is required to execute the queue watchdog exactly as the workflow does"
    )


def test_all_four_domains_are_extracted(workflow: str) -> None:
    """A restructure of the classifier must break this file, not silence it.

    Every assertion below indexes this dict; if the lines were reworded the tests
    would otherwise pass vacuously on an empty map. A superset check, not equality:
    a legitimate fifth domain must not redden this, but a rename or a removal of
    one of the four still does.
    """
    assert set(_domain_patterns(workflow)) >= {"py", "ts", "rust", "go"}


@pytest.mark.parametrize(("path", "domain", "why"), GATE_INPUTS)
def test_every_gate_input_is_classified(
    workflow: str, path: str, domain: str, why: str
) -> None:
    """A file a gated check reads must switch on the domain that runs that check."""
    assert (REPO / path).exists(), (
        f"the table names {path}, which no longer exists — fix the row"
    )
    pattern = _domain_patterns(workflow)[domain]
    assert _classifies(pattern, path), (
        f"{path} does not set dom_{domain}, so its gate skips green for a "
        f"diff that touches only it — {why}"
    )


@pytest.mark.parametrize("domain", ["py", "ts", "rust", "go"])
@pytest.mark.parametrize("path", NON_DOMAIN_PATHS)
def test_a_prose_only_change_claims_no_language_domain(
    workflow: str, path: str, domain: str
) -> None:
    """The classifier must still discriminate; a catch-all would pass everything.

    Run for every domain, not just ``py``: a pattern degenerated to ``.`` in any
    of the four would otherwise satisfy that domain's positive rows for free.
    """
    pattern = _domain_patterns(workflow)[domain]
    assert not _classifies(pattern, path), (
        f"{path} sets dom_{domain} — the pattern has become a catch-all, which "
        f"makes every dom_{domain} assertion in this file meaningless"
    )


@pytest.mark.parametrize("path", NON_PYTHON_DOMAIN_PATHS)
def test_inputs_without_python_readers_do_not_claim_the_python_domain(
    workflow: str, path: str
) -> None:
    """Cross-language ownership must stay narrower than whole input subtrees."""
    assert (REPO / path).exists(), f"{path} moved; update this test"
    assert not _classifies(_domain_patterns(workflow)["py"], path), (
        f"{path} sets dom_py even though no Python gate reads it; narrow the "
        "cross-language viewer pattern"
    )


def _viewer_source_calls(
    source_roots: tuple[Path, ...] = _PYTHON_SOURCE_ROOTS,
    *,
    repo: Path = REPO,
) -> tuple[set[str], list[str]]:
    paths: set[str] = set()
    non_literal_calls: list[str] = []
    for source_root in source_roots:
        for source_path in source_root.rglob("*.py"):
            source = source_path.read_text(encoding="utf-8")
            if "viewer_source" not in source:
                continue
            relative = source_path.relative_to(repo)
            tree = ast.parse(source, filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    aliased_import = any(
                        alias.name == "viewer_source"
                        and alias.asname not in (None, "viewer_source")
                        for alias in node.names
                    )
                    if aliased_import:
                        non_literal_calls.append(f"{relative}:{node.lineno}")
                    continue
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                is_viewer_source = (
                    isinstance(function, ast.Name) and function.id == "viewer_source"
                ) or (
                    isinstance(function, ast.Attribute)
                    and function.attr == "viewer_source"
                )
                if not is_viewer_source:
                    continue
                if (
                    len(node.args) != 1
                    or not isinstance(node.args[0], ast.Constant)
                    or not isinstance(node.args[0].value, str)
                ):
                    non_literal_calls.append(f"{relative}:{node.lineno}")
                    continue
                paths.add(f"packages/luxar-viewer/{node.args[0].value}")
    return paths, non_literal_calls


def test_viewer_source_scan_covers_every_pytest_source_root() -> None:
    with (REPO / "pyproject.toml").open("rb") as stream:
        pytest_paths = tomllib.load(stream)["tool"]["pytest"]["ini_options"][
            "testpaths"
        ]

    uncovered = [
        path
        for path in pytest_paths
        if not any(
            (REPO / path) == source_root or (REPO / path).is_relative_to(source_root)
            for source_root in _PYTHON_SOURCE_ROOTS
        )
    ]
    assert not uncovered, (
        f"pytest source roots missing from the viewer_source() scan: {uncovered}"
    )


def test_viewer_source_scan_finds_shared_helpers(tmp_path: Path) -> None:
    helper = tmp_path / "shared_helper.py"
    helper.write_text('viewer_source("src/shared.ts")\n', encoding="utf-8")

    paths, errors = _viewer_source_calls((tmp_path,), repo=tmp_path)

    assert paths == {"packages/luxar-viewer/src/shared.ts"}
    assert errors == []


def test_viewer_source_scan_rejects_aliased_imports(tmp_path: Path) -> None:
    helper = tmp_path / "shared_helper.py"
    helper.write_text(
        'from luxar.conftest import viewer_source as source\nsource("src/hidden.ts")\n',
        encoding="utf-8",
    )

    paths, errors = _viewer_source_calls((tmp_path,), repo=tmp_path)

    assert paths == set()
    assert errors == ["shared_helper.py:1"]


def _assert_viewer_source_paths_are_owned(paths: set[str]) -> None:
    python_gate_inputs = {path for path, domain, _why in GATE_INPUTS if domain == "py"}
    negative_readers = sorted(paths & set(NON_PYTHON_DOMAIN_PATHS))
    assert not negative_readers, (
        "NON_PYTHON_DOMAIN_PATHS contains viewer sources now read by Python tests; "
        f"move them into GATE_INPUTS: {negative_readers}"
    )

    missing = sorted(paths - python_gate_inputs)
    assert not missing, (
        "viewer sources read by Python tests must have dom_py GATE_INPUTS rows: "
        f"{missing}"
    )

    python_viewer_inputs = {
        path for path in python_gate_inputs if path.startswith("packages/luxar-viewer/")
    }
    exceptions = set(_NON_SCANNED_PYTHON_VIEWER_INPUTS)
    stale_exceptions = sorted(exceptions - python_viewer_inputs)
    unaccounted_inputs = sorted(python_viewer_inputs - paths - exceptions)
    scanned_exceptions = sorted(paths & exceptions)
    assert not stale_exceptions, (
        "non-scanned viewer input exceptions must name dom_py GATE_INPUTS rows: "
        f"{stale_exceptions}"
    )
    assert not unaccounted_inputs, (
        "dom_py viewer GATE_INPUTS must be discovered through viewer_source() or "
        f"documented as non-scannable: {unaccounted_inputs}"
    )
    assert not scanned_exceptions, (
        "viewer inputs now discovered through viewer_source() must leave the "
        f"non-scanned exception table: {scanned_exceptions}"
    )


def test_negative_viewer_source_reader_reports_negative_control() -> None:
    with pytest.raises(AssertionError, match="NON_PYTHON_DOMAIN_PATHS contains"):
        _assert_viewer_source_paths_are_owned(
            {"packages/luxar-viewer/src/rendering/display-range.ts"}
        )


def test_viewer_source_readers_are_statically_owned_by_the_python_gate() -> None:
    """Every shared viewer-source reader must have one checked classifier row."""
    paths, non_literal_calls = _viewer_source_calls()
    assert not non_literal_calls, (
        "viewer_source() must be called by that name with one string-literal path "
        "so classifier ownership is statically discoverable; invalid uses: "
        f"{non_literal_calls}"
    )
    assert paths, (
        "no viewer_source() calls found; the ownership guard would pass vacuously"
    )
    _assert_viewer_source_paths_are_owned(paths)


@pytest.mark.parametrize("path", NON_DOCS_PATHS)
def test_a_non_documentation_change_does_not_claim_the_docs_gate(
    workflow: str, path: str
) -> None:
    """The docs gate's own anti-catch-all control, on its own axis."""
    assert (REPO / path).exists(), f"{path} moved; update this test"
    assert not _classifies(_docs_pattern(workflow), path), (
        f"{path} sets docs_relevant — docs_pattern has become a catch-all, so "
        "the positive assertions about it prove nothing"
    )


def test_the_workflow_itself_selects_every_language_domain(workflow: str) -> None:
    """``ci.yml`` matches none of the four domain patterns; only this rule saves it.

    Deleting the block leaves a workflow edit classified into no domain at all, so
    a step whose command or condition it broke would be caught by the next
    unrelated PR in that language rather than by the run that contains it.
    """
    blocks = _ALL_DOMAINS_BLOCK_RE.findall(workflow)
    assert len(blocks) == 1, (
        "expected exactly one `if printf ... | grep -qE '...'; then` block in the "
        f"classifier (the all-domains rule), found {len(blocks)}"
    )
    pattern, body = blocks[0]
    assert _classifies(pattern, ".github/workflows/ci.yml"), (
        "the all-domains rule no longer matches .github/workflows/ci.yml, which "
        f"no other pattern classifies either — pattern: {pattern}"
    )
    for domain in ("py", "ts", "rust", "go"):
        assert f"dom_{domain}=true" in body, (
            f"a ci.yml-only diff no longer sets dom_{domain}, so that suite would "
            f"not run for the edit that changed how it is invoked — body: {body!r}"
        )


def test_the_docs_gate_names_its_own_checker_and_baselines(workflow: str) -> None:
    """The sibling invariant that made the complexity hole visible.

    ``docs_pattern`` already names both halves of its gate — the checker and the
    ratchet files it compares against — which is exactly what the complexity
    ratchet was missing. Pinning it keeps the precedent from eroding. The TypeDoc
    warning baseline belongs here, not in the language table: no ``typescript-tests``
    step reads it, the ratchet that does runs inside ``docs-quality``.
    """
    pattern = _docs_pattern(workflow)
    for path in (
        "scripts/check_documentation.py",
        "scripts/docs_baseline.json",
        "packages/luxar-viewer/typedoc-warnings-baseline.json",
    ):
        assert (REPO / path).exists(), f"{path} moved; update this test"
        assert _classifies(pattern, path), (
            f"{path} no longer triggers docs-quality, so a change to the docs "
            "gate's own input would skip the gate"
        )


def test_ci_jobs_respect_the_three_slot_obsidian_admission_contract(
    workflow: str,
) -> None:
    """Obsidian limits per-run slot use and preserves Python memory headroom."""
    jobs = yaml.safe_load(workflow)["jobs"]
    max_parallel = re.sub(r"\s+", "", jobs["python-tests"]["strategy"]["max-parallel"])
    branches = re.fullmatch(
        r"\$\{\{needs\.pick-runner\.outputs\.label=='obsidian'&&(\d+)\|\|(\d+)\}\}",
        max_parallel,
    )
    assert branches is not None, (
        "python-tests max-parallel must branch on pick-runner's obsidian label"
    )
    obsidian_cap, hosted_cap = map(int, branches.groups())
    assert obsidian_cap == 2, (
        "one run's Python matrix must not monopolise obsidian's three slots"
    )

    matrix_expression = jobs["python-tests"]["strategy"]["matrix"]["python-version"]
    matrix_lists = re.findall(r"fromJSON\('([^']+)'\)", matrix_expression)
    assert matrix_lists, "python-tests must declare its event-specific version matrices"
    largest_matrix_size = max(len(json.loads(matrix)) for matrix in matrix_lists)
    assert hosted_cap >= largest_matrix_size, (
        "the hosted max-parallel branch must not throttle the full Python matrix"
    )

    pytest_addopts = jobs["python-tests"]["env"]["PYTEST_ADDOPTS"]
    worker_branches = re.fullmatch(
        r"\$\{\{\s*needs\.pick-runner\.outputs\.label\s*==\s*'obsidian'\s*"
        r"&&\s*'([^']*)'\s*\|\|\s*'([^']*)'\s*\}\}",
        pytest_addopts,
    )
    assert worker_branches is not None, (
        "python-tests must enable xdist only on pick-runner's obsidian label"
    )
    obsidian_args = shlex.split(worker_branches.group(1))
    hosted_args = shlex.split(worker_branches.group(2))
    assert hosted_args == [], "GitHub-hosted Python coverage must stay serial"
    worker_flags = [
        obsidian_args[index + 1]
        for index, arg in enumerate(obsidian_args[:-1])
        if arg == "-n"
    ]
    assert worker_flags == ["2"], (
        "python-tests must leave memory headroom in obsidian's 12 GiB runner slot"
    )
    dist_flags = [
        obsidian_args[index + 1]
        for index, arg in enumerate(obsidian_args[:-1])
        if arg == "--dist"
    ]
    assert dist_flags == ["loadfile"], (
        "parallel coverage must keep each file's shared fixtures on one worker"
    )

    for hosted_job in ("release-readiness", "wheel-viewer"):
        assert jobs[hosted_job]["runs-on"] == "ubuntu-latest", (
            f"{hosted_job} must stay off the capacity-constrained obsidian pool"
        )
        assert jobs[hosted_job]["needs"] == ["changes"], (
            f"{hosted_job} must not wait for unrelated runner-selected jobs"
        )

    assert jobs["queue-watchdog"]["needs"] == ["pick-runner"], (
        "queue-watchdog must still run when diff classification fails"
    )

    pick_runner = jobs["pick-runner"]
    assert pick_runner["permissions"] == {}, (
        "runner selection needs no repository access"
    )
    pick_steps = {
        step.get("id", step.get("name")): step for step in pick_runner["steps"]
    }
    assert set(pick_steps) == {"guard", "route"}
    assert pick_steps["route"]["if"] == "steps.guard.outputs.label == ''"
    assert "label=obsidian" in pick_steps["route"]["run"]
    assert "ubuntu-latest" not in pick_steps["route"]["run"]
    assert "github.event" not in pick_steps["route"]["if"], (
        "schedule and rerun events must not acquire an automatic hosted exception"
    )
    assert "ci_queue_scan" not in str(pick_runner)
    assert "LUXAR_CI_HEARTBEAT" not in str(pick_runner)
    assert "LUXAR_CI_MAX_QUEUED_OBSIDIAN" not in str(pick_runner)

    watchdog = jobs["queue-watchdog"]
    assert watchdog["if"] == "needs.pick-runner.outputs.label == 'obsidian'"
    assert watchdog["permissions"] == {"actions": "write", "contents": "read"}
    watchdog_checkout = watchdog["steps"][0]
    assert watchdog_checkout["continue-on-error"] is True
    assert watchdog_checkout["with"] == {
        # Pinned to dev's tip on the schedule event; empty (== default) otherwise.
        "ref": "${{ github.event_name == 'schedule' && 'refs/heads/dev' || '' }}",
        "persist-credentials": False,
        "sparse-checkout": "scripts",
    }
    assert "not cancelling" in watchdog["steps"][1]["run"]
    assert watchdog["steps"][2]["if"] == ("steps.scanner-checkout.outcome == 'success'")


def test_live_ci_checkouts_share_one_scheduled_dev_sha(workflow: str) -> None:
    """A scheduled run captures one dev SHA and reuses it across every suite job."""
    parsed = yaml.safe_load(workflow)
    jobs = parsed["jobs"]
    expected_ref = "${{ needs.changes.outputs.dev_sha || '' }}"

    scheduled_suite_jobs = {
        job_name
        for job_name, job in jobs.items()
        if job_name == "changes"
        or "changes"
        in (
            [job["needs"]]
            if isinstance(job.get("needs"), str)
            else job.get("needs", [])
        )
    }
    checkout_jobs = {
        job_name: next(
            (
                step
                for step in job["steps"]
                if "actions/checkout" in step.get("uses", "")
            ),
            None,
        )
        for job_name, job in jobs.items()
        if job_name in scheduled_suite_jobs and job.get("if") is not False
    }
    checkout_jobs = {
        job_name: checkout
        for job_name, checkout in checkout_jobs.items()
        if checkout is not None
    }
    assert set(checkout_jobs) == scheduled_suite_jobs

    assert checkout_jobs["changes"]["with"]["ref"] == (
        "${{ github.event_name == 'schedule' && 'refs/heads/dev' || '' }}"
    )
    changes = jobs["changes"]
    assert changes["outputs"]["dev_sha"] == (
        "${{ steps.scheduled-dev.outputs.dev_sha }}"
    )
    capture_step = next(
        step for step in changes["steps"] if step.get("id") == "scheduled-dev"
    )
    assert capture_step["if"] == "github.event_name == 'schedule'"
    assert "git rev-parse HEAD" in capture_step["run"]
    for job_name, checkout in checkout_jobs.items():
        if job_name == "changes":
            continue
        assert checkout["with"]["ref"] == expected_ref, job_name


def test_mypy_gate_targets_stay_synchronized(workflow: str) -> None:
    """Every documented and enforced mypy entry point checks the same files."""
    with (REPO / "pyproject.toml").open("rb") as stream:
        lint_commands = tomllib.load(stream)["tool"]["hatch"]["envs"]["default"][
            "scripts"
        ]["lint"]

    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    make_command = re.search(
        r"^type-check-python:.*\n\t(.+)$", makefile, flags=re.MULTILINE
    )
    assert make_command is not None

    precommit = yaml.safe_load(
        (REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    precommit_hooks = [
        hook for repo in precommit["repos"] for hook in repo.get("hooks", [])
    ]

    claude = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    claude_command = re.search(
        r"^hatch run mypy .+  # Type check$", claude, flags=re.MULTILINE
    )
    assert claude_command is not None

    ci_steps = yaml.safe_load(workflow)["jobs"]["python-tests"]["steps"]
    commands = {
        "pyproject.toml": next(
            command for command in lint_commands if command.startswith("mypy ")
        ),
        "Makefile": make_command.group(1),
        ".pre-commit-config.yaml": next(
            hook["entry"] for hook in precommit_hooks if hook.get("id") == "mypy"
        ),
        "CLAUDE.md": claude_command.group(0).removesuffix("  # Type check"),
        ".github/workflows/ci.yml": next(
            step["run"] for step in ci_steps if step.get("name") == "Type check (mypy)"
        ),
    }

    targets = {}
    for source, command in commands.items():
        arguments = shlex.split(command)
        mypy_index = arguments.index("mypy")
        targets[source] = arguments[mypy_index + 1 :]

    assert len({tuple(paths) for paths in targets.values()}) == 1, (
        f"mypy target lists diverged: {targets}"
    )


def test_obsidian_routed_jobs_have_timeout_headroom(workflow: str) -> None:
    """Every dynamically routed job needs headroom for obsidian starvation."""
    jobs = yaml.safe_load(workflow)["jobs"]
    routed_timeouts = {
        name: job.get("timeout-minutes")
        for name, job in jobs.items()
        if "pick-runner.outputs.label" in str(job.get("runs-on", ""))
    }
    assert routed_timeouts, "CI must keep at least one job behind pick-runner"
    insufficient = {
        name: timeout
        for name, timeout in routed_timeouts.items()
        if not isinstance(timeout, int) or timeout < 120
    }
    assert not insufficient, (
        "every pick-runner-routed job needs at least 120 minutes of timeout "
        f"headroom; under-budget jobs: {insufficient}"
    )


def test_scheduled_ci_supplies_a_green_window_every_three_hours(
    workflow: str,
) -> None:
    """Promotion must not depend on a merge-free hour appearing by chance."""
    # BaseLoader preserves the YAML 1.1 ``on`` key instead of coercing it to True.
    parsed = yaml.load(workflow, Loader=yaml.BaseLoader)
    schedules = [entry["cron"] for entry in parsed["on"]["schedule"]]
    matrix_line = next(
        line for line in workflow.splitlines() if "python-version: ${{" in line
    )
    match = re.search(r"github\.event\.schedule == '([^']+)'", matrix_line)
    assert match is not None, "the full Python matrix must name a daily schedule"
    assert match.group(1) in schedules, (
        "one scheduled window must retain the full daily Python matrix"
    )

    scheduled_hours: list[int] = []
    for schedule in schedules:
        minute, hour, day, month, weekday = schedule.split()
        assert (minute, day, month, weekday) == ("17", "*", "*", "*")
        if hour == "*/3":
            scheduled_hours.extend(range(0, 24, 3))
        else:
            scheduled_hours.extend(int(value) for value in hour.split(","))
    assert sorted(scheduled_hours) == list(range(0, 24, 3))


def test_green_schedule_repairs_cancelled_push_contexts(workflow: str) -> None:
    """A green cron must clear cancelled duplicate contexts on the same SHA."""
    parsed = yaml.safe_load(workflow)
    assert parsed["concurrency"]["group"] == (
        "${{ github.workflow }}-${{ github.event_name }}-${{ github.ref }}-"
        "${{ github.run_attempt == 1 && 'fresh' || github.run_id }}"
    )
    assert parsed["concurrency"]["cancel-in-progress"] is True

    jobs = parsed["jobs"]
    repair = jobs["repair-cancelled-push-checks"]

    assert set(repair["needs"]) == {
        "changes",
        "python-tests",
        "typescript-tests",
        "release-readiness",
        "wheel-viewer",
        "docs-quality",
    }
    condition = re.sub(r"\s+", "", repair["if"])
    assert "github.event_name=='schedule'" in condition
    assert "always()" in condition
    assert "!cancelled()" in condition
    assert repair["permissions"] == {"actions": "write", "contents": "read"}

    # The job checks out the captured dev commit so the guard can cross-check it.
    checkout_step = next(step for step in repair["steps"] if "checkout" in step["uses"])
    assert checkout_step["with"] == {
        "ref": "${{ needs.changes.outputs.dev_sha || '' }}",
    }

    run_step = next(step for step in repair["steps"] if "run" in step)
    assert run_step["env"]["GH_TOKEN"] == "${{ github.token }}"

    script = run_step["run"]
    assert "actions/runs/${GITHUB_RUN_ID}/jobs?filter=latest" in script
    # REGRESSION TRIPWIRE: the walk must resolve dev's tip explicitly and must
    # NOT anchor on ${GITHUB_SHA} (the run's own ref == main's tip after the
    # default-branch flip), which would silently stall promotion.
    assert "git/refs/heads/dev" in script
    assert "compare/main...${dev_sha}" in script
    assert "compare/main...${GITHUB_SHA}" not in script
    # The scheduled event SHA belongs to main after the default-branch flip, so
    # the guard must verify the captured checkout remains on dev's ancestry even
    # if dev advances mid-window. It must not reject main as the default branch.
    assert 'gh api "repos/${GITHUB_REPOSITORY}"' not in script
    assert "'.default_branch'" not in script
    assert "rev-parse HEAD" in script
    assert "rev-parse origin/dev" in script
    assert "git/refs/heads/main" not in script
    assert "merge-base --is-ancestor" in script
    assert "event=push" in script
    assert "status=completed" in script
    assert "head_sha=${candidate_sha}" in script
    assert 'select(.name == "CI")' in script
    assert "max_by(.id).id // empty" in script
    assert "actions/runs/${push_run}/jobs?filter=latest" in script
    assert '.conclusion == "cancelled"' in script
    for context in (
        "python-tests (3.12)",
        "typescript-tests",
        "release-readiness",
        "wheel-viewer",
        "docs-quality",
    ):
        assert context in script
    assert "actions/jobs/$job_id/rerun" in script
    assert "actions/runs/${push_run}/rerun-failed-jobs" in script


def _run_cancelled_push_repair(
    workflow: str,
    tmp_path: Path,
    *,
    scheduled_python: str = "success",
    push_python_latest: str = "cancelled",
    push_typescript: str = "cancelled",
    push_release: str = "cancelled",
    rejected_endpoint: str = "",
    failed_get_endpoint: str = "",
    candidate_shas: tuple[str, ...] = ("deadbeef",),
    dev_ref_sha: str = "deadbeef",
    git_origin_dev: str = "deadbeef",
    git_head_sha: str = "deadbeef",
    github_sha: str = "deadbeef",
    git_is_ancestor: str = "1",
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Execute the repair shell against deterministic workflow/job snapshots."""
    steps = yaml.safe_load(workflow)["jobs"]["repair-cancelled-push-checks"]["steps"]
    script = next(step for step in steps if "run" in step)["run"]
    calls_path = tmp_path / "calls"
    fake_gh = tmp_path / "gh"
    fake_gh.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

endpoint = next((arg for arg in sys.argv if arg.startswith("repos/")), "")
if "--method" in sys.argv:
    with open(os.environ["CALLS_PATH"], "a", encoding="utf-8") as stream:
        stream.write(endpoint + "\\n")
    if endpoint == os.environ["REJECTED_ENDPOINT"]:
        raise SystemExit(1)
elif endpoint == os.environ["FAILED_GET_ENDPOINT"]:
    raise SystemExit(1)
elif endpoint.endswith("/git/refs/heads/dev"):
    # `gh --jq .object.sha` is applied by real gh; the fake prints the value the
    # jq filter would yield. This is the walk anchor (dev's tip).
    print(os.environ["DEV_REF_SHA"])
elif f"/runs/{os.environ['GITHUB_RUN_ID']}/jobs?" in endpoint:
    print(json.dumps([{"jobs": [
        {"id": 10, "name": "python-tests (3.12)", "conclusion": "cancelled"},
        {"id": 11, "name": "python-tests (3.12)", "conclusion": os.environ["SCHEDULED_PYTHON"]},
        {"id": 12, "name": "typescript-tests", "conclusion": "success"},
        {"id": 13, "name": "release-readiness", "conclusion": "success"},
        {"id": 14, "name": "wheel-viewer", "conclusion": "success"},
        {"id": 15, "name": "docs-quality", "conclusion": "success"},
        {"id": 16, "name": "python-tests (3.14)", "conclusion": "failure"},
    ]}]))
elif "/compare/main..." in endpoint:
    print(json.dumps([{"commits": [
        {"sha": sha} for sha in os.environ["CANDIDATE_SHAS"].split(",") if sha
    ]}]))
elif "/actions/runs?" in endpoint:
    if "head_sha=deadbeef" in endpoint:
        print("900")
    elif "head_sha=older" in endpoint:
        print("901")
    elif "head_sha=oldest" in endpoint:
        print("902")
elif "/runs/900/jobs?" in endpoint:
    print(json.dumps([{"jobs": [
        {"id": 21, "name": "python-tests (3.12)", "conclusion": "cancelled"},
        {"id": 22, "name": "typescript-tests", "conclusion": os.environ["PUSH_TYPESCRIPT"]},
        {"id": 23, "name": "release-readiness", "conclusion": os.environ["PUSH_RELEASE"]},
        {"id": 24, "name": "python-tests (3.14)", "conclusion": "cancelled"},
        {"id": 25, "name": "python-tests (3.12)", "conclusion": os.environ["PUSH_PYTHON_LATEST"]},
    ]}]))
elif "/runs/901/jobs?" in endpoint:
    print(json.dumps([{"jobs": [
        {"id": 31, "name": "docs-quality", "conclusion": "cancelled"},
        {"id": 32, "name": "python-tests (3.12)", "conclusion": "success"},
        {"id": 33, "name": "typescript-tests", "conclusion": "success"},
        {"id": 34, "name": "release-readiness", "conclusion": "success"},
        {"id": 35, "name": "wheel-viewer", "conclusion": "success"},
    ]}]))
elif "/runs/902/jobs?" in endpoint:
    print(json.dumps([{"jobs": [
        {"id": 41, "name": "wheel-viewer", "conclusion": "cancelled"},
        {"id": 42, "name": "python-tests (3.12)", "conclusion": "success"},
        {"id": 43, "name": "typescript-tests", "conclusion": "success"},
        {"id": 44, "name": "release-readiness", "conclusion": "success"},
        {"id": 45, "name": "docs-quality", "conclusion": "success"},
    ]}]))
else:
    raise SystemExit(f"unexpected endpoint: {endpoint}")
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    # A fake `git`: `rev-parse HEAD` echoes the checked-out commit and
    # `rev-parse origin/dev` echoes dev's tip (two INDEPENDENT observables, so
    # the guard's HEAD-vs-dev comparison can be exercised for real);
    # `merge-base --is-ancestor` exits 0/1 per GIT_IS_ANCESTOR so the
    # mismatch-classification branch can be driven; `fetch` (and anything else)
    # is a silent no-op so the guard runs without a real repo.
    fake_git = tmp_path / "git"
    fake_git.write_text(
        """#!/usr/bin/env python3
import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "rev-parse":
    target = sys.argv[2] if len(sys.argv) > 2 else ""
    if target == "HEAD":
        print(os.environ["GIT_HEAD_SHA"])
    else:
        print(os.environ["GIT_ORIGIN_DEV"])
elif len(sys.argv) > 2 and sys.argv[1] == "merge-base" and sys.argv[2] == "--is-ancestor":
    raise SystemExit(0 if os.environ["GIT_IS_ANCESTOR"] == "1" else 1)
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    result = subprocess.run(
        ["bash", "-e", "-c", script],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=os.environ
        | {
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CALLS_PATH": str(calls_path),
            "GITHUB_REPOSITORY": "royerlab/luxar",
            "GITHUB_RUN_ID": "800",
            "GITHUB_SHA": github_sha,
            "SCHEDULED_PYTHON": scheduled_python,
            "PUSH_PYTHON_LATEST": push_python_latest,
            "PUSH_TYPESCRIPT": push_typescript,
            "PUSH_RELEASE": push_release,
            "REJECTED_ENDPOINT": rejected_endpoint,
            "FAILED_GET_ENDPOINT": failed_get_endpoint,
            "CANDIDATE_SHAS": ",".join(candidate_shas),
            # DEV_REF_SHA = the gh-resolved walk anchor. GIT_HEAD_SHA = the
            # commit the run actually checked out. GIT_ORIGIN_DEV = dev's tip.
            # HEAD == origin/dev is the common healthy path; an older ancestor
            # models a merge landing after the captured checkout.
            "DEV_REF_SHA": dev_ref_sha,
            "GIT_ORIGIN_DEV": git_origin_dev,
            "GIT_HEAD_SHA": git_head_sha,
            "GIT_IS_ANCESTOR": git_is_ancestor,
        },
    )
    calls = calls_path.read_text().splitlines() if calls_path.exists() else []
    return result, calls


def test_green_schedule_reruns_all_latest_cancelled_required_push_jobs(
    workflow: str, tmp_path: Path
) -> None:
    result, calls = _run_cancelled_push_repair(workflow, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == [
        "repos/royerlab/luxar/actions/runs/900/rerun-failed-jobs",
    ]


def test_green_schedule_caps_repairs_at_two_newest_candidates(
    workflow: str, tmp_path: Path
) -> None:
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        candidate_shas=("oldest", "older", "missing", "deadbeef"),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == [
        "repos/royerlab/luxar/actions/runs/900/rerun-failed-jobs",
        "repos/royerlab/luxar/actions/jobs/31/rerun",
    ]
    assert "No completed push CI run found for missing" in result.stdout
    assert "Repair cap reached; older candidates intentionally skipped" in result.stdout


def test_green_schedule_ignores_older_cancelled_push_attempt(
    workflow: str, tmp_path: Path
) -> None:
    result, calls = _run_cancelled_push_repair(
        workflow, tmp_path, push_python_latest="success", push_release="success"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == ["repos/royerlab/luxar/actions/jobs/22/rerun"]


def test_single_cancelled_required_job_uses_job_rerun(
    workflow: str, tmp_path: Path
) -> None:
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        push_typescript="success",
        push_release="success",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == ["repos/royerlab/luxar/actions/jobs/25/rerun"]


def test_rejected_multi_job_rerun_is_reported(workflow: str, tmp_path: Path) -> None:
    endpoint = "repos/royerlab/luxar/actions/runs/900/rerun-failed-jobs"
    result, calls = _run_cancelled_push_repair(
        workflow, tmp_path, rejected_endpoint=endpoint
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == [endpoint]
    assert (
        "rerun rejected for push run 900 (deadbeef; 3 cancelled required jobs)"
        in result.stdout
    )


def test_cancelled_count_is_not_padded_by_a_bsd_wc(
    workflow: str, tmp_path: Path
) -> None:
    """BSD wc pads its count; the warning text must not carry that padding."""
    padding_wc = tmp_path / "wc"
    padding_wc.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        'print("%8d" % sys.stdin.buffer.read().count(b"\\n"))\n',
        encoding="utf-8",
    )
    padding_wc.chmod(0o755)
    endpoint = "repos/royerlab/luxar/actions/runs/900/rerun-failed-jobs"
    result, calls = _run_cancelled_push_repair(
        workflow, tmp_path, rejected_endpoint=endpoint
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == [endpoint]
    assert (
        "rerun rejected for push run 900 (deadbeef; 3 cancelled required jobs)"
        in result.stdout
    )


def test_rejected_rerun_does_not_consume_repair_cap(
    workflow: str, tmp_path: Path
) -> None:
    endpoint = "repos/royerlab/luxar/actions/runs/900/rerun-failed-jobs"
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        rejected_endpoint=endpoint,
        candidate_shas=("oldest", "older", "deadbeef"),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == [
        endpoint,
        "repos/royerlab/luxar/actions/jobs/31/rerun",
        "repos/royerlab/luxar/actions/jobs/41/rerun",
    ]


@pytest.mark.parametrize(
    "failed_get_endpoint",
    [
        "repos/royerlab/luxar/actions/runs?head_sha=deadbeef&event=push&status=completed&per_page=100",
        "repos/royerlab/luxar/actions/runs/900/jobs?filter=latest&per_page=100",
    ],
)
def test_candidate_api_failure_does_not_abort_backlog_repair(
    workflow: str, tmp_path: Path, failed_get_endpoint: str
) -> None:
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        failed_get_endpoint=failed_get_endpoint,
        candidate_shas=("older", "deadbeef"),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == ["repos/royerlab/luxar/actions/jobs/31/rerun"]
    assert "API query failed for deadbeef; continuing backlog repair" in result.stdout


def test_non_green_schedule_does_not_repair_push_jobs(
    workflow: str, tmp_path: Path
) -> None:
    result, calls = _run_cancelled_push_repair(
        workflow, tmp_path, scheduled_python="failure"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == []
    assert "python-tests (3.12)=failure" in result.stdout


# --- Precondition guard: the scheduled checkout must remain on dev, loudly. ----
# GitHub runs `schedule:` on the default branch, so after the dev->main flip the
# run's own ref/SHA is main's tip even though the suite checks out captured dev.
# The guard asserts that checkout remains on origin/dev's line before walking the
# explicit dev tip. Its failure arms are tested because they have no other signal.


def test_promotion_guard_green_when_dev_is_ahead_of_main(
    workflow: str, tmp_path: Path
) -> None:
    """Healthy, dev AHEAD of main -> GREEN and the backlog is repaired.

    Dev being ahead is modelled by a non-empty candidate list; ``deadbeef`` is
    the fake push-run map's head SHA so a real rerun is issued.
    """
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        dev_ref_sha="deadbeef",
        git_origin_dev="deadbeef",
        candidate_shas=("deadbeef",),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Walking commits ahead of main through dev tip deadbeef" in result.stdout
    assert calls == ["repos/royerlab/luxar/actions/runs/900/rerun-failed-jobs"]


def test_promotion_guard_green_when_dev_equals_main(
    workflow: str, tmp_path: Path
) -> None:
    """Healthy post-promote-ff steady state (dev == main) -> GREEN, no repair.

    This is the false positive the guard is written to AVOID: a naive
    ``dev_sha != main-tip`` check would redden this normal, nothing-to-do path.
    """
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        dev_ref_sha="deadbeef",
        git_origin_dev="deadbeef",
        candidate_shas=(),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == []
    assert "dev is level with main" in result.stdout


def test_promotion_guard_green_when_schedule_is_dispatched_on_main(
    workflow: str, tmp_path: Path
) -> None:
    """A main event SHA does not block repair of the captured dev checkout."""
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        dev_ref_sha="devtip",
        git_origin_dev="devtip",
        git_head_sha="devtip",
        github_sha="maintip",
        candidate_shas=(),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "scheduled run SHA maintip" in result.stdout
    assert "dev is level with main" in result.stdout
    assert calls == []


def test_promotion_guard_green_on_merge_during_checkout(
    workflow: str, tmp_path: Path
) -> None:
    """Merge landed BETWEEN checkout and read -> HEAD is an ANCESTOR of dev's
    tip (not main): benign GREEN with a note. This is the race the re-read could
    not fix (HEAD is frozen at an older dev commit; re-fetching only moves the
    tip further ahead) -- classification tolerates it.
    """
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        dev_ref_sha="devtip",
        git_origin_dev="devtip",
        git_head_sha="olddev",  # an earlier dev commit
        git_is_ancestor="1",  # HEAD is an ancestor of dev's tip
        candidate_shas=(),  # nothing further to promote in this scenario
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "dev advanced since checkout (olddev -> devtip)" in result.stdout
    assert calls == []


def test_promotion_guard_green_when_main_catches_up_during_window(
    workflow: str, tmp_path: Path
) -> None:
    """Promotion reaching the tested SHA does not invalidate an older dev checkout."""
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        dev_ref_sha="devtip",
        git_origin_dev="devtip",
        git_head_sha="deadbeef",
        git_is_ancestor="1",
        candidate_shas=("deadbeef",),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "dev advanced since checkout (deadbeef -> devtip)" in result.stdout
    assert calls == ["repos/royerlab/luxar/actions/runs/900/rerun-failed-jobs"]


def test_promotion_guard_reds_on_unknown_branch(workflow: str, tmp_path: Path) -> None:
    """HEAD outside dev's ancestry fails loudly."""
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        dev_ref_sha="devtip",
        git_origin_dev="devtip",
        git_head_sha="rogue",
        git_is_ancestor="0",  # not an ancestor of dev either
        candidate_shas=("devtip",),
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "unknown branch" in output
    assert "checked out rogue is not an ancestor of origin/dev" in output
    assert calls == []


def test_promotion_guard_reds_when_dev_ref_is_unresolvable(
    workflow: str, tmp_path: Path
) -> None:
    """A scheduled promotion run that cannot resolve dev fails loudly, not idle."""
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        dev_ref_sha="",
    )

    assert result.returncode != 0
    assert "could not resolve refs/heads/dev" in (result.stdout + result.stderr)
    assert calls == []


def _run_pick_runner(
    workflow: str,
    tmp_path: Path,
    *,
    head_repo: str = "royerlab/luxar",
    force_hosted: str = "0",
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the real inline router for a same-repo or hosted-override case."""
    steps = yaml.safe_load(workflow)["jobs"]["pick-runner"]["steps"]
    guard = next(step["run"] for step in steps if step.get("id") == "guard")
    route = next(step["run"] for step in steps if step.get("id") == "route")
    output_path = tmp_path / "github-output"
    script = f"""{guard}
if ! grep -q '^label=' \"$GITHUB_OUTPUT\" 2>/dev/null; then
{route}
fi
"""
    env = os.environ | {
        "GITHUB_OUTPUT": str(output_path),
        "GITHUB_REPOSITORY": "royerlab/luxar",
        "HEAD_REPO": head_repo,
        "FORCE_HOSTED": force_hosted,
    }
    result = subprocess.run(
        ["bash", "-eu", "-o", "pipefail", "-c", script],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=env,
        cwd=REPO,
    )
    label = ""
    if output_path.exists():
        match = re.search(r"^label=(.+)$", output_path.read_text(), re.MULTILINE)
        if match:
            label = match.group(1)
    return result, label


def test_pick_runner_routes_same_repo_to_obsidian(
    workflow: str, tmp_path: Path
) -> None:
    """Same-repo PR, push, schedule, and rerun events must stay on obsidian."""
    result, label = _run_pick_runner(workflow, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert label == "obsidian"
    assert "no automatic paid fallback" in result.stdout


@pytest.mark.parametrize(
    ("head_repo", "force_hosted"),
    [("someone/fork", "0"), ("royerlab/luxar", "1")],
)
def test_pick_runner_hosted_overrides(
    workflow: str,
    tmp_path: Path,
    head_repo: str,
    force_hosted: str,
) -> None:
    """Forks and the explicit repository override must select hosted CI."""
    result, label = _run_pick_runner(
        workflow, tmp_path, head_repo=head_repo, force_hosted=force_hosted
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert label == "ubuntu-latest"


def _run_queue_watchdog(
    workflow: str,
    tmp_path: Path,
    job_snapshots: list[list[dict[str, object]] | str],
    *,
    date_step: int = 0,
    other_run_active: bool = False,
    other_run_active_snapshots: list[bool] | None = None,
    other_run_status: str = "in_progress",
    run_api_error: bool = False,
    run_api_error_once: bool = False,
    job_api_error: bool = False,
    own_job_api_error_call: int = 0,
    other_job_api_error_call: int = 0,
    scanner_error: str = "",
) -> tuple[subprocess.CompletedProcess[str], int, bool]:
    """Run the real inline watchdog against deterministic GitHub API snapshots."""
    watchdog_steps = yaml.safe_load(workflow)["jobs"]["queue-watchdog"]["steps"]
    watchdog = next(
        step["run"]
        for step in watchdog_steps
        if step.get("name", "").startswith("Cancel the run")
    )
    hosted_jobs = [
        _hosted_job("changes", "completed"),
        _hosted_job("pick-runner", "completed"),
        _hosted_job("queue-watchdog", "in_progress"),
        _hosted_job("docs-quality", "queued"),
    ]
    job_snapshots = [
        snapshot
        if isinstance(snapshot, str)
        or any(job["name"] == "queue-watchdog" for job in snapshot)
        else [*hosted_jobs, *snapshot]
        for snapshot in job_snapshots
    ]
    snapshots_path = tmp_path / "snapshots.json"
    counter_path = tmp_path / "jobs-api-calls"
    run_counter_path = tmp_path / "runs-api-calls"
    run_first_status_path = tmp_path / "runs-api-first-status"
    other_job_counter_path = tmp_path / "other-jobs-api-calls"
    date_counter_path = tmp_path / "date-calls"
    cancel_path = tmp_path / "cancelled"
    snapshots_path.write_text(json.dumps(job_snapshots), encoding="utf-8")
    other_active_path = tmp_path / "other-active.json"
    other_active_path.write_text(
        json.dumps(other_run_active_snapshots or [other_run_active]), encoding="utf-8"
    )

    fake_gh = tmp_path / "gh"
    fake_gh.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

endpoint = next((arg for arg in sys.argv if "/actions/" in arg), "")
if "/actions/runs?" in endpoint:
    query = dict(part.partition("=")[::2] for part in endpoint.partition("?")[2].split("&"))
    status = query.get("status", "")
    counter = Path(os.environ["WATCHDOG_RUN_COUNTER"])
    first_status = Path(os.environ["WATCHDOG_RUN_FIRST_STATUS"])
    if not first_status.exists():
        first_status.write_text(status)
    # Index by SCAN, not by call: one scan asks for every requested run-level
    # status in turn, and only the first of them opens a new scan.
    if status == first_status.read_text():
        call = int(counter.read_text() or "0") if counter.exists() else 0
        counter.write_text(str(call + 1))
    else:
        call = max(int(counter.read_text() or "0") - 1, 0)
    error_mode = os.environ["WATCHDOG_RUN_API_ERROR"]
    if error_mode == "1" or (error_mode == "once" and call == 1):
        raise SystemExit(1)
    active_snapshots = json.loads(Path(os.environ["WATCHDOG_OTHER_ACTIVE"]).read_text())
    active = active_snapshots[min(call, len(active_snapshots) - 1)]
    # The cross-run 9999 appears under exactly one run-level status.
    active_here = active and status == os.environ["WATCHDOG_OTHER_RUN_STATUS"]
    run_ids = [2038, 9999] if active_here else [2038]
    print(json.dumps({"workflow_runs": [{"id": run_id} for run_id in run_ids]}))
elif "/runs/9999/jobs?" in endpoint:
    counter = Path(os.environ["WATCHDOG_OTHER_JOB_COUNTER"])
    call = int(counter.read_text() or "0") if counter.exists() else 0
    counter.write_text(str(call + 1))
    error_call = int(os.environ["WATCHDOG_OTHER_JOB_API_ERROR_CALL"])
    if os.environ["WATCHDOG_JOB_API_ERROR"] == "1" or error_call == call + 1:
        raise SystemExit(1)
    print(json.dumps({"jobs": [{"name": "other-python", "status": "in_progress", "labels": ["obsidian"]}]}))
elif "/jobs?" in endpoint:
    if "--jq" in sys.argv:
        raise SystemExit(f"unexpected --jq for jobs endpoint: {sys.argv!r}")
    counter = Path(os.environ["WATCHDOG_COUNTER"])
    call = int(counter.read_text() or "0") if counter.exists() else 0
    counter.write_text(str(call + 1))
    if int(os.environ["WATCHDOG_OWN_JOB_API_ERROR_CALL"]) == call + 1:
        raise SystemExit(1)
    snapshots = json.loads(Path(os.environ["WATCHDOG_SNAPSHOTS"]).read_text())
    jobs = snapshots[min(call, len(snapshots) - 1)]
    if jobs == "invalid-json":
        print("{")
        raise SystemExit(0)
    print(json.dumps({"jobs": jobs}))
elif endpoint.endswith("/cancel"):
    Path(os.environ["WATCHDOG_CANCELLED"]).write_text("yes")
else:
    raise SystemExit(f"unexpected gh invocation: {sys.argv!r}")
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    fake_date = tmp_path / "date"
    fake_date.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path

counter = Path(os.environ["WATCHDOG_DATE_COUNTER"])
call = int(counter.read_text() or "0") if counter.exists() else 0
counter.write_text(str(call + 1))
print(1000 + call * int(os.environ["WATCHDOG_DATE_STEP"]))
""",
        encoding="utf-8",
    )
    fake_date.chmod(0o755)
    for name, body in (("sleep", "#!/bin/sh\nexit 0\n"),):
        command = tmp_path / name
        command.write_text(body, encoding="utf-8")
        command.chmod(0o755)
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        """#!/bin/sh
case "$SCANNER_ERROR:$*" in
  scan:*scripts/ci_queue_scan.py*scan*|classify:*scripts/ci_queue_scan.py*classify*|all:*scripts/ci_queue_scan.py*) exit 1 ;;
esac
exec "$REAL_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "royerlab/luxar",
        "GITHUB_RUN_ID": "2038",
        "WATCHDOG_SNAPSHOTS": str(snapshots_path),
        "WATCHDOG_COUNTER": str(counter_path),
        "WATCHDOG_RUN_COUNTER": str(run_counter_path),
        "WATCHDOG_RUN_FIRST_STATUS": str(run_first_status_path),
        "WATCHDOG_OTHER_JOB_COUNTER": str(other_job_counter_path),
        "WATCHDOG_OTHER_ACTIVE": str(other_active_path),
        "WATCHDOG_OTHER_RUN_STATUS": other_run_status,
        "WATCHDOG_RUN_API_ERROR": (
            "once" if run_api_error_once else "1" if run_api_error else "0"
        ),
        "WATCHDOG_JOB_API_ERROR": "1" if job_api_error else "0",
        "WATCHDOG_OWN_JOB_API_ERROR_CALL": str(own_job_api_error_call),
        "WATCHDOG_OTHER_JOB_API_ERROR_CALL": str(other_job_api_error_call),
        "WATCHDOG_DATE_COUNTER": str(date_counter_path),
        "WATCHDOG_DATE_STEP": str(date_step),
        "WATCHDOG_CANCELLED": str(cancel_path),
        "REAL_PYTHON": sys.executable,
        "SCANNER_ERROR": scanner_error,
    }
    result = subprocess.run(
        ["bash", "-e", "-c", watchdog],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
        env=env,
        cwd=REPO,
    )
    calls = (
        int(counter_path.read_text(encoding="utf-8") or "0")
        if counter_path.exists()
        else 0
    )
    return result, calls, cancel_path.exists()


def _obsidian_job(name: str, status: str) -> dict[str, object]:
    """Build a watchdog API job resolved to the obsidian runner label."""
    return {"name": name, "status": status, "labels": ["obsidian"]}


def _hosted_job(name: str, status: str) -> dict[str, object]:
    return {"name": name, "status": status, "labels": ["ubuntu-latest"]}


def test_queue_watchdog_waits_for_obsidian_jobs_to_materialize(
    workflow: str, tmp_path: Path
) -> None:
    """The hosted watchdog must not win its startup race with dependent jobs."""
    snapshots = [
        [],
        [
            _obsidian_job("python-tests (3.12)", "in_progress"),
            _obsidian_job("python-tests (3.14)", "queued"),
        ],
        [
            _obsidian_job("python-tests (3.12)", "completed"),
            _obsidian_job("python-tests (3.14)", "in_progress"),
        ],
    ]
    result, calls, cancelled = _run_queue_watchdog(workflow, tmp_path, snapshots)

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 3, "watchdog exited before obsidian-routed jobs appeared"
    assert not cancelled


def test_queue_watchdog_waits_past_grace_while_fanout_is_still_pending(
    workflow: str, tmp_path: Path
) -> None:
    """Hosted queue delay must not consume the obsidian materialization grace."""
    queued = [
        _hosted_job("changes", "queued"),
        _hosted_job("pick-runner", "completed"),
        _hosted_job("queue-watchdog", "in_progress"),
    ]
    running = [
        _hosted_job("changes", "in_progress"),
        _hosted_job("pick-runner", "completed"),
        _hosted_job("queue-watchdog", "in_progress"),
    ]
    snapshots = [
        queued,
        queued,
        queued,
        queued,
        running,
        [
            _hosted_job("changes", "completed"),
            _hosted_job("pick-runner", "completed"),
            _hosted_job("queue-watchdog", "in_progress"),
            _obsidian_job("python-tests (3.12)", "in_progress"),
            _obsidian_job("python-tests (3.14)", "queued"),
        ],
        [
            _hosted_job("changes", "completed"),
            _hosted_job("pick-runner", "completed"),
            _hosted_job("queue-watchdog", "in_progress"),
            _obsidian_job("python-tests (3.14)", "in_progress"),
        ],
    ]
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        snapshots,
        date_step=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 7, "watchdog exited while the dependent fan-out was still pending"
    assert not cancelled


def test_queue_watchdog_rearmed_grace_stays_bounded_after_fanout_finishes(
    workflow: str, tmp_path: Path
) -> None:
    """A completed fan-out must start a fresh but still bounded startup grace."""
    snapshots = [
        [
            _hosted_job("changes", "queued"),
            _hosted_job("pick-runner", "completed"),
            _hosted_job("queue-watchdog", "in_progress"),
        ],
        [
            _hosted_job("changes", "completed"),
            _hosted_job("pick-runner", "completed"),
            _hosted_job("queue-watchdog", "in_progress"),
        ],
    ]
    result, calls, cancelled = _run_queue_watchdog(
        workflow, tmp_path, snapshots, date_step=60
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 4
    assert "did not appear during the startup grace" in result.stdout
    assert not cancelled


def test_queue_watchdog_keeps_held_matrix_leg_covered_while_siblings_run(
    workflow: str, tmp_path: Path
) -> None:
    """Own running jobs cover a matrix leg held behind the three slots."""
    running = [
        _obsidian_job("typescript-tests", "in_progress"),
        _obsidian_job("python-tests (3.12)", "in_progress"),
        _obsidian_job("python-tests (3.13)", "in_progress"),
        _obsidian_job("python-tests (3.14)", "queued"),
    ]
    snapshots = [
        running,
        running,
        running,
        [
            _obsidian_job("typescript-tests", "completed"),
            _obsidian_job("python-tests (3.14)", "in_progress"),
        ],
    ]
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        snapshots,
        date_step=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 4
    assert "this run has active obsidian jobs" in result.stdout
    assert not cancelled, "busy capacity was mistaken for a dead obsidian host"


def test_queue_watchdog_cancels_without_active_jobs(
    workflow: str, tmp_path: Path
) -> None:
    """A wholly queued run is cancelled after two clean repository scans."""
    queued = [_obsidian_job("python-tests (3.12)", "queued")]
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        [queued] * 6,
        date_step=30,
    )

    assert result.returncode == 1
    assert calls == 6
    assert "no active jobs on two consecutive checks" in result.stdout
    assert cancelled


def test_queue_watchdog_stops_waiting_for_jobs_that_never_materialize(
    workflow: str, tmp_path: Path
) -> None:
    """The jobs API startup grace must not consume the full watchdog window."""
    result, calls, cancelled = _run_queue_watchdog(
        workflow, tmp_path, [[]], date_step=60
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 3
    assert "did not appear during the startup grace" in result.stdout
    assert not cancelled


def test_queue_watchdog_retries_unparseable_jobs_response(
    workflow: str, tmp_path: Path
) -> None:
    """A transient malformed API response must not red or cancel a healthy run."""
    snapshots: list[list[dict[str, object]] | str] = [
        "invalid-json",
        [
            _obsidian_job("python-tests (3.12)", "in_progress"),
            _obsidian_job("python-tests (3.14)", "queued"),
        ],
        [_obsidian_job("python-tests (3.14)", "in_progress")],
    ]
    result, calls, cancelled = _run_queue_watchdog(workflow, tmp_path, snapshots)

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 3
    assert "not parseable; retrying" in result.stdout
    assert not cancelled


def test_queue_watchdog_retries_when_classifier_crashes(
    workflow: str, tmp_path: Path
) -> None:
    """A broken classifier must not retire or cancel the watchdog."""
    queued = [_obsidian_job("python-tests (3.12)", "queued")]
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        [queued],
        date_step=60,
        scanner_error="classify",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls > 1
    assert "not parseable; retrying" in result.stdout
    assert not cancelled


def test_queue_watchdog_accepts_cross_run_activity_when_every_job_is_queued(
    workflow: str, tmp_path: Path
) -> None:
    """Other runs may saturate every slot while this run remains wholly queued."""
    queued = [_obsidian_job("python-tests (3.12)", "queued")]
    snapshots = [
        queued,
        queued,
        queued,
        [_obsidian_job("python-tests (3.12)", "in_progress")],
    ]
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        snapshots,
        other_run_active=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 4
    assert "other runs have active obsidian jobs" in result.stdout
    assert not cancelled


def test_queue_watchdog_sees_obsidian_work_inside_a_queued_run(
    workflow: str, tmp_path: Path
) -> None:
    """A run reads `queued` while its obsidian jobs run; that is live capacity."""
    queued = [_obsidian_job("python-tests (3.12)", "queued")]
    snapshots: list[list[dict[str, object]] | str] = [
        *[queued] * 6,
        [_obsidian_job("python-tests (3.12)", "in_progress")],
    ]
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        snapshots,
        other_run_active=True,
        other_run_status="queued",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 7
    assert "other runs have active obsidian jobs (other-python)" in result.stdout
    assert not cancelled


@pytest.mark.parametrize(
    ("api_error", "expected_message"),
    [("runs", "run liveness unreadable"), ("jobs", "job liveness unreadable")],
)
def test_queue_watchdog_fails_liveness_reads_open(
    workflow: str, tmp_path: Path, api_error: str, expected_message: str
) -> None:
    """An unreadable cross-run signal must never cancel queued obsidian work."""
    queued = [_obsidian_job("python-tests (3.12)", "queued")]
    snapshots = [
        queued,
        queued,
        queued,
        [_obsidian_job("python-tests (3.12)", "in_progress")],
    ]
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        snapshots,
        other_run_active=True,
        run_api_error=api_error == "runs",
        job_api_error=api_error == "jobs",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 4
    assert expected_message in result.stdout
    assert not cancelled


def test_queue_watchdog_fails_liveness_helper_crash_open(
    workflow: str, tmp_path: Path
) -> None:
    """A broken repository scanner must never contribute cancellation evidence."""
    queued = [_obsidian_job("python-tests (3.12)", "queued")]
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        [queued],
        date_step=30,
        scanner_error="scan",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls >= 3
    assert "run liveness unreadable" in result.stdout
    assert "cancelling the run" not in result.stdout
    assert not cancelled


def test_queue_watchdog_api_error_breaks_no_activity_streak(
    workflow: str, tmp_path: Path
) -> None:
    """An unreadable scan must break consecutive no-activity evidence."""
    queued = [_obsidian_job("python-tests (3.12)", "queued")]
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        [queued] * 12,
        run_api_error_once=True,
        date_step=30,
    )

    assert result.returncode == 1
    assert calls == 12
    assert "run liveness unreadable" in result.stdout
    assert cancelled


@pytest.mark.parametrize(
    ("interruption", "expected_calls", "expected_message"),
    [
        ("cross-run activity", 12, "other runs have active obsidian jobs"),
        ("own-run activity", 10, "this run has active obsidian jobs"),
        ("own-jobs API error", 10, "jobs API read failed"),
        ("other-jobs API error", 12, "job liveness unreadable"),
    ],
)
def test_queue_watchdog_interruption_breaks_no_activity_streak(
    workflow: str,
    tmp_path: Path,
    interruption: str,
    expected_calls: int,
    expected_message: str,
) -> None:
    """Activity or an unreadable signal must break consecutive clean scans."""
    queued = [_obsidian_job("python-tests (3.12)", "queued")]
    snapshots = [queued] * 12
    kwargs: dict[str, object] = {}
    if interruption == "cross-run activity":
        kwargs["other_run_active_snapshots"] = [False, True, False, False]
    elif interruption == "own-run activity":
        snapshots[3] = [
            _obsidian_job("python-tests (3.12)", "queued"),
            _obsidian_job("typescript-tests", "in_progress"),
        ]
    elif interruption == "own-jobs API error":
        kwargs["own_job_api_error_call"] = 4
    else:
        kwargs["other_run_active_snapshots"] = [False, True, False, False]
        kwargs["other_job_api_error_call"] = 1
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        snapshots,
        date_step=30,
        **kwargs,
    )

    assert result.returncode == 1
    assert calls == expected_calls
    assert expected_message in result.stdout
    assert cancelled


def test_queue_watchdog_leaves_live_busy_run_alone_when_window_closes(
    workflow: str, tmp_path: Path
) -> None:
    """Continuously reported active work may outlive the hosted watchdog window."""
    snapshots = [
        [
            _obsidian_job("python-tests (3.12)", "in_progress"),
            _obsidian_job("python-tests (3.14)", "queued"),
        ]
    ]
    result, calls, cancelled = _run_queue_watchdog(
        workflow,
        tmp_path,
        snapshots,
        date_step=100,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 5
    assert "watchdog window over" in result.stdout
    assert not cancelled


def test_queue_watchdog_reports_when_window_closes_before_jobs_materialize(
    workflow: str, tmp_path: Path
) -> None:
    """Window expiry before fan-out must report that no routed jobs appeared."""
    snapshots = [
        [
            _hosted_job("changes", "queued"),
            _hosted_job("pick-runner", "completed"),
            _hosted_job("queue-watchdog", "in_progress"),
        ]
    ]
    result, calls, cancelled = _run_queue_watchdog(
        workflow, tmp_path, snapshots, date_step=100
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == 5
    assert "watchdog window over before obsidian-routed jobs appeared" in result.stdout
    assert "GitHub still reports active obsidian work" not in result.stdout
    assert not cancelled
