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
COVERAGE_WORKFLOW = REPO / ".github/workflows/coverage.yml"
CUDA_WORKFLOW = REPO / ".github/workflows/cuda-nightly.yml"

#: One row per gate input whose required domain is not guaranteed by its ordinary
#: source extension or package path, so an explicit pattern alternative is required.
#: ``(path, domain, why)`` — the reason is quoted back in the failure message.
#: Static scans cover ``viewer_source()`` calls, whole-path Python literals,
#: repo-rooted Python reads, and repo-rooted TypeScript ``readFileSync`` calls.
#:
#: Not every row is load-bearing to the same degree: some are matched by a broad
#: alternative that could not plausibly be removed (``Cargo.lock`` via the whole
#: ``^packages/luxar-viewer/`` prefix, say), so they are belt-and-braces rather
#: than the single thing keeping their gate alive. Do not read the table as a list
#: of narrow escapes.
GATE_INPUTS: list[tuple[str, str, str]] = [
    (
        ".github/workflows/cadence-liveness.yml",
        "py",
        "test_daily_workflow_has_the_permissions_and_token_to_enforce_the_table parses it",
    ),
    (
        ".github/workflows/cuda-nightly.yml",
        "py",
        "test_cuda_cadence_is_dispatch_only_and_requires_two_gpus parses it",
    ),
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
        ".nvmrc",
        "ts",
        "check:node-types compares it against the declared @types/node major",
    ),
    (
        ".gitignore",
        "py",
        "test_wheel_source_completeness.py guards package sources against broad "
        "ignore rules",
    ),
    (
        ".github/workflows/docs.yml",
        "py",
        "test_docs_workflow.py guards the Pages workflow itself",
    ),
    (
        ".github/workflows/coverage.yml",
        "py",
        "this module parses it to pin the coverage split",
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
        "Makefile",
        "ts",
        "generated-fixture-freshness.test.ts pins the ensure-viewer-fixtures E2E "
        "wiring",
    ),
    (
        "packages/luxar-launcher/pkgconfig/webkit2gtk-4.0.pc",
        "py",
        "test_linux_launcher_build_is_configured_for_webkitgtk_4_1 parses the "
        "compatibility module",
    ),
    (
        "packages/luxar-launcher/pkgconfig/webkit2gtk-4.0.pc",
        "go",
        "the launcher cgo build resolves webview_go's pkg-config request through it",
    ),
    (
        "packages/luxar-viewer/src/tests/unit/gallery-selection.test.ts",
        "py",
        "the classifier scans it for repo-rooted readFileSync inputs",
    ),
    (
        "packages/luxar-viewer/src/tests/unit/config/generated-fixture-freshness.test.ts",
        "py",
        "the classifier scans it for repo-rooted readFileSync inputs",
    ),
    (
        "scripts/generate_builtin_colormaps.py",
        "ts",
        "the viewer's third-party notices test scrapes its colormap tables",
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
        "scripts/gallery/media-manifest.json",
        "py",
        "test_verify_media.py checks the committed media inventory against README.md",
    ),
    (
        "scripts/gallery/media-manifest.json",
        "ts",
        "gallery-selection.test.ts validates README media keys against it",
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
        "scripts/zenodo_record_text/records.json",
        "py",
        "test_zenodo_record_text.py checks the committed snapshot index and HTML "
        "digests",
    ),
    (
        "docs/guides/user/CLI_REFERENCE.md",
        "py",
        "test_docs_command_coverage.py drift-guards it against the live Typer app; "
        "its .md only selects docs-quality, which runs no pytest",
    ),
    (
        "docs/guides/developer/DEMO_SITE_RUNBOOK.md",
        "py",
        "test_docs_workflow.py checks its section numbering and archive digest pins",
    ),
    (
        "docs/index.rst",
        "py",
        "test_readme_demo_docs.py drift-guards the published demo count",
    ),
    (
        "docs/api/gsplats.rst",
        "py",
        "test_format_contract.py checks its current gsplats format claim",
    ),
    (
        "docs/guides/user/FORMAT_AND_MIGRATION.md",
        "py",
        "test_format_contract.py checks its current gsplats format claims",
    ),
    (
        "docs/guides/user/LUXAR_ZARR_FORMAT.md",
        "py",
        "test_format_contract.py checks its current gsplats format claims",
    ),
    (
        "docs/specs/GSPLATS_ZARR_FORMAT.md",
        "py",
        "test_format_contract.py checks its current gsplats format claim",
    ),
    (
        "packages/luxar/src/luxar/cli/README.md",
        "py",
        "test_format_contract.py checks its current gsplats format claim",
    ),
    (
        "packages/luxar/src/luxar/gsplats/README.md",
        "py",
        "test_format_contract.py checks its current gsplats format claim",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        "py",
        "test_format_contract.py checks its current gsplats format claims",
    ),
    (
        "packages/luxar/src/luxar/gsplats/lod/README.md",
        "py",
        "test_format_contract.py checks its current gsplats format claim",
    ),
    (
        "packages/luxar/src/luxar/core/group/lod/README.md",
        "py",
        "test_format_contract.py checks its current gsplats format claim",
    ),
    (
        "packages/luxar/src/luxar/core/group/gsplats_pipeline/README.md",
        "py",
        "test_format_contract.py checks its current gsplats format claim",
    ),
    (
        "packages/luxar-viewer/src/data/codecs/README.md",
        "py",
        "test_format_contract.py checks its current gsplats format claim",
    ),
    (
        "packages/luxar-viewer/src/data/scene-loader/lifecycle/load-scene.ts",
        "py",
        "test_format_contract.py checks its readable gsplats format range",
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
        "packages/luxar/README.md",
        "py",
        "test_readme_demo_docs.py drift-guards the published demo count",
    ),
    (
        "packages/luxar-viewer/README.md",
        "py",
        "test_readme_demo_docs.py drift-guards the published demo count",
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
        "control-contract/contract.yaml",
        "py",
        "the source of truth `hatch run check-control-contract` projects from",
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
        "packages/luxar-viewer/scripts/bake-env.mjs",
        "py",
        "test_env_cli.py locks the Python bake URL to the Node driver guard",
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
        "packages/luxar-viewer/src/config/control-contract.ts",
        "py",
        "a generated TypeScript projection `check-control-contract` judges",
    ),
    (
        "packages/luxar-launcher/control_contract.go",
        "py",
        "a generated Go projection `check-control-contract` judges",
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
#: a separate axis). Without these the table above proves nothing: a pattern that
#: matched everything would satisfy every positive row.
NON_DOMAIN_PATHS: list[str] = ["CHANGELOG.md"]

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

_TYPESCRIPT_TEST_READER_INPUTS = frozenset(
    {
        "packages/luxar-viewer/src/tests/unit/gallery-selection.test.ts",
        "packages/luxar-viewer/src/tests/unit/config/generated-fixture-freshness.test.ts",
    }
)

_NON_SCANNED_PYTHON_VIEWER_INPUTS = {
    "packages/luxar-viewer/package.json": "read by check_version_consistency.py",
    "packages/luxar-viewer/README.md": "read by test_readme_demo_docs.py",
    "packages/luxar-viewer/src/data/codecs/README.md": (
        "read by test_format_contract.py through CURRENT_VERSION_CLAIMS"
    ),
    "packages/luxar-viewer/src/data/scene-loader/lifecycle/load-scene.ts": (
        "read by test_format_contract.py through CURRENT_VERSION_CLAIMS"
    ),
    "packages/luxar-viewer/src/types/format-contract.ts": (
        "generated and checked by scripts/gen_format_contract.py"
    ),
    "packages/luxar-viewer/src/config/control-contract.ts": (
        "generated and checked by scripts/gen_control_contract.py"
    ),
    "packages/luxar-viewer/src/tests/global-setup.ts": (
        "matched by test_fixture_environment.py through git grep"
    ),
    "packages/luxar-viewer/src/tests/unit/gallery-selection.test.ts": (
        "scanned for repo-rooted readFileSync inputs by this module"
    ),
    "packages/luxar-viewer/src/tests/unit/config/generated-fixture-freshness.test.ts": (
        "scanned for repo-rooted readFileSync inputs by this module"
    ),
    "packages/luxar-viewer/src/tests/README.md": (
        "matched by test_fixture_environment.py through git grep"
    ),
    "packages/luxar-viewer/tests/fixtures/README.md": (
        "matched by test_fixture_environment.py through git grep"
    ),
}

_NON_GATE_PYTHON_TEST_PATH_LITERAL_EXCLUSIONS = {
    ".github/workflows/ci.yml": "owned by the explicit all-domains workflow block",
    ".github/workflows/publish.yml": (
        "test_set_version.py writes a fixture workflow at this path"
    ),
    ".github/workflows/publish-npm.yml": (
        "test_set_version.py writes a fixture workflow at this path"
    ),
    "CHANGELOG.md": "read by a whole-tree prose vocabulary scan",
    "packages/luxar-viewer/playwright.gallery.config.ts": (
        "test_check_tile_staleness.py writes a fixture file at this path"
    ),
    "packages/luxar-viewer/src/tests/screenshots/crop-policy.ts": (
        "test_check_tile_staleness.py writes a fixture file at this path"
    ),
    "packages/luxar-viewer/src/tests/screenshots/gallery-dimension-readiness.ts": (
        "test_check_tile_staleness.py writes a fixture file at this path"
    ),
    "packages/luxar-viewer/src/tests/screenshots/gallery-media-reporting.ts": (
        "test_check_tile_staleness.py writes a fixture file at this path"
    ),
    "packages/luxar-viewer/src/tests/screenshots/gallery-timelapse-settle.ts": (
        "test_check_tile_staleness.py writes a fixture file at this path"
    ),
    "packages/luxar-viewer/src/tests/screenshots/orbit-axis.ts": (
        "test_check_tile_staleness.py writes a fixture file at this path"
    ),
    "packages/luxar/src/luxar/shading/README.md": (
        "test_check_tile_staleness.py writes a fixture file at this path"
    ),
}

_NON_GATE_PYTHON_TEST_PATH_READ_EXCLUSIONS = {
    ".github/workflows/ci.yml": "owned by the explicit all-domains workflow block",
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


def _tracked_paths(repo: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return set(proc.stdout.split("\0")) - {""}


def _is_python_test_source(source_root: Path, source_path: Path) -> bool:
    relative_parts = source_path.relative_to(source_root).parts
    return (
        source_path.name.startswith("test_")
        or source_path.name == "conftest.py"
        or source_root.name == "tests"
        or "tests" in relative_parts[:-1]
    )


def _tracked_non_python_test_path_literals(
    source_roots: tuple[Path, ...] = _PYTHON_SOURCE_ROOTS,
    *,
    repo: Path = REPO,
    tracked_paths: set[str] | None = None,
) -> set[str]:
    """Find whole tracked non-``.py`` path literals in pytest test modules.

    Test modules, ``conftest.py`` files, and helper modules below ``tests/``
    directories are scanned; other modules commonly name generated outputs rather
    than committed inputs. This file is excluded because its control tables name
    paths with every ownership shape. This literal-only pass cannot derive joined
    paths or f-strings; ``_tracked_python_test_path_reads`` separately resolves
    repo-rooted ``Path`` joins that flow into file reads.
    """
    tracked_paths = _tracked_paths(repo) if tracked_paths is None else tracked_paths
    paths: set[str] = set()
    this_file = Path(__file__).resolve()
    for source_root in source_roots:
        for source_path in source_root.rglob("*.py"):
            if not _is_python_test_source(source_root, source_path):
                continue
            if source_path.resolve() == this_file:
                continue
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"), filename=str(source_path)
            )
            paths.update(
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in tracked_paths
                and not node.value.endswith(".py")
            )
    return paths


def _static_python_path(
    node: ast.expr,
    bindings: dict[str, Path],
    source_path: Path,
) -> Path | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Path(node.value)
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return source_path
        return bindings.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _static_python_path(node.left, bindings, source_path)
        right = _static_python_path(node.right, bindings, source_path)
        return left / right if left is not None and right is not None else None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        value = _static_python_path(node.value, bindings, source_path)
        return value.parent if value is not None else None
    if isinstance(node, ast.Subscript):
        return _static_python_parent(node, bindings, source_path)
    if isinstance(node, ast.Call):
        return _static_python_path_call(node, bindings, source_path)
    return None


def _static_python_parent(
    node: ast.Subscript,
    bindings: dict[str, Path],
    source_path: Path,
) -> Path | None:
    if (
        isinstance(node.value, ast.Attribute)
        and node.value.attr == "parents"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ):
        value = _static_python_path(node.value.value, bindings, source_path)
        if value is None:
            return None
        try:
            return value.parents[node.slice.value]
        except IndexError:
            return None
    return None


def _static_python_path_call(
    node: ast.Call,
    bindings: dict[str, Path],
    source_path: Path,
) -> Path | None:
    if len(node.args) == 1 and not node.keywords:
        if isinstance(node.func, ast.Name) and node.func.id == "Path":
            return _static_python_path(node.args[0], bindings, source_path)
    if (
        not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "resolve"
    ):
        value = _static_python_path(node.func.value, bindings, source_path)
        return value.resolve() if value is not None else None
    return None


def _python_path_bindings(tree: ast.Module, source_path: Path) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for statement in tree.body:
        targets: list[ast.expr]
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value_node = statement.value
        else:
            continue
        if value_node is None:
            continue
        value = _static_python_path(value_node, bindings, source_path)
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = value
    return bindings


def _python_open_is_read(call: ast.Call, mode_index: int) -> bool:
    mode_node: ast.expr | None = None
    if len(call.args) > mode_index:
        mode_node = call.args[mode_index]
    else:
        mode_node = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "mode"), None
        )
    if mode_node is None:
        return True
    return (
        isinstance(mode_node, ast.Constant)
        and isinstance(mode_node.value, str)
        and not set(mode_node.value) & set("wax+")
    )


def _python_read_path(
    call: ast.Call,
    bindings: dict[str, Path],
    source_path: Path,
) -> Path | None:
    if isinstance(call.func, ast.Attribute):
        if call.func.attr in {"read_text", "read_bytes"}:
            return _static_python_path(call.func.value, bindings, source_path)
        if call.func.attr == "open" and _python_open_is_read(call, 0):
            return _static_python_path(call.func.value, bindings, source_path)
    if (
        isinstance(call.func, ast.Name)
        and call.func.id == "open"
        and call.args
        and _python_open_is_read(call, 1)
    ):
        return _static_python_path(call.args[0], bindings, source_path)
    return None


def _tracked_python_test_path_reads(
    source_roots: tuple[Path, ...] = _PYTHON_SOURCE_ROOTS,
    *,
    repo: Path = REPO,
    tracked_paths: set[str] | None = None,
) -> set[str]:
    """Find committed-file reads through statically resolvable ``Path`` chains.

    Module-level bindings may derive from ``__file__`` through ``Path``, ``resolve``,
    ``parent``/``parents``, and ``/``. Only ``read_text``, ``read_bytes``, and
    read-only ``open`` calls whose final path stays below ``repo``, names a tracked
    file, and does not name Python source are retained.
    """
    tracked_paths = _tracked_paths(repo) if tracked_paths is None else tracked_paths
    repo_resolved = repo.resolve()
    paths: set[str] = set()
    this_file = Path(__file__).resolve()
    for source_root in source_roots:
        for source_path in source_root.rglob("*.py"):
            if not _is_python_test_source(source_root, source_path):
                continue
            source_path_resolved = source_path.resolve()
            if source_path_resolved == this_file:
                continue
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"), filename=str(source_path)
            )
            bindings = _python_path_bindings(tree, source_path_resolved)
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                path = _python_read_path(call, bindings, source_path_resolved)
                if path is None:
                    continue
                try:
                    relative = path.resolve().relative_to(repo_resolved).as_posix()
                except ValueError:
                    continue
                if relative in tracked_paths and not relative.endswith(".py"):
                    paths.add(relative)
    return paths


_TYPESCRIPT_REPO_READ_RE = re.compile(
    r"""\breadFileSync\(\s*
    (?:(?:path\.)?(?:join|resolve))\(\s*REPO_ROOT
    (?P<arguments>(?:\s*,\s*(?:'[^'\r\n]*'|"[^"\r\n]*"))+)
    \s*\)""",
    re.VERBOSE,
)
_TYPESCRIPT_PATH_PART_RE = re.compile(r"'([^'\r\n]*)'|\"([^\"\r\n]*)\"")


def _tracked_typescript_test_path_reads(
    source_roots: tuple[Path, ...] = (REPO / "packages/luxar-viewer/src",),
    *,
    repo: Path = REPO,
    tracked_paths: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Find literal ``readFileSync(join|resolve(REPO_ROOT, ...))`` test inputs.

    This intentionally scans only ``src/**/*.test.ts`` calls rooted at the literal
    ``REPO_ROOT`` identifier. It does not cover ``import.meta``-rooted reads,
    ``*.spec.ts`` files, or ``scripts/*.test.mjs``. Existing matched reader files
    therefore need explicit ``dom_py`` ownership so edits to the guard run pytest.
    A brand-new reader is reported the next time another Python-relevant change runs
    this repository-wide check.
    """
    tracked_paths = _tracked_paths(repo) if tracked_paths is None else tracked_paths
    paths: set[str] = set()
    reader_paths: set[str] = set()
    for source_root in source_roots:
        for source_path in source_root.rglob("*.test.ts"):
            source = source_path.read_text(encoding="utf-8")
            matches = list(_TYPESCRIPT_REPO_READ_RE.finditer(source))
            if matches:
                reader_paths.add(source_path.relative_to(repo).as_posix())
            for match in matches:
                parts = [
                    single or double
                    for single, double in _TYPESCRIPT_PATH_PART_RE.findall(
                        match.group("arguments")
                    )
                ]
                relative = Path(*parts).as_posix()
                if relative in tracked_paths:
                    paths.add(relative)
    return paths, reader_paths


def _assert_unclassified_test_paths_are_owned(
    literal_paths: set[str], read_paths: set[str]
) -> None:
    python_gate_inputs = {path for path, domain, _why in GATE_INPUTS if domain == "py"}
    literal_exceptions = set(_NON_GATE_PYTHON_TEST_PATH_LITERAL_EXCLUSIONS)
    read_exceptions = set(_NON_GATE_PYTHON_TEST_PATH_READ_EXCLUSIONS)
    missing = sorted(
        (literal_paths - python_gate_inputs - literal_exceptions)
        | (read_paths - python_gate_inputs - read_exceptions)
    )
    stale_literal_exceptions = sorted(literal_exceptions - literal_paths)
    stale_read_exceptions = sorted(read_exceptions - read_paths)
    assert not missing and not stale_literal_exceptions and not stale_read_exceptions, (
        "tracked non-Python paths named by pytest tests must have dom_py "
        f"GATE_INPUTS rows or justified exclusions: {missing}; Python test "
        "literal exclusions no longer describe unclassified literal results; "
        f"remove or update them: {stale_literal_exceptions}; Python test read "
        "exclusions no longer describe unclassified read results; remove or "
        f"update them: {stale_read_exceptions}"
    )


def _assert_typescript_test_paths_are_owned(paths: set[str], pattern: str) -> None:
    unclassified = {path for path in paths if not _classifies(pattern, path)}
    typescript_gate_inputs = {
        path for path, domain, _why in GATE_INPUTS if domain == "ts"
    }
    missing = sorted(unclassified - typescript_gate_inputs)
    assert not missing, (
        "tracked paths read by TypeScript tests must have dom_ts GATE_INPUTS rows: "
        f"{missing}"
    )


def _assert_typescript_test_reader_paths_are_owned(
    reader_paths: set[str],
    declared_reader_paths: set[str] | frozenset[str] = _TYPESCRIPT_TEST_READER_INPUTS,
) -> None:
    python_gate_inputs = {path for path, domain, _why in GATE_INPUTS if domain == "py"}
    missing = sorted(reader_paths - python_gate_inputs)
    undeclared = sorted(reader_paths - declared_reader_paths)
    stale = sorted(declared_reader_paths - reader_paths)
    assert not missing and not undeclared and not stale, (
        "TypeScript test reader sources must have dom_py GATE_INPUTS rows: "
        f"{missing}; update the declared reader set for new sources: {undeclared}; "
        f"remove stale reader declarations: {stale}"
    )


def test_python_gate_input_scan_uses_only_whole_tracked_test_source_literals(
    tmp_path: Path,
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "test_reader.py").write_text(
        'INPUT = "tracked.json"\nSOURCE = "tracked.py"\nMISSING = "missing.json"\n',
        encoding="utf-8",
    )
    (tmp_path / "conftest.py").write_text('INPUT = "fixture.json"\n', encoding="utf-8")
    (tmp_path / "tests" / "_gate_helpers.py").write_text(
        'INPUT = "helper.json"\n', encoding="utf-8"
    )
    (tmp_path / "generator.py").write_text(
        'OUTPUT = "generated.json"\n', encoding="utf-8"
    )

    paths = _tracked_non_python_test_path_literals(
        (tmp_path,),
        repo=tmp_path,
        tracked_paths={
            "tracked.json",
            "tracked.py",
            "generated.json",
            "fixture.json",
            "helper.json",
        },
    )

    assert paths == {"tracked.json", "fixture.json", "helper.json"}


def test_python_gate_input_scan_includes_helpers_when_source_root_is_tests(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "tests"
    source_root.mkdir()
    (source_root / "_gate_helpers.py").write_text(
        'INPUT = "helper.json"\n', encoding="utf-8"
    )

    paths = _tracked_non_python_test_path_literals(
        (source_root,),
        repo=tmp_path,
        tracked_paths={"helper.json"},
    )

    assert paths == {"helper.json"}


def test_python_gate_input_scan_resolves_only_repo_rooted_reads(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_reader.py").write_text(
        """\
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAPTURE = REPO / "scripts" / "capture.py"
SNAPSHOTS = CAPTURE.parent

(SNAPSHOTS / "records.json").read_text()
(REPO / "bytes.bin").read_bytes()
(REPO / "opened.json").open("rb")
open(REPO / "builtin.json", encoding="utf-8")
(REPO.parent / "outside.json").read_text()
(REPO / "untracked.json").read_text()
(REPO / "tracked.py").read_text()
(REPO / "written.json").write_text("generated")
(REPO / "appended.json").open("a")
(REPO / "keyword-written.json").open(mode="w")
(REPO / "exclusive.json").open("x")
(REPO / "update.json").open("r+")
(REPO / dynamic_name).read_text()
""",
        encoding="utf-8",
    )

    paths = _tracked_python_test_path_reads(
        (tmp_path,),
        repo=tmp_path,
        tracked_paths={
            "scripts/records.json",
            "bytes.bin",
            "opened.json",
            "builtin.json",
            "written.json",
            "appended.json",
            "keyword-written.json",
            "tracked.py",
            "exclusive.json",
            "update.json",
        },
    )

    assert paths == {
        "scripts/records.json",
        "bytes.bin",
        "opened.json",
        "builtin.json",
    }


def test_typescript_gate_input_scan_resolves_only_repo_rooted_reads(
    tmp_path: Path,
) -> None:
    tests = tmp_path / "src/tests/unit"
    tests.mkdir(parents=True)
    (tests / "reader.test.ts").write_text(
        """\
readFileSync(path.join(REPO_ROOT, 'scripts', 'manifest.json'), 'utf-8');
readFileSync(resolve(REPO_ROOT, "Makefile"), "utf8");
readFileSync(path.resolve(REPO_ROOT, 'README.md'));
readFileSync(path.join(REPO_ROOT, 'untracked.json'), 'utf-8');
writeFileSync(path.join(REPO_ROOT, 'written.json'), 'generated');
readFileSync(path.join(REPO_ROOT, dynamicName), 'utf-8');
""",
        encoding="utf-8",
    )
    (tests / "ignored.spec.ts").write_text(
        "readFileSync(resolve(REPO_ROOT, 'ignored.json'), 'utf8');\n",
        encoding="utf-8",
    )

    paths, reader_paths = _tracked_typescript_test_path_reads(
        (tmp_path / "src",),
        repo=tmp_path,
        tracked_paths={
            "scripts/manifest.json",
            "Makefile",
            "README.md",
            "written.json",
            "ignored.json",
        },
    )

    assert paths == {"scripts/manifest.json", "Makefile", "README.md"}
    assert reader_paths == {"src/tests/unit/reader.test.ts"}


def test_python_gate_input_scan_rejects_missing_ownership() -> None:
    with pytest.raises(AssertionError, match="must have dom_py GATE_INPUTS rows"):
        _assert_unclassified_test_paths_are_owned(
            set(_NON_GATE_PYTHON_TEST_PATH_LITERAL_EXCLUSIONS) | {"unowned.json"},
            set(_NON_GATE_PYTHON_TEST_PATH_READ_EXCLUSIONS),
        )


def test_typescript_gate_input_scan_rejects_missing_ownership() -> None:
    with pytest.raises(AssertionError, match="must have dom_ts GATE_INPUTS rows"):
        _assert_typescript_test_paths_are_owned({"unowned.json"}, "^owned\\.json$")


def test_typescript_gate_input_reader_scan_rejects_missing_python_ownership() -> None:
    reader = "packages/luxar-viewer/src/tests/unit/unowned-reader.test.ts"
    with pytest.raises(AssertionError, match="must have dom_py GATE_INPUTS rows"):
        _assert_typescript_test_reader_paths_are_owned({reader}, {reader})


def test_typescript_gate_input_reader_scan_rejects_stale_declarations() -> None:
    with pytest.raises(AssertionError, match="remove stale reader declarations"):
        _assert_typescript_test_reader_paths_are_owned(
            set(),
            {"packages/luxar-viewer/src/tests/unit/removed-reader.test.ts"},
        )


def test_python_gate_input_scan_rejects_stale_exclusions() -> None:
    with pytest.raises(AssertionError, match="exclusions no longer describe"):
        _assert_unclassified_test_paths_are_owned(set(), set())


def test_python_gate_input_scan_reports_missing_and_stale_together() -> None:
    with pytest.raises(AssertionError) as error:
        _assert_unclassified_test_paths_are_owned({"unowned.json"}, set())
    assert "must have dom_py GATE_INPUTS rows" in str(error.value)
    assert "exclusions no longer describe" in str(error.value)


def test_python_gate_input_scan_literal_exclusion_cannot_excuse_read() -> None:
    fixture_path = "packages/luxar-viewer/src/tests/screenshots/crop-policy.ts"
    with pytest.raises(AssertionError, match=re.escape(fixture_path)):
        _assert_unclassified_test_paths_are_owned(
            set(_NON_GATE_PYTHON_TEST_PATH_LITERAL_EXCLUSIONS),
            set(_NON_GATE_PYTHON_TEST_PATH_READ_EXCLUSIONS) | {fixture_path},
        )


def test_python_gate_input_scan_exclusions_have_one_line_reasons() -> None:
    assert all(
        reason and "\n" not in reason
        for reason in (
            *_NON_GATE_PYTHON_TEST_PATH_LITERAL_EXCLUSIONS.values(),
            *_NON_GATE_PYTHON_TEST_PATH_READ_EXCLUSIONS.values(),
        )
    )


@pytest.mark.parametrize(
    "path",
    [
        ".gitignore.bak",
        "scripts/gallery/media-manifest.json.bak",
        "docs/api/gsplats.rst.bak",
        "docs/guides/developer/archive/DEMO_SITE_RUNBOOK.md",
        "docs/guides/user/archive/FORMAT_AND_MIGRATION.md",
        "docs/specs/GSPLATS_ZARR_FORMAT.md.bak",
        "packages/luxar/subdir/README.md",
        "packages/luxar-viewer/docs/README.md",
        "packages/luxar/src/luxar/gsplats/archive/README.md",
        "packages/luxar-viewer/src/data/codecs/archive/README.md",
        "packages/luxar-viewer/src/data/scene-loader/lifecycle/load-scene.ts.bak",
        "packages/luxar-viewer/src/tests/unit/whatever-new.test.ts",
        "scripts/zenodo_record_text_backup/records.json",
        "scripts/zenodo_record_text.md",
    ],
)
def test_derived_python_gate_input_patterns_are_exact(workflow: str, path: str) -> None:
    assert not _classifies(_domain_patterns(workflow)["py"], path)


@pytest.mark.parametrize(
    "path",
    [
        "subdir/Makefile",
        "docs/Makefile.md",
        "scripts/generate_builtin_colormaps_extra.py",
        "scripts/gallery/media-manifest.json.bak",
    ],
)
def test_derived_typescript_gate_input_patterns_are_exact(
    workflow: str, path: str
) -> None:
    assert not _classifies(_domain_patterns(workflow)["ts"], path)


def test_python_test_inputs_are_statically_owned_by_the_python_gate(
    workflow: str,
) -> None:
    """Every discovered Python test input is owned or explained."""
    literal_paths = _tracked_non_python_test_path_literals()
    read_paths = _tracked_python_test_path_reads()
    assert literal_paths, (
        "no Python test path literals found; the ownership guard is vacuous"
    )
    assert read_paths, "no Python test path reads found; the ownership guard is vacuous"
    assert {
        "docs/guides/developer/DEMO_SITE_RUNBOOK.md",
        "scripts/zenodo_record_text/records.json",
    } <= read_paths, (
        "these paths pin the Python read scan against going vacuous; if a genuine "
        "reader was removed, replace its path with a current committed-file canary"
    )
    python_pattern = _domain_patterns(workflow)["py"]
    unclassified_literals = {
        path for path in literal_paths if not _classifies(python_pattern, path)
    }
    unclassified_reads = {
        path for path in read_paths if not _classifies(python_pattern, path)
    }
    _assert_unclassified_test_paths_are_owned(unclassified_literals, unclassified_reads)


def test_typescript_test_path_reads_are_statically_owned_by_the_typescript_gate(
    workflow: str,
) -> None:
    paths, reader_paths = _tracked_typescript_test_path_reads()
    assert paths, "no TypeScript test path reads found; the ownership guard is vacuous"
    assert reader_paths, (
        "no TypeScript test reader sources found; the ownership guard is vacuous"
    )
    assert {
        "Makefile",
        "README.md",
        "scripts/gallery/manifest.json",
        "scripts/gallery/media-manifest.json",
    } <= paths, (
        "these paths pin the TypeScript read scan against going vacuous; if a "
        "genuine reader was removed, replace its path with a current committed-file "
        "canary"
    )
    _assert_typescript_test_reader_paths_are_owned(reader_paths)
    typescript_pattern = _domain_patterns(workflow)["ts"]
    _assert_typescript_test_paths_are_owned(paths, typescript_pattern)


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


def test_mobile_e2e_suite_is_enabled_in_ci(workflow: str) -> None:
    """The mobile suite must remain live, correctly gated, and diagnosable."""
    jobs = yaml.safe_load(workflow)["jobs"]
    job = jobs["e2e-tests"]

    assert set(job["needs"]) == {"changes", "python-tests", "typescript-tests"}
    assert job["if"] == (
        "${{ !cancelled() && needs.python-tests.result != 'failure' && "
        "needs.typescript-tests.result != 'failure' && "
        "needs.changes.outputs.dom_ts != 'false' }}"
    )
    reclaim_step = job["steps"][0]
    assert reclaim_step == jobs["typescript-tests"]["steps"][0]
    assert (reclaim_step["name"], reclaim_step["if"]) == (
        "Free disk space on hosted runners",
        "runner.environment == 'github-hosted'",
    )
    mobile_step = next(
        step for step in job["steps"] if step.get("run") == "make test-e2e-mobile"
    )
    assert mobile_step["timeout-minutes"] == 45
    assert job["timeout-minutes"] == 75

    mobile_config = (
        REPO / "packages/luxar-viewer/playwright.mobile.config.ts"
    ).read_text(encoding="utf-8")
    output_dir = re.search(r"outputDir:\s*['\"]([^'\"]+)['\"]", mobile_config)
    assert output_dir, "mobile Playwright config must declare outputDir"
    upload_step = next(
        step
        for step in job["steps"]
        if "actions/upload-artifact" in step.get("uses", "")
    )
    assert upload_step["with"]["path"] == (
        f"packages/luxar-viewer/{output_dir.group(1)}"
    )


def test_the_docs_gate_names_its_own_checkers_and_baselines(workflow: str) -> None:
    """The sibling invariant that made the complexity hole visible.

    ``docs_pattern`` names its checkers and the ratchet files they compare against,
    which is exactly what the complexity ratchet was missing. Pinning it keeps the
    precedent from eroding. The TypeDoc warning baseline belongs here, not in the
    language table: no ``typescript-tests`` step reads it, the ratchet that does runs
    inside ``docs-quality``.
    """
    pattern = _docs_pattern(workflow)
    for path in (
        "scripts/check_documentation.py",
        "scripts/changelog_build.py",
        "scripts/docs_baseline.json",
        "packages/luxar-viewer/scripts/check-overrides.mjs",
        "packages/luxar-viewer/scripts/check-typedoc-warnings.mjs",
        "packages/luxar-viewer/scripts/check-typedoc-warnings-tests.mjs",
        "packages/luxar-viewer/typedoc.json",
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
        "dispatch and rerun events must not acquire an automatic hosted exception"
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
        # Pinned to dev's tip on the dispatch event; empty (== default) otherwise.
        "ref": "${{ github.event_name == 'workflow_dispatch' && 'refs/heads/dev' || '' }}",
        "persist-credentials": False,
        "sparse-checkout": "scripts",
    }
    assert "not cancelling" in watchdog["steps"][1]["run"]
    assert watchdog["steps"][2]["if"] == ("steps.scanner-checkout.outcome == 'success'")


def test_cuda_cadence_is_dispatch_only_and_requires_two_gpus() -> None:
    """CUDA parity must run only on its dedicated, two-GPU obsidian slot."""
    # BaseLoader keeps YAML 1.1 from coercing the key ``on`` to boolean True.
    parsed = yaml.load(
        CUDA_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    assert parsed["on"] == {"workflow_dispatch": ""}
    assert parsed["concurrency"] == {
        "group": "cuda-native-${{ github.sha }}",
        "cancel-in-progress": "false",
    }

    assert set(parsed["jobs"]) == {"cuda-native"}
    job = parsed["jobs"]["cuda-native"]
    assert job["runs-on"] == ["self-hosted", "obsidian-cuda"]
    assert job["permissions"] == {"contents": "read", "issues": "write"}
    assert job["timeout-minutes"] == "75"
    assert job["env"]["MAX_JOBS"] == "2"
    assert job["env"]["LUXAR_REQUIRE_CUDA"] == "1"

    steps = {step["name"]: step for step in job["steps"] if "name" in step}
    assert steps["Install Hatch"]["timeout-minutes"] == "5"
    assert steps["Require the two-GPU CUDA runner"]["timeout-minutes"] == "15"
    assert steps["Compile-check CUDA translation units"]["timeout-minutes"] == "5"
    assert steps["Build CUDA extensions"]["timeout-minutes"] == "20"
    assert steps["Run CUDA parity suites"]["timeout-minutes"] == "20"
    assert steps["Report cadence failure"]["timeout-minutes"] == "5"

    run_steps = [step["run"] for step in steps.values() if "run" in step]
    commands = "\n".join(run_steps)
    assert "torch.cuda.device_count() >= 2" in commands
    assert "hatch run check-native --only nvcc --require nvcc" in commands
    assert "make build-cuda" in commands
    assert "make build-nlm-cuda" in commands
    assert "make test-cuda" in commands
    assert "make test-nlm-cuda" in commands
    report = steps["Report cadence failure"]
    assert report["if"] == "${{ failure() || cancelled() }}"
    assert re.fullmatch(r"actions/github-script@[0-9a-f]{40}", report["uses"])
    assert report["with"]["github-token"] == "${{ github.token }}"
    script = report["with"]["script"]
    assert "CUDA native cadence failure" in script
    assert "github.paginate(github.rest.issues.listForRepo" in script
    assert "github.rest.issues.createComment" in script
    assert "github.rest.issues.create" in script
    assert 'assignees: ["royerloic"]' in script

    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "pytest $(CUDA_EXT_DIR)/tests/ -v -rs" in makefile
    assert (
        "pytest packages/luxar/src/luxar/gsplats/preprocessing/tests/test_nlm_cuda.py -v -rs"
        in makefile
    )


def test_live_ci_checkouts_attest_one_dispatched_dev_sha(workflow: str) -> None:
    """A dispatched run tests and attests one immutable dev SHA in every suite."""
    parsed = yaml.safe_load(workflow)
    jobs = parsed["jobs"]
    expected_ref = "${{ needs.changes.outputs.dev_sha || '' }}"

    dispatched_suite_jobs = {
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
        if job_name in dispatched_suite_jobs and job.get("if") is not False
    }
    checkout_jobs = {
        job_name: checkout
        for job_name, checkout in checkout_jobs.items()
        if checkout is not None
    }
    assert set(checkout_jobs) == dispatched_suite_jobs

    assert checkout_jobs["changes"]["with"] == {"fetch-depth": 0}
    changes = jobs["changes"]
    assert changes["outputs"]["dev_sha"] == (
        "${{ steps.dispatched-dev.outputs.dev_sha }}"
    )
    capture_step = next(
        step for step in changes["steps"] if step.get("id") == "dispatched-dev"
    )
    assert capture_step["if"] == "github.event_name == 'workflow_dispatch'"
    assert "git rev-parse HEAD" in capture_step["run"]
    for job_name, checkout in checkout_jobs.items():
        if job_name == "changes":
            continue
        assert checkout["with"]["ref"] == expected_ref, job_name


def test_linux_launcher_build_is_configured_for_webkitgtk_4_1(
    workflow: str,
) -> None:
    """The required CI and local build paths must select WebKitGTK 4.1."""
    job = yaml.safe_load(workflow)["jobs"]["go-launcher"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["PKG_CONFIG_PATH"] == (
        "${{ github.workspace }}/packages/luxar-launcher/pkgconfig"
    )

    install_step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Install WebView build deps (cgo)"
    )
    assert "libwebkit2gtk-4.1-dev" in install_step["run"]
    assert "libwebkit2gtk-4.0-dev" not in install_step["run"]

    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert (
        "LAUNCHER_PKG_CONFIG_DIR := $(CURDIR)/$(LAUNCHER_SRC_DIR)/pkgconfig" in makefile
    )
    assert "pkg-config --exists webkit2gtk-4.1" in makefile
    for launcher_command in (
        '"$$GO_BIN" test ./...',
        '"$$GO_BIN" vet ./...',
        "GOOS=linux GOARCH=$$GOARCH CGO_ENABLED=1 $$GO_BIN build",
    ):
        before_command, separator, _ = makefile.partition(launcher_command)
        assert separator, launcher_command
        nearby_lines = "\n".join(before_command.splitlines()[-2:])
        assert "$(LAUNCHER_WEBKIT_ENV)" in nearby_lines, launcher_command

    compatibility_module = (
        REPO / "packages/luxar-launcher/pkgconfig/webkit2gtk-4.0.pc"
    ).read_text(encoding="utf-8")
    assert "Requires: webkit2gtk-4.1" in compatibility_module


@pytest.mark.skipif(
    shutil.which("pkg-config") is None,
    reason="pkg-config is required to exercise launcher module selection",
)
@pytest.mark.parametrize(
    ("module_name", "uses_shim"),
    (("webkit2gtk-4.1.pc", True), ("webkit2gtk-4.0.pc", False)),
)
def test_linux_launcher_makefile_selects_compatibility_module_conditionally(
    tmp_path: Path,
    module_name: str,
    uses_shim: bool,
) -> None:
    """The local launcher commands must preserve 4.0-only hosts and bridge 4.1."""
    pkgconfig_dir = tmp_path / "pkgconfig"
    pkgconfig_dir.mkdir()
    (pkgconfig_dir / module_name).write_text(
        "\n".join(
            (
                "Name: WebKitGTK test module",
                "Description: test module",
                "Version: 4.1",
                "",
            )
        ),
        encoding="utf-8",
    )
    probe = """\
.PHONY: print-launcher-pkg-config-path
print-launcher-pkg-config-path:
\t@$(LAUNCHER_WEBKIT_ENV) printf '%s' "$${PKG_CONFIG_PATH-}"
"""
    base_env = {
        **os.environ,
        "PKG_CONFIG_LIBDIR": str(pkgconfig_dir),
        "PKG_CONFIG_PATH": "/existing/pkgconfig",
    }

    proc = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-f",
            str(REPO / "Makefile"),
            "-f",
            "-",
            "print-launcher-pkg-config-path",
        ],
        input=probe,
        text=True,
        capture_output=True,
        check=True,
        cwd=REPO,
        env=base_env,
    )
    expected_shim = REPO / "packages/luxar-launcher/pkgconfig"
    expected_path = (
        f"{expected_shim}:/existing/pkgconfig" if uses_shim else "/existing/pkgconfig"
    )
    assert proc.stdout == expected_path
    expected_diagnostic = (
        "using the bundled webkit2gtk-4.0 → 4.1 compatibility module"
        if uses_shim
        else "falling back to system webkit2gtk-4.0"
    )
    if uses_shim or sys.platform.startswith("linux"):
        assert expected_diagnostic in proc.stderr


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
    lint_steps = [step for step in ci_steps if step.get("run") == "hatch run lint"]
    assert len(lint_steps) == 1

    commands = {
        "pyproject.toml": next(
            command for command in lint_commands if command.startswith("mypy ")
        ),
        "Makefile": make_command.group(1),
        ".pre-commit-config.yaml": next(
            hook["entry"] for hook in precommit_hooks if hook.get("id") == "mypy"
        ),
        "CLAUDE.md": claude_command.group(0).removesuffix("  # Type check"),
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


def test_ci_repair_window_is_dispatched_on_dev(workflow: str) -> None:
    """Promotion requests repair windows explicitly instead of default-branch crons."""
    # BaseLoader preserves the YAML 1.1 ``on`` key instead of coercing it to True.
    parsed = yaml.load(workflow, Loader=yaml.BaseLoader)
    assert "schedule" not in parsed["on"]
    dispatch = parsed["on"]["workflow_dispatch"]
    full_matrix = dispatch["inputs"]["full_python_matrix"]
    assert full_matrix["required"] == "false"
    assert full_matrix["default"] == "false"
    assert full_matrix["type"] == "boolean"
    assert "dev" in full_matrix["description"]
    assert "repairs cancelled push checks" in full_matrix["description"]
    matrix_line = next(
        line for line in workflow.splitlines() if "python-version: ${{" in line
    )
    assert "github.event.schedule" not in matrix_line
    assert (
        "github.event_name == 'workflow_dispatch' && inputs.full_python_matrix"
        in matrix_line
    )

    changes = yaml.safe_load(workflow)["jobs"]["changes"]
    guard = next(
        step for step in changes["steps"] if step["name"] == "Require a dev dispatch"
    )
    assert guard["if"] == (
        "github.event_name == 'workflow_dispatch' && github.ref != 'refs/heads/dev'"
    )
    assert "--ref dev" in guard["run"]
    assert "exit 1" in guard["run"]


def test_pull_requests_skip_coverage_while_dev_paths_enforce_and_observe_it(
    workflow: str,
) -> None:
    """Pin the coverage split so the 89% gate cannot be lost silently.

    Coverage instrumentation is the dominant cost of ``python-tests``, so PRs run
    the ``-m 'not slow'`` suite plain (``test-nocov``). Pushes to ``dev`` retain
    ``test-cov`` under the protected ``python-tests (3.12)`` context, while the
    separate ``coverage.yml`` workflow records the same result without cancellation.
    Its context is advisory until repository protection requires it. Without this
    test either path could be lost silently -- the fail-open class this module exists
    to catch.
    """
    # (a) ci.yml's test step and hatch scripts: test-nocov on pull_request,
    # test-cov otherwise, with identical target defaults and coverage only in cov.
    steps = yaml.safe_load(workflow)["jobs"]["python-tests"]["steps"]
    run = next(step["run"] for step in steps if "test-nocov" in step.get("run", ""))
    assert '"${{ github.event_name }}" = "pull_request"' in run
    assert "hatch run test-nocov" in run
    assert "hatch run test-cov" in run
    # test-nocov only on the pull_request branch, test-cov only on the else branch.
    assert run.index("test-nocov") < run.index("else") < run.index("hatch run test-cov")
    with (REPO / "pyproject.toml").open("rb") as stream:
        scripts = tomllib.load(stream)["tool"]["hatch"]["envs"]["default"]["scripts"]
    test_cov = scripts["test-cov"]
    test_nocov = scripts["test-nocov"]
    args_pattern = re.compile(r"\{args:([^}]*)\}")
    assert args_pattern.search(test_cov).group(1) == args_pattern.search(
        test_nocov
    ).group(1)
    assert "--cov" in test_cov
    assert "--cov" not in test_nocov

    # (b) the dedicated workflow starts coverage on every push to dev.
    assert COVERAGE_WORKFLOW.exists(), "the dedicated coverage workflow is missing"
    coverage_text = COVERAGE_WORKFLOW.read_text(encoding="utf-8")
    coverage = yaml.safe_load(coverage_text)
    triggers = yaml.load(coverage_text, Loader=yaml.BaseLoader)["on"]
    assert triggers["push"]["branches"] == ["dev"]
    # A schedule trigger would attach its verdict to the default branch, not dev.
    assert "schedule" not in triggers
    # Per-COMMIT group, so a newer dev push cannot cancel an older run.
    assert coverage["concurrency"]["group"] == "coverage-${{ github.sha }}"
    assert coverage["concurrency"]["cancel-in-progress"] is False
    coverage_job = coverage["jobs"]["coverage"]
    assert coverage_job["runs-on"] == (
        "${{ vars.LUXAR_CI_FORCE_HOSTED == '1' && 'ubuntu-latest' || 'obsidian' }}"
    )
    assert coverage_job["env"]["PYTEST_ADDOPTS"] == (
        "${{ vars.LUXAR_CI_FORCE_HOSTED != '1' && '-n 2 --dist loadfile' || '' }}"
    )
    coverage_runs = "\n".join(step.get("run", "") for step in coverage_job["steps"])
    assert "hatch run test-cov" in coverage_runs


def test_green_dispatch_repairs_cancelled_push_contexts(workflow: str) -> None:
    """A green dev dispatch must clear cancelled duplicate contexts on that SHA."""
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
    assert "github.event_name=='workflow_dispatch'" in condition
    assert "always()" in condition
    assert "!cancelled()" in condition
    assert repair["permissions"] == {"actions": "write", "contents": "read"}

    # The job checks out the attested dev commit so the guard can cross-check it.
    checkout_step = next(step for step in repair["steps"] if "checkout" in step["uses"])
    assert checkout_step["with"] == {
        "ref": "${{ needs.changes.outputs.dev_sha || '' }}",
    }

    run_step = next(step for step in repair["steps"] if "run" in step)
    assert run_step["env"]["GH_TOKEN"] == "${{ github.token }}"

    script = run_step["run"]
    assert "actions/runs/${GITHUB_RUN_ID}/jobs?filter=latest" in script
    # REGRESSION TRIPWIRE: the walk must resolve dev's tip explicitly and must
    # NOT replace the explicit dev-tip walk with ${GITHUB_SHA}; the dispatch SHA
    # may lag dev's tip while the window runs.
    assert "git/refs/heads/dev" in script
    assert "compare/main...${dev_sha}" in script
    assert "compare/main...${GITHUB_SHA}" not in script
    # The dispatched event SHA belongs to dev, so
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
    dispatched_python: str = "success",
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
        {"id": 11, "name": "python-tests (3.12)", "conclusion": os.environ["DISPATCHED_PYTHON"]},
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
            "DISPATCHED_PYTHON": dispatched_python,
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


def test_green_dispatch_reruns_all_latest_cancelled_required_push_jobs(
    workflow: str, tmp_path: Path
) -> None:
    result, calls = _run_cancelled_push_repair(workflow, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == [
        "repos/royerlab/luxar/actions/runs/900/rerun-failed-jobs",
    ]


def test_green_dispatch_caps_repairs_at_two_newest_candidates(
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


def test_green_dispatch_ignores_older_cancelled_push_attempt(
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


def test_non_green_dispatch_does_not_repair_push_jobs(
    workflow: str, tmp_path: Path
) -> None:
    result, calls = _run_cancelled_push_repair(
        workflow, tmp_path, dispatched_python="failure"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == []
    assert "python-tests (3.12)=failure" in result.stdout


# --- Precondition guard: the dispatched checkout must remain on dev, loudly. ----
# A `--ref dev` dispatch SHA is a dev commit that may lag dev's tip while the
# window runs. The guard asserts that checkout remains on origin/dev's line before
# walking the explicit dev tip. Its failure arms have no other signal.


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


def test_promotion_guard_green_when_dispatch_targets_dev(
    workflow: str, tmp_path: Path
) -> None:
    """A dev dispatch attests the same commit that the repair job checks out."""
    result, calls = _run_cancelled_push_repair(
        workflow,
        tmp_path,
        dev_ref_sha="devtip",
        git_origin_dev="devtip",
        git_head_sha="devtip",
        github_sha="devtip",
        candidate_shas=(),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "dispatched run SHA devtip" in result.stdout
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
    """A dispatched promotion run that cannot resolve dev fails loudly, not idle."""
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
    """Same-repo PR, push, dispatch, and rerun events must stay on obsidian."""
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
