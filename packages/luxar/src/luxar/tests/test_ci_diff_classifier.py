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

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
WORKFLOW = REPO / ".github/workflows/ci.yml"

#: One row per gate input that carries NO classified source extension, so the only
#: thing standing between it and a silent skip is an explicit pattern alternative.
#: ``(path, domain, why)`` — the reason is quoted back in the failure message.
#:
#: Not every row is load-bearing to the same degree: some are matched by a broad
#: alternative that could not plausibly be removed (``Cargo.lock`` via the whole
#: ``^packages/luxar-viewer/`` prefix, say), so they are belt-and-braces rather
#: than the single thing keeping their gate alive. Do not read the table as a list
#: of narrow escapes.
GATE_INPUTS: list[tuple[str, str, str]] = [
    (
        "scripts/complexity_baseline.json",
        "py",
        "the C901 ratchet's only non-.py input; test_check_complexity.py is what "
        "proves the baseline still matches the tree",
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
        "docs/guides/user/CLI_REFERENCE.md",
        "py",
        "test_docs_command_coverage.py drift-guards it against the live Typer app; "
        "its .md only selects docs-quality, which runs no pytest",
    ),
    (
        "README.md",
        "py",
        "test_readme_demo_docs.py drift-guards the root demo documentation",
    ),
    (
        "packages/luxar/src/luxar/demos/README.md",
        "py",
        "test_demo_import_spelling.py validates its shared-helper inventory",
    ),
    (
        "pyproject.toml",
        "py",
        "hatch envs, ruff/mypy targets and the import-linter contracts",
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
]

#: Paths that belong to no LANGUAGE domain (they do select the docs gate, which is
#: a separate axis — ``docs/index.rst`` matches ``docs_pattern`` by design). Without
#: these the table above proves nothing: a pattern that matched everything would
#: satisfy every positive row.
NON_DOMAIN_PATHS: list[str] = ["CHANGELOG.md", "docs/index.rst"]

#: The docs gate's own negative control: real tracked files that must NOT set
#: ``docs_relevant``. Kept separate from ``NON_DOMAIN_PATHS`` because the two
#: controls test opposite gates — these DO belong to a language domain.
NON_DOCS_PATHS: list[str] = [
    "packages/luxar-launcher/go.mod",
    "packages/luxar-viewer/src/wasm/rust/src/lib.rs",
]

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
    spelling a maintainer edits is the thing under test.
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
def _require_grep() -> None:
    """A real ``grep`` must be on PATH; Python's ``re`` is not a substitute.

    Only the presence of an executable named ``grep`` is checked — which flavour
    it is (GNU here and in CI, BSD on macOS) is not, and does not need to be: what
    matters is that the verdict comes from a POSIX ``grep -E`` rather than from
    ``re``, whose ERE dialect differs in escapes, intervals and backreferences.
    """
    assert shutil.which("grep"), (
        "a POSIX `grep -E` is required to evaluate the CI patterns the way the "
        "workflow does; Python's `re` is a different dialect"
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
