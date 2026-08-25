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

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

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
    """Obsidian reserves a TypeScript slot without delaying required checks."""
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
        "python-tests must leave one of obsidian's three slots for TypeScript"
    )

    matrix_expression = jobs["python-tests"]["strategy"]["matrix"]["python-version"]
    matrix_lists = re.findall(r"fromJSON\('([^']+)'\)", matrix_expression)
    assert matrix_lists, "python-tests must declare its event-specific version matrices"
    largest_matrix_size = max(len(json.loads(matrix)) for matrix in matrix_lists)
    assert hosted_cap >= largest_matrix_size, (
        "the hosted max-parallel branch must not throttle the off-PR Python matrix"
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
    assert pick_runner["permissions"] == {"actions": "read"}, (
        "pick-runner needs only actions:read to inspect repository run activity"
    )
    assert pick_runner["steps"][0]["env"]["GH_TOKEN"] == "${{ github.token }}", (
        "pick-runner must authenticate gh api with the workflow token"
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


def _run_pick_runner(
    workflow: str,
    tmp_path: Path,
    *,
    head_repo: str = "royerlab/luxar",
    heartbeat: str = "0",
    force_hosted: str = "0",
    other_run_active: bool = False,
    api_error: str = "",
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the real inline router against deterministic repository activity."""
    router = yaml.safe_load(workflow)["jobs"]["pick-runner"]["steps"][0]["run"]
    output_path = tmp_path / "github-output"

    fake_gh = tmp_path / "gh"
    fake_gh.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

endpoint = next((arg for arg in sys.argv if "/actions/" in arg), "")
if os.environ["ROUTER_API_ERROR"] == "runs" and "/actions/runs?" in endpoint:
    raise SystemExit(1)
if os.environ["ROUTER_API_ERROR"] == "jobs" and "/runs/" in endpoint and "/jobs?" in endpoint:
    raise SystemExit(1)
if "/actions/runs?" in endpoint:
    run_ids = [2038, 9999] if os.environ["ROUTER_OTHER_ACTIVE"] == "1" else [2038]
    print(json.dumps({"workflow_runs": [{"id": run_id} for run_id in run_ids]}))
elif "/runs/9999/jobs?" in endpoint:
    print(json.dumps({"jobs": [{"status": "in_progress", "labels": ["obsidian"]}]}))
else:
    print(json.dumps({"jobs": []}))
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    fake_date = tmp_path / "date"
    fake_date.write_text("#!/bin/sh\necho 1000\n", encoding="utf-8")
    fake_date.chmod(0o755)

    env = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(output_path),
        "GITHUB_REPOSITORY": "royerlab/luxar",
        "HEAD_REPO": head_repo,
        "FORCE_HOSTED": force_hosted,
        "HEARTBEAT": heartbeat,
        "ROUTER_API_ERROR": api_error,
        "ROUTER_OTHER_ACTIVE": "1" if other_run_active else "0",
    }
    result = subprocess.run(
        ["bash", "-e", "-c", router],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=env,
    )
    label = ""
    if output_path.exists():
        match = re.search(r"^label=(.+)$", output_path.read_text(), re.MULTILINE)
        if match:
            label = match.group(1)
    return result, label


@pytest.mark.parametrize(
    ("heartbeat", "expected"),
    [
        ("950", "obsidian"),
        ("0", "ubuntu-latest"),
        ("700", "ubuntu-latest"),
        ("1031", "ubuntu-latest"),
        ("not-a-number", "ubuntu-latest"),
    ],
)
def test_pick_runner_routes_same_repo_on_fresh_capacity_heartbeat(
    workflow: str,
    tmp_path: Path,
    heartbeat: str,
    expected: str,
) -> None:
    """Fresh capacity routes obsidian; unavailable, stale, or future does not."""
    result, label = _run_pick_runner(workflow, tmp_path, heartbeat=heartbeat)

    assert result.returncode == 0, result.stdout + result.stderr
    assert label == expected
    assert "integer expression expected" not in result.stderr


@pytest.mark.parametrize("api_error", ["runs", "jobs"])
def test_pick_runner_fails_api_read_toward_obsidian(
    workflow: str, tmp_path: Path, api_error: str
) -> None:
    """A transient GitHub API failure must not restart the paid overflow."""
    result, label = _run_pick_runner(
        workflow, tmp_path, other_run_active=True, api_error=api_error
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert label == "obsidian"


def test_pick_runner_routes_busy_box_to_obsidian(workflow: str, tmp_path: Path) -> None:
    """An active obsidian job proves a zero-capacity box is live and busy."""
    result, label = _run_pick_runner(
        workflow, tmp_path, heartbeat="0", other_run_active=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert label == "obsidian"


@pytest.mark.parametrize(
    ("head_repo", "force_hosted"),
    [("someone/fork", "0"), ("royerlab/luxar", "1")],
)
def test_pick_runner_hosted_overrides_bypass_heartbeat(
    workflow: str,
    tmp_path: Path,
    head_repo: str,
    force_hosted: str,
) -> None:
    """Fork safety and the operator kill switch always select hosted runners."""
    result, label = _run_pick_runner(
        workflow,
        tmp_path,
        head_repo=head_repo,
        api_error="runs",
        force_hosted=force_hosted,
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
    run_api_error: bool = False,
    run_api_error_once: bool = False,
    job_api_error: bool = False,
    own_job_api_error_call: int = 0,
    other_job_api_error_call: int = 0,
) -> tuple[subprocess.CompletedProcess[str], int, bool]:
    """Run the real inline watchdog against deterministic GitHub API snapshots."""
    watchdog = yaml.safe_load(workflow)["jobs"]["queue-watchdog"]["steps"][0]["run"]
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
    counter = Path(os.environ["WATCHDOG_RUN_COUNTER"])
    call = int(counter.read_text() or "0") if counter.exists() else 0
    counter.write_text(str(call + 1))
    error_mode = os.environ["WATCHDOG_RUN_API_ERROR"]
    if error_mode == "1" or (error_mode == "once" and call == 1):
        raise SystemExit(1)
    active_snapshots = json.loads(Path(os.environ["WATCHDOG_OTHER_ACTIVE"]).read_text())
    active = active_snapshots[min(call, len(active_snapshots) - 1)]
    run_ids = [2038, 9999] if active else [2038]
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

    env = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "royerlab/luxar",
        "GITHUB_RUN_ID": "2038",
        "WATCHDOG_SNAPSHOTS": str(snapshots_path),
        "WATCHDOG_COUNTER": str(counter_path),
        "WATCHDOG_RUN_COUNTER": str(run_counter_path),
        "WATCHDOG_OTHER_JOB_COUNTER": str(other_job_counter_path),
        "WATCHDOG_OTHER_ACTIVE": str(other_active_path),
        "WATCHDOG_RUN_API_ERROR": (
            "once" if run_api_error_once else "1" if run_api_error else "0"
        ),
        "WATCHDOG_JOB_API_ERROR": "1" if job_api_error else "0",
        "WATCHDOG_OWN_JOB_API_ERROR_CALL": str(own_job_api_error_call),
        "WATCHDOG_OTHER_JOB_API_ERROR_CALL": str(other_job_api_error_call),
        "WATCHDOG_DATE_COUNTER": str(date_counter_path),
        "WATCHDOG_DATE_STEP": str(date_step),
        "WATCHDOG_CANCELLED": str(cancel_path),
    }
    result = subprocess.run(
        ["bash", "-e", "-c", watchdog],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
        env=env,
    )
    calls = (
        int(counter_path.read_text(encoding="utf-8") or "0")
        if counter_path.exists()
        else 0
    )
    return result, calls, cancel_path.exists()


def _obsidian_job(name: str, status: str) -> dict[str, object]:
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
