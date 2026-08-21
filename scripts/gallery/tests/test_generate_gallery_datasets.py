"""Unit tests for scripts/gallery/generate_gallery_datasets.py.

Collected by the default suite: ``scripts/gallery/tests`` is listed both on
pytest's ``testpaths`` AND on the explicit path arguments of the ``test`` /
``test-cov`` / ``test-cov-all`` hatch scripts — the latter matters, because an
explicit path argument OVERRIDES ``testpaths``. Run in isolation with:
    hatch run pytest scripts/gallery/tests/test_generate_gallery_datasets.py -q

What is guarded is the *demote-on-failure* rule. A demo whose ``DEMO_META``
declares machine-local ``local_data`` (``manual-file`` / ``kaggle-auth`` /
``git-lfs``) exits 1 wherever that input is absent, which used to make the whole
gallery build report a hard failure and return 1 — on a fresh clone that meant
``make generate-gallery`` aborted. Such an entry is now still RUN (so the machine
that has the input keeps regenerating its tile, ``--force`` included) and only
demoted to the soft ``manual-data`` bucket if it actually fails. The scoping is
the delicate part: a ``timeout`` / ``no-output`` / death by signal stays hard even
for a demoted mode.

Behavioral cases run against a synthetic manifest + synthetic demo files; one
separate invariant test reads the real manifest to keep ``script: null`` honest.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_gallery_datasets as gen  # noqa: E402

_DEMO_TEMPLATE = '''"""Synthetic demo used by the gallery-harness tests."""

DEMO_META = {{
    "key": "{key}",
    "title": "Synthetic {key}",
    "description": "Synthetic demo for the gallery harness tests.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {{
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": {local_data},
    }},
    "caches": [],
    "outputs": ["{key}"],
}}
'''


def _write_demo(demos_dir: Path, key: str, local_data: Optional[str]) -> str:
    """Write a synthetic ``demo_<key>.py``; return its filename."""
    name = f"demo_{key}.py"
    (demos_dir / name).write_text(
        _DEMO_TEMPLATE.format(
            key=key,
            local_data="None" if local_data is None else f'"{local_data}"',
        )
    )
    return name


class _Recorder(list):
    """The spawned scripts, in order — with every full invocation kept beside them.

    A list subclass so the readable ``calls == ["demo_x.py"]`` assertions stay as
    they are, while ``calls.invocations`` can pin the child *command* too: a fake
    that only looks at ``cmd[1]`` is blind to a dropped ``--no-serve`` (every
    gallery demo would then start a web server and hang the harness), a dropped
    ``cwd`` / ``timeout`` / ``capture_output`` / ``text``, and so on.
    """

    def __init__(self) -> None:
        super().__init__()
        self.invocations: list[tuple[tuple[str, ...], dict[str, Any]]] = []


def _fake_run(repo: Path, calls: _Recorder, outcomes: dict[str, str]):
    """A ``subprocess.run`` that records the call and plays a scripted outcome.

    ``fail`` = the real "input not on this machine" shape (exit 1 with the demo's
    own message on stderr); ``ok`` writes the dataset the harness then looks for;
    ``silent`` exits 0 writing nothing (the ``no-output`` bug shape); ``signal``
    is a child killed by a signal (POSIX: a NEGATIVE return code); ``timeout``
    raises like the real call does.

    Faithful to ``check=``: with it set, the real ``subprocess.run`` RAISES on a
    non-zero exit instead of returning it, which would destroy demote-on-failure
    (and turn the first missing input into a traceback) — a fake that ignored the
    kwarg would grade that as fine.
    """

    def run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        script = Path(cmd[1]).name
        calls.append(script)
        calls.invocations.append((tuple(cmd), dict(kwargs)))
        outcome = outcomes[script]
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1)
        if outcome == "fail":
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(1, cmd)
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Gaia star catalog not found.\nPlace it in the cache by hand.",
            )
        if outcome == "signal":
            # SIGKILL, as `subprocess` reports it on POSIX.
            return SimpleNamespace(
                returncode=-9,
                stdout="",
                stderr="Fitting 3M splats...",
            )
        if outcome == "ok":
            key = script.removeprefix("demo_").removesuffix(".py")
            (repo / f"datasets/demos/{key}.luxar.zarr").mkdir(
                parents=True, exist_ok=True
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: list[tuple[str, Optional[str], str]],
    *,
    present: tuple[str, ...] = (),
    missing_script: tuple[str, ...] = (),
    capture_only: tuple[str, ...] = (),
) -> _Recorder:
    """Point the harness at a synthetic tree; return the recorded-calls list.

    ``specs`` is ``(id, local_data, outcome)`` per manifest entry; ``present``
    names ids whose dataset already exists on disk; ``missing_script`` names ids
    whose demo file is deliberately never written (the ``missing-script`` shape);
    ``capture_only`` names ids whose manifest entry declares ``script: null`` (the
    feature-branch shape — three real entries look like this).
    """
    demos_dir = tmp_path / "demos"
    demos_dir.mkdir()
    monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gen, "DEMOS_DIR", demos_dir)

    manifest: list[dict[str, Any]] = []
    outcomes: dict[str, str] = {}
    for demo_id, local_data, outcome in specs:
        script: Optional[str]
        if demo_id in capture_only:
            script = None
        elif demo_id in missing_script:
            script = f"demo_{demo_id}.py"  # never written to disk
        else:
            script = _write_demo(demos_dir, demo_id, local_data)
        if script is not None:
            outcomes[script] = outcome
        manifest.append(
            {
                "id": demo_id,
                "script": script,
                "dataset": f"datasets/demos/{demo_id}.luxar.zarr",
            }
        )
    for demo_id in present:
        (tmp_path / f"datasets/demos/{demo_id}.luxar.zarr").mkdir(parents=True)

    monkeypatch.setattr(gen, "load_manifest", lambda: manifest)
    calls = _Recorder()
    monkeypatch.setattr(gen.subprocess, "run", _fake_run(tmp_path, calls, outcomes))
    return calls


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["generate_gallery_datasets", *argv])
    return gen.main()


@pytest.mark.parametrize("mode", ["manual-file", "kaggle-auth", "git-lfs"])
def test_a_failing_local_input_demo_is_demoted_not_failed(
    tmp_path, monkeypatch, capsys, mode
) -> None:
    # Each mode can legitimately be absent on the machine running the gallery:
    # hand-placed files, Kaggle credentials, and Git LFS payloads are all
    # provisioned outside the demo itself.
    calls = _setup(tmp_path, monkeypatch, [("needs_input", mode, "fail")])

    code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert code == 0, "a missing machine-local input must not red the whole build"
    assert "manual-data" in out
    assert "failed" not in out  # the summary lists only non-empty buckets
    # Demoted, not pre-skipped: the demo really ran…
    assert calls == ["demo_needs_input.py"]
    # …and its own error tail is still on screen, so a real bug stays visible.
    assert "Gaia star catalog not found." in out
    # The advice names the mode that was actually declared: telling a Kaggle
    # failure to hand-place a file would send the reader down the wrong path.
    assert mode in out


def test_the_child_command_contract(tmp_path, monkeypatch) -> None:
    """What the harness actually spawns — the fake sees only what it records.

    ``--no-serve`` is load-bearing (without it every gallery demo starts a viewer
    server and the harness hangs until the timeout), and so are the run kwargs:
    ``cwd`` decides where relative dataset paths land, ``timeout`` is what makes
    the ``TimeoutExpired`` handler reachable at all, and ``capture_output``/``text``
    are what let the error tail be printed as text.
    """
    calls = _setup(tmp_path, monkeypatch, [("plain", None, "ok")])

    _run_main(monkeypatch)

    (cmd, kwargs) = calls.invocations[0]
    assert cmd[0] == sys.executable
    assert Path(cmd[1]).name == "demo_plain.py"
    assert list(cmd[2:]) == ["--no-serve"]
    # The exact kwarg SET, not just presence: `check=True` would raise on the
    # first failing demo and take demote-on-failure with it.
    assert set(kwargs) == {"cwd", "timeout", "capture_output", "text"}
    assert kwargs["cwd"] == gen.REPO_ROOT
    assert kwargs["timeout"] == gen.GEN_TIMEOUT_S
    # …and a floor, so a retuned constant cannot make the timeout meaningless.
    assert gen.GEN_TIMEOUT_S >= 600
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_a_local_input_demo_that_succeeds_is_generated(
    tmp_path, monkeypatch, capsys
) -> None:
    # The maintainer's machine DOES have the inputs, and two of the five
    # manual-file manifest entries (`gsplats_4d_neuromast_2ch`,
    # `human_multiome_peak_umap`) have committed media proving they build there.
    # Demotion must never cost them their tile.
    calls = _setup(tmp_path, monkeypatch, [("needs_input", "manual-file", "ok")])

    code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert code == 0
    assert calls == ["demo_needs_input.py"]
    assert "generated" in out
    assert "manual-data" not in out


@pytest.mark.parametrize(
    "outcome,bucket", [("timeout", "timeout"), ("silent", "no-output")]
)
@pytest.mark.parametrize("mode", ["manual-file", "kaggle-auth", "git-lfs"])
def test_timeout_and_no_output_stay_hard_for_a_local_input_demo(
    tmp_path, monkeypatch, capsys, outcome, bucket, mode
) -> None:
    # Demotion is scoped to a non-zero exit. A demo that hangs, or that exits 0
    # having written nothing, is a bug in the demo — not a missing input.
    _setup(tmp_path, monkeypatch, [("needs_input", mode, outcome)])

    code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert code == 1
    assert bucket in out
    assert "manual-data" not in out


@pytest.mark.parametrize("mode", ["manual-file", "kaggle-auth", "git-lfs"])
def test_a_signal_killed_local_input_demo_stays_hard(
    tmp_path, monkeypatch, capsys, mode
) -> None:
    # A negative return code is POSIX for "killed by a signal" — the OOM killer
    # on the ~30 GB Kaggle download, a segfault in a native dependency. That is
    # not how a demo reports a missing input (it exits 1 with a message), and it
    # only happens to a run that got far enough to allocate, i.e. one that HAD
    # its input. Demoting it would report a real failure as "this machine
    # appears not to have the input" and return 0.
    calls = _setup(tmp_path, monkeypatch, [("needs_input", mode, "signal")])

    code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert calls == ["demo_needs_input.py"]
    assert code == 1
    assert "failed" in out
    assert "manual-data" not in out
    assert "appears not to have" not in out


def test_capture_only_entries_have_no_runnable_demo_on_disk() -> None:
    """``script: null`` must mean the demo is genuinely absent from this branch."""
    from luxar.demos.registry import extract_demo_meta

    on_disk = {
        extract_demo_meta(path)["key"]: path.name
        for path in gen.DEMOS_DIR.glob("demo_*.py")
    }
    for entry in gen.load_manifest():
        if entry.get("script") is None:
            assert entry["id"] not in on_disk, (
                f"{entry['id']}: manifest says capture-only, but "
                f"{on_disk[entry['id']]} is on disk and can generate the dataset"
            )


def test_a_present_local_input_entry_reports_already_present(
    tmp_path, monkeypatch, capsys
) -> None:
    calls = _setup(
        tmp_path,
        monkeypatch,
        [("needs_input", "manual-file", "fail")],
        present=("needs_input",),
    )

    code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert code == 0
    assert not calls  # idempotence still wins over everything else
    assert "already-present" in out
    assert "manual-data" not in out


def test_force_regenerates_a_local_input_entry(tmp_path, monkeypatch, capsys) -> None:
    # --force used to be defeated outright by the pre-skip: it reported
    # manual-data and never spawned the demo.
    calls = _setup(
        tmp_path,
        monkeypatch,
        [("needs_input", "manual-file", "ok")],
        present=("needs_input",),
    )

    code = _run_main(monkeypatch, "--force")
    out = capsys.readouterr().out

    assert code == 0
    assert calls == ["demo_needs_input.py"]
    assert "generated" in out


def test_force_on_a_failing_local_input_entry_is_still_soft(
    tmp_path, monkeypatch, capsys
) -> None:
    # The fresh-clone shape with --force: the dataset is stale/absent and the
    # input is not here either. --force must not turn the soft bucket hard.
    calls = _setup(
        tmp_path,
        monkeypatch,
        [("needs_input", "manual-file", "fail")],
        present=("needs_input",),
    )

    code = _run_main(monkeypatch, "--force")
    out = capsys.readouterr().out

    assert calls == ["demo_needs_input.py"]
    assert code == 0
    assert "manual-data" in out


def test_only_on_a_failing_local_input_entry_is_still_soft(
    tmp_path, monkeypatch, capsys
) -> None:
    # Same for a targeted run: `--only <id>` is how a maintainer regenerates one
    # tile, and it must not report a hard failure where a full run would not.
    calls = _setup(
        tmp_path,
        monkeypatch,
        [("needs_input", "manual-file", "fail"), ("plain", None, "ok")],
    )

    code = _run_main(monkeypatch, "--only", "needs_input")
    out = capsys.readouterr().out

    assert calls == ["demo_needs_input.py"]
    assert code == 0
    assert "manual-data" in out


def test_a_capture_only_entry_is_reported_and_is_not_a_hard_failure(
    tmp_path, monkeypatch, capsys
) -> None:
    # `script: null` describes three real manifest entries (feature-branch demos
    # whose dataset is kept locally). Nothing to spawn, and never a failure —
    # folding this bucket into the hard list would red every machine forever.
    calls = _setup(
        tmp_path,
        monkeypatch,
        [("branch_demo", None, "ok")],
        capture_only=("branch_demo",),
    )

    code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert not calls
    assert "capture-only" in out
    assert code == 0


def test_each_entry_is_generated_on_its_own(tmp_path, monkeypatch, capsys) -> None:
    """Two runnable entries in one unfiltered run, each judged on its own result.

    Every other spawning case has exactly one runnable entry, which cannot tell a
    per-entry loop from one that keeps re-running the first row.
    """
    calls = _setup(
        tmp_path,
        monkeypatch,
        [("plain", None, "ok"), ("needs_input", "manual-file", "fail")],
    )

    code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert calls == ["demo_plain.py", "demo_needs_input.py"]
    assert code == 0
    # Each id in its OWN bucket line of the summary.
    generated = next(line for line in out.splitlines() if "generated:" in line)
    demoted = next(line for line in out.splitlines() if "manual-data:" in line)
    assert "plain" in generated and "needs_input" not in generated
    assert "needs_input" in demoted and " plain" not in demoted


def test_list_reports_which_datasets_are_present(tmp_path, monkeypatch, capsys) -> None:
    # The presence column is the whole point of --list; inverted, it would send
    # the reader to regenerate what is already there (and vice versa).
    _setup(
        tmp_path,
        monkeypatch,
        [("here", None, "ok"), ("absent", None, "ok")],
        present=("here",),
    )

    _run_main(monkeypatch, "--list")
    out = capsys.readouterr().out

    here = next(line for line in out.splitlines() if "here" in line)
    absent = next(line for line in out.splitlines() if "absent" in line)
    assert "ready" in here and "MISSING" not in here
    assert "MISSING" in absent and "ready" not in absent


def test_a_manifest_entry_whose_script_is_gone_is_reported_but_is_not_hard(
    tmp_path, monkeypatch, capsys
) -> None:
    # `missing-script` is a manifest/tree mismatch — always a repo bug, never a
    # machine's missing input, so it stays out of the soft bucket and out of the
    # hard-failure list it was never on.
    calls = _setup(
        tmp_path,
        monkeypatch,
        [("gone", "manual-file", "ok")],
        missing_script=("gone",),
    )

    code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert not calls  # nothing to spawn
    assert "missing-script" in out
    assert "manual-data" not in out
    assert code == 0, "missing-script has never been a hard failure; don't widen it"


def test_only_still_warns_about_an_unknown_id(tmp_path, monkeypatch, capsys) -> None:
    calls = _setup(tmp_path, monkeypatch, [("plain", None, "ok")])

    code = _run_main(monkeypatch, "--only", "plain,nope")
    out = capsys.readouterr().out

    assert code == 0
    assert calls == ["demo_plain.py"]
    assert "Unknown demo ids ignored" in out
    assert "nope" in out


@pytest.mark.parametrize("mode", gen.LOCAL_INPUT_MODES)
def test_list_marks_local_input_entries_and_generates_nothing(
    tmp_path, monkeypatch, capsys, mode
) -> None:
    # Parametrized over BOTH modes: a mark narrowed to manual-file would silently
    # leave the two real kaggle-auth entries unannotated.
    calls = _setup(
        tmp_path,
        monkeypatch,
        [("needs_input", mode, "ok"), ("plain", None, "ok")],
    )

    code = _run_main(monkeypatch, "--list")
    out = capsys.readouterr().out

    assert code == 0
    assert not calls
    marked = next(line for line in out.splitlines() if "needs_input" in line)
    plain = next(line for line in out.splitlines() if "demo_plain.py" in line)
    assert mode in marked and "manual-data" in marked
    assert mode not in plain
    # The old opt-in wording must be gone: naming it is no longer required.
    assert "--only" not in out


def test_a_broken_demo_meta_entry_still_fails_hard(
    tmp_path, monkeypatch, capsys
) -> None:
    # A DemoMetaError must neither crash the harness nor excuse the entry: it is
    # not a local-input demo, so it keeps the hard `failed` verdict.
    calls = _setup(tmp_path, monkeypatch, [("broken", None, "fail")])
    (tmp_path / "demos" / "demo_broken.py").write_text("DEMO_META = 'not a dict'\n")

    code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert calls == ["demo_broken.py"]
    assert code == 1
    assert "failed" in out
    assert "manual-data" not in out


def test_needs_local_input_ignores_entries_with_no_runnable_script(
    tmp_path, monkeypatch
) -> None:
    _setup(tmp_path, monkeypatch, [])

    # capture-only (script: null) and a manifest pointing at a deleted demo.
    assert gen.needs_local_input({"id": "cap", "script": None}) is None
    assert gen.needs_local_input({"id": "gone", "script": "demo_absent.py"}) is None


class TestUnbuildableEntries:
    """``UNBUILDABLE_IDS`` — soft-skipped WITHOUT being spawned.

    A demo that cannot be built on any machine (today: the Visible Human head,
    whose shipped sidecar is misordered against its fit, #1670) would otherwise
    be RUN like any other: a ~1.1 GB download plus a 4M-splat fit that exceeds
    ``GEN_TIMEOUT_S``, landing in the hard ``timeout`` bucket, returning 1 and
    aborting ``make generate-gallery`` before a single tile is captured.
    """

    def test_an_unbuildable_entry_is_skipped_without_running(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        calls = _setup(
            tmp_path,
            monkeypatch,
            [("cannot_build", None, "timeout"), ("plain", None, "ok")],
        )
        monkeypatch.setattr(gen, "UNBUILDABLE_IDS", {"cannot_build": "because #1670"})

        code = _run_main(monkeypatch)
        out = capsys.readouterr().out

        # Never spawned — its scripted outcome is `timeout`, which would be hard.
        assert calls == ["demo_plain.py"]
        assert code == 0
        assert "unbuildable" in out
        assert "because #1670" in out
        assert "timeout" not in out
        # …and its neighbour is unaffected.
        assert "generated" in out

    def test_force_does_not_run_an_unbuildable_entry(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        # --force is the route that would otherwise re-run it on every machine,
        # even one whose tile is already captured and committed.
        calls = _setup(
            tmp_path,
            monkeypatch,
            [("cannot_build", None, "timeout")],
            present=("cannot_build",),
        )
        monkeypatch.setattr(gen, "UNBUILDABLE_IDS", {"cannot_build": "because #1670"})

        code = _run_main(monkeypatch, "--force")
        out = capsys.readouterr().out

        assert not calls
        assert code == 0
        assert "unbuildable" in out

    def test_list_marks_an_unbuildable_entry(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        _setup(tmp_path, monkeypatch, [("cannot_build", None, "ok")])
        monkeypatch.setattr(gen, "UNBUILDABLE_IDS", {"cannot_build": "because #1670"})

        _run_main(monkeypatch, "--list")
        out = capsys.readouterr().out

        marked = next(line for line in out.splitlines() if "cannot_build" in line)
        assert "unbuildable" in marked

    def test_every_unbuildable_id_is_a_real_manifest_id(self) -> None:
        """A typo'd id would silently do nothing at all.

        Read from the REAL manifest (not the synthetic one the other tests use):
        this list names specific entries, so it must stay coupled to them.
        """
        ids = {d["id"] for d in gen.load_manifest()}
        assert set(gen.UNBUILDABLE_IDS) <= ids, sorted(set(gen.UNBUILDABLE_IDS) - ids)

    def test_each_reason_cites_its_issue(self) -> None:
        """Every entry must say WHY and be deletable on that basis."""
        for demo_id, reason in gen.UNBUILDABLE_IDS.items():
            assert "#" in reason, f"{demo_id}: no issue cited in {reason!r}"
            assert "elete" in reason, f"{demo_id}: no removal condition in {reason!r}"


def test_the_luxar_import_stays_deferred() -> None:
    """No module-level ``luxar`` import: the script is stdlib + arbol at import.

    `luxar.demos.registry` is import-light by design, but this script has to run
    on a partial environment — a module-level import would traceback before
    argparse could even print ``--help``.
    """
    import ast

    tree = ast.parse(Path(gen.__file__).read_text(encoding="utf-8"))
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [
        alias.name
        for node in top_level
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [node.module or "" for node in top_level if isinstance(node, ast.ImportFrom)]
    assert not any(n.split(".")[0] == "luxar" for n in names), names
