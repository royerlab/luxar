from __future__ import annotations

import dataclasses
import email.message
import http.client
import importlib.util
import io
import json
import sys
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import pytest

SCRIPT = Path(__file__).parents[1] / "check_cadence_liveness.py"
SPEC = importlib.util.spec_from_file_location("check_cadence_liveness", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

CADENCE = module.CADENCES[0]
# Before CADENCE.not_before (2026-09-14), so an absent workflow or a missing
# success is still a notice unless a test moves the clock forward explicitly.
NOW = datetime(2026, 9, 11, 20, tzinfo=UTC)
AFTER_NOT_BEFORE = CADENCE.not_before + timedelta(days=2)

# Literals, not references to the module being tested: comparing the module
# against itself would let a 1-second timeout or a page size of 1 pass.
WORKFLOWS_URL = "https://api.github.com/repos/royerlab/luxar/actions/workflows"
RUNS_URL = f"{WORKFLOWS_URL}/42/runs"
EXPECTED_TIMEOUT = 30
EXPECTED_RETRY_DELAY = 2.0
EXPECTED_RUNS_QUERY = {
    "branch": ["dev"],
    "event": ["workflow_dispatch"],
    "status": ["success"],
    "per_page": ["5"],
}


def test_the_constants_are_the_values_the_fixtures_assert() -> None:
    assert module.API_ROOT == "https://api.github.com"
    assert module.DEFAULT_REPOSITORY == "royerlab/luxar"
    assert module.REQUEST_TIMEOUT_SECONDS == EXPECTED_TIMEOUT
    assert module.RETRY_DELAY_SECONDS == EXPECTED_RETRY_DELAY
    assert module.WORKFLOW_PAGE_SIZE == 100
    assert module.RUN_PAGE_SIZE == 5
    assert module.MAX_WORKFLOW_PAGES == 5
    assert module.MAX_CLOCK_SKEW == timedelta(minutes=5)
    assert CADENCE.max_age == timedelta(days=3)


def _response(payload: object) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode())


def _workflow(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": 42,
        "name": CADENCE.workflow_name,
        "path": CADENCE.workflow_path,
        "state": "active",
        "created_at": "2026-09-12T01:17:05.000-07:00",
    }
    entry.update(overrides)
    return entry


def _listing(*workflows: dict[str, object], total_count: int | None = None) -> dict:
    return {
        "total_count": len(workflows) if total_count is None else total_count,
        "workflows": list(workflows),
    }


def _run(
    conclusion: str = "success",
    created_at: str = "2026-09-10T03:00:00Z",
    **overrides: object,
) -> dict[str, object]:
    """A run record whose three timestamps are deliberately distinct.

    `created_at` is the only one the watchdog may read. The other two default to
    much fresher values (a re-run bumps `updated_at`), so a test that expects
    STALE fails if either of them is used instead.
    """
    entry: dict[str, object] = {
        "conclusion": conclusion,
        "created_at": created_at,
        "run_started_at": "2026-09-11T12:00:00Z",
        "updated_at": "2026-09-11T19:00:00Z",
    }
    entry.update(overrides)
    return entry


def _opener(workflow_pages: list[object], runs: object) -> Callable[..., Any]:
    """Serve the paginated workflow listing, then the filtered runs query.

    Both queries are asserted as exact `parse_qs` dicts against literals, so a
    dropped filter, a changed value, and an extra parameter all fail -- unlike a
    substring check, which `branch=development` and `per_page=100` satisfy.
    """

    def open_request(request: Any, timeout: int) -> io.BytesIO:
        assert timeout == EXPECTED_TIMEOUT
        assert request.headers["User-agent"] == "luxar-cadence-watchdog"
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.headers["Accept"] == "application/vnd.github+json"
        split = urlsplit(request.full_url)
        base = f"{split.scheme}://{split.netloc}{split.path}"
        query = parse_qs(split.query)
        if base == WORKFLOWS_URL:
            page = int(query.get("page", ["0"])[0])
            assert query == {"per_page": ["100"], "page": [str(page)]}
            assert 1 <= page <= len(workflow_pages), request.full_url
            return _response(workflow_pages[page - 1])
        if base == RUNS_URL:
            assert query == EXPECTED_RUNS_QUERY
            return _response(runs)
        raise AssertionError(request.full_url)

    return open_request


def _no_sleep(seconds: float) -> None:
    raise AssertionError(f"the watchdog slept {seconds}s when it should not have")


def _check(
    *,
    workflow_pages: list[object],
    runs: object = None,
    now: datetime = NOW,
    repo_root: Path | None = None,
) -> Any:
    return module.check_cadence(
        CADENCE,
        repository="royerlab/luxar",
        now=now,
        token="test-token",
        opener=_opener(workflow_pages, {} if runs is None else runs),
        repo_root=module.REPO_ROOT if repo_root is None else repo_root,
        sleep=_no_sleep,
    )


def _check_all(
    *,
    opener: Callable[..., Any],
    now: datetime = NOW,
    repo_root: Path | None = None,
    sleep: Callable[[float], None] = _no_sleep,
) -> list[Any]:
    return module.check_all(
        repository="royerlab/luxar",
        now=now,
        token="test-token",
        opener=opener,
        repo_root=module.REPO_ROOT if repo_root is None else repo_root,
        sleep=sleep,
    )


def _capturing_checker(
    captured: dict[str, Any], level: str = "OK"
) -> Callable[..., list[Any]]:
    def checker(**kwargs: Any) -> list[Any]:
        captured.update(kwargs)
        return [module.Result(CADENCE, level, "detail")]

    return checker


# --------------------------------------------------------------------------
# The table has to describe files that actually exist, on a bounded snooze.
# --------------------------------------------------------------------------


def test_every_cadence_path_names_a_file_in_this_checkout() -> None:
    for cadence in module.CADENCES:
        assert (module.REPO_ROOT / cadence.workflow_path).is_file(), (
            f"{cadence.workflow_path} does not exist; the presence probe would "
            "resolve False forever and its wording would never be produced"
        )


def test_cadence_path_matches_the_workflow_display_name_on_disk() -> None:
    text = (module.REPO_ROOT / CADENCE.workflow_path).read_text()
    assert text.splitlines()[0] == f"name: {CADENCE.workflow_name}"


def test_not_before_is_not_an_unbounded_snooze() -> None:
    """A stalled promotion is a red no code change can fix.

    Bumping `not_before` is therefore the path of least resistance, so the
    horizon is bounded here. Measured: the file reached `dev` on 2026-09-12 and
    `promote-ff` carried it to `main` at 2026-09-12T08:17:05Z the same day, with
    the out-of-repo dispatcher enabled for 02:36 PDT daily. Anything past two
    daily cycles from that is a blind spot over exactly the failure this script
    exists to catch, and has to be argued for in the diff that widens this
    bound.
    """
    assert datetime(2026, 9, 12, tzinfo=UTC) <= CADENCE.not_before
    assert CADENCE.not_before <= datetime(2026, 9, 15, tzinfo=UTC)


# --------------------------------------------------------------------------
# Only declared levels can be emitted, and the guard can itself fail.
# --------------------------------------------------------------------------


def test_the_module_emits_only_declared_levels() -> None:
    levels, stray_result_calls = module.declared_result_levels()

    assert stray_result_calls == 0, "a Result was built outside the _result factory"
    assert levels == {"OK", "NOTICE", "UNKNOWN", "STALE", "CONFIG"}
    assert levels == module.LEVELS
    assert module.RED_LEVELS < module.LEVELS


@pytest.mark.parametrize(
    ("scope", "source"),
    [
        ("module level", "x = Result(c, 'OK', 'd')\n"),
        ("class body", "class A:\n    y = Result(c, 'OK', 'd')\n"),
        ("async def", "async def f():\n    return Result(c, 'OK', 'd')\n"),
        ("lambda", "f = lambda c: Result(c, 'OK', 'd')\n"),
        ("comprehension", "xs = [Result(c, 'OK', 'd') for c in cs]\n"),
        ("plain function", "def f(c):\n    return Result(c, 'OK', 'd')\n"),
        (
            "nested function",
            "def o():\n    def i(c):\n        return Result(c, 'x', 'd')\n",
        ),
    ],
)
def test_a_stray_result_is_counted_in_every_scope(scope: str, source: str) -> None:
    """The guard must be able to fail; `if False:` here would pass otherwise."""
    factory = (
        "def _result(cadence, level, detail):\n    return Result(c, level, detail)\n"
    )

    levels, strays = module.declared_result_levels(source=factory + source)

    assert strays == 1, scope
    assert levels == set()


def test_the_stray_guard_does_not_flag_the_factorys_own_call() -> None:
    factory = (
        "def _result(cadence, level, detail):\n    return Result(c, level, detail)\n"
    )

    levels, strays = module.declared_result_levels(
        source=factory + "def f(c):\n    return _result(c, 'STALE', 'd')\n"
    )

    assert strays == 0
    assert levels == {"STALE"}


def test_an_undeclared_level_is_refused_rather_than_defaulting_to_green() -> None:
    with pytest.raises(module.ContractError):
        module._result(CADENCE, "STALLE", "a typo in a red level")


# --------------------------------------------------------------------------
# Identity is the checked-in path, exactly.
# --------------------------------------------------------------------------


def test_lookup_matches_on_path_despite_a_renamed_display_name() -> None:
    result = _check(
        workflow_pages=[_listing(_workflow(name="Renamed by somebody"))],
        runs={"workflow_runs": [_run()]},
    )

    assert result.level == "OK"


def test_lookup_ignores_a_right_name_at_a_different_path(tmp_path: Path) -> None:
    result = _check(
        workflow_pages=[_listing(_workflow(path=".github/workflows/impostor.yml"))],
        repo_root=tmp_path,
    )

    assert result.level == "NOTICE"
    assert "absent from this checkout and from the Actions listing" in result.detail


def test_lookup_rejects_near_miss_paths(tmp_path: Path) -> None:
    """A suffix or case-insensitive match would accept a backup or an archive."""
    result = _check(
        workflow_pages=[
            _listing(
                _workflow(id=7, path=".github/workflows/archive/cuda-nightly.yml"),
                _workflow(id=8, path=f"{CADENCE.workflow_path}.bak"),
                _workflow(id=9, path=CADENCE.workflow_path.upper()),
            )
        ],
        repo_root=tmp_path,
    )

    assert result.level == "NOTICE"
    assert "absent from this checkout and from the Actions listing" in result.detail


def test_two_active_workflows_at_one_path_are_a_contract_failure() -> None:
    results = _check_all(opener=_opener([_listing(_workflow(), _workflow(id=43))], {}))

    assert results[0].level == "CONFIG"
    assert "multiple active workflows are registered" in results[0].detail


def test_a_deleted_record_beside_a_live_one_does_not_red_a_healthy_cadence() -> None:
    """A delete-then-re-add leaves a `state: "deleted"` record at the same path."""
    result = _check(
        workflow_pages=[_listing(_workflow(id=99, state="deleted"), _workflow(id=42))],
        runs={"workflow_runs": [_run()]},
    )

    assert result.level == "OK"


def test_an_only_deleted_record_reads_as_unregistered(tmp_path: Path) -> None:
    result = _check(
        workflow_pages=[_listing(_workflow(state="deleted"))], repo_root=tmp_path
    )

    assert result.level == "NOTICE"


# --------------------------------------------------------------------------
# The absent-workflow branch is gated on not_before, not on the checkout.
# The two dated listing measurements in the module docstring are why.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("file_in_checkout", [True, False])
@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (NOW, "NOTICE"),
        (CADENCE.not_before - timedelta(seconds=1), "NOTICE"),
        (CADENCE.not_before, "CONFIG"),
        (AFTER_NOT_BEFORE, "CONFIG"),
    ],
)
def test_an_absent_workflow_is_red_exactly_from_not_before(
    tmp_path: Path, file_in_checkout: bool, now: datetime, expected: str
) -> None:
    if file_in_checkout:
        workflow_file = tmp_path / CADENCE.workflow_path
        workflow_file.parent.mkdir(parents=True)
        workflow_file.write_text(f"name: {CADENCE.workflow_name}\n")

    result = _check(workflow_pages=[_listing()], repo_root=tmp_path, now=now)

    assert result.level == expected
    assert (result.level in module.RED_LEVELS) == (expected == "CONFIG")
    if expected == "NOTICE":
        assert CADENCE.not_before.date().isoformat() in result.detail


def test_the_presence_probe_chooses_only_the_wording(tmp_path: Path) -> None:
    workflow_file = tmp_path / CADENCE.workflow_path
    workflow_file.parent.mkdir(parents=True)
    workflow_file.write_text(f"name: {CADENCE.workflow_name}\n")

    present = _check(workflow_pages=[_listing()], repo_root=tmp_path, now=NOW)
    absent = _check(workflow_pages=[_listing()], repo_root=tmp_path / "empty", now=NOW)

    assert present.level == absent.level == "NOTICE"
    assert "present in this checkout" in present.detail
    assert "not promoted to the default branch" in present.detail
    assert "absent from this checkout" in absent.detail
    assert "renamed without updating this table" in absent.detail


# --------------------------------------------------------------------------
# not_before separates bootstrap from genuine staleness, for a registered
# workflow that has never succeeded.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (NOW, "NOTICE"),
        (CADENCE.not_before - timedelta(seconds=1), "NOTICE"),
        (CADENCE.not_before, "STALE"),
        (AFTER_NOT_BEFORE, "STALE"),
    ],
)
def test_no_successful_run_is_red_exactly_from_not_before(
    now: datetime, expected: str
) -> None:
    result = _check(
        workflow_pages=[_listing(_workflow())], runs={"workflow_runs": []}, now=now
    )

    assert result.level == expected
    assert (result.level in module.RED_LEVELS) == (expected == "STALE")
    assert "no successful run" in result.detail


def test_a_stale_bootstrap_reports_how_long_it_has_been_overdue() -> None:
    result = _check(
        workflow_pages=[_listing(_workflow())],
        runs={"workflow_runs": []},
        now=AFTER_NOT_BEFORE,
    )

    assert result.level == "STALE"
    assert "2 days" in result.detail


def test_not_before_is_not_derived_from_the_workflow_creation_date() -> None:
    """An old registration must not by itself expire the bootstrap window."""
    result = _check(
        workflow_pages=[_listing(_workflow(created_at="2024-01-01T00:00:00Z"))],
        runs={"workflow_runs": []},
        now=NOW,
    )

    assert result.level == "NOTICE"


# --------------------------------------------------------------------------
# Staleness boundary and the field the age is measured from.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (timedelta(hours=1), "OK"),
        (timedelta(days=3) - timedelta(minutes=1), "OK"),
        (timedelta(days=3), "OK"),
        (timedelta(days=3) + timedelta(minutes=1), "STALE"),
        (timedelta(days=30), "STALE"),
    ],
)
def test_staleness_is_decided_at_max_age(offset: timedelta, expected: str) -> None:
    result = _check(
        workflow_pages=[_listing(_workflow())],
        runs={"workflow_runs": [_run(created_at=(NOW - offset).isoformat())]},
    )

    assert result.level == expected
    # A literal, not a restatement of the module's own format expression: a
    # 12-hour max_age used to print "maximum 0 days".
    assert "maximum 3 days" in result.detail


def test_the_max_age_bound_is_rendered_from_the_value_not_from_whole_days() -> None:
    twelve_hours = dataclasses.replace(CADENCE, max_age=timedelta(hours=12))

    result = module.check_cadence(
        twelve_hours,
        repository="royerlab/luxar",
        now=NOW,
        token="test-token",
        opener=_opener(
            [_listing(_workflow())],
            {
                "workflow_runs": [
                    _run(created_at=(NOW - timedelta(hours=2)).isoformat())
                ]
            },
        ),
        sleep=_no_sleep,
    )

    assert result.level == "OK"
    assert "maximum 12 hours" in result.detail


def test_a_stale_verdict_names_the_fires_and_fails_ambiguity() -> None:
    result = _check(
        workflow_pages=[_listing(_workflow())],
        runs={"workflow_runs": [_run(created_at="2026-03-01T00:00:00Z")]},
    )

    assert result.level == "STALE"
    assert "fires and fails looks the same here" in result.detail


def test_the_age_is_measured_from_created_at_not_from_updated_at() -> None:
    """A re-run bumps `updated_at`, which would buy fake freshness."""
    stale_run = _run(
        created_at="2026-03-01T00:00:00Z",
        run_started_at="2026-09-11T12:00:00Z",
        updated_at="2026-09-11T19:00:00Z",
    )

    result = _check(
        workflow_pages=[_listing(_workflow())], runs={"workflow_runs": [stale_run]}
    )

    assert result.level == "STALE"
    assert "194 days" in result.detail


def test_a_fresh_created_at_is_current_even_with_an_ancient_updated_at() -> None:
    fresh_run = _run(
        created_at="2026-09-11T03:00:00Z",
        run_started_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )

    result = _check(
        workflow_pages=[_listing(_workflow())], runs={"workflow_runs": [fresh_run]}
    )

    assert result.level == "OK"
    assert "17 hours" in result.detail


def test_recent_success_reports_its_age() -> None:
    result = _check(
        workflow_pages=[_listing(_workflow())],
        runs={"workflow_runs": [_run(created_at="2026-09-10T03:00:00Z")]},
    )

    assert result.level == "OK"
    assert "1 day, 17 hours" in result.detail


def test_a_non_utc_offset_is_converted_rather_than_taken_at_face_value() -> None:
    """05:00+02:00 is 03:00Z -- the same instant as the fixture above."""
    result = _check(
        workflow_pages=[_listing(_workflow())],
        runs={"workflow_runs": [_run(created_at="2026-09-10T05:00:00+02:00")]},
    )

    assert result.level == "OK"
    assert "1 day, 17 hours" in result.detail


@pytest.mark.parametrize("order", ["newest-first", "oldest-first"])
def test_the_newest_success_wins_regardless_of_listing_order(order: str) -> None:
    fresh = _run(created_at="2026-09-11T03:00:00Z")
    stale = _run(created_at="2026-03-01T00:00:00Z")
    runs = [fresh, stale] if order == "newest-first" else [stale, fresh]

    result = _check(
        workflow_pages=[_listing(_workflow())], runs={"workflow_runs": runs}
    )

    assert result.level == "OK"


@pytest.mark.parametrize(
    ("skew", "expected"),
    [
        (timedelta(minutes=4, seconds=59), "OK"),
        (timedelta(minutes=5), "OK"),
        (timedelta(minutes=5, seconds=1), "CONFIG"),
        (timedelta(days=181), "CONFIG"),
    ],
)
def test_a_future_dated_success_is_red_beyond_the_clock_skew_tolerance(
    skew: timedelta, expected: str
) -> None:
    result = _check(
        workflow_pages=[_listing(_workflow())],
        runs={"workflow_runs": [_run(created_at=(NOW + skew).isoformat())]},
    )

    assert result.level == expected
    if expected == "CONFIG":
        assert "in the future" in result.detail


def test_a_small_future_skew_reports_minutes_not_zero_hours() -> None:
    result = _check(
        workflow_pages=[_listing(_workflow())],
        runs={
            "workflow_runs": [
                _run(created_at=(NOW + timedelta(minutes=10)).isoformat())
            ]
        },
    )

    assert result.level == "CONFIG"
    assert "dated 10 minutes in the future" in result.detail


def test_disabled_workflow_is_a_configuration_failure() -> None:
    result = _check(workflow_pages=[_listing(_workflow(state="disabled_manually"))])

    assert result.level == "CONFIG"
    assert "disabled_manually" in result.detail


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (timedelta(0), "0 minutes"),
        (timedelta(minutes=1), "1 minute"),
        (timedelta(minutes=59), "59 minutes"),
        (timedelta(hours=1), "1 hour"),
        (timedelta(hours=2), "2 hours"),
        (timedelta(days=1), "1 day"),
        (timedelta(days=1, hours=1), "1 day, 1 hour"),
        (timedelta(days=2), "2 days"),
        (timedelta(days=2, hours=3), "2 days, 3 hours"),
    ],
)
def test_age_text_pluralises_every_unit(age: timedelta, expected: str) -> None:
    assert module._age_text(age) == expected


# --------------------------------------------------------------------------
# HTTP status classification: weather is green, a permanent status is red.
# --------------------------------------------------------------------------


def _http_error(status: int, headers: object = None) -> Callable[..., Any]:
    def raise_status(request: Any, timeout: int) -> Any:
        raise urllib.error.HTTPError(
            request.full_url, status, "nope", headers, io.BytesIO(b"{}")
        )

    return raise_status


def _rate_limit_headers() -> email.message.Message:
    headers = email.message.Message()
    headers["X-RateLimit-Remaining"] = "0"
    headers["Retry-After"] = "60"
    return headers


@pytest.mark.parametrize(
    ("status", "headers", "level"),
    [
        (401, None, "CONFIG"),  # expired or missing token
        (404, None, "CONFIG"),  # wrong --repository, or the repo was renamed
        (422, None, "CONFIG"),
        (403, None, "CONFIG"),  # host job forgot `permissions: actions: read`
        (403, {"x-ratelimit-remaining": "4998"}, "CONFIG"),
        (403, _rate_limit_headers(), "UNKNOWN"),  # documented rate-limit shape
        (403, {"Retry-After": "60"}, "UNKNOWN"),  # header lookup is case-blind
        (403, {"X-RateLimit-Remaining": "0"}, "UNKNOWN"),
        (408, None, "UNKNOWN"),
        (425, None, "UNKNOWN"),
        (429, None, "UNKNOWN"),  # rate limited: weather
        (500, None, "UNKNOWN"),
        (502, None, "UNKNOWN"),
        (503, None, "UNKNOWN"),
    ],
)
def test_http_statuses_are_split_into_defects_and_weather(
    status: int, headers: object, level: str
) -> None:
    slept: list[float] = []

    results = _check_all(opener=_http_error(status, headers), sleep=slept.append)

    assert results[0].level == level
    assert (results[0].level in module.RED_LEVELS) == (level == "CONFIG")
    if level == "CONFIG":
        assert f"HTTP {status}" in results[0].detail
        # A permanent status is not retried; retrying and then calling it
        # weather is exactly how a dead token stays green.
        assert slept == []
    else:
        assert slept == [EXPECTED_RETRY_DELAY]


def test_a_retryable_status_followed_by_a_good_response_gives_a_real_verdict() -> None:
    inner = _opener([_listing(_workflow())], {"workflow_runs": [_run()]})
    calls = {"n": 0}
    slept: list[float] = []

    def flaky(request: Any, timeout: int) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                request.full_url, 503, "unavailable", None, io.BytesIO(b"{}")
            )
        return inner(request, timeout)

    result = module.check_cadence(
        CADENCE,
        repository="royerlab/luxar",
        now=NOW,
        token="test-token",
        opener=flaky,
        sleep=slept.append,
    )

    assert result.level == "OK"
    assert slept == [EXPECTED_RETRY_DELAY]


def test_a_permanent_http_status_makes_main_exit_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = module.main(
        [],
        checker=lambda **kwargs: _check_all(opener=_http_error(401)),
        environ={"GITHUB_TOKEN": "expired"},
    )

    assert code == 1
    assert "[CONFIG]" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Transport failure is weather; a malformed payload is a defect.
# --------------------------------------------------------------------------


def test_a_transport_blip_is_retried_once_and_then_succeeds() -> None:
    inner = _opener([_listing(_workflow())], {"workflow_runs": [_run()]})
    calls = {"n": 0}
    slept: list[float] = []

    def flaky(request: Any, timeout: int) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection reset")
        return inner(request, timeout)

    result = module.check_cadence(
        CADENCE,
        repository="royerlab/luxar",
        now=NOW,
        token="test-token",
        opener=flaky,
        sleep=slept.append,
    )

    assert result.level == "OK"
    # The failed listing, its retry, and then the runs query.
    assert calls["n"] == 3
    assert slept == [EXPECTED_RETRY_DELAY]


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("offline"),
        http.client.IncompleteRead(b"half a body"),
        http.client.RemoteDisconnected("closed"),
    ],
)
def test_a_persistent_transport_error_is_unknown_and_not_red(
    error: Exception,
) -> None:
    slept: list[float] = []

    def failing(request: Any, timeout: int) -> Any:
        raise error

    results = _check_all(opener=failing, sleep=slept.append)

    assert results[0].level == "UNKNOWN"
    assert results[0].level not in module.RED_LEVELS
    assert "could not verify the cadence" in results[0].detail
    assert slept == [EXPECTED_RETRY_DELAY]


@pytest.mark.parametrize(
    ("body", "why"),
    [
        (b"<html><body>502 Bad Gateway</body></html>", "a proxy HTML error page"),
        # UnicodeDecodeError is a ValueError SIBLING of JSONDecodeError, so this
        # used to escape the retry and land as a permanent red -- a red decided
        # by the byte values of somebody's error page.
        (b"<html>Gateway error \x92 oops</html>", "a latin-1 error page"),
        (b"\x1f\x8b\x08\x00mangled gzip", "a mis-decoded gzip body"),
    ],
)
def test_an_undecodable_body_is_weather_and_is_retried(body: bytes, why: str) -> None:
    slept: list[float] = []

    def bad_body(request: Any, timeout: int) -> Any:
        return io.BytesIO(body)

    results = _check_all(opener=bad_body, sleep=slept.append)

    assert results[0].level == "UNKNOWN", why
    assert results[0].level not in module.RED_LEVELS
    assert slept == [EXPECTED_RETRY_DELAY]


@pytest.mark.parametrize("body", [b"not json at all", b"<html>oops \x92</html>"])
def test_an_undecodable_body_that_recovers_on_retry_gives_a_real_verdict(
    body: bytes,
) -> None:
    inner = _opener([_listing(_workflow())], {"workflow_runs": [_run()]})
    calls = {"n": 0}
    slept: list[float] = []

    def flaky(request: Any, timeout: int) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return io.BytesIO(body)
        return inner(request, timeout)

    result = module.check_cadence(
        CADENCE,
        repository="royerlab/luxar",
        now=NOW,
        token="test-token",
        opener=flaky,
        sleep=slept.append,
    )

    assert result.level == "OK"
    assert slept == [EXPECTED_RETRY_DELAY]


def test_an_unclassified_exception_is_red_and_names_its_type() -> None:
    def exploding(request: Any, timeout: int) -> Any:
        raise RuntimeError("boom")

    results = _check_all(opener=exploding)

    assert results[0].level == "CONFIG"
    assert "RuntimeError" in results[0].detail


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ([], "non-object response"),
        ({"total_count": 0}, "omitted workflows"),
        ({"workflows": []}, "omitted the workflow total_count"),
        ({"total_count": True, "workflows": []}, "omitted the workflow total_count"),
        ({"total_count": 1, "workflows": ["nope"]}, "non-object workflow"),
    ],
)
def test_a_malformed_listing_payload_is_red(payload: object, fragment: str) -> None:
    results = _check_all(opener=_opener([payload], {}))

    assert results[0].level == "CONFIG"
    assert results[0].level in module.RED_LEVELS
    assert fragment in results[0].detail


@pytest.mark.parametrize(
    ("workflow_overrides", "runs", "fragment"),
    [
        ({"id": "42"}, None, "omitted the workflow id"),
        ({"id": True}, None, "omitted the workflow id"),
        ({}, {"total_count": 0}, "omitted workflow runs"),
        ({}, {"workflow_runs": ["nope"]}, "non-object workflow run"),
        ({}, {"workflow_runs": [{"conclusion": "success"}]}, "omitted created_at"),
        (
            {},
            {"workflow_runs": [_run(created_at="not-a-date")]},
            "unparsable created_at",
        ),
        (
            {},
            {"workflow_runs": [_run(created_at="2026-09-10T03:00:00")]},
            "timezone-naive created_at",
        ),
        (
            {},
            # A good success row does not excuse a malformed sibling: the
            # payload is broken either way.
            {
                "workflow_runs": [
                    _run(created_at="2026-09-11T03:00:00Z"),
                    _run(created_at="garbage"),
                ]
            },
            "unparsable created_at",
        ),
    ],
)
def test_a_malformed_run_payload_is_red(
    workflow_overrides: dict[str, object], runs: object, fragment: str
) -> None:
    results = _check_all(
        opener=_opener(
            [_listing(_workflow(**workflow_overrides))], {} if runs is None else runs
        )
    )

    assert results[0].level == "CONFIG"
    assert fragment in results[0].detail


def test_a_malformed_timestamp_on_a_skipped_row_is_not_read() -> None:
    """Non-success rows are skipped before their timestamps are parsed."""
    result = _check(
        workflow_pages=[_listing(_workflow())],
        runs={
            "workflow_runs": [
                _run(conclusion="failure", created_at="garbage"),
                _run(created_at="2026-09-11T03:00:00Z"),
            ]
        },
    )

    assert result.level == "OK"


# --------------------------------------------------------------------------
# Pagination: a truncated listing must not read as an absence.
# --------------------------------------------------------------------------


def test_the_listing_is_paginated_until_it_is_exhausted() -> None:
    page_one = _listing({"id": 7, "path": ".github/workflows/ci.yml"}, total_count=2)
    page_two = _listing(_workflow(), total_count=2)

    result = _check(
        workflow_pages=[page_one, page_two], runs={"workflow_runs": [_run()]}
    )

    assert result.level == "OK"


def test_a_truncated_listing_is_a_contract_failure_not_an_absence(
    tmp_path: Path,
) -> None:
    truncated = _listing({"id": 7, "path": ".github/workflows/ci.yml"}, total_count=500)

    # An exhausted listing that still falls short of total_count: page 2 comes
    # back empty, so pagination stops with 1 of the claimed 500.
    results = _check_all(
        opener=_opener([truncated, _listing(total_count=500)], {}), repo_root=tmp_path
    )

    assert results[0].level == "CONFIG"
    assert "listed 1 workflows but reported 500" in results[0].detail


def test_exhausting_the_page_budget_is_a_contract_failure(tmp_path: Path) -> None:
    """Six pages against a five-page budget: the shortfall must not read benign."""
    pages = [
        _listing({"id": n, "path": f".github/workflows/w{n}.yml"}, total_count=6)
        for n in range(6)
    ]

    results = _check_all(opener=_opener(pages, {}), repo_root=tmp_path)

    assert results[0].level == "CONFIG"
    assert "listed 5 workflows but reported 6" in results[0].detail


# --------------------------------------------------------------------------
# The server-side success filter is not trusted blindly.
# --------------------------------------------------------------------------


def test_a_non_success_conclusion_is_skipped_not_raised() -> None:
    result = _check(
        workflow_pages=[_listing(_workflow())],
        runs={"workflow_runs": [_run(conclusion="failure")]},
        now=AFTER_NOT_BEFORE,
    )

    assert result.level == "STALE"
    assert "no successful run" in result.detail


def test_a_leaked_failure_does_not_hide_a_real_success() -> None:
    """With per_page=1 a single leaked row would report STALE on a live cadence."""
    result = _check(
        workflow_pages=[_listing(_workflow())],
        runs={
            "workflow_runs": [
                _run(conclusion="failure", created_at="2026-09-11T03:00:00Z"),
                _run(conclusion="cancelled", created_at="2026-09-11T02:00:00Z"),
                _run(conclusion="success", created_at="2026-09-10T03:00:00Z"),
            ]
        },
    )

    assert result.level == "OK"


# --------------------------------------------------------------------------
# The whole table is walked, and one bad row does not take the others down.
# --------------------------------------------------------------------------


def _second_row() -> Any:
    return dataclasses.replace(
        CADENCE,
        workflow_name="Second cadence",
        workflow_path=".github/workflows/second.yml",
        not_before=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_every_table_row_is_checked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(module, "CADENCES", (CADENCE, _second_row()))

    results = _check_all(
        opener=_opener([_listing(_workflow())], {"workflow_runs": [_run()]}),
        repo_root=tmp_path,
    )

    assert [result.level for result in results] == ["OK", "CONFIG"]
    assert [result.cadence.workflow_name for result in results] == [
        CADENCE.workflow_name,
        "Second cadence",
    ]


def test_a_failing_row_does_not_abort_the_rest_of_the_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "CADENCES", (CADENCE, _second_row()))
    inner = _opener([_listing(_workflow())], {"workflow_runs": [_run()]})
    calls = {"n": 0}

    def fails_for_the_second_row(request: Any, timeout: int) -> Any:
        calls["n"] += 1
        # Row one spends two calls (listing, runs); everything after belongs to
        # row two.
        if calls["n"] > 2:
            raise urllib.error.URLError("offline")
        return inner(request, timeout)

    results = _check_all(opener=fails_for_the_second_row, sleep=lambda _seconds: None)

    assert [result.level for result in results] == ["OK", "UNKNOWN"]


# --------------------------------------------------------------------------
# main(). Nothing in this suite may touch the network, so every test that
# reaches main passes the `checker` seam.
# --------------------------------------------------------------------------


def test_main_defaults_are_the_real_collaborators() -> None:
    """Asserted offline: calling main() without `checker` would hit GitHub."""
    defaults = module.main.__kwdefaults__

    assert defaults["checker"] is module.check_all
    assert defaults["clock"] is module._utc_now
    assert defaults["environ"] is None


def test_main_help_text_is_the_first_docstring_line_not_the_whole_thing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        module.main(["--help"], checker=_capturing_checker({}))

    out = capsys.readouterr().out
    assert module.__doc__ is not None
    assert module.__doc__.splitlines()[0] in out
    assert "systemd timer" not in out


@pytest.mark.parametrize(
    ("level", "expected"),
    [("OK", 0), ("NOTICE", 0), ("UNKNOWN", 0), ("STALE", 1), ("CONFIG", 1)],
)
def test_main_only_fails_for_stale_or_broken_cadences(
    capsys: pytest.CaptureFixture[str], level: str, expected: int
) -> None:
    code = module.main(
        [],
        checker=lambda **kwargs: [module.Result(CADENCE, level, "detail")],
        environ={"GITHUB_TOKEN": "test-token"},
    )

    assert code == expected
    assert f"[{level}] {CADENCE.workflow_name}: detail" in capsys.readouterr().out


def test_main_fails_when_any_single_row_is_red(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def checker(**kwargs: object) -> list[Any]:
        return [
            module.Result(CADENCE, "OK", "fine"),
            module.Result(CADENCE, "STALE", "dead clock"),
        ]

    assert module.main([], checker=checker, environ={"GITHUB_TOKEN": "t"}) == 1
    out = capsys.readouterr().out
    assert "[OK]" in out
    assert "[STALE]" in out


def test_main_passes_the_real_clock_the_repository_and_the_token() -> None:
    """The wiring, not a mock that discards it.

    A frozen `now`, a wrong `repository`, and an empty `token` are all
    permanently-green mutants that no assertion on a kwargs-discarding lambda
    can see.
    """
    captured: dict[str, Any] = {}

    before = datetime.now(UTC)
    code = module.main(
        [], checker=_capturing_checker(captured), environ={"GITHUB_TOKEN": "real"}
    )
    after = datetime.now(UTC)

    assert code == 0
    assert captured["token"] == "real"
    assert captured["repository"] == "royerlab/luxar"
    assert captured["now"].tzinfo is not None
    assert before <= captured["now"] <= after


@pytest.mark.parametrize(
    ("argv", "environ", "expected"),
    [
        ([], {"GITHUB_TOKEN": "t"}, "royerlab/luxar"),
        ([], {"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": ""}, "royerlab/luxar"),
        ([], {"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "other/repo"}, "other/repo"),
        (
            ["--repository", "flag/wins"],
            {"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "other/repo"},
            "flag/wins",
        ),
    ],
)
def test_main_resolves_the_repository_by_flag_then_environment(
    argv: list[str], environ: dict[str, str], expected: str
) -> None:
    captured: dict[str, Any] = {}

    module.main(argv, checker=_capturing_checker(captured), environ=environ)

    assert captured["repository"] == expected


@pytest.mark.parametrize(
    "repository", ["not-a-repo", "too/many/parts", "owner/", "/name", "owner name"]
)
def test_main_rejects_a_malformed_repository(
    capsys: pytest.CaptureFixture[str], repository: str
) -> None:
    def checker(**kwargs: object) -> list[Any]:
        raise AssertionError("main must not query a malformed repository")

    code = module.main(
        ["--repository", repository], checker=checker, environ={"GITHUB_TOKEN": "t"}
    )

    assert code == 1
    assert "is not an owner/name repository" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"GITHUB_TOKEN": "primary", "GH_TOKEN": "fallback"}, "primary"),
        ({"GH_TOKEN": "fallback"}, "fallback"),
        ({"GITHUB_TOKEN": "", "GH_TOKEN": "fallback"}, "fallback"),
    ],
)
def test_main_prefers_github_token_and_falls_back_to_gh_token(
    environ: dict[str, str], expected: str
) -> None:
    captured: dict[str, Any] = {}

    assert module.main([], checker=_capturing_checker(captured), environ=environ) == 0
    assert captured["token"] == expected


@pytest.mark.parametrize("environ", [{}, {"GITHUB_TOKEN": ""}, {"GH_TOKEN": ""}])
def test_main_requires_an_actions_read_token(
    capsys: pytest.CaptureFixture[str], environ: dict[str, str]
) -> None:
    def checker(**kwargs: object) -> list[Any]:
        raise AssertionError("main must not query anything without a token")

    assert module.main([], checker=checker, environ=environ) == 1
    assert "actions:read is required" in capsys.readouterr().out


def test_main_reads_the_process_environment_when_none_is_passed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    def checker(**kwargs: object) -> list[Any]:
        raise AssertionError("main must not query anything without a token")

    assert module.main([], checker=checker) == 1
    assert "actions:read is required" in capsys.readouterr().out
