#!/usr/bin/env python3
"""Fail when a repository-owned workflow cadence stops succeeding.

Some checks in this repository are load-bearing but are *clocked from outside
it*. The `CUDA native cadence` workflow is `workflow_dispatch`-only and is
invoked by a systemd timer living in `royerlab/luxar-ci`; nothing in this
repository notices if that timer dies. Every check here stays green, which is
exactly the failure mode that makes a green check worthless: the absence of a
red is being read as evidence, and no evidence is being produced.

This watchdog reads the most recent *successful* run of each row in `CADENCES`
and goes red when it is older than that row's `max_age`.

Red verdicts:

* `STALE` -- a success exists but is older than `max_age`, or none exists at
  all and the row's `not_before` has passed.
* `CONFIG` -- the workflow is registered but disabled, or it is missing from
  the Actions listing at or after `not_before`, or the API answered with a
  permanent HTTP status, or the payload violated the contract, or the newest
  success is dated in the future.

`NOTICE` (not red) is the bootstrap state: a workflow that is not in the
listing yet, or that has never succeeded, before `not_before`. `UNKNOWN` (not
red) means the watchdog could not reach GitHub -- see the weather paragraph
below. `OK` is a success inside `max_age`.

What a red does NOT tell you: this reads successes only, so "no success inside
the window" cannot distinguish a dispatcher that stopped firing from a cadence
that fires daily and fails. The compensating control lives in the workflow
itself -- `cuda-nightly.yml`'s `Report cadence failure` step files, or comments
on, an issue on every failure or cancellation -- so look for that issue before
hunting the timer. Disambiguating it here would cost a second API query per row
for information that already exists elsewhere, so it is deliberately not done.

Two measurements against the live Actions listing, kept because they bracket
the promotion this repository was in the middle of and because together they
justify the design:

* 2026-09-12, before promotion: `GET /repos/royerlab/luxar/actions/workflows`
  returned `total_count: 10` and did NOT include
  `.github/workflows/cuda-nightly.yml`, although the file was already on `dev`.
* 2026-09-12T08:17:05Z, after `promote-ff` fast-forwarded `main` to 8cc8f3df6:
  the same request returned `total_count: 11` and did include the workflow,
  `state: "active"`, `created_at: 2026-09-12T01:17:05.000-07:00`.

Same conclusion from both readings: the listing is effectively
default-branch-scoped, so absence from it tracks *promotion*, not the health of
the cadence. That is why the LEVEL for an absent workflow is decided by
`not_before` alone while the presence probe only picks the wording -- otherwise
the verdict would depend on which branch the host happened to check out.

Scope, stated plainly so the table is not read as a broader promise than it is:

* **Only workflow cadences are visible here.** The verdict comes from the
  Actions `workflow_runs` API, so a cadence that is not a workflow cannot be
  covered *in principle*. The precedent that motivated this script -- a
  dev-to-main promotion timer that was stopped and never rearmed -- is a systemd
  unit on a host, and a `workflow_runs` query structurally cannot see it. Read
  `CADENCES` as "these workflow cadences are watched", never as "cadences are
  watched".
* **Detection latency is bounded by the host, not by the table alone.** The
  dedicated `cadence-liveness.yml` job runs daily, and its contract test keeps
  that period no longer than the smallest `max_age` in `CADENCES`; otherwise a
  stale cadence would only be noticed on the host's next firing.

Transport failure is deliberately *not* a red, but "transport failure" is drawn
narrowly. A connection error, a body truncated mid-read, a body that is not
JSON or not even UTF-8 (a proxy error page in some other encoding), HTTP 408,
425, 429, and any 5xx are weather: the request is retried once, and if the
failure persists the row reports `UNKNOWN` -- loud in the log, exit code 0. On
a daily host, weather would cost at most a day of detection latency, whereas a
watchdog that cries wolf on every API hiccup is a watchdog that gets ignored.

Everything else is a permanent defect and stays red. A 4xx that is not a rate
limit fails identically forever, so laundering it into `UNKNOWN` would leave the
job green on a dead token -- verbatim the failure this script exists to kill.
403 is the ambiguous one: GitHub answers *both* a missing `actions: read` scope
*and* a primary or secondary rate limit with it, so a 403 is treated as weather
only when the response carries `Retry-After` or `x-ratelimit-remaining: 0`, and
stays red otherwise. A well-formed but wrong-shaped payload (no `workflows`
key, no workflow id, a non-object body, a `total_count` larger than what was
actually listed) is likewise a defect, not weather.

The module is import-safe and performs no network I/O at import time. The URL
opener, the clock, the retry sleep, and the checkout root used for the presence
probe are all injected, and `main` additionally takes the environment and the
checker it delegates to, so its wiring is asserted against real values instead
of a mock that discards them.
"""

from __future__ import annotations

import argparse
import ast
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

API_ROOT = "https://api.github.com"
DEFAULT_REPOSITORY = "royerlab/luxar"
REPOSITORY_PATTERN = re.compile(r"[\w.-]+/[\w.-]+")
REQUEST_TIMEOUT_SECONDS = 30
RETRY_DELAY_SECONDS = 2.0
WORKFLOW_PAGE_SIZE = 100
# The repository has eleven workflows; five pages is 500. A budget of thousands
# would turn a pathological listing into a very slow green.
MAX_WORKFLOW_PAGES = 5
# `status=success` is applied server-side and has been observed to leak other
# conclusions, so ask for a handful and pick the newest genuine success. With
# `per_page=1` a single leaked row would hide a real success entirely.
RUN_PAGE_SIZE = 5
# A success dated further than this into the future is a defect signal (clock
# skew, or reading the wrong field), not freshness.
MAX_CLOCK_SKEW = timedelta(minutes=5)
# 408/425/429 are transient by definition and 5xx is the server having a bad
# day; all are retried and end as UNKNOWN. Every other status is permanent and
# stays red, except a 403 that carries rate-limit headers (see `_is_retryable`).
RETRYABLE_STATUS_FLOOR = 500
RETRYABLE_STATUSES = frozenset({408, 425, 429})
AMBIGUOUS_STATUS = 403
REPO_ROOT = Path(__file__).resolve().parents[1]
LEVELS = frozenset({"OK", "NOTICE", "UNKNOWN", "STALE", "CONFIG"})
RED_LEVELS = frozenset({"STALE", "CONFIG"})
OpenUrl = Callable[..., Any]
Sleep = Callable[[float], None]


class ContractError(ValueError):
    """The GitHub payload or status was a permanent defect."""


class TransportError(OSError):
    """The request could not be completed, twice in a row."""


@dataclass(frozen=True)
class Cadence:
    # `workflow_name` is a human label for reports only. Identity is
    # `workflow_path`, which is the checked-in file path the Actions API reports
    # as `path` -- the same key `ci.yml`'s diff classifier uses. Matching on the
    # display `name` would make a one-line rename inside the workflow file read
    # as "not registered yet", i.e. silently green.
    workflow_name: str
    workflow_path: str
    branch: str
    event: str
    max_age: timedelta
    # Absence from the listing, and the absence of any successful run, are
    # failures only once `now >= not_before`. This is an explicit, snoozable
    # date rather than a grace period measured from the workflow's `created_at`:
    # registration time is not "the out-of-repo timer is wired up", and a
    # `workflow_dispatch` workflow cannot be invoked at all until its file
    # reaches the default branch through promotion.
    not_before: datetime


@dataclass(frozen=True)
class Result:
    cadence: Cadence
    level: str
    detail: str


CADENCES = (
    Cadence(
        workflow_name="CUDA native cadence",
        workflow_path=".github/workflows/cuda-nightly.yml",
        branch="dev",
        event="workflow_dispatch",
        # The dispatcher fires daily, so three days tolerates two missed fires
        # before going red -- one miss is weather (a busy self-hosted runner, a
        # rebooting host), two in a row is a dead clock.
        max_age=timedelta(days=3),
        # Measured, not guessed: `promote-ff` carried the file to `main` at
        # 2026-09-12T08:17:05Z, about three hours after it reached `dev`, and
        # the out-of-repo dispatcher is enabled for 02:36 PDT daily -- so the
        # first dispatch was due the same day. This is two daily cycles of slack
        # past that observed promotion. A longer window would be a blind spot
        # over precisely the failure this script exists to catch, and moving it
        # out is a snooze on a real red: argue for it in the diff, and see
        # `test_not_before_is_not_an_unbounded_snooze`, which bounds it.
        not_before=datetime(2026, 9, 14, tzinfo=UTC),
    ),
)


def _result(cadence: Cadence, level: str, detail: str) -> Result:
    """Build a `Result`, refusing any level `main` does not classify.

    Five level strings are emitted and `RED_LEVELS` names two of them, so a
    typo'd red level would otherwise default to exit 0.
    """
    if level not in LEVELS:
        raise ContractError(f"{level!r} is not a declared level")
    return Result(cadence, level, detail)


def _headers_lower(error: urllib.error.HTTPError) -> dict[str, str]:
    """Return the response headers keyed lower-case, or `{}` if there are none."""
    items = getattr(error.headers, "items", None)
    if items is None:
        return {}
    return {str(key).lower(): str(value) for key, value in items()}


def _is_rate_limited(error: urllib.error.HTTPError) -> bool:
    headers = _headers_lower(error)
    if "retry-after" in headers:
        return True
    return headers.get("x-ratelimit-remaining", "").strip() == "0"


def _is_retryable(error: urllib.error.HTTPError) -> bool:
    if error.code >= RETRYABLE_STATUS_FLOOR or error.code in RETRYABLE_STATUSES:
        return True
    # GitHub answers a primary or secondary rate limit with 403 as readily as
    # with 429, but a 403 is also how a missing `actions: read` scope and a dead
    # token present. Only the rate-limit shape is weather.
    return error.code == AMBIGUOUS_STATUS and _is_rate_limited(error)


def _read_json(
    url: str, opener: OpenUrl, token: str, *, sleep: Sleep
) -> dict[str, object]:
    """Fetch and decode one API page, retrying a weather failure once.

    A permanent HTTP status is raised immediately as a `ContractError`: retrying
    a 401 twice and then calling it weather is how a dead token stays green.
    """
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "luxar-cadence-watchdog",
            "Authorization": f"Bearer {token}",
        },
    )
    last_error: Exception | None = None
    for attempt in range(2):
        if attempt:
            sleep(RETRY_DELAY_SECONDS)
        try:
            with opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read()
            payload = json.loads(body)
        except urllib.error.HTTPError as error:
            # HTTPError subclasses OSError, so it must be caught first or every
            # status would be laundered into the retryable branch below.
            if not _is_retryable(error):
                raise ContractError(
                    f"GitHub answered HTTP {error.code} for {url}"
                ) from error
            last_error = error
        except (
            OSError,
            http.client.HTTPException,
            json.JSONDecodeError,
            # A ValueError SIBLING of JSONDecodeError, not a subclass: a proxy
            # error page in some other encoding must not be a permanent red.
            UnicodeDecodeError,
        ) as error:
            last_error = error
        else:
            if not isinstance(payload, dict):
                raise ContractError("GitHub returned a non-object response")
            return payload
    raise TransportError(
        f"GitHub request to {url} failed twice: "
        f"{type(last_error).__name__}: {last_error}"
    )


def _parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError(f"GitHub response omitted {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        message = f"GitHub returned an unparsable {field}: {error}"
        raise ContractError(message) from error
    if parsed.tzinfo is None:
        # `astimezone` would silently reinterpret a naive value as host-local
        # time, which on a UTC-7 box reads a day-old run as 16 hours old.
        raise ContractError(f"GitHub returned a timezone-naive {field}: {value!r}")
    return parsed.astimezone(UTC)


def _plural(count: int, unit: str) -> str:
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


def _age_text(age: timedelta) -> str:
    """Render a duration. Minutes below an hour, so clock skew reads sensibly."""
    total_minutes = max(0, int(age.total_seconds() // 60))
    hours, minutes = divmod(total_minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        if hours:
            return f"{_plural(days, 'day')}, {_plural(hours, 'hour')}"
        return _plural(days, "day")
    if hours:
        return _plural(hours, "hour")
    return _plural(minutes, "minute")


def _list_workflows(
    repository: str, opener: OpenUrl, token: str, *, sleep: Sleep
) -> list[dict[str, object]]:
    """Return every registered workflow, paginating until the listing agrees.

    Truncating the listing would make "absent" indistinguishable from "not
    registered", which resolves benign -- so a listing shorter than the
    payload's own `total_count` is a contract failure, not an absence.
    """
    workflows: list[dict[str, object]] = []
    total_count = 0
    for page in range(1, MAX_WORKFLOW_PAGES + 1):
        query = urlencode({"per_page": WORKFLOW_PAGE_SIZE, "page": page})
        payload = _read_json(
            f"{API_ROOT}/repos/{repository}/actions/workflows?{query}",
            opener,
            token,
            sleep=sleep,
        )
        items = payload.get("workflows")
        if not isinstance(items, list):
            raise ContractError("GitHub response omitted workflows")
        count = payload.get("total_count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ContractError("GitHub response omitted the workflow total_count")
        total_count = count
        for item in items:
            if not isinstance(item, dict):
                raise ContractError("GitHub listed a non-object workflow")
            workflows.append(item)
        if not items or len(workflows) >= total_count:
            break
    if len(workflows) < total_count:
        raise ContractError(
            f"GitHub listed {len(workflows)} workflows but reported {total_count}"
        )
    return workflows


def _matching_workflow(
    workflows: list[dict[str, object]], path: str
) -> dict[str, object] | None:
    # Exact equality, deliberately: `.github/workflows/archive/cuda-nightly.yml`
    # and `cuda-nightly.yml.bak` are different files, and a suffix or
    # case-insensitive match would accept either as the cadence.
    matches = [item for item in workflows if item.get("path") == path]
    # A delete-then-re-add leaves a `state: "deleted"` record behind at the same
    # path, and a pagination race can show one twice. Neither is a live cadence,
    # and reporting "multiple workflows registered" for them would red a healthy
    # cadence and blame the table.
    live = []
    seen_ids: set[object] = set()
    for item in matches:
        if item.get("state") == "deleted":
            continue
        workflow_id = item.get("id")
        if workflow_id is not None and workflow_id in seen_ids:
            continue
        live.append(item)
        if workflow_id is not None:
            seen_ids.add(workflow_id)
    if len(live) > 1:
        raise ContractError(f"multiple active workflows are registered at {path!r}")
    return live[0] if live else None


def _unregistered_result(cadence: Cadence, repo_root: Path, now: datetime) -> Result:
    """Classify a cadence the Actions listing does not contain.

    The level comes from `not_before` alone; the presence probe only picks the
    wording. See the module docstring's two dated listing measurements for why:
    absence tracks promotion to the default branch, not the health of the
    cadence. The scheduled host checks out that default branch, while a manual
    run may intentionally probe another ref; neither changes the level. At or
    after `not_before`,
    promotion has had its window and absence is real -- un-promoted, deleted, or
    renamed out from under the table.
    """
    if (repo_root / cadence.workflow_path).is_file():
        cause = (
            f"{cadence.workflow_path} is present in this checkout but absent "
            "from the Actions listing (not promoted to the default branch)"
        )
    else:
        cause = (
            f"{cadence.workflow_path} is absent from this checkout and from the "
            "Actions listing (deleted, or renamed without updating this table)"
        )
    not_before = cadence.not_before.date().isoformat()
    if now < cadence.not_before:
        return _result(cadence, "NOTICE", f"{cause}; still expected by {not_before}")
    return _result(
        cadence,
        "CONFIG",
        f"{cause}; it was expected by {not_before} "
        f"({_age_text(now - cadence.not_before)} ago)",
    )


def _latest_success(
    cadence: Cadence,
    *,
    repository: str,
    workflow_id: int,
    opener: OpenUrl,
    token: str,
    sleep: Sleep,
) -> datetime | None:
    """Return when the newest successful run *started*, or `None` if none did.

    `created_at` is the field read, not `updated_at`: `updated_at` is bumped by
    a re-run of the same record, so clicking Re-run on a six-month-old success
    would buy three days of fake freshness. `created_at` is immutable and is the
    instant the dispatcher fired.

    The query is already filtered to `status=success`, but that filter is
    server-side and has been observed to return runs concluding otherwise. A
    non-success conclusion is therefore skipped -- it falls through to the
    no-successful-run path -- rather than raising an exception that would page a
    human about a server quirk.
    """
    query = urlencode(
        {
            "branch": cadence.branch,
            "event": cadence.event,
            "status": "success",
            "per_page": RUN_PAGE_SIZE,
        }
    )
    payload = _read_json(
        f"{API_ROOT}/repos/{repository}/actions/workflows/{workflow_id}/runs?{query}",
        opener,
        token,
        sleep=sleep,
    )
    workflow_runs = payload.get("workflow_runs")
    if not isinstance(workflow_runs, list):
        raise ContractError("GitHub response omitted workflow runs")
    starts = []
    for entry in workflow_runs:
        if not isinstance(entry, dict):
            raise ContractError("GitHub listed a non-object workflow run")
        if entry.get("head_branch") != cadence.branch:
            continue
        if entry.get("event") != cadence.event:
            continue
        if entry.get("conclusion") != "success":
            continue
        starts.append(_parse_time(entry.get("created_at"), "created_at"))
    # Newest wins explicitly rather than by trusting the listing order.
    return max(starts) if starts else None


def _never_ran_result(cadence: Cadence, now: datetime) -> Result:
    not_before = cadence.not_before.date().isoformat()
    if now < cadence.not_before:
        return _result(
            cadence,
            "NOTICE",
            "no successful run yet; the first one is not expected before "
            f"{not_before} (promotion to the default branch has to land first)",
        )
    return _result(
        cadence,
        "STALE",
        f"no successful run, and one was expected by {not_before} "
        f"({_age_text(now - cadence.not_before)} ago)",
    )


def check_cadence(
    cadence: Cadence,
    *,
    repository: str,
    now: datetime,
    token: str,
    opener: OpenUrl = urllib.request.urlopen,
    repo_root: Path = REPO_ROOT,
    sleep: Sleep = time.sleep,
) -> Result:
    workflows = _list_workflows(repository, opener, token, sleep=sleep)
    workflow = _matching_workflow(workflows, cadence.workflow_path)
    if workflow is None:
        return _unregistered_result(cadence, repo_root, now)

    state = workflow.get("state")
    if state != "active":
        return _result(cadence, "CONFIG", f"workflow state is {state!r}, not 'active'")
    workflow_id = workflow.get("id")
    if not isinstance(workflow_id, int) or isinstance(workflow_id, bool):
        raise ContractError("GitHub response omitted the workflow id")

    last_success = _latest_success(
        cadence,
        repository=repository,
        workflow_id=workflow_id,
        opener=opener,
        token=token,
        sleep=sleep,
    )
    if last_success is None:
        return _never_ran_result(cadence, now)

    age = now - last_success
    if age < -MAX_CLOCK_SKEW:
        return _result(
            cadence,
            "CONFIG",
            f"the newest success is dated {_age_text(-age)} in the future; "
            "the clock or the field being read is wrong",
        )
    detail = (
        f"last success was {_age_text(age)} ago; maximum {_age_text(cadence.max_age)}"
    )
    if age > cadence.max_age:
        # Successes are all this can see, so a cadence that fires daily and
        # fails is indistinguishable from one that stopped firing.
        return _result(
            cadence,
            "STALE",
            f"{detail} (a cadence that fires and fails looks the same here -- "
            "check the workflow's failure issue before suspecting the timer)",
        )
    return _result(cadence, "OK", detail)


def check_all(
    *,
    repository: str,
    now: datetime,
    token: str,
    opener: OpenUrl = urllib.request.urlopen,
    repo_root: Path = REPO_ROOT,
    sleep: Sleep = time.sleep,
) -> list[Result]:
    results = []
    for cadence in CADENCES:
        try:
            result = check_cadence(
                cadence,
                repository=repository,
                now=now,
                token=token,
                opener=opener,
                repo_root=repo_root,
                sleep=sleep,
            )
        except TransportError as error:
            # Weather, not a defect: loud but not red. See the module docstring.
            result = _result(
                cadence,
                "UNKNOWN",
                f"could not verify the cadence; GitHub Actions was unreachable: {error}",
            )
        except (OSError, ValueError) as error:
            result = _result(
                cadence, "CONFIG", f"could not query GitHub Actions: {error}"
            )
        # A row that raises something unclassified must not abort the loop over
        # the rest of the table; it reports the exception type and goes red.
        except Exception as error:  # noqa: BLE001
            result = _result(
                cadence,
                "CONFIG",
                f"unexpected {type(error).__name__} while querying "
                f"GitHub Actions: {error}",
            )
        results.append(result)
    return results


def _utc_now() -> datetime:
    return datetime.now(UTC)


def main(
    argv: list[str] | None = None,
    *,
    checker: Callable[..., list[Result]] = check_all,
    clock: Callable[[], datetime] = _utc_now,
    environ: Mapping[str, str] | None = None,
) -> int:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--repository", default=env.get("GITHUB_REPOSITORY") or DEFAULT_REPOSITORY
    )
    args = parser.parse_args(argv)
    if not REPOSITORY_PATTERN.fullmatch(args.repository):
        print(
            f"[CONFIG] GitHub Actions: {args.repository!r} is not an "
            "owner/name repository"
        )
        return 1
    token = env.get("GITHUB_TOKEN") or env.get("GH_TOKEN")
    if not token:
        # The repository is private, so an unauthenticated Actions request 404s
        # and would be indistinguishable from a missing workflow.
        print("[CONFIG] GitHub Actions: GITHUB_TOKEN with actions:read is required")
        return 1
    results = checker(repository=args.repository, now=clock(), token=token)
    for result in results:
        print(f"[{result.level}] {result.cadence.workflow_name}: {result.detail}")
    return int(any(result.level in RED_LEVELS for result in results))


def _literal_levels(node: ast.expr | None) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _literal_levels(node.body) | _literal_levels(node.orelse)
    raise ContractError(f"the level argument is not a string literal: {node!r}")


def declared_result_levels(source: str | None = None) -> tuple[set[str], int]:
    """Return every level literal passed to `_result`, and stray `Result` calls.

    Exposed so a test can prove the module emits nothing outside `LEVELS`, and
    that every result goes through the validating `_result` factory rather than
    round the side of it, without re-implementing the parse in the test. A
    stray is any `Result(...)` call not lexically inside `_result` -- at module
    level, in a class body, in an `async def`, in a lambda, or in a
    comprehension included, which is why this walks calls rather than
    enumerating function scopes. `source` overrides the module's own text so the
    counting branch itself can be exercised.
    """
    tree = ast.parse(Path(__file__).read_text() if source is None else source)
    factory_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_result":
            factory_nodes = {id(inner) for inner in ast.walk(node)}
            break
    levels: set[str] = set()
    stray_result_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "Result" and id(node) not in factory_nodes:
            stray_result_calls += 1
        elif node.func.id == "_result":
            levels |= _literal_levels(node.args[1] if len(node.args) > 1 else None)
    return levels, stray_result_calls


if __name__ == "__main__":
    sys.exit(main())
