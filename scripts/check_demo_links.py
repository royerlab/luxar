"""Report whether demo click-through destinations still discriminate real keys.

This is an opt-in network audit, not a CI gate. Every destination prints one
line and request failures never change the process exit status.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from luxar.demos.tests.test_demo_link_templates import DEMO_LINK_AUDITS_BY_HOST

TIMEOUT_SECONDS = 20
USER_AGENT = "LuxarDemoLinkAudit/1.0"


class Response(NamedTuple):
    status: int
    url: str
    body: str


class AuditResult(NamedTuple):
    level: str
    message: str


Fetch = Callable[[str], Response]


def fetch_url(url: str) -> Response:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            return Response(
                response.status,
                response.url,
                response.read().decode("utf-8", errors="replace"),
            )
    except HTTPError as error:
        return Response(
            error.code,
            error.url,
            error.read().decode("utf-8", errors="replace"),
        )
    except URLError as error:
        raise OSError(str(error.reason)) from error


def _format_url(template: str, value: str) -> str:
    return template.format(value=quote(value, safe=""))


def _accepted(response: Response, spec: Mapping[str, Any]) -> bool:
    if not 200 <= response.status < 400:
        return False
    mode = spec["mode"]
    if mode == "status":
        return True
    if mode == "body-marker":
        return str(spec["good_marker"]) in response.body
    if mode == "redirect":
        return str(spec["good_final_marker"]) in response.url
    if mode == "json-count":
        value: Any = json.loads(response.body)
        for key in spec["count_path"]:
            value = value[key]
        return int(value) > 0
    raise ValueError(f"unknown audit mode: {mode}")


def audit_destination(host: str, spec: Mapping[str, Any], fetch: Fetch) -> AuditResult:
    if spec["mode"] == "human":
        return AuditResult(
            "HUMAN", f"{spec['reason']}; last checked {spec['last_checked']}"
        )

    good_url = _format_url(str(spec["url_template"]), str(spec["good"]))
    bad_url = _format_url(str(spec["url_template"]), str(spec["bad"]))
    try:
        if landing_template := spec.get("landing_template"):
            landing = fetch(_format_url(str(landing_template), str(spec["good"])))
            if not 200 <= landing.status < 400:
                return AuditResult(
                    "FAIL",
                    f"canonical page route rejected the good key ({landing.status})",
                )
        good = fetch(good_url)
        bad = fetch(bad_url)
        good_accepted = _accepted(good, spec)
        bad_accepted = _accepted(bad, spec)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return AuditResult("ERROR", str(error))

    statuses = f"{good.status}/{bad.status}"
    if good_accepted and not bad_accepted:
        return AuditResult("OK", f"good matched; bad rejected ({statuses})")
    if not good_accepted and not bad_accepted:
        return AuditResult("FAIL", f"good and bad both rejected ({statuses})")
    if good_accepted and bad_accepted:
        return AuditResult("FAIL", f"good and bad both accepted ({statuses})")
    return AuditResult("FAIL", f"good rejected while bad accepted ({statuses})")


def main(
    *,
    audits: Mapping[str, Mapping[str, Any]] = DEMO_LINK_AUDITS_BY_HOST,
    fetch: Fetch = fetch_url,
) -> int:
    for host in sorted(audits):
        result = audit_destination(host, audits[host], fetch)
        print(f"[{result.level}]".ljust(8) + f"{host}: {result.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
