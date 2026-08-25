"""Report whether demo click-through destinations still discriminate real keys.

This is an opt-in network audit, not a CI gate. Every destination prints one
line and request failures never change the process exit status.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from luxar.demos.link_registry import (
    CANONICAL_LINKS_BY_HOST,
    DEMO_LINK_AUDITS_BY_HOST,
    LINK_PLACEHOLDERS,
)

TIMEOUT_SECONDS = 20
USER_AGENT = "LuxarDemoLinkAudit/1.0"
HUMAN_CHECK_MAX_AGE_DAYS = 183
AUDIT_MODES = frozenset({"body-marker", "human", "json-count", "redirect", "status"})
REQUIRED_FIELDS_BY_MODE = {
    "body-marker": frozenset({"good_marker"}),
    "human": frozenset({"reason", "verified_in", "last_checked"}),
    "json-count": frozenset({"count_path"}),
    "redirect": frozenset({"good_final_marker"}),
    "status": frozenset(),
}


class Response(NamedTuple):
    status: int
    url: str
    body: str


class AuditResult(NamedTuple):
    level: str
    message: str


Fetch = Callable[[str], Response]


def fetch_url(url: str) -> Response:
    if urlsplit(url).scheme != "https":
        raise ValueError("demo link audits only allow HTTPS URLs")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # nosec B310
            if urlsplit(response.url).scheme != "https":
                raise ValueError("demo link audit redirected to a non-HTTPS URL")
            return Response(
                response.status,
                response.url,
                response.read().decode("utf-8", errors="replace"),
            )
    except HTTPError as error:
        if urlsplit(error.url).scheme != "https":
            raise ValueError("demo link audit redirected to a non-HTTPS URL") from error
        return Response(
            error.code,
            error.url,
            error.read().decode("utf-8", errors="replace"),
        )
    except URLError as error:
        raise OSError(str(error.reason)) from error


def _format_url(template: str, value: str) -> str:
    # Keep in sync with hover-template.ts's encodeURIComponent substitution.
    return template.format(value=quote(value, safe="!*'()"))


def _canonical_probe_templates(templates: frozenset[str]) -> tuple[str, ...]:
    normalized = set()
    for template in templates:
        for placeholder in LINK_PLACEHOLDERS:
            template = template.replace(f"{{{placeholder}}}", "{value}")
        normalized.add(template)
    return tuple(sorted(normalized))


def _same_destination_family(host: str, probe_host: str) -> bool:
    return host == probe_host or host.split(".")[-2:] == probe_host.split(".")[-2:]


def _literal_probe_template(templates: tuple[str, ...], values: tuple[Any, ...]) -> str:
    inferred = {
        template.replace(quote(str(value), safe="!*'()"), "{value}")
        for template, value in zip(templates, values, strict=True)
    }
    if len(inferred) != 1 or "{value}" not in next(iter(inferred)):
        raise ValueError("literal canonical links must share one identifier URL shape")
    return inferred.pop()


def _validate_request_spec(
    host: str, spec: Mapping[str, Any], canonical_templates: frozenset[str] | None
) -> None:
    if "good" not in spec or "bad" not in spec:
        raise KeyError("request audit requires good and bad identifiers")
    if spec["good"] == spec["bad"]:
        raise ValueError("good and bad identifiers must differ")
    if canonical_templates is None and "url_template" not in spec:
        raise KeyError("request audit requires canonical templates or url_template")
    if override := spec.get("url_template"):
        probe_host = urlsplit(str(override)).netloc
        if not _same_destination_family(host, probe_host):
            raise ValueError(
                f"probe host {probe_host!r} does not match destination {host!r}"
            )
    if canonical_templates:
        _validate_canonical_probe_values(
            spec, _canonical_probe_templates(canonical_templates)
        )


def _validate_canonical_probe_values(
    spec: Mapping[str, Any], canonical_probes: tuple[str, ...]
) -> None:
    good = spec["good"]
    markers = spec.get("good_marker")
    if isinstance(good, (tuple, list)) and isinstance(markers, (tuple, list)):
        if len(markers) != len(good):
            raise ValueError(
                "body-marker audits require one marker per good identifier"
            )
    if any("{value}" in template for template in canonical_probes):
        return
    if not isinstance(good, (tuple, list)) or len(good) != len(canonical_probes):
        raise ValueError("literal canonical links require one good identifier per URL")
    _literal_probe_template(canonical_probes, tuple(good))


def validate_spec(
    host: str, spec: Mapping[str, Any], canonical_templates: frozenset[str] | None
) -> None:
    mode = spec["mode"]
    if mode not in AUDIT_MODES:
        raise ValueError(f"unknown audit mode: {mode}")
    missing = REQUIRED_FIELDS_BY_MODE[mode] - spec.keys()
    if missing:
        raise KeyError(f"{mode} audit missing required fields: {sorted(missing)}")
    if mode == "human":
        date.fromisoformat(str(spec["last_checked"]))
        return
    _validate_request_spec(host, spec, canonical_templates)


def _accepted(
    response: Response, spec: Mapping[str, Any], marker: str | None = None
) -> bool:
    if not 200 <= response.status < 400:
        return False
    mode = spec["mode"]
    if mode == "status":
        return True
    if mode == "body-marker":
        markers = spec["good_marker"]
        if marker is not None:
            return marker in response.body
        if isinstance(markers, (tuple, list)):
            return any(str(candidate) in response.body for candidate in markers)
        return str(markers) in response.body
    if mode == "redirect":
        return str(spec["good_final_marker"]) in response.url
    if mode == "json-count":
        value: Any = json.loads(response.body)
        for key in spec["count_path"]:
            value = value[key]
        if not spec["count_path"]:
            value = len(value)
        return int(value) > 0
    raise ValueError(f"unknown audit mode: {mode}")


def _human_result(spec: Mapping[str, Any], today: date | None) -> AuditResult:
    checked = date.fromisoformat(str(spec["last_checked"]))
    age_days = ((today or date.today()) - checked).days
    level = "STALE" if age_days > HUMAN_CHECK_MAX_AGE_DAYS else "HUMAN"
    age_unit = "day" if age_days == 1 else "days"
    return AuditResult(
        level,
        (
            f"{spec['reason']}; verified in {spec['verified_in']} on {checked} "
            f"({age_days} {age_unit} ago)"
        ),
    )


def _request_urls(
    spec: Mapping[str, Any], canonical_probes: tuple[str, ...]
) -> tuple[list[str], str]:
    probe_template = (
        str(spec["url_template"]) if "url_template" in spec else canonical_probes[0]
    )
    if "{value}" not in probe_template:
        bad_template = _literal_probe_template(canonical_probes, tuple(spec["good"]))
        return list(canonical_probes), _format_url(bad_template, str(spec["bad"]))
    good_values = spec["good"]
    if not isinstance(good_values, (tuple, list)):
        good_values = (good_values,)
    good_urls = [_format_url(probe_template, str(value)) for value in good_values]
    return good_urls, _format_url(probe_template, str(spec["bad"]))


def _landing_failure(
    spec: Mapping[str, Any], canonical_probes: tuple[str, ...], fetch: Fetch
) -> AuditResult | None:
    landing_templates = canonical_probes
    if explicit_landing := spec.get("landing_template"):
        landing_templates = (str(explicit_landing),)
    if "url_template" not in spec or not landing_templates:
        return None
    landing_values = spec["good"]
    if isinstance(landing_values, (tuple, list)):
        landing_values = landing_values[:1]
    else:
        landing_values = (landing_values,)
    for landing_template in landing_templates:
        for value in landing_values:
            landing_url = (
                _format_url(landing_template, str(value))
                if "{value}" in landing_template
                else landing_template
            )
            landing = fetch(landing_url)
            if not 200 <= landing.status < 400:
                return AuditResult(
                    "FAIL",
                    f"canonical page route rejected the good key ({landing.status})",
                )
    return None


def _good_responses_accepted(
    responses: list[Response], spec: Mapping[str, Any]
) -> bool:
    markers = spec.get("good_marker")
    if isinstance(markers, (tuple, list)):
        return all(
            _accepted(response, spec, str(markers[index]))
            for index, response in enumerate(responses)
        )
    return all(_accepted(response, spec) for response in responses)


def audit_destination(
    host: str,
    spec: Mapping[str, Any],
    fetch: Fetch,
    canonical_templates: frozenset[str] | None = None,
    *,
    today: date | None = None,
) -> AuditResult:
    try:
        validate_spec(host, spec, canonical_templates)
        if spec["mode"] == "human":
            return _human_result(spec, today)

        canonical_probes = (
            _canonical_probe_templates(canonical_templates)
            if canonical_templates
            else ()
        )
        good_urls, bad_url = _request_urls(spec, canonical_probes)
        if landing_failure := _landing_failure(spec, canonical_probes, fetch):
            return landing_failure

        good = [fetch(url) for url in good_urls]
        bad = fetch(bad_url)
        good_accepted = _good_responses_accepted(good, spec)
        bad_accepted = _accepted(bad, spec)
    except OSError as error:
        return AuditResult("ERROR", str(error))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return AuditResult("CONFIG", str(error))

    statuses = f"{','.join(str(response.status) for response in good)}/{bad.status}"
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
    today: date | None = None,
) -> int:
    for host in sorted(audits):
        result = audit_destination(
            host,
            audits[host],
            fetch,
            CANONICAL_LINKS_BY_HOST.get(host),
            today=today,
        )
        print(f"[{result.level}]".ljust(8) + f"{host}: {result.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
