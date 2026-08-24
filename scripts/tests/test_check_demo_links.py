"""Tests for the opt-in demo click-through destination audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "check_demo_links.py"
    spec = importlib.util.spec_from_file_location("check_demo_links", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_passes_only_when_good_and_bad_discriminate() -> None:
    checker = _load_script()
    spec = {
        "mode": "body-marker",
        "url_template": "https://example.org/{value}",
        "good": "known",
        "bad": "missing",
        "good_marker": "canonical record",
    }
    responses = {
        "https://example.org/known": checker.Response(
            200, "https://example.org/known", "canonical record"
        ),
        "https://example.org/missing": checker.Response(
            200, "https://example.org/missing", "not found"
        ),
    }

    result = checker.audit_destination("example.org", spec, responses.__getitem__)

    assert result.level == "OK"
    assert "good matched; bad rejected" in result.message


def test_pair_reports_moved_path_when_both_fail_identically() -> None:
    checker = _load_script()
    spec = {
        "mode": "body-marker",
        "url_template": "https://example.org/{value}",
        "good": "known",
        "bad": "missing",
        "good_marker": "canonical record",
    }
    response = checker.Response(404, "https://example.org/gone", "not found")

    result = checker.audit_destination("example.org", spec, lambda _url: response)

    assert result.level == "FAIL"
    assert "good and bad both rejected" in result.message


def test_status_mode_uses_http_success_as_the_discriminator() -> None:
    checker = _load_script()
    spec = {
        "mode": "status",
        "url_template": "https://example.org/{value}",
        "good": "known",
        "bad": "missing",
    }
    responses = {
        "https://example.org/known": checker.Response(200, "", ""),
        "https://example.org/missing": checker.Response(404, "", ""),
    }

    result = checker.audit_destination("example.org", spec, responses.__getitem__)

    assert result == checker.AuditResult("OK", "good matched; bad rejected (200/404)")


def test_redirect_mode_requires_only_the_good_key_to_reach_the_record() -> None:
    checker = _load_script()
    spec = {
        "mode": "redirect",
        "url_template": "https://example.org/search?q={value}",
        "good": "known",
        "bad": "missing",
        "good_final_marker": "/record/known",
    }
    responses = {
        "https://example.org/search?q=known": checker.Response(
            200, "https://example.org/record/known", ""
        ),
        "https://example.org/search?q=missing": checker.Response(
            200, "https://example.org/search?q=missing", ""
        ),
    }

    result = checker.audit_destination("example.org", spec, responses.__getitem__)

    assert result.level == "OK"


def test_json_count_mode_and_canonical_landing_route_are_both_required() -> None:
    checker = _load_script()
    spec = {
        "mode": "json-count",
        "landing_template": "https://example.org/page/{value}",
        "url_template": "https://api.example.org/search?q={value}",
        "good": "known",
        "bad": "missing",
        "count_path": ("response", "count"),
    }
    responses = {
        "https://example.org/page/known": checker.Response(200, "", ""),
        "https://api.example.org/search?q=known": checker.Response(
            200, "", '{"response":{"count":1}}'
        ),
        "https://api.example.org/search?q=missing": checker.Response(
            200, "", '{"response":{"count":0}}'
        ),
    }

    result = checker.audit_destination("example.org", spec, responses.__getitem__)

    assert result.level == "OK"
    responses["https://example.org/page/known"] = checker.Response(404, "", "")
    result = checker.audit_destination("example.org", spec, responses.__getitem__)
    assert result == checker.AuditResult(
        "FAIL", "canonical page route rejected the good key (404)"
    )


def test_identifiers_are_encoded_like_viewer_url_substitutions() -> None:
    checker = _load_script()

    assert checker._format_url("https://doi.org/{value}", "10.1/example:42") == (
        "https://doi.org/10.1%2Fexample%3A42"
    )


def test_fetch_rejects_non_https_urls_before_opening_them() -> None:
    checker = _load_script()

    try:
        checker.fetch_url("file:///etc/passwd")
    except ValueError as error:
        assert str(error) == "demo link audits only allow HTTPS URLs"
    else:
        raise AssertionError("non-HTTPS URL was accepted")


def test_request_outage_is_reported_without_raising() -> None:
    checker = _load_script()
    spec = {
        "mode": "status",
        "url_template": "https://example.org/{value}",
        "good": "known",
        "bad": "missing",
    }

    def unavailable(_url: str) -> checker.Response:
        raise OSError("temporary DNS failure")

    result = checker.audit_destination("example.org", spec, unavailable)

    assert result.level == "ERROR"
    assert "temporary DNS failure" in result.message


def test_human_only_destination_reports_review_state() -> None:
    checker = _load_script()
    spec = {
        "mode": "human",
        "reason": "Cloudflare returns the same response for every identifier",
        "checked_by": "@maintainer",
        "last_checked": "2026-08-24",
    }

    result = checker.audit_destination("example.org", spec, lambda _url: None)

    assert result.level == "HUMAN"
    assert result.message == (
        "Cloudflare returns the same response for every identifier; "
        "last checked by @maintainer on 2026-08-24"
    )


def test_main_reports_every_destination_and_always_returns_zero(capsys) -> None:
    checker = _load_script()
    audits = {
        "manual.example": {
            "mode": "human",
            "reason": "browser-only",
            "checked_by": "@maintainer",
            "last_checked": "never",
        },
        "broken.example": {
            "mode": "status",
            "url_template": "https://broken.example/{value}",
            "good": "known",
            "bad": "missing",
        },
    }
    response = checker.Response(404, "https://broken.example/gone", "not found")

    exit_code = checker.main(audits=audits, fetch=lambda _url: response)

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        "[FAIL]  broken.example: good and bad both rejected (404/404)",
        "[HUMAN] manual.example: browser-only; last checked by @maintainer on never",
    ]
