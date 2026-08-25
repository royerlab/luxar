"""Tests for the opt-in demo click-through destination audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from luxar.demos.link_registry import CANONICAL_LINKS_BY_HOST, DEMO_LINK_AUDITS_BY_HOST


def _load_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "check_demo_links.py"
    spec = importlib.util.spec_from_file_location("check_demo_links", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load_script()


def test_pair_passes_only_when_good_and_bad_discriminate() -> None:
    checker = CHECKER
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
    checker = CHECKER
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
    checker = CHECKER
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
    checker = CHECKER
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
    checker = CHECKER
    spec = {
        "mode": "json-count",
        "url_template": "https://api.example.org/search?q={value}",
        "good": "known",
        "bad": "missing",
        "count_path": ("response", "count"),
    }
    canonical = frozenset({"https://example.org/page/{hover_key}"})
    responses = {
        "https://example.org/page/known": checker.Response(200, "", ""),
        "https://api.example.org/search?q=known": checker.Response(
            200, "", '{"response":{"count":1}}'
        ),
        "https://api.example.org/search?q=missing": checker.Response(
            200, "", '{"response":{"count":0}}'
        ),
    }

    result = checker.audit_destination(
        "example.org", spec, responses.__getitem__, canonical
    )

    assert result.level == "OK"
    responses["https://example.org/page/known"] = checker.Response(404, "", "")
    result = checker.audit_destination(
        "example.org", spec, responses.__getitem__, canonical
    )
    assert result == checker.AuditResult(
        "FAIL", "canonical page route rejected the good key (404)"
    )


def test_identifiers_are_encoded_like_viewer_url_substitutions() -> None:
    checker = CHECKER

    assert checker._format_url("https://doi.org/{value}", "10.1/example!'()") == (
        "https://doi.org/10.1%2Fexample!'()"
    )


def test_fetch_rejects_non_https_urls_before_opening_them() -> None:
    checker = CHECKER

    try:
        checker.fetch_url("file:///etc/passwd")
    except ValueError as error:
        assert str(error) == "demo link audits only allow HTTPS URLs"
    else:
        raise AssertionError("non-HTTPS URL was accepted")


def test_request_outage_is_reported_without_raising() -> None:
    checker = CHECKER
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
    checker = CHECKER
    spec = {
        "mode": "human",
        "reason": "Cloudflare returns the same response for every identifier",
        "verified_in": "#2091",
        "last_checked": "2026-08-24",
    }

    result = checker.audit_destination(
        "example.org", spec, lambda _url: None, today=checker.date(2026, 8, 25)
    )

    assert result.level == "HUMAN"
    assert result.message == (
        "Cloudflare returns the same response for every identifier; "
        "verified in #2091 on 2026-08-24 (1 day ago)"
    )


def test_human_only_destination_becomes_stale_after_six_months() -> None:
    checker = CHECKER
    spec = {
        "mode": "human",
        "reason": "browser-only",
        "verified_in": "#2091",
        "last_checked": "2026-01-01",
    }

    result = checker.audit_destination(
        "example.org", spec, lambda _url: None, today=checker.date(2026, 8, 25)
    )

    assert result.level == "STALE"
    assert "236 days ago" in result.message


def test_registry_specs_are_complete_and_use_canonical_landing_routes() -> None:
    checker = CHECKER
    assert DEMO_LINK_AUDITS_BY_HOST.keys() == CANONICAL_LINKS_BY_HOST.keys()
    for host, spec in DEMO_LINK_AUDITS_BY_HOST.items():
        checker.validate_spec(host, spec, CANONICAL_LINKS_BY_HOST[host])


def test_api_override_cannot_probe_an_unrelated_host() -> None:
    checker = CHECKER
    spec = {
        "mode": "status",
        "url_template": "https://unrelated.example/api/{value}",
        "good": "known",
        "bad": "missing",
    }

    result = checker.audit_destination("example.org", spec, lambda _url: None)

    assert result == checker.AuditResult(
        "CONFIG",
        "probe host 'unrelated.example' does not match destination 'example.org'",
    )


def test_api_override_rejects_an_unrelated_two_label_suffix() -> None:
    checker = CHECKER
    four_label_spec = {
        "mode": "status",
        "url_template": "https://www.ox.ac.uk/api/{value}",
        "good": "known",
        "bad": "missing",
    }
    two_label_spec = {
        "mode": "status",
        "url_template": "https://evil.org/api/{value}",
        "good": "known",
        "bad": "missing",
    }

    four_label_result = checker.audit_destination(
        "www.ebi.ac.uk", four_label_spec, lambda _url: None
    )
    two_label_result = checker.audit_destination(
        "doi.org", two_label_spec, lambda _url: None
    )

    assert four_label_result == checker.AuditResult(
        "CONFIG",
        "probe host 'www.ox.ac.uk' does not match destination 'www.ebi.ac.uk'",
    )
    assert two_label_result == checker.AuditResult(
        "CONFIG", "probe host 'evil.org' does not match destination 'doi.org'"
    )


def test_dotless_destination_host_reports_invalid_override_as_config() -> None:
    checker = CHECKER
    spec = {
        "mode": "status",
        "url_template": "https://other/api/{value}",
        "good": "known",
        "bad": "missing",
    }

    result = checker.audit_destination("localhost", spec, lambda _url: None)

    assert result == checker.AuditResult(
        "CONFIG", "probe host 'other' does not match destination 'localhost'"
    )


def test_empty_json_count_path_counts_top_level_results() -> None:
    checker = CHECKER
    spec = {"mode": "json-count", "count_path": ()}

    assert checker._accepted(checker.Response(200, "", '[{"gene": "TP53"}]'), spec)
    assert not checker._accepted(checker.Response(200, "", "[]"), spec)


def test_malformed_and_unknown_specs_report_config_without_raising() -> None:
    checker = CHECKER

    missing = checker.audit_destination(
        "example.org", {"mode": "status"}, lambda _: None
    )
    unknown = checker.audit_destination(
        "example.org", {"mode": "stauts"}, lambda _: None
    )

    assert missing.level == "CONFIG"
    assert unknown == checker.AuditResult("CONFIG", "unknown audit mode: stauts")


def test_body_marker_lists_require_one_marker_per_good_identifier() -> None:
    checker = CHECKER
    spec = {
        "mode": "body-marker",
        "good": ("one", "two"),
        "bad": "missing",
        "good_marker": ("record one",),
    }
    canonical = frozenset({"https://example.org/one", "https://example.org/two"})

    result = checker.audit_destination(
        "example.org", spec, lambda _url: None, canonical
    )

    assert result == checker.AuditResult(
        "CONFIG", "body-marker audits require one marker per good identifier"
    )


def test_literal_canonical_urls_are_all_probed() -> None:
    checker = CHECKER
    host = "simbad.cds.unistra.fr"
    spec = DEMO_LINK_AUDITS_BY_HOST[host]
    requested: list[str] = []

    def fetch(url: str) -> checker.Response:
        requested.append(url)
        star = url.rsplit("=", 1)[-1]
        body = "not found" if "LUXAR_NO_SUCH" in url else f"<h1>{star}</h1>"
        return checker.Response(200, url, body)

    result = checker.audit_destination(host, spec, fetch, CANONICAL_LINKS_BY_HOST[host])

    assert result.level == "OK"
    assert set(requested[:-1]) == CANONICAL_LINKS_BY_HOST[host]


def test_fetch_rejects_https_redirect_to_http(monkeypatch) -> None:
    checker = CHECKER

    class RedirectedResponse:
        status = 200
        url = "http://example.org/record"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b"record"

    monkeypatch.setattr(
        checker, "urlopen", lambda *_args, **_kwargs: RedirectedResponse()
    )

    try:
        checker.fetch_url("https://example.org/record")
    except ValueError as error:
        assert str(error) == "demo link audit redirected to a non-HTTPS URL"
    else:
        raise AssertionError("HTTPS to HTTP redirect was accepted")


def test_main_reports_every_destination_and_always_returns_zero(capsys) -> None:
    checker = CHECKER
    audits = {
        "manual.example": {
            "mode": "human",
            "reason": "browser-only",
            "verified_in": "#2091",
            "last_checked": "2026-08-24",
        },
        "broken.example": {
            "mode": "status",
            "url_template": "https://broken.example/{value}",
            "good": "known",
            "bad": "missing",
        },
    }
    response = checker.Response(404, "https://broken.example/gone", "not found")

    exit_code = checker.main(
        audits=audits,
        fetch=lambda _url: response,
        today=checker.date(2026, 8, 25),
    )

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        "[FAIL]  broken.example: good and bad both rejected (404/404)",
        "[HUMAN] manual.example: browser-only; verified in #2091 on 2026-08-24 "
        "(1 day ago)",
    ]
