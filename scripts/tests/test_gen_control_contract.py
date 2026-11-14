"""Guard the generated token policy against invalid saved-token lengths."""

from __future__ import annotations

import importlib.util
import secrets
from pathlib import Path

import pytest

from luxar.cli import _export_serve_template as template

_SCRIPT = Path(__file__).resolve().parents[1] / "gen_control_contract.py"
_SPEC = importlib.util.spec_from_file_location("gen_control_contract", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_token_policy = _MODULE._token_policy


@pytest.mark.parametrize("token_bytes", [6, 24, 96])
def test_token_policy_accepts_bytes_that_fit_saved_token_format(
    token_bytes: int,
) -> None:
    assert _token_policy(
        {"token_policy": {"required_off_loopback": True, "bytes": token_bytes}}
    ) == (True, token_bytes)


@pytest.mark.parametrize("token_bytes", [0, 5, 97, True, "24"])
def test_token_policy_rejects_bytes_outside_saved_token_format(
    token_bytes: object,
) -> None:
    with pytest.raises(TypeError, match="token_policy.bytes"):
        _token_policy(
            {"token_policy": {"required_off_loopback": True, "bytes": token_bytes}}
        )


@pytest.mark.parametrize("required", [False, 0, "true", None])
def test_token_policy_requires_enabled_default(required: object) -> None:
    with pytest.raises(TypeError, match="token_policy.required_off_loopback"):
        _token_policy(
            {"token_policy": {"required_off_loopback": required, "bytes": 24}}
        )


@pytest.mark.parametrize(
    "token_bytes, accepted", [(5, False), (6, True), (96, True), (97, False)]
)
def test_token_byte_bounds_match_saved_token_pattern(
    token_bytes: int, accepted: bool
) -> None:
    token = secrets.token_urlsafe(token_bytes)
    assert (template._TOKEN_PATTERN.fullmatch(token) is not None) is accepted
