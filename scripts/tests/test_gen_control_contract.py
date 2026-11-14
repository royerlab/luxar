"""Guard the generated token policy against invalid saved-token lengths."""

from __future__ import annotations

import pytest

from scripts.gen_control_contract import _token_policy


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


@pytest.mark.parametrize("required", [0, "true", None])
def test_token_policy_requires_boolean_default(required: object) -> None:
    with pytest.raises(TypeError, match="token_policy.required_off_loopback"):
        _token_policy(
            {"token_policy": {"required_off_loopback": required, "bytes": 24}}
        )
