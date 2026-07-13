"""The CUSTOM-encoder dispatch table must stay within the format contract.

`StructuralEncoderMixin._CUSTOM_DISPATCH` is the registry that replaced the
CUSTOM-mode string if/elif. Every encoding name it can emit must be a member of
the single-sourced cross-language contract (``format-contract/contract.yaml``),
so a name the Python writer produces is always one the TypeScript decoder
recognises — closing the encoder-emits ⊆ contract half of the #10 loop.
"""

from __future__ import annotations

from luxar.encoding._encoders.structural import StructuralEncoderMixin
from luxar.typing_utils._format_contract import ENCODING_NAMES


def test_custom_dispatch_names_are_contract_encodings() -> None:
    dispatch_names = set(StructuralEncoderMixin._CUSTOM_DISPATCH)
    unknown = dispatch_names - set(ENCODING_NAMES)
    assert not unknown, (
        f"CUSTOM dispatch emits encoding names absent from the format contract: "
        f"{sorted(unknown)}. Add them to format-contract/contract.yaml (and run "
        f"`make gen-contract`) or fix the dispatch table."
    )


def test_every_custom_handler_resolves() -> None:
    """Each dispatch target must name a real bound method on the mixin."""
    for encoder_name, handler in StructuralEncoderMixin._CUSTOM_DISPATCH.items():
        assert callable(getattr(StructuralEncoderMixin, handler, None)), (
            f"custom dispatch for {encoder_name!r} points at missing handler "
            f"{handler!r}"
        )
