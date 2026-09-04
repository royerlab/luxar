"""No document may name an on-disk encoding that does not exist.

Luxar's encoding vocabulary carries two bit-width suffix conventions that are
NOT interchangeable: the per-channel family writes ``_uN``, the scalar / bounded
/ lut / rgb families write ``_uintN``. ``format-contract/contract.yaml`` says so
in a comment, and the split is deliberate -- both spellings are live in
published stores, so neither can be renamed without invalidating every one of
them.

What the split costs is that the wrong sibling of a real name reads exactly as
plausibly as the real one, and is not a name at all. The confusion has a
specific source: the WASM/TS decode kernels are named with the short form
uniformly, because that is the Rust convention, while the value those kernels
read off disk may use the long one. An author reading the kernel writes the
kernel's spelling into prose, and the next author copies the prose into an
f-string.

So this gate derives, for every quantized name in the contract, its
wrong-convention sibling, and asserts no document uses one. It deliberately
covers prose, because prose is upstream of the bug: no runtime test can catch an
encoding name that so far only appears in a docstring, and by the time it
reaches an f-string it is writing stores no decoder can read.

Companion to ``test_emitted_names_contract`` (every name the encoder emits at
RUNTIME is in the contract) and ``format-contract.test.ts`` (the decoder
recognises every contract name). This is the third side: every name any document
CLAIMS.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from luxar.typing_utils._format_contract import ENCODING_NAMES

REPO = Path(__file__).resolve().parents[6]

SEARCH_ROOTS = (
    "packages/luxar/src/luxar",
    "packages/luxar-viewer/src",
    "docs",
    "format-contract",
)

DOC_SUFFIXES = frozenset({".py", ".ts", ".rs", ".md", ".rst", ".yaml"})

#: Markdown/rst/yaml are prose throughout. In a source file, only a BACKTICKED
#: token is a claim about the vocabulary -- a bare one is an ordinary
#: identifier, and `rgb_u8 = self._lod(...)` is a local variable in a gsplat
#: test, not an encoding name. Flagging it would be a false RED.
PROSE_SUFFIXES = frozenset({".md", ".rst", ".yaml"})
BACKTICKED = re.compile(r"``?([^`\n]+?)``?")

#: `decode_<family>_u8` / `encode_...` are KERNEL FUNCTION names, which use the
#: short form for every family by Rust convention. Matched by prefix so a new
#: kernel needs no allowlist entry.
KERNEL_PREFIXES = ("decode_", "encode_")

#: This module must name the wrong spellings in order to explain them.
SELF = Path(__file__).resolve()


def wrong_convention_siblings() -> dict[str, str]:
    """Map each non-existent sibling spelling to the real contract name.

    Derived from the contract, so a new encoding is covered the moment it is
    added -- there is no list here to forget to update.
    """
    contract = set(ENCODING_NAMES)
    siblings: dict[str, str] = {}
    for name in contract:
        if m := re.fullmatch(r"(.+)_uint(8|16)", name):
            siblings[f"{m.group(1)}_u{m.group(2)}"] = name
        elif m := re.fullmatch(r"(.+)_u(8|16)", name):
            siblings[f"{m.group(1)}_uint{m.group(2)}"] = name
    # Both conventions genuinely exist across different families, so a
    # "sibling" that is itself a real name is not a violation.
    return {wrong: real for wrong, real in siblings.items() if wrong not in contract}


def _documents() -> list[Path]:
    """Every file in the search roots that could name an encoding."""
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        found.extend(
            p
            for p in (REPO / root).rglob("*")
            if p.is_file()
            and p.suffix in DOC_SUFFIXES
            and "node_modules" not in p.parts
            and p.resolve() != SELF
        )
    return found


def _violations_in(path: Path, siblings: dict[str, str]) -> list[tuple[int, str, str]]:
    """``(lineno, wrong_token, real_name)`` for each bad claim in ``path``."""
    prose = path.suffix in PROSE_SUFFIXES
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
    ):
        haystacks = [line] if prose else [m.group(1) for m in BACKTICKED.finditer(line)]
        for haystack in haystacks:
            for wrong, real in siblings.items():
                # `..._u{bits}` stands for both widths, so check the template
                # form too -- that is how the encoder builds these names.
                stem = re.sub(r"(8|16)$", "", wrong)
                for pattern in (
                    rf"\b\w*{re.escape(wrong)}\b",
                    rf"\b\w*{re.escape(stem)}\{{bits\}}",
                ):
                    for m in re.finditer(pattern, haystack):
                        token = m.group(0)
                        if token.startswith(KERNEL_PREFIXES):
                            continue
                        if token not in (wrong, f"{stem}{{bits}}"):
                            continue  # part of a longer identifier
                        out.append((lineno, token, real))
    return out


def test_the_scan_reached_the_documents() -> None:
    """Fail closed: an empty scan would make the real assertion vacuous."""
    docs = _documents()
    assert len(docs) > 500, f"only found {len(docs)} documents under {REPO}"
    siblings = wrong_convention_siblings()
    assert len(siblings) >= 15, (
        f"derived only {len(siblings)} sibling spellings from "
        f"{len(ENCODING_NAMES)} contract names — the derivation is broken"
    )


def test_no_document_names_a_nonexistent_encoding() -> None:
    siblings = wrong_convention_siblings()
    violations = [
        (path, lineno, token, real)
        for path in _documents()
        for lineno, token, real in _violations_in(path, siblings)
    ]
    if not violations:
        return
    lines = "\n".join(
        f"  {p.relative_to(REPO)}:{n}  says '{tok}', but the on-disk name is '{real}'"
        for p, n, tok, real in sorted(violations, key=lambda v: (str(v[0]), v[1]))
    )
    pytest.fail(
        f"{len(violations)} document(s) name an on-disk encoding that is not in "
        f"format-contract/contract.yaml:\n{lines}\n\n"
        "The two suffix conventions are NOT interchangeable: the per-channel "
        "family is `_uN`, the scalar/bounded/lut/rgb families are `_uintN`. "
        "If you are naming a decode KERNEL rather than an on-disk encoding, "
        "prefix it `decode_`/`encode_`."
    )
