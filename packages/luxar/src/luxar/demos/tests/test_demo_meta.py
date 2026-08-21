"""Enforcement + registry tests for DEMO_META.

Every demo script must carry a valid DEMO_META literal — these tests are the
single durable gate (the block generator that seeded them was a one-off).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import textwrap
import unicodedata
from collections.abc import Mapping
from pathlib import Path

import pytest

from luxar.demos import registry
from luxar.demos.registry import (
    DemoMetaError,
    extract_demo_meta,
    get_demo,
    inventory_caches,
    iter_demos,
)

DEMO_PATHS = sorted(registry._DEMOS_DIR.glob("demo_*.py"))
MANIFEST_PATH = registry._DEMOS_DIR.parents[4] / "scripts" / "gallery" / "manifest.json"


def test_demos_exist() -> None:
    assert len(DEMO_PATHS) >= 70, "demo files disappeared?"


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.name)
def test_every_demo_has_valid_meta(path: Path) -> None:
    """Schema-validates DEMO_META in every demo file (AST, no import)."""
    meta = extract_demo_meta(path)  # raises DemoMetaError on any violation
    assert meta["key"]


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.name)
def test_every_demo_compiles(path: Path) -> None:
    """The file must compile to bytecode — not merely ast.parse.

    Inserting DEMO_META ahead of a ``from __future__`` import parses fine but
    is a hard SyntaxError at compile/import time; compiling catches it.
    """
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.name)
def test_every_demo_supports_no_serve(path: Path) -> None:
    """Every demo must honor ``--no-serve`` (generate, don't block on a server).

    ``luxar demo run-all`` and ``make run-demos`` pass ``--no-serve`` to every
    demo; one demo that ignores it blocks the whole batch on a serving viewer
    (this bit us: demo_nd_transforms launched the viewer unconditionally).
    Source-level check — the flag must be consulted either literally
    (``"--no-serve" in sys.argv``) or via the ``parse_demo_flags()`` helper.
    """
    source = path.read_text(encoding="utf-8")
    assert "--no-serve" in source or "parse_demo_flags" in source, (
        f"{path.name} never consults --no-serve; `luxar demo run-all` would "
        "hang on it. Gate serving on '--no-serve' in sys.argv (see "
        "demo_lorenz) or use parse_demo_flags()."
    )


def test_keys_unique_and_resolvable() -> None:
    demos = iter_demos(refresh=True)
    assert len(demos) == len(DEMO_PATHS)
    keys = [d.key for d in demos]
    assert len(set(keys)) == len(keys)
    for demo in demos:
        assert get_demo(demo.key) == demo
        assert get_demo(str(demo.index)) == demo


def test_output_stems_unique_across_registry() -> None:
    """No two demos may resolve an output to the same scene path.

    Outputs land side by side in ``datasets/demos/``, so a shared path means two
    demos overwrite each other's scene and ``luxar demo`` STATUS reports one as
    built because the other ran (issue #1363: the Kaggle arXiv demo listed
    ``arxiv_papers`` alongside ``arxiv_papers_kaggle``, colliding with the
    Semantic Scholar demo's only output). Compared on RESOLVED paths, not raw
    stems, because ``demo_output_paths`` passes a ``.zarr``-suffixed stem through
    verbatim — ``"foo"`` and ``"foo.luxar.zarr"`` are two spellings of one file.
    """
    dummy_dir = Path("/nonexistent-demos-dir")  # resolve only; touch nothing
    owners: dict[str, list[str]] = {}
    for demo in iter_demos():
        for path in registry.demo_output_paths(demo, demos_dir=dummy_dir):
            owners.setdefault(path.name, []).append(demo.key)
    shared = {name: keys for name, keys in owners.items() if len(keys) > 1}
    assert not shared, f"output scenes claimed by multiple demos: {shared}"


def test_descriptions_unique_and_non_placeholder() -> None:
    """Field-quality guard: descriptions must be real, distinct one-liners.

    The schema validator enforces non-empty single-line strings; this pins
    the softer qualities nothing else guards — no copy-pasted descriptions
    and no placeholder text surviving into the registry/CLI table.
    """
    demos = iter_demos()
    descriptions = [d.description for d in demos]
    dupes = {x for x in descriptions if descriptions.count(x) > 1}
    assert not dupes, f"duplicated demo descriptions: {sorted(dupes)}"
    placeholder = ("todo", "tbd", "fixme", "placeholder", "a demo.")
    offenders = [
        d.key
        for d in demos
        if d.description.strip().lower() in placeholder
        or any(p in d.description.lower() for p in ("todo:", "fixme:"))
    ]
    assert not offenders, f"placeholder descriptions: {offenders}"


def test_get_demo_suggests_close_matches() -> None:
    with pytest.raises(KeyError, match="unknown demo"):
        get_demo("lorentz-attractor-oops")
    with pytest.raises(KeyError, match="out of range"):
        get_demo("99999")


def test_iter_demos_does_not_import_demo_modules() -> None:
    """The demo table must come from AST extraction, never module imports.

    Asserts the MECHANISM, not a stopwatch. Importing the demo modules is the
    thing that would actually make the table slow, and ``sys.modules`` detects
    that exactly, under any load.

    Both timing proxies were tried and both proved unusable in parallel CI. A
    wall-clock budget measures the machine, not the work (it failed at 1.48s and
    at 1.01s — the latter by 10ms). CPU time is *less* load-sensitive but not
    immune: the identical work cost 0.13s on an idle box and 1.14s under 16
    competing workers, because cache and memory-bandwidth contention inflate the
    cycle count. (A parent-side ``time.process_time()`` bound was also tried,
    and was worse than useless: it excludes the child, where all the work
    happens.) The only remaining time bound is the subprocess timeout below — a
    catastrophic-regression backstop in the spirit of the 60s bounds in
    test_substitutive.py.

    A SUBPROCESS, because an in-process ``sys.modules`` diff passes vacuously:
    ``test_all_demos_import.py`` collects earlier in a serial run and imports
    every demo module, so the "before" snapshot already contains everything the
    regression would import and the difference is empty no matter what
    ``iter_demos`` does. Only a fresh interpreter guarantees a clean baseline.
    """
    probe = textwrap.dedent(
        """
        import sys
        from luxar.demos.registry import iter_demos
        iter_demos(refresh=True)
        print(",".join(sorted(m for m in sys.modules
                              if m.startswith("luxar.demos.demo_"))))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"probe failed: {proc.stderr[-2000:]}"

    imported = [m for m in proc.stdout.strip().split(",") if m]
    assert not imported, (
        f"iter_demos imported {len(imported)} demo module(s); the table must "
        f"come from AST extraction: {imported[:5]}"
    )


def test_cache_root_matches_utils_demos() -> None:
    """registry.DEMO_CACHE_ROOT duplicates utils.demos' constant (import-weight);
    they must never diverge."""
    from luxar.utils.demos import _DEFAULT_CACHE_ROOT

    assert registry.DEMO_CACHE_ROOT == _DEFAULT_CACHE_ROOT


def _manifest_citation_problems(
    script: str, entry: Mapping[str, object], citation: Mapping[str, object] | None
) -> list[str]:
    """Complaints about one gallery entry's ``citation`` field.

    The manifest carries the credit so a tile's attribution is reviewable in the
    repo instead of only on the rendered page, which only works if it cannot
    drift: it must be the demo's own ``citation["short"]``, character for
    character.

    Absence is meaningful in BOTH directions. A demo with no real credit must
    leave the key off — inventing one would put an unsourced attribution in front
    of readers — and a demo that HAS one must not have the manifest drop it, which
    is how a tile silently loses its credit. Note that "no key" here means
    "unknown or not yet recorded", not "nothing to credit"; that distinction is
    the reason ``DEMO_META`` spells ``None`` out (see luxar/core/citation.py).
    """
    declared = citation["short"] if citation else None
    present = entry.get("citation")
    if declared is None:
        if present is not None:
            return [
                f"{script}: manifest declares citation {present!r} but the demo "
                "declares no credit; a tile must not invent an attribution"
            ]
        return []
    if present is None:
        return [
            f"{script}: demo declares citation {declared!r} but the manifest "
            "entry has none; the tile would drop the credit"
        ]
    if present != declared:
        return [
            f"{script}: manifest citation {present!r} != demo citation.short "
            f"{declared!r}"
        ]
    return []


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="gallery manifest not in tree")
def test_meta_cross_validates_against_gallery_manifest() -> None:
    """For every gallery entry with a script: key == id, geometry/category
    match, the manifest dataset stem is among the demo's outputs, and the
    manifest's credit is the demo's own ``citation["short"]`` (or absent)."""
    manifest = json.loads(MANIFEST_PATH.read_text())["demos"]
    by_path = {d.path.name: d for d in iter_demos()}

    problems: list[str] = []
    for entry in manifest:
        script = entry.get("script")
        if not script:
            continue
        demo = by_path.get(script)
        if demo is None:
            problems.append(f"{script}: in manifest but not on disk")
            continue
        if demo.key != entry["id"]:
            problems.append(
                f"{script}: key {demo.key!r} != manifest id {entry['id']!r}"
            )
        if demo.geometry != entry["geometry"]:
            problems.append(
                f"{script}: geometry {demo.geometry!r} != manifest {entry['geometry']!r}"
            )
        if demo.category != entry["category"]:
            problems.append(
                f"{script}: category {demo.category!r} != manifest {entry['category']!r}"
            )
        stem = Path(entry["dataset"]).name.removesuffix(".luxar.zarr")
        if stem not in demo.outputs:
            problems.append(
                f"{script}: manifest dataset stem {stem!r} not in outputs {demo.outputs}"
            )
        problems += _manifest_citation_problems(script, entry, demo.citation)
    assert not problems, "\n".join(problems)


def test_manifest_citation_check_catches_drift() -> None:
    """The manifest credit rule must fail on each way a tile's credit can go
    wrong — a dropped credit, an invented one, and a mismatched one."""
    real = {"short": "Kim et al. 2024"}
    assert not _manifest_citation_problems(
        "d.py", {"citation": "Kim et al. 2024"}, real
    )
    assert not _manifest_citation_problems("d.py", {}, None)

    assert _manifest_citation_problems("d.py", {}, real)  # credit dropped
    assert _manifest_citation_problems("d.py", {"citation": "Kim et al. 2024"}, None)
    assert _manifest_citation_problems("d.py", {"citation": "Kim et al. 2023"}, real)
    # Whitespace/spelling drift is drift: the field is a verbatim copy.
    assert _manifest_citation_problems("d.py", {"citation": "Kim et al.  2024"}, real)


def test_validate_meta_rejects_bad_blocks(tmp_path: Path) -> None:
    good = {
        "key": "x-demo",
        "title": "X",
        "description": "A demo.",
        "category": "synthetic",
        "geometry": "points",
        "requirements": {
            "download_mb": 0,
            "compute": "light",
            "gpu": "none",
            "local_data": None,
        },
        "caches": [],
        "outputs": ["x"],
    }
    registry.validate_meta(good, tmp_path / "demo_x.py")

    def variant(**changes):
        m = {**good, **changes}
        if "requirements" in changes and isinstance(changes["requirements"], dict):
            m["requirements"] = {**good["requirements"], **changes["requirements"]}
        return m

    bad_cases = [
        variant(key="Bad Key"),
        variant(category="cooking"),
        variant(geometry="voxels"),
        variant(description="two\nlines"),
        variant(requirements={"compute": "instant"}),
        variant(requirements={"gpu": "maybe"}),
        variant(requirements={"download_mb": -1}),
        variant(requirements={"local_data": "usb-stick"}),
        variant(caches=[1]),
        # Path containment: caches/outputs names are joined onto roots that
        # `demo cache clear` rmtree's — separators / '..' / absolute paths
        # must be rejected.
        variant(caches=["../escape"]),
        variant(caches=["/absolute"]),
        variant(outputs=["a/b"]),
        variant(outputs=[".."]),
        {k: v for k, v in good.items() if k != "outputs"},
        # `citation` is optional, but a present one must be usable: a tile has
        # nothing to render without `short`, and a DOI-shaped URL is the mistake
        # most likely to be pasted in by hand.
        variant(citation={"doi": "10.1000/x"}),
        variant(citation={"short": ""}),
        variant(citation={"short": "A et al. 2020", "authors": "A, B"}),
        variant(
            citation={"short": "A et al. 2020", "doi": "https://doi.org/10.1000/x"}
        ),
        variant(citation="A et al. 2020"),
    ]
    for bad in bad_cases:
        with pytest.raises(DemoMetaError):
            registry.validate_meta(bad, tmp_path / "demo_x.py")

    # Both ways of saying "credited" and "nothing to credit" are accepted, as is
    # omitting the key entirely while the corpus is still being populated.
    for ok in (
        variant(citation=None),
        variant(citation={"short": "Yeh 2022"}),
        variant(
            citation={
                "short": "Bui et al. 2013",
                "doi": "10.1016/j.cell.2013.10.055",
                "license": "CC BY 4.0",
                "url": "https://example.org/npc",
            }
        ),
    ):
        registry.validate_meta(ok, tmp_path / "demo_x.py")


def test_get_demo_rejects_malformed_numeric_tokens() -> None:
    # "--5" passes an isdigit-after-lstrip gate but is not an int — it must
    # fall through to the unknown-key path (KeyError), not raise ValueError.
    with pytest.raises(KeyError, match="unknown demo"):
        get_demo("--5")


def test_extract_rejects_missing_and_non_literal(tmp_path: Path) -> None:
    no_meta = tmp_path / "demo_none.py"
    no_meta.write_text('"""Doc."""\nX = 1\n')
    with pytest.raises(DemoMetaError, match="no top-level DEMO_META"):
        extract_demo_meta(no_meta)

    non_literal = tmp_path / "demo_expr.py"
    non_literal.write_text('"""Doc."""\nDEMO_META = {"key": "a" + "b"}\n')
    with pytest.raises(DemoMetaError, match="pure literal"):
        extract_demo_meta(non_literal)


def test_inventory_caches_maps_and_flags_orphans(tmp_path: Path) -> None:
    demos = iter_demos()
    claimed_name = next((d.caches[0] for d in demos if d.caches), None)
    assert claimed_name is not None, "no demo declares a cache namespace?"

    (tmp_path / claimed_name).mkdir()
    (tmp_path / claimed_name / "blob.bin").write_bytes(b"x" * 2048)
    (tmp_path / "totally-orphaned-dir").mkdir()

    entries = {e.path.name: e for e in inventory_caches(cache_root=tmp_path)}
    assert entries[claimed_name].demo_keys, "claimed dir reported as orphan"
    assert entries[claimed_name].size_bytes == 2048
    assert entries["totally-orphaned-dir"].demo_keys == ()


def test_output_and_cache_paths_resolve(tmp_path: Path) -> None:
    demo = get_demo("lorenz")
    outs = registry.demo_output_paths(demo, demos_dir=tmp_path)
    assert outs and outs[0] == tmp_path / "lorenz.luxar.zarr"
    dirs = registry.demo_cache_dirs(demo, cache_root=tmp_path)
    assert all(d.parent == tmp_path for d in dirs)


# --------------------------------------------------------------------------- #
# DEMO_META.caches vs the cache directory a demo actually writes
# --------------------------------------------------------------------------- #
def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Map every NAME bound to a string literal anywhere in *tree* (last wins)."""
    consts: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = node.value.value
    return consts


def _resolve_str(node: ast.expr, consts: Mapping[str, str]) -> str | None:
    """Resolve one path-chain operand to a string, or None if it is neither a
    string literal nor a name bound to one in *consts*."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    return None


def _truediv_chain(node: ast.BinOp) -> list[ast.expr]:
    """Flatten the left spine of an ``a / b / c`` chain into its operand list."""
    parts: list[ast.expr] = []
    cur: ast.expr = node
    while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
        parts.insert(0, cur.right)
        cur = cur.left
    parts.insert(0, cur)
    return parts


def _cache_dirs_written(path: Path) -> set[str]:
    """Cache subdirectory names *path* joins onto the luxar cache root.

    Matches the ``... / ".cache" / "luxar" / X`` chain every demo uses, resolving
    ``X`` through the module's own string constants (they spell it
    ``CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME``, so a literal-only
    match finds nothing).

    Deliberately AST-only and local to this test: the check has to run without
    importing 84 demo modules, and keeping it here avoids widening the registry's
    public surface for one invariant.
    """
    tree = ast.parse(path.read_text())
    consts = _string_constants(tree)

    found: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            continue
        vals = [_resolve_str(p, consts) for p in _truediv_chain(node)]
        if "luxar" not in vals:
            continue
        i = vals.index("luxar")
        seg = vals[i + 1] if i + 1 < len(vals) else None
        # A dotted segment is a FILE sitting at the cache root, not a cache
        # directory (arxiv parks `arxiv_embeddings.zip` and
        # `arxiv_metadata.json` there). `inventory_caches` walks directories
        # only, and `caches` names are joined then rmtree'd, so a filename
        # does not belong in it.
        if seg and "." not in seg:
            found.add(str(seg))
    return found


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.stem)
def test_written_cache_dirs_are_declared(path: Path) -> None:
    """A cache directory a demo writes must be declared in its ``caches``.

    ``luxar demo cache list`` and ``cache clear`` both work off ``caches``, so an
    undeclared directory is one the user can neither see nor free. Three demos had
    drifted this way and were holding ~2 GB between them (the INRIA garden capture
    alone is 1.5 GB), reported only as anonymous orphans in the cache inventory.
    """
    meta = extract_demo_meta(path)
    written = _cache_dirs_written(path)
    declared = set(meta["caches"])
    undeclared = sorted(written - declared)
    assert not undeclared, (
        f"{path.name} writes {undeclared} under the cache root but declares "
        f"caches={sorted(declared)}; `luxar demo cache list/clear` cannot "
        f"see or free it"
    )


# --------------------------------------------------------------------------- #
# On-scene credit footers vs DEMO_META["citation"]
# --------------------------------------------------------------------------- #
# A demo credits its data twice: structurally in ``DEMO_META["citation"]`` and
# visibly in an ``add_text`` footer painted onto the scene. Nothing keeps the two
# in step, and the drift is not hypothetical — PR #1745 fixed three footers that
# named the wrong paper. This is the guard for that class.
#
# Deliberately narrow: it compares only the two facts a credit footer states in a
# machine-checkable way, a YEAR and the LEADING NAME of the group in front of it.
# Anything softer would either miss the real mistakes (a whole other paper) or
# fire on the legitimate spellings the corpus already uses.

#: A publication year. Bounded on both sides so a plain number (``2048`` splats,
#: ``1024`` bins) inside a longer token is not read as a date.
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

#: Punctuation stripped off a token's ends before it is judged. A token that is
#: *nothing but* punctuation is a field separator (``•``, ``—``, ``|``) and ends
#: the name group — that is what keeps "3D UMAP • Kim et al. 2024" from reading
#: "UMAP" as part of the credit.
_TOKEN_TRIM = "()[]{}<>«»\"'“”‘’.,;:!?•—–-…/|*&"

#: Tokens that sit *inside* a name group without being names.
_CREDIT_GLUE = {"et", "al", "and", "with"}

#: Pure-punctuation tokens that join two names rather than separating fields.
_CREDIT_CONJUNCTIONS = {"&", "+"}

#: Capitalized words that start a sentence or a phrase rather than a name.
_NAME_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "by",
    "with",
    "via",
    "see",
    "press",
    "hover",
    "click",
    "data",
    "dataset",
    "version",
    "release",
}

#: How many names a credit group may hold before we stop walking back. Four
#: covers "The Tabula Sapiens Consortium 2022" with room to spare.
_MAX_CREDIT_NAMES = 4

#: Stands in for an f-string hole. Runtime-interpolated text is unknowable, so it
#: must never be *elided* — eliding it would splice two strings that are never
#: adjacent on screen and invent a credit nobody wrote. This placeholder is
#: neither a name nor glue, so it ends a name group like any other foreign token.
_INTERPOLATION = "￼"  # OBJECT REPLACEMENT CHARACTER


def _fold(text: str) -> str:
    """Case- and accent-insensitive form of *text*.

    ``casefold`` first, because it is the step that maps ``ß`` to ``ss`` — NFKD
    leaves ``ß`` alone, so ``Enßlin`` would never match ``Ensslin`` without it.
    NFKD then strips the combining marks (``Muñoz`` ≡ ``Munoz``).
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _overlay_literal(node: ast.expr, consts: Mapping[str, str]) -> str | None:
    """The statically knowable text of one overlay argument, or ``None``.

    Handles a plain string, a module-level ``NAME = "..."`` constant, implicit
    and explicit concatenation, and an f-string (literal segments only, with
    every ``{...}`` replaced by :data:`_INTERPOLATION`).
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
            else f" {_INTERPOLATION} "
            for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _overlay_literal(node.left, consts)
        right = _overlay_literal(node.right, consts)
        if left is None and right is None:
            return None
        return (left or f" {_INTERPOLATION} ") + (right or f" {_INTERPOLATION} ")
    return None


def _overlay_strings(source: str) -> list[str]:
    """Every statically resolvable ``add_text`` / ``add_html`` string in *source*.

    AST-only, for the same reason the registry is: importing a demo module runs
    heavy optional imports and import-time side effects.
    """
    tree = ast.parse(source)
    consts = _string_constants(tree)

    found: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("add_text", "add_html"):
            continue
        args = list(node.args[:1])
        args += [kw.value for kw in node.keywords if kw.arg in ("text", "html")]
        for arg in args:
            text = _overlay_literal(arg, consts)
            if text:
                found.append(text)
    return found


def _credit_claims(text: str) -> list[tuple[str | None, str]]:
    """Every ``(leading name | None, year)`` credit claim *text* makes.

    Walks back from each year over the tokens that make up its credit group and
    returns the group's FIRST name — never every name in it. "Leike & Enßlin
    2020" is one credit for a paper whose short form is "Leike et al. 2020", so
    demanding that every name appear would flag a perfectly correct footer.
    """
    tokens = [(m.group(), m.start()) for m in re.finditer(r"\S+", text)]
    claims: list[tuple[str | None, str]] = []
    for match in _YEAR_RE.finditer(text):
        index = next(
            (
                i
                for i, (tok, start) in enumerate(tokens)
                if start <= match.start() < start + len(tok)
            ),
            None,
        )
        if index is None:  # pragma: no cover - finditer spans are always in a token
            continue
        names: list[str] = []
        cursor = index - 1
        while cursor >= 0 and len(names) < _MAX_CREDIT_NAMES:
            token = tokens[cursor][0]
            cursor -= 1
            word = token.strip(_TOKEN_TRIM)
            if not word:
                if token in _CREDIT_CONJUNCTIONS:
                    continue
                break  # a bare separator: the credit group starts after it
            folded = _fold(word)
            if folded in _CREDIT_GLUE:
                continue
            name_shaped = (
                word[0].isupper()
                and sum(ch.isalpha() for ch in word) >= 3
                and folded not in _NAME_STOPWORDS
            )
            if not name_shaped:
                break
            names.append(word)
        claims.append((names[-1] if names else None, match.group()))
    return claims


def _credit_contradictions(short: str, overlays: list[str]) -> list[str]:
    """Overlay credit claims that ``short`` does not back."""
    folded_short = _fold(short)
    problems: list[str] = []
    for text in overlays:
        for name, year in _credit_claims(text):
            if year not in short:
                problems.append(f"year {year!r} in overlay {text.strip()!r}")
            elif name is not None and _fold(name) not in folded_short:
                problems.append(f"author {name!r} in overlay {text.strip()!r}")
    return problems


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.stem)
def test_overlay_credits_agree_with_declared_citation(path: Path) -> None:
    """An on-scene credit footer may not contradict the demo's own citation.

    Only demos that declare a real (non-``None``) citation are checked: with no
    declared credit there is nothing to contradict, and the 45 demos that have
    not been credited yet must not be blocked on their footers.
    """
    citation = extract_demo_meta(path).get("citation")
    if not citation:
        return
    short = citation["short"]
    problems = _credit_contradictions(short, _overlay_strings(path.read_text()))
    assert not problems, (
        f"{path.name} paints a credit its DEMO_META does not back "
        f"(citation.short = {short!r}):\n  " + "\n  ".join(problems)
    )


def test_credit_claim_extraction_reads_the_corpus_shapes() -> None:
    """The extractor must actually find the credits it is meant to police.

    Without this the sweep above could pass by finding nothing at all.
    """
    source = textwrap.dedent(
        '''
        """Doc."""
        FOOTER = "Tan et al. 2018 • chromosomes as 3D polylines"
        n = 3

        def build(scene):
            scene.add_text("Single-Cell 3D Genome")
            scene.add_text(FOOTER)
            scene.add_text(f"{n} cells • Kim et al. 2024")
            scene.add_text("Leike & " "Enßlin 2020 • 3D dust density")
        '''
    )
    overlays = _overlay_strings(source)
    assert "Tan et al. 2018 • chromosomes as 3D polylines" in overlays
    claims = [c for text in overlays for c in _credit_claims(text)]
    assert claims == [("Tan", "2018"), ("Kim", "2024"), ("Leike", "2020")]


@pytest.mark.parametrize(
    "short,overlay",
    [
        # A different first author, and a different year: the two mistakes the
        # guard exists for (both are real shapes PR #1745 had to fix by hand).
        ("Kim et al. 2024", "95K cells • 32 cell types • Lee et al. 2024"),
        ("Kim et al. 2024", "95K cells • 32 cell types • Kim et al. 2023"),
        ("Yeh 2022", "HCP-1065 atlas (Tournier 2019) — 87 tracts"),
        # An f-string whose literal tail still names a paper outright.
        ("Bui et al. 2013", "  proteins • rendered after Beck et al. 2016"),
    ],
)
def test_credit_guard_flags_a_contradicting_footer(short: str, overlay: str) -> None:
    assert _credit_contradictions(short, [overlay]), (
        f"guard missed a contradiction: {overlay!r} vs {short!r}"
    )


@pytest.mark.parametrize(
    "short,overlay",
    [
        # Every one of these is a real (short, footer) pair from the corpus. They
        # are the shapes a naive rule gets wrong, so they are pinned as tests
        # rather than merely observed to pass in the sweep.
        ("Tan et al. 2018", "Tan et al. 2018 • chromosomes as 3D polylines"),
        # A leading dataset token that is NOT the credited name, and a licence
        # whose "4.0" must not read as a year.
        ("Yeh 2022", "HCP-1065 atlas (Yeh 2022, CC BY-SA 4.0) — 87 tracts"),
        # "et al." on one side, "&" plus an eszett on the other.
        ("Leike et al. 2020", "Leike & Enßlin 2020 • 3D dust density • ~1 pc/voxel"),
        ("Kim et al. 2024", "95K cells • 32 cell types • Kim et al. 2024"),
        # A leading "The", and the bullet-separated fields before it.
        (
            "Tabula Sapiens Consortium 2022",
            "￼ cells • ￼ tissues • The Tabula Sapiens Consortium 2022",
        ),
        # The credit sits at the end of a long sentence of lowercase prose.
        (
            "CAFA5 (Kaggle); embeddings by ProtT5 (Elnaggar et al. 2022)",
            "￼ proteins • ProtT5 embeddings • 3D UMAP • clusters named by "
            "UniProt keyword enrichment • Elnaggar et al. 2022",
        ),
        # The credited "author" is a dataset acronym, and the footer names a
        # second source the short form also lists.
        (
            "ETOPO 2022 / HydroSHEDS",
            "ETOPO 2022 topography • HydroRIVERS (HydroSHEDS) river networks",
        ),
    ],
)
def test_credit_guard_passes_legitimate_corpus_footers(
    short: str, overlay: str
) -> None:
    assert not _credit_contradictions(short, [overlay])
