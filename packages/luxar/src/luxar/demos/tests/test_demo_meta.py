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
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from luxar.demos import registry
from luxar.demos.registry import (
    DemoInfo,
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
    Because those are two different claims, a written-out ``"citation": null`` is
    rejected outright rather than read as absence — as is any non-string value.
    """
    declared = citation["short"] if citation else None
    has_key = "citation" in entry
    present = entry.get("citation")
    if has_key and not isinstance(present, str):
        return [
            f"{script}: manifest citation is {present!r}; the key must carry the "
            "demo's citation.short string, or be left out entirely (absence is "
            "the only spelling of 'no credit recorded')"
        ]
    if declared is None:
        if has_key:
            return [
                f"{script}: manifest declares citation {present!r} but the demo "
                "declares no credit; a tile must not invent an attribution"
            ]
        return []
    if not has_key:
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


def _manifest_problems(
    manifest: Sequence[Mapping[str, object]], demos: Sequence[DemoInfo]
) -> list[str]:
    """Every disagreement between the gallery manifest and the demo registry.

    An entry is matched to its demo by script name, or — for a ``script: null``
    entry, one whose demo lives only on a feature branch and whose dataset is
    pre-generated — by ``id`` against the demo key. On a branch that DOES carry
    the demo, the scriptless entry therefore still resolves, and once it does, the
    manifest and the demo are two descriptions of the same thing: every
    comparison runs, credit and structure alike. All three of today's scriptless
    entries resolve, and running the structural checks on them is what caught
    ``gsplats_3d_cryoem_virus``, filed ``microscopy`` in the manifest against the
    ``structural`` its demo (and both its peers, ``nuclear_pore_complex`` and
    ``atp_synthase``) declares.

    Only "the script is on disk" stays script-gated: a scriptless entry has no
    script to be missing. The id/key comparison is kept for both paths for
    symmetry, though it is tautological on the id-resolved one — the key is how
    that demo was found.
    """
    by_path = {d.path.name: d for d in demos}
    by_key = {d.key: d for d in demos}

    problems: list[str] = []
    for entry in manifest:
        script = entry.get("script")
        demo = by_path.get(str(script)) if script else by_key.get(str(entry.get("id")))
        if demo is None:
            if script:
                problems.append(f"{script}: in manifest but not on disk")
            continue  # a scriptless entry for a demo that is not on this branch
        label = str(script) if script else f"id={entry.get('id')}"
        problems += _manifest_citation_problems(label, entry, demo.citation)
        if demo.key != entry["id"]:
            problems.append(f"{label}: key {demo.key!r} != manifest id {entry['id']!r}")
        if demo.geometry != entry["geometry"]:
            problems.append(
                f"{label}: geometry {demo.geometry!r} != manifest {entry['geometry']!r}"
            )
        if demo.category != entry["category"]:
            problems.append(
                f"{label}: category {demo.category!r} != manifest {entry['category']!r}"
            )
        stem = Path(str(entry["dataset"])).name.removesuffix(".luxar.zarr")
        if stem not in demo.outputs:
            problems.append(
                f"{label}: manifest dataset stem {stem!r} not in outputs {demo.outputs}"
            )
    return problems


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="gallery manifest not in tree")
def test_meta_cross_validates_against_gallery_manifest() -> None:
    """For every gallery entry that resolves to a demo in this tree — by script
    name or, for a ``script: null`` entry, by ``id`` — key == id, geometry and
    category match, the manifest dataset stem is among the demo's outputs, and the
    credit is the demo's own."""
    manifest = json.loads(MANIFEST_PATH.read_text())["demos"]
    problems = _manifest_problems(manifest, iter_demos())
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
    # A written-out null is not absence: absence says "not recorded yet", null
    # would be a second spelling of it, and neither is a string to compare.
    assert _manifest_citation_problems("d.py", {"citation": None}, None)
    assert _manifest_citation_problems("d.py", {"citation": None}, real)
    assert _manifest_citation_problems("d.py", {"citation": 2024}, real)


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="gallery manifest not in tree")
def test_scriptless_manifest_entries_are_fully_checked() -> None:
    """A ``script: null`` entry must not escape the cross-check.

    It used to: the cross-check skipped every entry without a script, and one of
    them (``gsplats_3d_ct_totalsegmentator``, a README tile) had already lost the
    ``Wasserthal et al. 2023`` its demo declares. Resolving the entry by ``id``
    against the demo key brings all three of today's scriptless entries back under
    the rule — credit AND structure, which is how ``gsplats_3d_cryoem_virus``'s
    wrong category surfaced. This pins that they resolve, and drives the real
    cross-check with a perturbed credit and a perturbed category on each to prove
    both halves reach them.
    """
    manifest = json.loads(MANIFEST_PATH.read_text())["demos"]
    demos = list(iter_demos())
    keys = {d.key for d in demos}

    scriptless = [e for e in manifest if not e.get("script")]
    assert scriptless, "the scriptless-entry shape this test guards is gone"
    for entry in scriptless:
        assert entry["id"] in keys, (
            f"manifest id {entry['id']!r} has no script and resolves to no demo; "
            "if that is deliberate (a feature-branch demo) this test needs to "
            "allow it, but today all three resolve"
        )
        wrong_credit = {**entry, "citation": "Nobody et al. 1999"}
        assert _manifest_problems([wrong_credit], demos), (
            f"a scriptless entry (id={entry['id']}) escaped the credit check"
        )
        wrong_category = {**entry, "category": "not-a-category"}
        assert _manifest_problems([wrong_category], demos), (
            f"a scriptless entry (id={entry['id']}) escaped the structural check"
        )


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
    importing 86 demo modules, and keeping it here avoids widening the registry's
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
# visibly in an ``add_text`` / ``add_html`` footer painted onto the scene. Nothing
# keeps the two in step, and the drift is not hypothetical — PR #1745 fixed three
# footers that named the wrong paper. This is the guard for that class.
#
# Every corpus count written into this section (86 demos, 26 credited, 8 painting
# a dated credit, and the shape tallies below) is indicative of the corpus at the
# time of writing: nothing derives them, and only the two pinned constants —
# :data:`_KNOWN_CORPUS_CREDITS` and :data:`_DEMOS_PAINTING_A_CREDIT` — go red when
# they move.
#
# What it checks, exactly: in a demo that declares a credit, every YEAR a statically
# resolvable overlay string paints, together with the NAMES immediately in front of
# that year. The year must appear in ``citation["short"]``, and at least one of
# those names must too. That is all a credit footer states in a machine-checkable
# way; anything softer either misses the real mistake (a whole other paper) or
# fires on the legitimate spellings the corpus already uses.
#
# Which strings count as overlay text: the first (or ``text=`` / ``html=``)
# argument of an ``add_text`` / ``add_html`` call, plus a ``credit=`` keyword
# passed to ANY call — six ``demo_gsplats_interop_*`` demos paint their footer
# through ``_interop_common.build_interop_scene(credit=...)``, and reading the
# demo file alone would have covered none of them. A credit a helper assembles by
# other means (from its own constants, from a parameter that is not spelled
# ``credit``) is still out of reach.
#
# What it does NOT check, and why it is still worth having:
#   * a footer that names the wrong dataset without giving a year — no claim to
#     compare;
#   * a credit assembled at runtime (an f-string hole, a ``list.append`` joined
#     later) — nothing statically resolvable to read;
#   * a demo that declares no credit at all — there is no declared fact to
#     contradict, so a footer painting an uncredited attribution is invisible here
#     (that is the inverse hazard, and it is why ``citation`` is worth backfilling
#     rather than a second rule);
#   * a footer that names a dataset the ``short`` form also names while attributing
#     it to the wrong author — the name group matches on the dataset token and the
#     wrong surname rides along. Worked: a short of "HCP-1065 / Yeh 2022" against a
#     footer of "HCP-1065 (Tournier 2022)" passes. (Note it is the YEAR that has to
#     agree for the hole to open: the same footer dated 2019 is caught, on the year.)
#     Measured cost of the any-name rule on today's corpus: 5 of the 26 credited
#     demos would accept a wrong author at the right year, because their ``short``
#     carries two or more name-shaped tokens for a footer to match instead of the
#     surname — ``gaia_milky_way_3m``, ``global_rivers_earth``,
#     ``gsplats_4d_nexrad_supercell``, ``protein_embeddings_cafa5``,
#     ``tabula_sapiens`` (and ``esm3_protein_landscape``, whose ``short`` this
#     reader cannot follow anyway). The other 21 still catch it.
#
# Why keep the any-name rule at that cost: the alternative — demand the LEADING
# name of the group — rejects ZERO of today's real footers, but it is one deleted
# lowercase word away from rejecting the real dmri one. "HCP-1065 atlas (Yeh 2022,
# CC BY-SA 4.0)" passes a leading-name rule only because the lowercase "atlas"
# stops the walk; drop that word and the group leads with "HCP-1065", which the
# short "Yeh 2022" does not contain, and a correct credit turns red. A guard that
# an ordinary copy-edit can falsify gets switched off. The other two candidates
# are worse: ``_MAX_CREDIT_GROUP_NAMES = 1`` rejects the real
# "Leike & Enßlin 2020" footer, and a year-only rule drops the wrong-author class
# the guard exists for.

#: A publication year. Bounded on both sides so a longer digit run (``12020241``
#: in an accession) is not read as a date. A four-digit number that IS bounded
#: (``2048³`` voxels, ``1920 x 1080``) still matches here — it is
#: :func:`_credit_claims`, not this pattern, that refuses to call a year with no
#: name in front of it a credit.
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
#:
#: Deliberately defensive, NOT tuned against observations: a second reviewer
#: measured that removing this whole set changes no outcome on today's corpus. It
#: exists so an ordinary footer edit cannot go red for the wrong reason — a
#: capitalized non-name in front of a year still forms a claim, so "… Kim et al.
#: 2024 • Copyright 2024 Zebrahub", "… • Released 2023", "Snapshot March 2024 • …"
#: and "Collected 2018–2024 • …" all fired against a CORRECT ``short`` before the
#: date-adjacent words were listed.
#:
#: A stopword is a missed claim, never a false one, so the cost of listing a word
#: that is also a surname is that the guard stops checking it: "May et al. 2024"
#: and "March et al. 2024" are now unclaimable. The list is knowingly incomplete —
#: "Winter 2023 build" still forms a claim — because every addition pays that
#: cost, and calendar words earn it more clearly than season words do.
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
    # Date-adjacent words: a footer routinely dates itself as well as its source.
    "released",
    "updated",
    "accessed",
    "published",
    "recorded",
    "collected",
    "imaged",
    "captured",
    "snapshot",
    "since",
    "copyright",
    "build",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

#: How many names a credit group may hold before we stop walking back. A claim is
#: backed when ANY name in the group appears in the declared ``short``, so a wider
#: window lowers the false-positive rate (a footer led by a title or a dataset
#: name) at the cost of a slightly wider blind spot (a wrong surname sitting next
#: to a word the ``short`` form happens to use too). Four covers the longest real
#: group, "The Tabula Sapiens Consortium 2022", plus a leading dataset token.
_MAX_CREDIT_GROUP_NAMES = 4

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


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Map every MODULE-LEVEL name bound to a string literal (last wins).

    Module level only, unlike :func:`_string_constants`: a whole-tree walk is
    last-wins across the entire file, so an unrelated ``FOOTER = "..."`` rebound
    inside some other function would beat the module constant the overlay call
    actually paints — a false credit in, and the real one out.
    """
    consts: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = node.value.value
    return consts


def _overlay_literal(node: ast.expr, consts: Mapping[str, str]) -> str | None:
    """The statically knowable text of one overlay argument, or ``None``.

    Handles a plain string, a module-level ``NAME = "..."`` constant, explicit
    ``+`` concatenation, and an f-string (literal segments only, with every
    ``{...}`` replaced by :data:`_INTERPOLATION`). Implicit concatenation needs no
    branch of its own: CPython folds adjacent literals into one ``Constant``
    before this ever sees them.
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


def _overlay_branches(node: ast.expr) -> list[ast.expr]:
    """The expressions an overlay argument can evaluate to.

    Only ``a if cond else b`` needs unfolding, and both branches are harvested as
    separate overlay strings — either can be the one on screen, and merging them
    would splice text that is never adjacent.
    ``demo_gsplats_4d_nexrad_supercell`` writes its per-frame timestamp this way
    (``f"2013-{stamp}" if stamp else ""``) and is the corpus's only case.
    """
    return [node.body, node.orelse] if isinstance(node, ast.IfExp) else [node]


def _overlay_strings(source: str) -> list[str]:
    """Every statically resolvable overlay string in *source*.

    Two shapes are read: an ``add_text`` / ``add_html`` call's first (or ``text=``
    / ``html=``) argument, and a ``credit=`` keyword passed to ANY call. The
    second is not decoration — the six ``demo_gsplats_interop_*`` demos paint
    their footer through ``_interop_common.build_interop_scene(credit=...)``, so a
    reader that only looked at ``add_text`` in the demo's own file would report
    green having read nothing (the same demo-file-only blind spot that
    ``tests/_scanned_modules.py`` exists for). A credit a helper builds some other
    way — from its own constants, or from a parameter not spelled ``credit`` — is
    still out of reach.

    AST-only, for the same reason the registry is: importing a demo module runs
    heavy optional imports and import-time side effects.
    """
    tree = ast.parse(source)
    consts = _module_string_constants(tree)

    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        args: list[ast.expr] = []
        if isinstance(node.func, ast.Attribute) and node.func.attr in (
            "add_text",
            "add_html",
        ):
            args += node.args[:1]
            args += [kw.value for kw in node.keywords if kw.arg in ("text", "html")]
        args += [kw.value for kw in node.keywords if kw.arg == "credit"]
        for arg in args:
            for branch in _overlay_branches(arg):
                text = _overlay_literal(branch, consts)
                if text:
                    found.append(text)
    return found


def _credit_claims(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Every ``(names, year)`` credit claim *text* makes, newest name first.

    Walks back from each year over the tokens that make up its credit group and
    returns ALL the names it collected, in walk order (so "Leike & Enßlin 2020"
    gives ``("Enßlin", "Leike")``). The caller treats the claim as backed when ANY
    of them appears in the declared credit: a footer routinely leads with a title
    or a dataset name ("Dip-C, Tan et al. 2018"), so requiring one particular name
    — the first, the last — rejects correct credits, while requiring all of them
    rejects "Leike & Enßlin 2020" against a short of "Leike et al. 2020".

    A year the walk finds no name for yields NO claim: a bare number is not an
    attribution, and "2048³ voxels" or "1920 x 1080" would otherwise be read as
    one.
    """
    tokens = [(m.group(), m.start()) for m in re.finditer(r"\S+", text)]
    claims: list[tuple[tuple[str, ...], str]] = []
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
        while cursor >= 0 and len(names) < _MAX_CREDIT_GROUP_NAMES:
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
        if names:
            claims.append((tuple(names), match.group()))
    return claims


def _mentions(needle: str, folded_short: str) -> bool:
    """True when *needle* appears in the already-folded *folded_short* as a token.

    Token-bounded, not a substring: "Kim" must not be satisfied by "Kimura", "Tan"
    by "Constant", "Lee" by "Leeuwenhoek", and a year must not be satisfied by a
    digit run it happens to sit inside. "Kim" and "Tan" are live surnames in this
    corpus; the others are the same shape, kept as tests rather than observations.
    Punctuation still bounds a token, so "Elnaggar" matches inside "(Elnaggar et
    al. 2022)" and "2013" inside "(KTLX, 2013-05-31)".

    The boundary is ``\\w``-based rather than an ASCII ``[0-9a-z]`` class, because
    :func:`_fold` does not fold a non-ASCII letter away — it strips combining
    marks and maps ``ß``, but leaves ``ø``/``λ`` standing. An ASCII class treats
    those as boundaries, so "Tan" would match inside "Tanø" and the collision this
    function closes would reopen one letter later.
    """
    pattern = rf"(?<!\w){re.escape(_fold(needle))}(?!\w)"
    return re.search(pattern, folded_short) is not None


def _credit_contradictions(short: str, overlays: list[str]) -> list[str]:
    """Overlay credit claims that ``short`` does not back."""
    folded_short = _fold(short)
    problems: list[str] = []
    for text in overlays:
        for names, year in _credit_claims(text):
            if not _mentions(year, folded_short):
                problems.append(f"year {year!r} in overlay {text.strip()!r}")
            elif not any(_mentions(name, folded_short) for name in names):
                problems.append(
                    f"none of the names {list(names)!r} credited beside {year} "
                    f"in overlay {text.strip()!r}"
                )
    return problems


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.stem)
def test_overlay_credits_agree_with_declared_citation(path: Path) -> None:
    """An on-scene credit footer may not contradict the demo's own citation.

    Checked: every year a statically resolvable overlay string paints, plus the
    names in front of it — the year and at least one of those names must appear in
    ``citation["short"]``. Not checked: a footer with no year in its literal text,
    and a credit built at runtime.

    Only demos that declare a real (non-``None``) citation are swept: with no
    declared credit there is nothing to contradict, and the 60 demos that have not
    been credited yet must not be blocked on their footers. Of the 26 that do
    declare one, 8 paint a year-bearing credit today (pinned by
    :func:`test_corpus_yields_the_credits_the_sweep_judges`) — the rest are visited
    and found to claim nothing.

    Known limitation: ``DEMO_META`` is the proxy for the credit, and
    ``demo_esm3_protein_landscape.py`` overrides its ``short`` at ``create_scene``
    time with a runtime model name. It is the only one of the 26 that does, and a
    static reader cannot follow it.
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


def test_the_sweep_itself_fails_on_a_planted_contradiction(tmp_path: Path) -> None:
    """Drive the sweep, as a whole, over a demo file written to fail it.

    The sweep is parametrized over 86 real demos that are all correct, so nothing
    else proves it can still fail. Blanking its citation lookup — the one line
    that decides whether a demo is judged at all — left the entire suite green.
    """
    meta = {
        "key": "planted-demo",
        "title": "Planted",
        "description": "A demo whose footer contradicts its own credit.",
        "category": "synthetic",
        "geometry": "points",
        "requirements": {
            "download_mb": 0,
            "compute": "light",
            "gpu": "none",
            "local_data": None,
        },
        "caches": [],
        "outputs": ["planted"],
        "citation": {"short": "Kim et al. 2024"},
    }
    planted = tmp_path / "demo_planted.py"
    planted.write_text(
        f'"""Planted demo."""\n\nDEMO_META = {meta!r}\n\n\n'
        'def build(scene):\n    scene.add_text("95K cells • Lange et al. 2024")\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="does not back"):
        test_overlay_credits_agree_with_declared_citation(planted)


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
    assert claims == [
        (("Tan",), "2018"),
        (("Kim",), "2024"),
        (("Enßlin", "Leike"), "2020"),
    ]


def test_overlay_extraction_covers_the_call_and_argument_shapes() -> None:
    """``add_html``, the keyword forms, and explicit ``+`` concatenation.

    Only ``add_html`` is live in the corpus today (37 calls; the first-argument
    shapes are 148 ``Constant``, 47 f-string, 20 ``Name``, 11 ``Call``, 1
    conditional, and zero ``+``, with ``text=``/``html=`` never used). The keyword
    and ``+`` branches are therefore defensive coverage of shapes the helper
    claims to handle, pinned so they cannot rot before something writes them.
    Implicit concatenation needs no case: CPython folds it into one ``Constant``
    before the helper runs, so it never reaches the ``+`` branch.
    """
    source = textwrap.dedent(
        """
        HEAD = "Dip-C, "

        def build(scene, tail):
            scene.add_html(html="<b>Tan et al. 2018</b>")
            scene.add_text(text="Kim et al. 2024")
            scene.add_text(HEAD + "Tan et al. 2018")
            scene.add_text("Leike 2020 • " + tail)
        """
    )
    overlays = _overlay_strings(source)
    assert "<b>Tan et al. 2018</b>" in overlays
    assert "Kim et al. 2024" in overlays
    assert "Dip-C, Tan et al. 2018" in overlays
    # An unresolvable operand becomes a break, never an elision.
    spliced = next(t for t in overlays if t.startswith("Leike 2020"))
    assert spliced.endswith(f" {_INTERPOLATION} ")


def test_a_credit_painted_through_a_shared_helper_is_read() -> None:
    """A ``credit=`` argument counts as overlay text wherever it is passed.

    Six ``demo_gsplats_interop_*`` demos hand their footer to
    ``_interop_common.build_interop_scene(credit=...)``, which is the call that
    runs ``scene.add_text``. Reading only the demo's own ``add_text`` calls saw
    none of them: the sweep would report green having read nothing the moment one
    of those demos backfills its ``citation``.
    """
    source = textwrap.dedent(
        """
        CREDIT = "Barron 2022 • research use"

        def build(path):
            build_interop_scene(path, credit=CREDIT)
            build_interop_scene(path, credit="Kerbl 2023 • 3DGS")
        """
    )
    overlays = _overlay_strings(source)
    assert "Barron 2022 • research use" in overlays
    assert "Kerbl 2023 • 3DGS" in overlays
    assert _credit_contradictions("Kerbl et al. 2023", overlays), (
        "a helper-painted credit must be judged like any other footer"
    )


def test_a_conditional_overlay_yields_both_branches() -> None:
    """``a if cond else b`` is two possible footers, not an unreadable one.

    ``demo_gsplats_4d_nexrad_supercell`` writes its per-frame timestamp this way;
    before this the whole argument resolved to ``None`` and the demo's only dated
    overlay was invisible to the reader.
    """
    source = textwrap.dedent(
        """
        def build(scene, stamp):
            scene.add_text("Tan et al. 2018" if stamp else "Kim et al. 2024")
        """
    )
    assert sorted(_overlay_strings(source)) == ["Kim et al. 2024", "Tan et al. 2018"]


def test_overlay_constants_resolve_at_module_level_only() -> None:
    """A function-local rebinding must not beat the module constant.

    ``ast.walk`` is breadth-first and last-wins, so an unrelated ``FOOTER`` inside
    some other function used to win — painting a credit the demo never shows and
    hiding the one it does.
    """
    source = textwrap.dedent(
        """
        FOOTER = "Tan et al. 2018 • chromosomes"

        def unrelated():
            FOOTER = "Lee et al. 2019"
            return FOOTER

        def build(scene):
            scene.add_text(FOOTER)
        """
    )
    assert _overlay_strings(source) == ["Tan et al. 2018 • chromosomes"]


def test_fstring_holes_break_a_credit_instead_of_splicing_one() -> None:
    """A name and a year that are never adjacent on screen are not a credit."""
    source = 'def build(scene, n):\n    scene.add_text(f"3D genome • Lange {n} 2024")\n'
    (overlay,) = _overlay_strings(source)
    assert _INTERPOLATION in overlay
    assert _credit_claims(overlay) == [], (
        "the f-string hole was elided, splicing 'Lange' onto a year it never "
        "sits beside"
    )


def test_a_bare_year_is_not_a_credit_claim() -> None:
    """A four-digit number with no name in front of it is a number.

    ``2048`` (a volume edge) and ``2016`` (an epoch) both match :data:`_YEAR_RE`;
    only the name walk separates them from a date.
    """
    assert _credit_claims("Gaia DR3 • epoch 2016.0") == []
    assert _credit_claims("recorded 1999–2013") == []
    assert _credit_claims("1920 x 1080") == []
    # A real credit in the same string is still read.
    assert _credit_claims("2048³ voxels • Kim et al. 2024") == [(("Kim",), "2024")]
    assert not _credit_contradictions(
        "Kim et al. 2024", ["2048³ voxels • Kim et al. 2024"]
    )


def test_credit_matching_is_token_bounded() -> None:
    """A name or year matches ``short`` as a whole token, never as a substring."""
    # A shorter surname must not be satisfied by a longer one it prefixes.
    assert _credit_contradictions("Kimura et al. 2024", ["3D genome • Kim et al. 2024"])
    assert _credit_contradictions(
        "Leeuwenhoek et al. 2021", ["cells • Lee et al. 2021"]
    )
    assert _credit_contradictions(
        "Constant et al. 2020", ["polylines • Tan et al. 2020"]
    )
    # The boundary is unicode-aware: an ASCII-only class ends the token at the
    # first non-ASCII letter, so "Tan" used to be satisfied by "Tanø".
    assert _credit_contradictions("Tanø et al. 2020", ["polylines • Tan et al. 2020"])
    assert _credit_contradictions("Leeλ 2020", ["cells • Lee 2020"])
    # A year must be its own number, not a run inside an accession.
    assert _credit_contradictions(
        "EMPIAR-12020241 (Bui et al. 2013)", ["pores • Bui et al. 2024"]
    )
    # Punctuation still bounds a token, so these correct credits still pass.
    assert not _credit_contradictions(
        "NOAA NEXRAD Level II (KTLX, 2013-05-31)", ["storm • NEXRAD 2013 sweep"]
    )
    assert not _credit_contradictions(
        "CAFA5 (Kaggle); embeddings by ProtT5 (Elnaggar et al. 2022)",
        ["proteins • Elnaggar et al. 2022"],
    )
    # A hyphenated surname, an apostrophe, and a folded eszett are all one token.
    assert not _credit_contradictions(
        "Sánchez-Ruiz & O'Brien 2021", ["cells • Sanchez-Ruiz & O'Brien 2021"]
    )
    assert not _credit_contradictions("Enßlin 2020", ["dust • Ensslin 2020"])


#: ``(demo file stem, one name in the credit group, year)`` triples the real demo
#: corpus paints today. The sweep above is parametrized over all 86 demos, but 78
#: of those parametrizations assert on input that cannot produce a problem, so it
#: stays green even if extraction silently stops working (blanking the citation
#: lookup left the whole suite passing). This set is the anchor: it fails if a
#: credit the guard is known to read stops being readable.
_KNOWN_CORPUS_CREDITS = frozenset(
    {
        ("demo_dipc_3d_genome", "Tan", "2018"),
        ("demo_dmri_tractography", "Yeh", "2022"),
        ("demo_global_rivers_earth", "ETOPO", "2022"),
        ("demo_gsplats_3d_milky_way_dust", "Leike", "2020"),
        ("demo_protein_embeddings_cafa5", "Elnaggar", "2022"),
        ("demo_tabula_sapiens", "Tabula", "2022"),
        ("demo_zebrahub_multiome", "Kim", "2024"),
        ("demo_zebrahub_multiome_peak_umap", "Kim", "2024"),
    }
)

#: How many credited demos paint a year-bearing credit today. A floor, not an
#: equality: crediting another demo, or giving one a dated footer, may only push
#: it up. Not an independent signal — it is the same extraction counted a second
#: way, so it can only fire once someone edits :data:`_KNOWN_CORPUS_CREDITS`; what
#: it adds is that a demo dropping out of coverage cannot be waved through by
#: deleting its line from that set.
_DEMOS_PAINTING_A_CREDIT = 8


def test_corpus_yields_the_credits_the_sweep_judges() -> None:
    """The real corpus must keep yielding the credit claims it yields today.

    If you rewrote a footer on purpose, update :data:`_KNOWN_CORPUS_CREDITS` (and
    :data:`_DEMOS_PAINTING_A_CREDIT` if a demo now paints no dated credit at all)
    in the same commit — and check the rewrite did not move the credit into an
    f-string hole or a runtime join, which is how a demo silently leaves the
    guard's coverage.
    """
    observed: set[tuple[str, str, str]] = set()
    painting = 0
    for path in DEMO_PATHS:
        if not extract_demo_meta(path).get("citation"):
            continue
        claims = [
            claim
            for text in _overlay_strings(path.read_text())
            for claim in _credit_claims(text)
        ]
        painting += bool(claims)
        for names, year in claims:
            observed |= {(path.stem, name, year) for name in names}

    missing = sorted(_KNOWN_CORPUS_CREDITS - observed)
    assert not missing, (
        f"credits the guard used to read are no longer extracted: {missing}. "
        "Either the footer changed (update _KNOWN_CORPUS_CREDITS) or extraction "
        "broke (the sweep would go green while checking nothing)."
    )
    assert painting >= _DEMOS_PAINTING_A_CREDIT, (
        f"only {painting} credited demos paint a readable dated credit, down from "
        f"{_DEMOS_PAINTING_A_CREDIT}. A demo dropped out of the guard's coverage; "
        "lower the floor only if that was deliberate."
    )


@pytest.mark.parametrize(
    "short,overlay",
    [
        # A different first author with the right year — the shape of every wrong
        # footer PR #1745 fixed by hand.
        ("Kim et al. 2024", "95K cells • 32 cell types • Lange et al., Cell 2024"),
        # A wrong year. Not a shape seen in the corpus, but the other half of what
        # a credit states, and cheap to check.
        ("Kim et al. 2024", "95K cells • 32 cell types • Kim et al. 2023"),
        ("Yeh 2022", "HCP-1065 atlas (Tournier 2019) — 87 tracts"),
        # A second paper named in a footer's tail, after the credited one.
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
        # Correct credits the guard must not reject. The first seven are real
        # (short, footer) pairs from the corpus; they are the shapes a naive rule
        # gets wrong, so they are pinned rather than merely observed to pass.
        ("Tan et al. 2018", "Tan et al. 2018 • chromosomes as 3D polylines"),
        # A leading dataset token that is NOT the credited name, and a licence
        # whose "4.0" must not read as a year.
        ("Yeh 2022", "HCP-1065 atlas (Yeh 2022, CC BY-SA 4.0) — 87 tracts"),
        # "et al." on one side, "&" plus an eszett on the other.
        ("Leike et al. 2020", "Leike & Enßlin 2020 • 3D dust density • ~1 pc/voxel"),
        ("Kim et al. 2024", "95K cells • 32 cell types • Kim et al. 2024"),
        # A leading "The", and the bullet-separated fields before it. The f-string
        # holes are spelled through the constant so this row cannot silently stop
        # testing the shape if the placeholder ever changes.
        (
            "Tabula Sapiens Consortium 2022",
            f"{_INTERPOLATION} cells • {_INTERPOLATION} tissues • "
            "The Tabula Sapiens Consortium 2022",
        ),
        # The credit sits at the end of a long sentence of lowercase prose.
        (
            "CAFA5 (Kaggle); embeddings by ProtT5 (Elnaggar et al. 2022)",
            f"{_INTERPOLATION} proteins • ProtT5 embeddings • 3D UMAP • clusters "
            "named by UniProt keyword enrichment • Elnaggar et al. 2022",
        ),
        # The credited "author" is a dataset acronym, and the footer names a
        # second source the short form also lists.
        (
            "ETOPO 2022 / HydroSHEDS",
            "ETOPO 2022 topography • HydroRIVERS (HydroSHEDS) river networks",
        ),
        # The rest are the shape a footer takes when a title or a dataset name
        # leads the credit. They are CONSTRUCTED, not painted: each pairs a real
        # ``short`` with a title-led footer nobody writes today (three of the four
        # demos paint no dated footer at all, and the Dip-C row is not the footer
        # `demo_dipc_3d_genome` paints — that one is the first row above). A
        # leading-name rule rejects all four as a wrong
        # author ('Nuclear', 'OpenCell', 'Dip-C', 'Human'); what saves them is that
        # ANY name in the group may back the claim. They are hypotheticals about a
        # rule, which is why the argument for keeping that rule rests on the row
        # below them — a real footer minus one word — and not on these.
        ("Bui et al. 2013", "Nuclear Pore Complex (Bui et al. 2013)"),
        ("Cho et al. 2022", "OpenCell MAP4 (Cho et al. 2022)"),
        ("Tan et al. 2018", "Dip-C, Tan et al. 2018"),
        ("Luck et al. 2020", "Human Reference Interactome, Luck et al. 2020"),
        # The "HCP-1065 atlas" corpus footer near the top of this list, with its
        # one lowercase word removed: under a leading-name rule 'atlas' is the only
        # thing stopping the walk, so deleting a word turns a correct credit red.
        ("Yeh 2022", "HCP-1065 (Yeh 2022, CC BY-SA 4.0) — 87 tracts"),
    ],
)
def test_credit_guard_passes_legitimate_corpus_footers(
    short: str, overlay: str
) -> None:
    assert not _credit_contradictions(short, [overlay])
