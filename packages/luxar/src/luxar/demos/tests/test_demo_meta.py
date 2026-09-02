"""Enforcement + registry tests for DEMO_META.

Every demo script must carry a valid DEMO_META literal — these tests are the
single durable gate (the block generator that seeded them was a one-off).
"""

from __future__ import annotations

import ast
import html
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
def test_credited_demo_wires_citation(path: Path) -> None:
    """Every credited demo must pass its citation into each scene constructor."""
    meta = extract_demo_meta(path)
    if meta.get("citation") is None:
        return

    tree = ast.parse(path.read_text(encoding="utf-8"))
    scene_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "create_scene")
            or (
                isinstance(node.func, ast.Name)
                and node.func.id == "build_interop_scene"
            )
        )
    ]
    assert scene_calls, f"{path.name} declares a citation but creates no scene"
    unwired = [
        call.lineno
        for call in scene_calls
        if not any(keyword.arg == "citation" for keyword in call.keywords)
    ]
    assert not unwired, f"{path.name} omits citation= at scene calls on lines {unwired}"


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


def test_cache_root_matches_demo_cache() -> None:
    """The registry and dataset-cache roots must never diverge."""
    from luxar.demos._support.datasets.cache import _DEFAULT_CACHE_ROOT

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
    entry, documented to mean a demo that lives only on a feature branch — by
    ``id`` against the demo key. Once an entry resolves either way, the manifest
    and the demo are two descriptions of the same thing, so every comparison runs
    on it: credit and structure alike. Running the structural half on the
    scriptless entries is what caught ``gsplats_3d_cryoem_virus``, filed
    ``microscopy`` in the manifest against the ``structural`` its demo (and both
    its peers, ``nuclear_pore_complex`` and ``atp_synthase``) declares.

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

    (problem,) = _manifest_citation_problems("d.py", {}, real)
    assert "would drop the credit" in problem, problem
    assert _manifest_citation_problems("d.py", {"citation": "Kim et al. 2024"}, None)
    assert _manifest_citation_problems("d.py", {"citation": "Kim et al. 2023"}, real)
    # Whitespace/spelling drift is drift: the field is a verbatim copy.
    assert _manifest_citation_problems("d.py", {"citation": "Kim et al.  2024"}, real)
    # A written-out null is not absence: absence says "not recorded yet", null
    # would be a second spelling of it, and neither is a string to compare. Every
    # non-string is already caught downstream, as an invented or a mismatched
    # credit, so the non-string branch earns its place only through the message it
    # writes — assert on that, or it is untested code.
    for entry, declared in (
        ({"citation": None}, None),
        ({"citation": None}, real),
        ({"citation": 2024}, real),
    ):
        (problem,) = _manifest_citation_problems("d.py", entry, declared)
        assert "left out entirely" in problem, problem


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="gallery manifest not in tree")
def test_scriptless_manifest_entries_are_fully_checked() -> None:
    """A ``script: null`` entry must not escape the cross-check.

    It used to: the cross-check skipped every entry without a script, so the
    ``citation`` backfill passed over those entries and left them un-backfilled —
    ``gsplats_3d_ct_totalsegmentator`` (a README tile) carries the
    ``Wasserthal et al. 2023`` its demo declares only because resolving these
    entries by ``id`` put them back under the rule. The structural half caught a
    genuinely pre-existing drift at the same time: ``gsplats_3d_cryoem_virus`` was
    filed ``microscopy`` against the ``structural`` its demo declares.

    The three entries that exposed this hole now carry their on-disk script paths
    after #1809. The fallback remains part of the manifest contract, so this test
    drives any scriptless entries that exist and synthesizes the shape when there
    are none.
    """
    manifest = json.loads(MANIFEST_PATH.read_text())["demos"]
    demos = list(iter_demos())
    keys = {d.key for d in demos}

    scriptless = [e for e in manifest if not e.get("script") and e["id"] in keys]
    if not scriptless:  # every entry has been given its script back: still guard
        scriptless = [{**e, "script": None} for e in manifest if e["id"] in keys][:1]
    assert scriptless, "no gallery entry resolves to a demo in this tree"

    for entry in scriptless:
        assert not _manifest_problems([entry], demos), (
            f"scriptless entry id={entry['id']} disagrees with the demo it names"
        )
        # Perturb each half so neither can be quietly gated back behind `script`.
        for field, wrong in (
            ("citation", "Nobody et al. 1999"),
            ("category", "not-a-category"),
            ("geometry", "not-a-geometry"),
        ):
            assert _manifest_problems([{**entry, field: wrong}], demos), (
                f"a scriptless entry (id={entry['id']}) escaped the {field} check"
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
# footers that named the wrong paper. This is the guard for that class, and the
# canonical account of its reach: the code below points back here instead of
# restating it.
#
# WHAT IT CHECKS: in a demo that declares a credit, every YEAR a statically
# resolvable overlay string paints, together with the NAMES immediately in front
# of it. The year must appear in ``citation["short"]``, and at least one of those
# names must too. That is all a credit footer states in a machine-checkable way;
# anything softer either misses the real mistake (a whole other paper) or fires on
# the legitimate spellings the corpus already uses. Overlay text means the first
# (or ``text=`` / ``html=``) argument of an ``add_text`` / ``add_html`` call, plus
# a ``credit=`` keyword passed to ANY call — six ``demo_gsplats_interop_*`` demos
# paint their footer through ``_interop_common.build_interop_scene(credit=...)``.
# They are uncredited today; this branch starts judging one as soon as it declares
# a citation, when reading only the demo's own overlay calls would cover none of it.
#
# WHAT IT DOES NOT CHECK:
#   * a year that yields no name group: a footer with no year at all, a year cut
#     off from its names by a field bullet ("Lange et al. • 2024") or written in
#     front of them ("2024 Lange et al."), and a lowercase surname, which the walk
#     reads as prose and stops on;
#   * a two-letter surname, which the walk treats as a short token rather than a
#     name, and a spelled-out group longer than four names, whose earliest names
#     fall outside the bounded lookback;
#   * a credit that is not statically resolvable — an f-string hole, a
#     ``list.append`` joined later, a conditional nested inside another (only the
#     outer two branches unfold), a helper that assembles the footer from its own
#     constants or from a parameter not spelled ``credit``;
#   * a demo that declares no credit at all — there is no declared fact to
#     contradict, so a footer painting an uncredited attribution is invisible here
#     (that is the inverse hazard, and it is why ``citation`` is worth backfilling
#     rather than a second rule);
#   * a footer that names a dataset the ``short`` form also names while attributing
#     it to the wrong author — the name group matches on the dataset token and the
#     wrong surname rides along. Worked: a short of "HCP-1065 / Yeh 2022" against a
#     footer of "HCP-1065 (Tournier 2022)" passes. (It is the YEAR that has to
#     agree for the hole to open: the same footer dated 2019 is caught, on the
#     year.) Measured cost of that any-name rule today: 6 of the 26 credited demos
#     have a ``short`` carrying a year AND two or more name-shaped tokens, so a
#     wrong author at the right year would pass on them — ``gaia_milky_way_3m``,
#     ``global_rivers_earth``, ``gsplats_4d_nexrad_supercell``,
#     ``protein_embeddings_cafa5``, ``tabula_sapiens``, and
#     ``esm3_protein_landscape`` (whose ``short`` this reader cannot follow
#     anyway). The other 20 still catch it.
#
# WHY KEEP THE ANY-NAME RULE at that cost: the alternative — demand the LEADING
# name of the group — rejects ZERO of today's real footers, but it is one deleted
# lowercase word away from rejecting the real dmri one. "HCP-1065 atlas (Yeh 2022,
# CC BY-SA 4.0)" passes a leading-name rule only because the lowercase "atlas"
# stops the walk; drop that word and the group leads with "HCP-1065", which the
# short "Yeh 2022" does not contain, and a correct credit turns red. A guard that
# an ordinary copy-edit can falsify gets switched off. The other two candidates
# are worse: ``_MAX_CREDIT_GROUP_NAMES = 1`` rejects the synthetic
# "Leike & Enßlin 2020" shape pinned below, and a year-only rule drops the
# wrong-author class the guard exists for.
#
# Every corpus count here (86 demos, 26 credited, 8 painting a dated credit, the
# AST tallies below) describes the corpus at the time of writing; nothing derives
# them, and only :data:`_KNOWN_CORPUS_CREDITS` goes red when they move.

#: A publication year. Bounded on both sides so a longer digit run (``12020241``
#: in an accession) is not read as a date. A four-digit number that IS bounded
#: (``2048³`` voxels, ``1920 x 1080``) still matches here — it is
#: :func:`_credit_claims`, not this pattern, that refuses to call a year with no
#: name in front of it a credit.
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

#: Punctuation stripped off a token's ends before it is judged. A token that is
#: *nothing but* punctuation is a field separator (``•``, ``—``, ``|``) and ends
#: the name group — that is what keeps "3D UMAP • Kim et al. 2024" from reading
#: "UMAP" as part of the credit. Every character of
#: :data:`_CREDIT_CONJUNCTIONS` must be listed here: a conjunction is re-admitted
#: below only once it has trimmed to nothing, so a ``+`` missing from this string
#: is read as a name-shaped token, breaks the walk, and drops the name in front
#: of it ("Leike + Enßlin 2020" would yield only "Enßlin").
_TOKEN_TRIM = "()[]{}<>«»\"'“”‘’.,;:!?•—–-…/|*&+"

#: Tokens that sit *inside* a name group without being names.
_CREDIT_GLUE = {"et", "al", "and", "with"}

#: Pure-punctuation tokens that join two names rather than separating fields.
_CREDIT_CONJUNCTIONS = {"&", "+"}

#: Capitalized words that start a sentence or a phrase rather than a name.
#:
#: Deliberately defensive, NOT tuned against observations: removing this whole
#: set changes no outcome on today's corpus. It exists so an ordinary footer edit
#: cannot go red for the wrong reason — a capitalized non-name in front of a year
#: still forms a claim, so "… Kim et al. 2024 • Copyright 2024 Zebrahub",
#: "… • Released 2023", "Snapshot March 2024 • …" and "Collected 2018–2024 • …"
#: all fired against a CORRECT ``short`` before the date-adjacent words were
#: listed. :func:`test_a_capitalized_non_name_beside_a_year_is_not_a_credit` pins
#: the first of those, so emptying the set cannot go unnoticed.
#:
#: A stopword is SKIPPED, not treated as the end of the group, so listing a word
#: that is also a surname costs only that name: "Bühlmann & May 2024" is still
#: claimed on "Bühlmann", and only a footer whose credit group is stopwords all
#: the way down ("May et al. 2024") becomes unclaimable. The list is knowingly
#: incomplete — "Winter 2023 build" still forms a claim — because calendar words
#: earn their place more clearly than season words do.
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


def _caption_reference(tree: ast.Module) -> str | None:
    """Return the compact reference declared by a module's ``DEMO_META``."""
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "DEMO_META"
            for target in statement.targets
        ):
            continue
        citation = ast.literal_eval(statement.value).get("citation")
        return citation.get("ref", citation["short"]) if citation else None
    return None


def _overlay_args(node: ast.Call) -> tuple[list[ast.expr], bool]:
    """Return overlay-bearing arguments and whether this is a demo caption."""
    args: list[ast.expr] = []
    keyword_names = {kw.arg for kw in node.keywords}
    is_demo_caption = (
        isinstance(node.func, ast.Name) and node.func.id == "add_demo_caption"
    ) or ("credit" in keyword_names and "citation" in keyword_names)
    if isinstance(node.func, ast.Name) and node.func.id == "add_demo_caption":
        args += node.args[1:2]
        args += [kw.value for kw in node.keywords if kw.arg == "caption"]
    if isinstance(node.func, ast.Attribute) and node.func.attr in (
        "add_text",
        "add_html",
    ):
        args += node.args[:1]
        args += [kw.value for kw in node.keywords if kw.arg in ("text", "html")]
    args += [kw.value for kw in node.keywords if kw.arg == "credit"]
    return args, is_demo_caption


def _overlay_strings(source: str) -> list[str]:
    """Every statically resolvable overlay string in *source*.

    Three shapes are read (see the section comment above for the full reach): an
    ``add_text`` / ``add_html`` call's first (or ``text=`` / ``html=``) argument,
    an ``add_demo_caption`` caption argument, and a ``credit=`` keyword passed to
    ANY call. The last is not decoration — it is the only way the six
    ``demo_gsplats_interop_*`` helpers' footers are seen here. This is the same
    demo-file-only blind spot ``tests/_scanned_modules.py`` exists for. A
    module's compact citation reference is appended to caption text when needed,
    matching ``format_demo_caption``.

    AST-only, for the same reason the registry is: importing a demo module runs
    heavy optional imports and import-time side effects.
    """
    tree = ast.parse(source)
    consts = _module_string_constants(tree)
    caption_reference = _caption_reference(tree)

    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        args, is_demo_caption = _overlay_args(node)
        for arg in args:
            for branch in _overlay_branches(arg):
                text = _overlay_literal(branch, consts)
                if text:
                    if (
                        is_demo_caption
                        and caption_reference
                        and not text.endswith(caption_reference)
                    ):
                        text = f"{text} • {caption_reference}"
                    found.append(text)
    return found


def _credit_claims(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Every ``(names, year)`` credit claim *text* makes, newest name first.

    Walks back from each year over the tokens that make up its credit group and
    returns ALL the names it collected, in walk order (so "Leike & Enßlin 2020"
    gives ``("Enßlin", "Leike")``). The caller treats the claim as backed when ANY
    of them appears in the declared credit — the section comment above argues that
    rule and prices it.

    A year the walk finds no name for yields NO claim: a bare number is not an
    attribution, and "2048³ voxels" or "1920 x 1080" would otherwise be read as
    one. HTML tags render as no text, so they become separators before entities
    are decoded and the rendered text is tokenized. That order keeps escaped tags
    visible rather than mistaking them for markup.
    """
    text = html.unescape(re.sub(r"<[^>]*>", " ", text))
    text = re.sub(r"([•—–·|/])", r" \1 ", text)
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
            if not (word[0].isupper() and sum(ch.isalpha() for ch in word) >= 3):
                break  # lowercase prose or a short token: the group starts after it
            if folded in _NAME_STOPWORDS:
                continue  # capitalized, but not a name; keep walking past it
            names.append(word)
        if names:
            claims.append((tuple(names), match.group()))
    return claims


def _mentions(needle: str, folded_short: str) -> bool:
    """True when *needle* appears in the already-folded *folded_short* as a token.

    Token-bounded, not a substring: "Kim" (a live surname here) must not be
    satisfied by "Kimura", nor a year by a digit run it sits inside. Punctuation
    still bounds a token, so "Elnaggar" matches inside "(Elnaggar et al. 2022)".

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

    The section comment above states exactly what is and is not compared. Only
    demos that declare a real (non-``None``) citation are swept: with no declared
    credit there is nothing to contradict, and the 60 demos that have not been
    credited yet must not be blocked on their footers. Of the 26 that do declare
    one, 8 paint a year-bearing credit today (pinned by
    :func:`test_corpus_yields_the_credits_the_sweep_judges`) — the rest are
    visited and found to claim nothing.

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

    Of those three, only ``add_html`` is live in the corpus today: 37 of the 227
    overlay calls (190 are ``add_text``), whose first arguments are 147
    ``Constant``, 48 f-string, 20 ``Name``, 11 ``Call`` and 1 conditional — zero
    ``+``, and ``text=``/``html=`` never used. The keyword and ``+`` branches are
    therefore defensive coverage of shapes the helper claims to handle, pinned so
    they cannot rot before something writes them.
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
    assert _credit_claims("<b>Tan et al. 2018</b>") == [(("Tan",), "2018")]
    assert _credit_contradictions("Lange et al. 2018", ["<b>Tan et al. 2018</b>"])
    assert "Kim et al. 2024" in overlays
    assert "Dip-C, Tan et al. 2018" in overlays
    # An unresolvable operand becomes a break, never an elision.
    spliced = next(t for t in overlays if t.startswith("Leike 2020"))
    assert spliced.endswith(f" {_INTERPOLATION} ")


def test_a_credit_painted_through_a_shared_helper_is_read() -> None:
    """A ``credit=`` argument counts as overlay text wherever it is passed.

    Six ``demo_gsplats_interop_*`` demos hand their footer to
    ``_interop_common.build_interop_scene(credit=...)``. They are uncredited
    today; this keeps their footer visible to the guard if one gains a citation.
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
    """``a if cond else b`` is two possible footers, not an unreadable one."""
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
    """A four-digit number with no name in front of it is a number."""
    assert _credit_claims("Gaia DR3 • epoch 2016.0") == []
    assert _credit_claims("recorded 1999–2013") == []
    assert _credit_claims("1920 x 1080") == []
    # A real credit in the same string is still read.
    assert _credit_claims("2048³ voxels • Kim et al. 2024") == [(("Kim",), "2024")]
    assert not _credit_contradictions(
        "Kim et al. 2024", ["2048³ voxels • Kim et al. 2024"]
    )


def test_either_conjunction_holds_a_name_group_together() -> None:
    """HTML ``&amp;``, plain ``&``, and ``+`` must hold the same name group.

    ``+`` was listed in :data:`_CREDIT_CONJUNCTIONS` but missing from
    :data:`_TOKEN_TRIM`, so it never trimmed to nothing, never reached the
    conjunction test, and broke the walk as a name-shaped token instead — dropping
    the name in front of it and turning a correct credit into a reported problem.
    Encoded HTML punctuation has the same failure mode unless entities are rendered
    before tokenization, which is why all three spellings are pinned.
    """
    assert _credit_claims("Leike & Enßlin 2020") == [(("Enßlin", "Leike"), "2020")]
    assert _credit_claims("Leike + Enßlin 2020") == [(("Enßlin", "Leike"), "2020")]
    assert _credit_claims("<b>Leike &amp; Enßlin 2020</b>") == [
        (("Enßlin", "Leike"), "2020")
    ]
    assert _credit_claims("<span>Kim&nbsp;et&nbsp;al.&nbsp;2024</span>") == [
        (("Kim",), "2024")
    ]
    # Tags are stripped before entities are decoded: escaped markup is visible text.
    assert _credit_claims("&lt;b&gt;Kim et al. 2024&lt;/b&gt;") == []
    assert not _credit_contradictions(
        "Leike et al. 2020", ["<b>dust • Leike &amp; Enßlin 2020</b>"]
    )


@pytest.mark.parametrize("separator", ["•", "—", "–", "·", "|", "/", "&mdash;"])
def test_unspaced_field_separator_ends_the_name_group(separator: str) -> None:
    footer = f"Zebrahub{separator}Kim et al. 2024"
    assert _credit_claims(footer) == [(("Kim",), "2024")]
    assert not _credit_contradictions("Kim et al. 2024", [footer])


def test_a_capitalized_non_name_beside_a_year_is_not_a_credit() -> None:
    """A footer routinely dates ITSELF as well as its source.

    Without :data:`_NAME_STOPWORDS` the self-date reads as an attribution to
    whatever capitalized word precedes it and a correct footer goes red; emptying
    the set leaves this test as the one that notices.
    """
    footer = "95K cells • Kim et al. 2024 • Copyright 2024 Zebrahub"
    assert _credit_claims(footer) == [(("Kim",), "2024")]
    assert not _credit_contradictions("Kim et al. 2024", [footer])
    # Skipped, not fatal: a real surname behind a stopword still forms its claim,
    # so listing a word that is also a surname costs that name and not the group.
    assert _credit_claims("Bühlmann & May 2024") == [(("Bühlmann",), "2024")]


def test_short_surnames_and_long_author_groups_are_known_limits() -> None:
    """The defensive name floor and bounded lookback can omit real surnames."""
    assert _credit_claims("single-cell atlas • Li et al. 2024") == []
    assert _credit_contradictions("Li et al. 2024", ["atlas • Li & Kim 2024"])
    assert _credit_contradictions(
        "Kerbl et al. 2023",
        ["Kerbl, Kopanas, Leimkühler, Schmid, Drettakis 2023"],
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
#: lookup left the whole suite passing). This set is the whole floor: one row per
#: demo that paints a dated credit, so a demo dropping out of the guard's
#: coverage — or a credit becoming unreadable — shows up as a missing row.
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


def test_corpus_yields_the_credits_the_sweep_judges() -> None:
    """The real corpus must keep yielding the credit claims it yields today.

    If you rewrote a footer on purpose, update :data:`_KNOWN_CORPUS_CREDITS` in
    the same commit — and check the rewrite did not move the credit into an
    f-string hole or a runtime join, which is how a demo silently leaves the
    guard's coverage.
    """
    observed: set[tuple[str, str, str]] = set()
    for path in DEMO_PATHS:
        if not extract_demo_meta(path).get("citation"):
            continue
        for text in _overlay_strings(path.read_text()):
            for names, year in _credit_claims(text):
                observed |= {(path.stem, name, year) for name in names}

    missing = sorted(_KNOWN_CORPUS_CREDITS - observed)
    assert not missing, (
        f"credits the guard used to read are no longer extracted: {missing}. "
        "Either the footer changed (update _KNOWN_CORPUS_CREDITS) or extraction "
        "broke (the sweep would go green while checking nothing)."
    )


def test_corpus_captions_do_not_repeat_their_compact_reference() -> None:
    duplicates = []
    for path in DEMO_PATHS:
        citation = extract_demo_meta(path).get("citation")
        if not citation:
            continue
        reference = citation.get("ref", citation["short"])
        duplicates += [
            (path.name, text)
            for text in _overlay_strings(path.read_text())
            if text.count(reference) > 1
        ]
    assert not duplicates, f"captions repeat their compact reference: {duplicates}"


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
        # A synthetic "et al." versus "&" pair, including an eszett.
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
        # ``short`` with a title-led footer nobody writes today, and a
        # leading-name rule would reject all four ('Nuclear', 'OpenCell',
        # 'Dip-C', 'Human'). Being hypotheticals, they are not the argument for
        # the any-name rule — the row below them, a real footer minus one word,
        # is (see the section comment).
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
