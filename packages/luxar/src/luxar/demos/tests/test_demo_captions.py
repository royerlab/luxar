"""Every authored demo scene carries one standard bottom-right caption."""

from __future__ import annotations

import ast

from luxar.demos import format_demo_caption, registry

# These demos delegate their only scene build and caption to
# ``_interop_common.build_interop_scene``. Keep the list explicit so a new
# indirect route cannot silently escape the direct create_scene/caption count.
_INTEROP_DEMOS = {
    "demo_gsplats_interop_inria_garden.py",
    "demo_gsplats_interop_macro_clusterfly.py",
    "demo_gsplats_interop_mipnerf_garden.py",
    "demo_gsplats_interop_observatory.py",
    "demo_gsplats_interop_sog_matrixcity.py",
    "demo_gsplats_interop_spz_scaniverse.py",
}


def _name_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _attribute_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_every_authored_scene_has_exactly_one_standard_caption() -> None:
    for path in sorted(registry._DEMOS_DIR.glob("demo_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        creates = _attribute_calls(tree, "create_scene")
        captions = _name_calls(tree, "add_demo_caption")
        interop = _name_calls(tree, "build_interop_scene")
        if path.name in _INTEROP_DEMOS:
            assert not creates and not captions and len(interop) == 1, path.name
            continue
        assert creates, f"{path.name} authors no scene"
        assert len(captions) == len(creates), (
            f"{path.name} authors {len(creates)} scene(s) but "
            f"{len(captions)} standard caption(s)"
        )


def test_demo_modules_do_not_hand_author_bottom_right_overlays() -> None:
    for path in sorted(registry._DEMOS_DIR.glob("demo_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"add_text", "add_html"}
            ):
                continue
            if any(
                keyword.arg == "anchor"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "bottom-right"
                for keyword in node.keywords
            ):
                offenders.append(node.lineno)
        assert not offenders, (
            f"{path.name} hand-authors bottom-right overlay(s) at {offenders}; "
            "use add_demo_caption"
        )


def test_every_cited_demo_has_a_footer_sized_reference() -> None:
    for demo in registry.iter_demos(refresh=True):
        if demo.citation is None:
            continue
        caption = format_demo_caption("caption", demo.citation)
        reference = demo.citation.get("ref", demo.citation["short"])
        assert caption == f"caption · {reference}", demo.key
