"""Every authored demo scene carries a standard or explicit custom credit."""

from __future__ import annotations

import ast
from pathlib import Path

from luxar.demos import format_demo_caption, registry

from ._scanned_modules import scanned_demo_modules

# These demos delegate their only scene build and caption to
# ``_interop_common.build_interop_scene``. Keep the list explicit so a new
# indirect route cannot silently escape the direct create_scene/caption count.
_INTEROP_DEMOS = {
    "demo_gsplats_interop_inria_bonsai.py",
    "demo_gsplats_interop_macro_clusterfly.py",
    "demo_gsplats_interop_mipnerf_garden.py",
    "demo_gsplats_interop_observatory.py",
    "demo_gsplats_interop_sog_matrixcity.py",
    "demo_gsplats_interop_spz_scaniverse.py",
}

# This kiosk intentionally reserves bottom-right for its bundled Biohub mark and
# renders the dataset/model/license credit as a persistent line under the title.
_CUSTOM_CAPTION_DEMOS = {"demo_esm3_protein_stories.py"}


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


def _demo_modules() -> list[Path]:
    return [path for path in scanned_demo_modules() if path.name.startswith("demo_")]


def test_every_authored_scene_has_exactly_one_standard_caption() -> None:
    for path in _demo_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        creates = _attribute_calls(tree, "create_scene")
        captions = _name_calls(tree, "add_demo_caption")
        interop = _name_calls(tree, "build_interop_scene")
        if path.name in _INTEROP_DEMOS:
            assert not creates and not captions and len(interop) == 1, path.name
            continue
        assert creates, f"{path.name} authors no scene"
        if path.name in _CUSTOM_CAPTION_DEMOS:
            assert not captions, path.name
            continue
        assert len(captions) == len(creates), (
            f"{path.name} authors {len(creates)} scene(s) but "
            f"{len(captions)} standard caption(s)"
        )


def test_demo_modules_do_not_hand_author_bottom_right_overlays() -> None:
    for path in scanned_demo_modules():
        if path.name == "_caption.py" or path.name in _CUSTOM_CAPTION_DEMOS:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"add_text", "add_html", "add_image"}
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
        assert caption == f"caption • {reference}", demo.key


def test_caption_fields_use_the_standard_separator() -> None:
    """Only caption expressions are normalized; labels and math may use ``·``."""
    for path in _demo_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        caption_exprs = []
        for call in _name_calls(tree, "add_demo_caption"):
            caption_exprs += call.args[1:2]
            caption_exprs += [kw.value for kw in call.keywords if kw.arg == "caption"]
        for call in _name_calls(tree, "build_interop_scene"):
            caption_exprs += [kw.value for kw in call.keywords if kw.arg == "credit"]
        offenders = [
            node.lineno
            for expr in caption_exprs
            for node in ast.walk(expr)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and " · " in node.value
        ]
        assert not offenders, f"{path.name} uses · in caption fields at {offenders}"

    for demo in registry.iter_demos(refresh=True):
        if demo.citation is None:
            continue
        reference = demo.citation.get("ref", demo.citation["short"])
        assert " · " not in reference, f"{demo.path.name} uses · in citation reference"


def test_nd_transform_channel_readouts_use_the_free_top_left_corner() -> None:
    path = next(
        path for path in _demo_modules() if path.name == "demo_nd_transforms.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    readouts = []
    for call in _attribute_calls(tree, "add_html"):
        names = [kw.value for kw in call.keywords if kw.arg == "name"]
        if not names or "expected_channel_" not in ast.unparse(names[0]):
            continue
        kwargs = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in call.keywords
            if kw.arg in {"position", "anchor"}
        }
        readouts.append((kwargs["position"], kwargs["anchor"]))
    assert readouts == [((0.015, 0.025), "top-left")]
