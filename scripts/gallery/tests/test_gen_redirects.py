"""Tests for the stable ``/d/<demo-key>`` route generator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "gen_redirects.py"
SPEC = importlib.util.spec_from_file_location("gen_redirects", SCRIPT)
assert SPEC and SPEC.loader
gen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gen
SPEC.loader.exec_module(gen)


DEMOS = [
    {"id": "lorenz", "dataset": "datasets/demos/lorenz.luxar.zarr"},
    # id != store: the case a mechanical mapping gets wrong
    {
        "id": "cosmicflows_laniakea",
        "dataset": "datasets/demos/cosmicflows_laniakea_full.luxar.zarr",
    },
    {"id": "not_published", "dataset": "datasets/demos/not_published.luxar.zarr"},
]
LIVE = {"lorenz", "cosmicflows_laniakea_full"}


def test_store_comes_from_dataset_not_id():
    entry = {
        "id": "cosmicflows_laniakea",
        "dataset": "d/cosmicflows_laniakea_full.luxar.zarr",
    }
    assert gen.store_of(entry) == "cosmicflows_laniakea_full"


def test_store_suffix_is_removed_only_at_the_end():
    entry = {"dataset": "d/my.luxar.zarr.backup"}
    assert gen.store_of(entry) == "my.luxar.zarr.backup"
    assert gen.normalise_stores(["my.luxar.zarr.backup"]) == {"my.luxar.zarr.backup"}


def test_route_targets_the_store_not_the_key():
    lines, routed, _ = gen.build_routes(DEMOS, LIVE, "2026-09-02")
    laniakea = [ln for ln in lines if ln.startswith("/d/cosmicflows_laniakea ")]
    assert len(laniakea) == 1
    # the mistake this guards: pointing /d/<id> at <id>.luxar.zarr
    assert "cosmicflows_laniakea_full.luxar.zarr" in laniakea[0]
    assert "/data/2026-09-02/cosmicflows_laniakea.luxar.zarr" not in laniakea[0]
    assert routed >= {"lorenz", "cosmicflows_laniakea"}


def test_store_spelling_gets_an_alias():
    lines, routed, _ = gen.build_routes(DEMOS, LIVE, "2026-09-02")
    assert any(ln.startswith("/d/cosmicflows_laniakea_full ") for ln in lines)
    assert "cosmicflows_laniakea_full" in routed
    # ...and a demo whose id already equals its store gets exactly one route
    assert sum(1 for ln in lines if ln.startswith("/d/lorenz ")) == 1


def test_demo_without_a_live_store_is_reported_not_emitted():
    lines, routed, skipped = gen.build_routes(DEMOS, LIVE, "2026-09-02")
    assert not any("/d/not_published " in ln for ln in lines)
    assert "not_published" not in routed
    assert ("not_published", "not_published") in skipped


def test_duplicate_route_path_is_rejected():
    demos = [
        {"id": "a", "dataset": "a_full.luxar.zarr"},
        {"id": "a_full", "dataset": "a_full.luxar.zarr"},
    ]
    with pytest.raises(gen.RouteError, match=r"duplicate route path: /d/a_full"):
        gen.build_routes(demos, {"a_full"}, "2026-09-02")


def test_normalise_stores_accepts_rclone_spellings():
    assert gen.normalise_stores(
        ["lorenz.luxar.zarr/", "  ocean.luxar.zarr", "", "cloud/"]
    ) == {"lorenz", "ocean", "cloud"}


def test_rendered_file_has_no_catch_all():
    lines, _, _ = gen.build_routes(DEMOS, LIVE, "2026-09-02")
    body = gen.render(lines, "2026-09-02")
    # a /d/* fallback would hide a typo'd link behind a page that looks fine
    assert "/d/*" not in body
    assert "prefix: 2026-09-02" in body


def _write_inputs(tmp_path, linked_keys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"demos": DEMOS}))
    stores = tmp_path / "stores.txt"
    stores.write_text("\n".join(sorted(LIVE)))
    media = tmp_path / "media-manifest.json"
    media.write_text(json.dumps({"tiles": {k: {} for k in linked_keys}}))
    readme = tmp_path / "README.md"
    readme.write_text("")
    return manifest, stores, media, readme


def test_readme_linked_keys_unions_tiles_and_readme_prose(tmp_path):
    _, _, media, readme = _write_inputs(tmp_path, ["lorenz"])
    readme.write_text("See https://demos.luxarviewer.dev/d/prose-only too.")
    assert gen.readme_linked_keys(media, readme) == {"lorenz", "prose-only"}


def test_contract_check_fails_when_a_readme_key_loses_its_route(tmp_path):
    manifest, stores, media, readme = _write_inputs(tmp_path, ["lorenz"])
    readme.write_text("See https://demos.luxarviewer.dev/d/a_renamed_key.")
    with pytest.raises(gen.RouteError, match="a_renamed_key"):
        gen.main(
            [
                "--prefix",
                "2026-09-02",
                "--live-stores",
                str(stores),
                "-o",
                str(tmp_path / "_redirects"),
                "--manifest",
                str(manifest),
                "--media-manifest",
                str(media),
                "--readme",
                str(readme),
                "--check-contract",
            ]
        )


def test_contract_check_passes_and_writes(tmp_path, capsys):
    manifest, stores, media, readme = _write_inputs(
        tmp_path, ["lorenz", "cosmicflows_laniakea"]
    )
    out = tmp_path / "_redirects"
    assert (
        gen.main(
            [
                "--prefix",
                "2026-09-02",
                "--live-stores",
                str(stores),
                "-o",
                str(out),
                "--manifest",
                str(manifest),
                "--media-manifest",
                str(media),
                "--readme",
                str(readme),
                "--check-contract",
            ]
        )
        == 0
    )
    body = out.read_text()
    assert "/d/lorenz" in body and "/d/cosmicflows_laniakea" in body
    assert "no live store" in capsys.readouterr().out  # not_published reported


@pytest.mark.parametrize("empty_sources", [False, True])
def test_contract_check_refuses_missing_or_empty_sources(tmp_path, empty_sources):
    manifest, stores, media, readme = _write_inputs(tmp_path, [])
    if not empty_sources:
        media.unlink()
        readme.unlink()
    with pytest.raises(gen.RouteError, match="contract sources"):
        gen.main(
            [
                "--prefix",
                "2026-09-02",
                "--live-stores",
                str(stores),
                "-o",
                str(tmp_path / "_redirects"),
                "--manifest",
                str(manifest),
                "--media-manifest",
                str(media),
                "--readme",
                str(readme),
                "--check-contract",
            ]
        )


def test_real_media_manifest_keys_are_manifest_demo_ids():
    demos = json.loads(gen.MANIFEST_PATH.read_text())["demos"]
    demo_ids = {entry["id"] for entry in demos}
    media_keys = set(json.loads(gen.MEDIA_MANIFEST_PATH.read_text())["tiles"])
    assert media_keys
    assert media_keys <= demo_ids


def test_empty_live_store_list_refuses_rather_than_emitting_nothing(tmp_path):
    manifest, _, _, _ = _write_inputs(tmp_path, [])
    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    with pytest.raises(gen.RouteError, match="no live stores"):
        gen.main(
            [
                "--prefix",
                "2026-09-02",
                "--live-stores",
                str(empty),
                "-o",
                str(tmp_path / "_redirects"),
                "--manifest",
                str(manifest),
            ]
        )
