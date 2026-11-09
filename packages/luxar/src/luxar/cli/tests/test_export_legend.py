"""A scene's colour legend in the exported folder.

The legend is authored once, into the scene's root attributes, and the export
puts it everywhere a reader of the folder might look: README.txt, COLOURS.txt,
COLOURS.pdf, and the compact key on the panel's printed sheet (serve.py draws
that one, from the legend substituted into it at export time).
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler
from luxar.cli import _qr
from luxar.cli import export as export_module
from luxar.cli.export import (
    SceneFacts,
    _get_serve_script_content,
    _write_colour_files,
    legend_text,
    read_scene_facts,
)
from luxar.core.legend import Legend, LegendEntry

#: A text line's y in a content stream (negative included: a line pushed off
#: the page must be caught, not skipped by the pattern).
_TEXT_Y = rb"Tf -?[\d.]+ (-?[\d.]+) Td"
_BOTTOM = 56 + 6 - 0.01  # the bottom margin plus the footer's band


def _streams(pdf: bytes) -> list[bytes]:
    return [
        zlib.decompress(m) for m in re.findall(rb"stream\n(.*?)\nendstream", pdf, re.S)
    ]


LEGEND = Legend(
    title="What the colours mean",
    entries=(
        LegendEntry("Proteobacteria", "#66E680", "lime green", "Pseudomonadota."),
        LegendEntry("Viruses", "#E63333", "red"),
    ),
    note="Dim points are the dark proteome. Light adds up where points overlap.",
)


def _store(tmp_path: Path, legend: object = LEGEND, *, raw: bool = False) -> Path:
    store = tmp_path / "scene.luxar.zarr"
    with LuxarZarrCompiler(str(store)) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(), legend=None if raw else legend
        )
        scene.add_points("pts", np.zeros((4, 3), dtype=np.float32))
        if raw:
            scene.attrs["legend"] = legend  # bypasses validation, like a hand edit
    return store


def test_the_exporter_reads_the_legend_from_the_scene(tmp_path: Path) -> None:
    assert read_scene_facts(_store(tmp_path)).legend == LEGEND


def test_a_malformed_stored_legend_is_left_out_not_fatal(tmp_path: Path) -> None:
    facts = read_scene_facts(_store(tmp_path, {"title": "t", "entries": "x"}, raw=True))
    assert facts.legend is None


def test_the_text_legend_names_hex_colour_and_meaning() -> None:
    text = legend_text(LEGEND)
    assert text.startswith("What the colours mean\n---------------------\n")
    assert re.search(r"#66E680  Proteobacteria\s+lime green", text)
    assert "Pseudomonadota." in text
    assert "dark proteome" in text
    assert all(len(line) <= 79 for line in text.splitlines())


def test_the_readme_carries_the_legend_and_lists_the_colour_files(
    tmp_path: Path,
) -> None:
    export_module._generate_readme(tmp_path, "data", SceneFacts(legend=LEGEND))
    readme = (tmp_path / "README.txt").read_text()
    assert legend_text(LEGEND) in readme
    assert "COLOURS.txt" in readme and "COLOURS.pdf" in readme
    export_module._generate_readme(tmp_path, "data", SceneFacts())
    plain = (tmp_path / "README.txt").read_text()
    assert "COLOURS" not in plain and "What the colours mean" not in plain


def test_the_colour_files_are_written(tmp_path: Path) -> None:
    _write_colour_files(tmp_path, LEGEND, "My Scene")
    assert (tmp_path / "COLOURS.txt").read_text() == legend_text(LEGEND)
    pdf = (tmp_path / "COLOURS.pdf").read_bytes()
    assert pdf.startswith(b"%PDF-1.4") and b"/Annots" not in pdf  # nothing to link
    content = zlib.decompress(
        re.search(rb"stream\n(.*?)\nendstream", pdf, re.S).group(1)
    )
    assert b"(#66E680)" not in content  # the hex sits next to the colour name...
    assert b"lime green  #66E680" in content  # ...as one line
    assert b"0.4000 0.9020 0.5020 rg" in content  # the swatch is that colour


def test_serve_py_gets_the_legend_for_its_sheet() -> None:
    script = _get_serve_script_content("data", "T", SceneFacts(legend=LEGEND))
    line = next(x for x in script.splitlines() if x.startswith("LEGEND_JSON = "))
    assert json.loads(eval(line.split(" = ", 1)[1])) == LEGEND.to_dict()  # noqa: S307
    plain = _get_serve_script_content("data", "T", SceneFacts())
    assert "LEGEND_JSON = ''" in plain


def test_the_sheet_carries_the_key_and_still_scans(tmp_path: Path, monkeypatch) -> None:
    from luxar.cli import _export_serve_template as template

    monkeypatch.setattr(template, "LEGEND_JSON", json.dumps(LEGEND.to_dict()))
    monkeypatch.setattr(template, "qr_pdf_bytes", _qr.qr_pdf_bytes, raising=False)
    title, rows, note = template._legend_key()
    assert title == LEGEND.title
    assert rows[0] == ("#66E680", "lime green", "Proteobacteria")
    assert note == "Dim points are the dark proteome."  # first sentence only
    url = "http://h:8000/viewer/control.html?control&controlToken=abcdefgh12"
    pdf = template.panel_sheet(_qr.qr_matrix(url), url, has_token=True)
    content = zlib.decompress(
        re.search(rb"stream\n(.*?)\nendstream", pdf, re.S).group(1)
    )
    assert b"(Proteobacteria \\(lime green, #66E680\\)) Tj" in content

    monkeypatch.setattr(template, "LEGEND_JSON", "{not json")
    assert template._legend_key() == ("", (), "")  # a key is never the gate


@pytest.mark.parametrize("rows", [1, 10, 24])
def test_the_sheet_fits_any_key_by_shrinking_text_not_the_code(rows: int) -> None:
    url = "http://h:8000/viewer/control.html?control&controlToken=abcdefgh12"
    legend = tuple(
        (f"#{i:02X}{i:02X}{i:02X}", "grey", f"Group {i}") for i in range(rows)
    )
    pdf = _qr.qr_pdf_bytes(
        _qr.qr_matrix(url),
        url=url,
        heading="Drive the display from your phone",
        intro="What this is. " * 6,
        steps=("One.", "Two.", "Three."),
        notes=("A note. " * 12,) * 3,
        legend=legend,
        legend_note="Each point is a cluster.",
    )
    content = zlib.decompress(
        re.search(rb"stream\n(.*?)\nendstream", pdf, re.S).group(1)
    )
    # Every text line sits above the bottom margin (the footer's band).
    ys = [float(y) for y in re.findall(_TEXT_Y, content)]
    assert ys and min(ys) >= _BOTTOM
    # The code is still half the page wide: its first module row starts there.
    assert b"148.820" in content or b"148.82" in content


def _key_template(monkeypatch, legend: dict):
    from luxar.cli import _export_serve_template as template

    monkeypatch.setattr(template, "LEGEND_JSON", json.dumps(legend))
    monkeypatch.setattr(template, "qr_pdf_bytes", _qr.qr_pdf_bytes, raising=False)
    return template


def _many(count: int, description: str = "") -> Legend:
    return Legend(
        title="What the colours mean",
        entries=tuple(
            LegendEntry(f"Group {i}", f"#{i:02X}80{i:02X}", "teal", description)
            for i in range(count)
        ),
        note="Dim points are the dark proteome. " * 4,
    )


def test_a_long_key_on_the_sheet_is_capped_and_points_to_colours_pdf(
    monkeypatch,
) -> None:
    """Driven through ``panel_sheet``, as serve.py does: a key longer than the
    sheet can hold shows its first rows and the count of the rest."""
    template = _key_template(monkeypatch, _many(30).to_dict())
    url = "http://h:8000/viewer/control.html?control&controlToken=abcdefgh12"
    pdf = template.panel_sheet(_qr.qr_matrix(url), url, has_token=True)
    (content,) = _streams(pdf)
    assert b"Group 14 " in content and b"Group 15 " not in content
    assert b"... and 15 more: see COLOURS.pdf" in content
    ys = [float(y) for y in re.findall(_TEXT_Y, content)]
    assert min(y for y in ys if y != 50.0) >= _BOTTOM  # the footer sits at 50


def test_the_sheet_with_a_key_still_scans(tmp_path: Path, monkeypatch) -> None:
    import shutil
    import subprocess

    zxingcpp = pytest.importorskip("zxingcpp")
    image = pytest.importorskip("PIL.Image")
    if shutil.which("pdftoppm") is None:
        pytest.skip("pdftoppm (poppler) is not installed")
    template = _key_template(monkeypatch, _many(16).to_dict())
    url = "http://h:8000/viewer/control.html?control&controlToken=abcdefgh12"
    (tmp_path / "sheet.pdf").write_bytes(
        template.panel_sheet(_qr.qr_matrix(url), url, has_token=True)
    )
    subprocess.run(
        ["pdftoppm", "-r", "100", "-png", "sheet.pdf", "page"],
        cwd=tmp_path,
        check=True,
    )
    page = image.open(next(tmp_path.glob("page*.png")))
    assert [r.text for r in zxingcpp.read_barcodes(page)] == [url]


def test_a_long_colours_pdf_continues_on_new_pages_inside_the_margins() -> None:
    legend = _many(64, "A description long enough to wrap onto lines. " * 8)
    pdf = _qr.legend_pdf_bytes(
        title=legend.title,
        rows=export_module._legend_rows(legend),
        note=legend.note,
        footer="Luxar",
    )
    pages = _streams(pdf)
    assert len(pages) > 1
    assert b"/Count %d" % len(pages) in pdf
    for content in pages:
        ys = [float(y) for y in re.findall(_TEXT_Y, content)]
        body = [y for y in ys if y != 50.0]  # the footer line sits at 50
        assert body and min(body) >= _BOTTOM
    text = b"".join(pages)
    assert all(b"(Group %d)" % i in text for i in range(64))  # every row, once
    assert b"page 1 of %d" % len(pages) in pages[0]


def test_a_long_label_wraps_instead_of_running_off_the_page() -> None:
    label = "A very long label " * 8
    pdf = _qr.legend_pdf_bytes(
        title="t", rows=(("#123456", "blue", label.strip(), ""),)
    )
    (content,) = _streams(pdf)
    lines = re.findall(rb"/F2 13 Tf [\d.]+ [\d.]+ Td \((.*?)\) Tj", content)
    assert len(lines) > 1
    assert all(_qr._text_width(x.decode(), "Helvetica-Bold", 13) <= 483 for x in lines)


@pytest.mark.parametrize(
    ("note", "first"),
    [
        ("Dim points are dark. Light adds up.", "Dim points are dark."),
        ("See Fig. 2 for the key. More.", "See Fig. 2 for the key."),
        ("Bright, e.g. the hubs, are big. More.", "Bright, e.g. the hubs, are big."),
        ("Is it dim? Then it is dark.", "Is it dim?"),
        ("No full stop", "No full stop."),
        ("", ""),
    ],
)
def test_the_sheet_note_is_the_first_whole_sentence(
    monkeypatch, note: str, first: str
) -> None:
    template = _key_template(monkeypatch, {**LEGEND.to_dict(), "note": note})
    assert template._legend_key()[2] == first


def test_a_hand_edited_key_row_without_a_real_colour_is_dropped(monkeypatch) -> None:
    stored = LEGEND.to_dict()
    stored["entries"] = [*stored["entries"], {"label": "x", "color": "red",
                                               "color_name": "red"}]  # fmt: skip
    _title, rows, _note = _key_template(monkeypatch, stored)._legend_key()
    assert [row[2] for row in rows] == ["Proteobacteria", "Viruses"]
    stored["entries"] = [{"label": "x", "color": "1.0", "color_name": "red"}]
    assert _key_template(monkeypatch, stored)._legend_key() == ("", (), "")


def test_quotes_and_backslashes_in_a_legend_survive_into_serve_py() -> None:
    legend = Legend(
        title='The "key" \\ here',
        entries=(LegendEntry("A\\B 'c'", "#010203", 'say "red"'),),
    )
    script = _get_serve_script_content("data", "T", SceneFacts(legend=legend))
    line = next(x for x in script.splitlines() if x.startswith("LEGEND_JSON = "))
    assert json.loads(eval(line.split(" = ", 1)[1])) == legend.to_dict()  # noqa: S307


def test_export_warns_when_the_pdf_fonts_cannot_draw_a_character(
    tmp_path: Path, capsys
) -> None:
    legend = Legend(title="Key", entries=(LegendEntry("α-helix", "#010203", "red"),))
    _write_colour_files(tmp_path, legend, None)
    out = capsys.readouterr().out
    assert "cannot draw α" in out
    assert "α-helix" in (tmp_path / "COLOURS.txt").read_text(encoding="utf-8")
    _write_colour_files(tmp_path, LEGEND, None)
    assert "cannot draw" not in capsys.readouterr().out
