"""Tests for the deterministic helpers in demo_bird_plumage_colorspace.

Colorimetry, column resolution, binning and spike geometry — no network, no
xlsx on disk, no scene I/O. The demo is loaded by file path (see
test_demo_ppi_flow_field for the rationale).
"""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_bird_plumage_colorspace.py"


def _load_demo_module():
    name = "_luxar_demo_bird_plumage_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
WAVELENGTHS = _demo.WAVELENGTHS
META_COLUMNS = _demo.META_COLUMNS
N_UV_BINS = _demo.N_UV_BINS
LAB_CENTRE_LIGHTNESS = _demo.LAB_CENTRE_LIGHTNESS
_ColourModel = _demo._ColourModel
_Accumulator = _demo._Accumulator
_column_index = _demo._column_index
_iter_rows = _demo._iter_rows
_resolve_columns = _demo._resolve_columns
_uv_bins = _demo._uv_bins
_spikes = _demo._spikes
_labels = _demo._labels


def _narrowband(centre_nm: float, width_nm: float = 20.0) -> np.ndarray:
    return np.exp(-0.5 * ((WAVELENGTHS - centre_nm) / width_nm) ** 2)


@pytest.fixture(scope="module")
def model() -> "_ColourModel":
    return _ColourModel()


class TestWavelengthGrid:
    def test_is_the_documented_bins(self) -> None:
        """201 columns, 300-700 nm, 2 nm — what every BirdColorBase file heads."""
        assert len(WAVELENGTHS) == 201
        assert WAVELENGTHS[0] == 300.0
        assert WAVELENGTHS[-1] == 700.0
        assert np.all(np.diff(WAVELENGTHS) == 2.0)


class TestColourModel:
    def test_perfect_reflector_is_reference_white(self, model) -> None:
        """R == 1 must land on L* 100, neutral a*/b*, and linear RGB (1, 1, 1).

        This is the white-balance check: the illuminant is a 6500 K Planckian
        rather than literal D65, so without the ``rgb_gain`` correction a flat
        reflector renders tinted and every colour in the scene carries the tint.
        """
        lab, rgb, _uv = model.evaluate(np.ones((1, len(WAVELENGTHS))))
        assert lab[0, 0] == pytest.approx(100.0, abs=1e-6)
        assert lab[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert lab[0, 2] == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(rgb[0], 1.0, atol=1e-5)

    def test_grey_is_neutral_and_linear(self, model) -> None:
        """An 18% grey card: L* ~ 49.5, no chroma, and LINEAR 0.18 RGB.

        The stored colour must be linear light — an sRGB-encoded 0.18 would be
        0.46 and the whole scene would render washed out.
        """
        lab, rgb, _uv = model.evaluate(np.full((1, len(WAVELENGTHS)), 0.18))
        assert lab[0, 0] == pytest.approx(49.5, abs=0.5)
        assert np.hypot(lab[0, 1], lab[0, 2]) == pytest.approx(0.0, abs=1e-5)
        np.testing.assert_allclose(rgb[0], 0.18, atol=1e-4)

    @pytest.mark.parametrize(
        ("centre_nm", "dominant"),
        [(450.0, 2), (550.0, 1), (620.0, 0)],
    )
    def test_narrowband_reflectors_get_the_right_hue(
        self, model, centre_nm, dominant
    ) -> None:
        _lab, rgb, _uv = model.evaluate(_narrowband(centre_nm)[None, :])
        assert int(np.argmax(rgb[0])) == dominant

    def test_uv_chroma_is_the_reflectance_ratio(self, model) -> None:
        """R300-400 / R300-700, and nothing about the illuminant or observer."""
        _lab, _rgb, uv = model.evaluate(_narrowband(350.0)[None, :])
        assert uv[0] > 0.98
        _lab, _rgb, uv = model.evaluate(_narrowband(620.0)[None, :])
        assert uv[0] < 0.01
        # A flat reflector's UV share is exactly the band's share of the grid.
        _lab, _rgb, uv = model.evaluate(np.ones((1, len(WAVELENGTHS))))
        expected = float((WAVELENGTHS < 400.0).sum()) / len(WAVELENGTHS)
        assert uv[0] == pytest.approx(expected, abs=1e-6)

    def test_ultraviolet_is_invisible_to_the_displayed_axes(self, model) -> None:
        """The claim the whole demo rests on, as an assertion.

        Adding a strong UV lobe to a red reflector must move UV chroma to ~0.5
        while leaving CIELAB and the rendered colour where they were: the human
        observer cannot see it, so the fourth dimension carries information the
        three displayed ones provably do not.
        """
        red = _narrowband(620.0)
        red_and_uv = np.clip(red + _narrowband(350.0), 0.0, 1.0)

        lab_a, rgb_a, uv_a = model.evaluate(red[None, :])
        lab_b, rgb_b, uv_b = model.evaluate(red_and_uv[None, :])

        np.testing.assert_allclose(lab_a, lab_b, atol=2.0)
        np.testing.assert_allclose(rgb_a, rgb_b, atol=0.01)
        assert uv_a[0] < 0.01
        assert uv_b[0] > 0.4

    def test_rgb_stays_in_gamut(self, model) -> None:
        """Out-of-gamut plumage is clipped, never written negative or over 1."""
        spectra = np.stack([_narrowband(c) for c in (320, 400, 470, 520, 580, 660)])
        _lab, rgb, _uv = model.evaluate(spectra)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0


def _workbook(
    rows: list[list[tuple[str, str]]],
    second_sheet: list[list[tuple[str, str]]] | None = None,
) -> zipfile.ZipFile:
    """A minimal in-memory xlsx whose cells carry explicit column references.

    Each row is a list of ``(column_letter, value)`` pairs, and ONLY those
    cells are emitted — which is how a real xlsx spells an empty cell, and the
    shape the positional reader got wrong.
    """

    def sheet_xml(sheet_rows: list[list[tuple[str, str]]]) -> str:
        cells = "".join(
            f'<row r="{r + 1}">'
            + "".join(
                f'<c r="{letter}{r + 1}"><v>{value}</v></c>' for letter, value in row
            )
            + "</row>"
            for r, row in enumerate(sheet_rows)
        )
        return (
            '<?xml version="1.0"?><worksheet '
            'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<sheetData>{cells}</sheetData></worksheet>"
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml(rows))
        if second_sheet is not None:
            archive.writestr("xl/worksheets/sheet2.xml", sheet_xml(second_sheet))
    return zipfile.ZipFile(buffer)


class TestColumnIndex:
    @pytest.mark.parametrize(
        ("letters", "index"),
        [("A", 0), ("B", 1), ("Z", 25), ("AA", 26), ("AZ", 51), ("HQ", 224)],
    )
    def test_base_26_with_no_zero_digit(self, letters, index) -> None:
        assert _column_index(letters) == index


class TestIterRows:
    """Cells are placed by column reference; omitted cells must not shift a row."""

    def test_dense_row_reads_in_order(self) -> None:
        book = _workbook(
            [
                [("A", "h1"), ("B", "h2"), ("C", "h3")],
                [("A", "1"), ("B", "2"), ("C", "3")],
            ]
        )
        rows = list(_iter_rows(book))
        assert rows[0] == ["h1", "h2", "h3"]
        assert rows[1] == ["1", "2", "3"]

    def test_omitted_cell_leaves_a_hole_rather_than_shifting(self) -> None:
        """The bug this guards: a data row missing one metadata cell.

        Every BirdColorBase file heads a 225-column table but writes 224-cell
        data rows, because one metadata column is blank throughout. Read
        positionally, the whole spectrum slides one column left — reflectance at
        302 nm read as 300 nm — and every value stays plausible. Two files
        (Maia, Stoddard-Prum) are 100% such rows, and were dropped entirely by
        the length check that stood in for this.
        """
        book = _workbook(
            [
                [("A", "h1"), ("B", "h2"), ("C", "h3"), ("D", "h4")],
                [("A", "1"), ("C", "3"), ("D", "4")],  # B omitted
            ]
        )
        rows = list(_iter_rows(book))
        assert rows[0] == ["h1", "h2", "h3", "h4"]
        assert rows[1] == ["1", None, "3", "4"]

    def test_short_row_is_padded_to_the_header_width(self) -> None:
        book = _workbook(
            [
                [("A", "h1"), ("B", "h2"), ("C", "h3")],
                [("A", "1")],
            ]
        )
        rows = list(_iter_rows(book))
        assert rows[1] == ["1", None, None]

    def test_multiple_worksheets_fail_loudly(self) -> None:
        book = _workbook([[("A", "header")]], second_sheet=[[("A", "other")]])
        with pytest.raises(ValueError, match="exactly one worksheet"):
            list(_iter_rows(book))


class TestAccumulator:
    def test_percent_scaling_is_decided_once_per_file(self) -> None:
        accumulator = _Accumulator(_ColourModel(), block_size=1)
        accumulator.begin_file()

        accumulator.add([2.0] * len(WAVELENGTHS), {})
        assert accumulator.flush() == (1, 0)
        accumulator.add([1.0] * len(WAVELENGTHS), {})
        assert accumulator.flush() == (1, 0)

        assert accumulator.scaled_from_percent
        assert np.max(accumulator.result()["lab"][:, 0]) < 25.0

    def test_flush_reports_lightness_drops(self) -> None:
        accumulator = _Accumulator(_ColourModel())
        accumulator.begin_file()
        accumulator.add([0.0] * len(WAVELENGTHS), {})

        assert accumulator.flush() == (0, 1)


class TestResolveColumns:
    """Column resolution is BY NAME; a fixed index is wrong on the Gomez file."""

    @staticmethod
    def _header(prefix: list[str], patch_name: str = "Patch") -> list[str]:
        return [
            *prefix,
            "English_BirdTree_Jetz",
            "Family3",
            "Order3",
            "JetzSpecies",
            "OriginalSpecies",
            "Sex",
            patch_name,
            *[f"{int(nm)}" for nm in WAVELENGTHS],
        ]

    def test_finds_the_spectral_block(self) -> None:
        header = self._header(["Lead_contributors"])
        meta, spectral = _resolve_columns(header)
        assert len(spectral) == len(WAVELENGTHS)
        assert header[spectral[0]] == "300"
        assert header[spectral[-1]] == "700"
        assert header[meta["english"]] == "English_BirdTree_Jetz"

    def test_offset_layout_resolves_to_the_same_fields(self) -> None:
        """The revised Gomez layout: four extra leading columns, renamed patch.

        Positional reading would hand back a museum code as a plumage patch for
        every row in that file, and no exception would be raised.
        """
        plain = self._header(["Lead_contributors"])
        offset = self._header(
            ["orig order", "BirdColorBase version", "Lead_contributors", "Source"],
            patch_name="Patch recoded 9Apr25",
        )
        meta_plain, spectral_plain = _resolve_columns(plain)
        meta_offset, spectral_offset = _resolve_columns(offset)

        for field in META_COLUMNS:
            assert field in meta_plain and field in meta_offset
        assert [plain[i] for i in spectral_plain] == [
            offset[i] for i in spectral_offset
        ]
        assert offset[meta_offset["patch"]] == "Patch recoded 9Apr25"
        assert offset[meta_offset["english"]] == "English_BirdTree_Jetz"

    def test_rejects_a_different_wavelength_grid(self) -> None:
        """A file binned at 5 nm must fail loudly, not integrate wrongly."""
        header = ["English_BirdTree_Jetz", *[str(nm) for nm in range(300, 701, 5)]]
        with pytest.raises(ValueError, match="300-700"):
            _resolve_columns(header)


class TestUvBins:
    def test_deciles_are_equal_population(self) -> None:
        rng = np.random.default_rng(0)
        # Right-skewed, like the real UV-chroma distribution.
        uv = rng.beta(2.0, 8.0, size=100_000).astype(np.float32)
        index, labels = _uv_bins(uv)

        assert len(labels) == N_UV_BINS
        counts = np.bincount(index, minlength=N_UV_BINS)
        assert len(counts) == N_UV_BINS
        assert counts.min() > 0.95 * counts.max()

    def test_bins_are_monotone_in_uv(self) -> None:
        uv = np.linspace(0.0, 1.0, 5000, dtype=np.float32)
        index, _labels = _uv_bins(uv)
        assert np.all(np.diff(index) >= 0)
        assert index.min() == 0
        assert index.max() == N_UV_BINS - 1

    def test_extremes_land_in_the_end_bins(self) -> None:
        """The maximum must not spill past the last bin (searchsorted off-by-one)."""
        uv = np.linspace(0.1, 0.9, 1000, dtype=np.float32)
        index, _labels = _uv_bins(uv)
        assert index[0] == 0
        assert index[-1] == N_UV_BINS - 1


class TestSpikes:
    @staticmethod
    def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lab = np.array(
            [[50.0, 30.0, -40.0], [80.0, -10.0, 5.0], [20.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        rgb = np.array(
            [[0.5, 0.1, 0.9], [0.2, 0.8, 0.3], [0.1, 0.1, 0.1]], dtype=np.float32
        )
        uv_bin = np.array([0, 4, 9], dtype=np.int64)
        return lab, rgb, uv_bin

    def test_two_vertices_per_reading(self) -> None:
        lab, rgb, uv_bin = self._sample()
        vertices, colors, widths = _spikes(lab, rgb, uv_bin)
        assert vertices.shape == (2 * len(lab), 4)
        assert colors.shape == (2 * len(lab), 3)
        assert widths.shape == (2 * len(lab),)

    def test_spike_length_is_chroma(self) -> None:
        """The geometric claim: |spike| == C*ab, so length reads as chroma.

        The inner end sits at the reading's OWN neutral grey, which is what
        makes this true; anchoring every spike at a shared origin would make the
        length a mix of chroma and lightness instead.
        """
        lab, rgb, uv_bin = self._sample()
        vertices, _colors, _widths = _spikes(lab, rgb, uv_bin)
        lengths = np.linalg.norm(vertices[1::2, :3] - vertices[0::2, :3], axis=1)
        np.testing.assert_allclose(lengths, np.hypot(lab[:, 1], lab[:, 2]), atol=1e-4)

    def test_both_endpoints_share_the_uv_coordinate(self) -> None:
        """A segment straddling two hidden coordinates would be culled in half."""
        lab, rgb, uv_bin = self._sample()
        vertices, _colors, _widths = _spikes(lab, rgb, uv_bin)
        np.testing.assert_array_equal(vertices[0::2, 3], vertices[1::2, 3])
        np.testing.assert_array_equal(vertices[0::2, 3], uv_bin)

    def test_lightness_is_recentred_on_mid_grey(self) -> None:
        lab, rgb, uv_bin = self._sample()
        vertices, _colors, _widths = _spikes(lab, rgb, uv_bin)
        np.testing.assert_allclose(
            vertices[0::2, 2], lab[:, 0] - LAB_CENTRE_LIGHTNESS, atol=1e-4
        )
        np.testing.assert_allclose(vertices[1::2, 2], vertices[0::2, 2], atol=1e-6)

    def test_colour_is_shared_by_both_endpoints(self) -> None:
        lab, rgb, uv_bin = self._sample()
        _vertices, colors, _widths = _spikes(lab, rgb, uv_bin)
        np.testing.assert_array_equal(colors[0::2], rgb)
        np.testing.assert_array_equal(colors[1::2], rgb)


class TestLabels:
    @staticmethod
    def _corpus() -> dict:
        return {
            "english": np.array(["Anna's Hummingbird", "", "Barn Owl"], dtype=object),
            "species": np.array(
                ["Calypte_anna", "Turdus_migratorius", "Tyto_alba"], dtype=object
            ),
            "patch": np.array(["crown", "breast", ""], dtype=object),
            "sex": np.array(["Male", "female", "unknown"], dtype=object),
            "uv_chroma": np.array([0.31, 0.05, 0.12], dtype=np.float32),
        }

    def test_label_carries_name_patch_and_uv(self) -> None:
        labels, _keys = _labels(self._corpus())
        assert labels[0] == "Anna's Hummingbird — crown ♂ — 31% UV"
        assert "breast ♀" in labels[1]

    def test_normalises_patch_placeholders_and_short_sex_codes(self) -> None:
        corpus = self._corpus()
        corpus["patch"] = np.array(["Crown", "not noted", "?"], dtype=object)
        corpus["sex"] = np.array(["M", "f", "unknown"], dtype=object)

        labels, _keys = _labels(corpus)

        assert labels[0] == "Anna's Hummingbird — crown ♂ — 31% UV"
        assert labels[1] == "Turdus migratorius — ♀ — 5% UV"
        assert labels[2] == "Barn Owl — 12% UV"

    @pytest.mark.parametrize("placeholder", ["not noted", "not recorded", "?", ""])
    def test_patch_placeholders_are_omitted(self, placeholder: str) -> None:
        corpus = self._corpus()
        corpus["patch"][0] = placeholder

        labels, _keys = _labels(corpus)

        assert labels[0] == "Anna's Hummingbird — ♂ — 31% UV"

    def test_falls_back_to_the_scientific_name(self) -> None:
        labels, _keys = _labels(self._corpus())
        assert labels[1].startswith("Turdus migratorius")

    def test_key_is_a_search_term_falling_back_to_the_binomial(self) -> None:
        """The key drives a Wikipedia search, so it is the plain name.

        A missing common name must fall back to the binomial rather than going
        empty — an empty substitution suppresses the link, and the species with
        no BirdTree common name are exactly the ones worth looking up.
        """
        _labels_out, keys = _labels(self._corpus())
        assert keys[0] == "Anna's Hummingbird"
        assert keys[1] == "Turdus migratorius"
        assert keys[2] == "Barn Owl"

    def test_key_is_empty_only_when_the_reading_is_unidentified(self) -> None:
        corpus = self._corpus()
        corpus["english"] = np.array(["", "", ""], dtype=object)
        corpus["species"] = np.array(["", "", ""], dtype=object)
        _labels_out, keys = _labels(corpus)
        assert keys == ["", "", ""]
