"""Guards for the nuclear-pore demo: architecture table, assembly, shading.

The demo's scientific claim is that every subunit of the human NPC is present and
correctly placed, so the tests that matter here are the ones that would catch a
MISSING or DOUBLE-COUNTED nucleoporin. Both failure modes are silent in a render
— a missing protein just looks like a slightly sparser complex.
"""

import numpy as np
import pytest

from luxar.demos import demo_nuclear_pore_complex as demo

#: The complete chain inventory of PDB 7R5J / 7R5K, as ``(entity description,
#: space-separated author chain ids)``. Both entries deposit the same 101 chains
#: under the same names (verified against the live depositions), so this fixture
#: pins the contract the curated architecture table has to satisfy. If RCSB ever
#: re-versions a chain name, `test_every_deposited_chain_is_assigned` is the
#: thing that goes red instead of the demo quietly dropping a protein.
DEPOSITED_CHAINS = (
    ("Aladin", "40 41"),
    ("E3 SUMO-protein ligase RanBP2", "00 01 02 03 04"),
    ("Nuclear pore complex protein Nup107", "L0 L1 L2 L3"),
    ("Nuclear pore complex protein Nup133", "K0 K1 K2 K3"),
    ("Nuclear pore complex protein Nup155", "D0 D1 D2 D3 D4 D5"),
    ("Nuclear pore complex protein Nup160", "R0 R1 R2 R3"),
    ("Nuclear pore complex protein Nup205", "C0 C1 C2 C3 C4"),
    ("Nuclear pore complex protein Nup214", "V0"),
    ("Nuclear pore complex protein Nup85", "P0 P1 P2 P3"),
    ("Nuclear pore complex protein Nup88", "W0"),
    ("Nuclear pore complex protein Nup93", "A0 A1 A2 A3 A4 A5 A6"),
    ("Nuclear pore complex protein Nup96", "M0 M1 M2 M3"),
    ("Nuclear pore complex protein Nup98", "U0 U1 U2 U3 U4 U5 U6"),
    ("Nuclear pore glycoprotein p62", "J0 J1 J2 J3 J4"),
    ("Nuclear pore membrane glycoprotein 210", "10 11 12 13 14 15 16 17"),
    ("Nucleoporin NDC1", "E0 E1"),
    ("Nucleoporin NUP188 homolog", "B0 B1"),
    ("Nucleoporin NUP35", "F0 F1 F2 F3"),
    ("Nucleoporin Nup37", "S0 S1 S2 S3"),
    ("Nucleoporin Nup43", "Q0 Q1 Q2 Q3"),
    ("Nucleoporin SEH1", "O0 O1 O2 O3"),
    ("Nucleoporin p54", "H0 H1 H2 H3"),
    ("Nucleoporin p58/p45", "I0 I1 I2 I3"),
    ("Protein ELYS", "T0 T1"),
    ("Protein SEC13 homolog", "N0 N1 N2 N3"),
)

#: 101 chains x 8 protomers = the deposition's declared ``808-meric`` assembly.
EXPECTED_CHAINS_PER_PROTOMER = 101


def _all_chains():
    """Yield every ``(description, chain)`` pair in the deposition."""
    for description, chains in DEPOSITED_CHAINS:
        for chain in chains.split():
            yield description, chain


def test_fixture_matches_the_declared_assembly() -> None:
    """The inventory really is the 101-chain protomer, so the rest means something."""
    assert len(list(_all_chains())) == EXPECTED_CHAINS_PER_PROTOMER


def test_every_deposited_chain_is_assigned() -> None:
    """No chain falls through the curated architecture table.

    A chain the table does not recognise raises, rather than being dropped — the
    whole point of the table being explicit.
    """
    for description, chain in _all_chains():
        module, slug, copy = demo.assign_chain(description, chain)
        assert module in demo.MODULE_COLORS, (description, chain, module)
        assert slug and copy


def test_unknown_chain_is_rejected_loudly() -> None:
    """An unrecognised protein must fail, not vanish from the scene."""
    with pytest.raises(KeyError):
        demo.assign_chain("Totally Novel Nucleoporin", "ZZ")


def test_module_and_copy_key_is_unique_per_chain_group() -> None:
    """``(module, node)`` keys never merge two distinct populations.

    The node NAME alone is ambiguous — ``nup160_inner`` belongs to both the
    cytoplasmic and the nuclear ring — and selecting on it wrote each Y-complex
    node twice, each copy holding both rings' atoms. This pins the composite key
    that fixed it: a bare name may repeat across modules, but the pair may not
    collapse two different chain sets into one key.
    """
    by_key: dict[tuple[str, str], set[str]] = {}
    for description, chain in _all_chains():
        module, slug, copy = demo.assign_chain(description, chain)
        by_key.setdefault((module, f"{slug}_{copy}"), set()).add(description)
    # Every key maps to exactly one nucleoporin.
    for key, descriptions in by_key.items():
        assert len(descriptions) == 1, (key, descriptions)
    # And the ambiguity the bug relied on is real, so the test is not vacuous:
    bare_names = [name for _, name in by_key]
    assert len(bare_names) != len(set(bare_names)), (
        "expected at least one node name to repeat across modules; if this ever "
        "becomes false the composite-key guard is no longer exercised"
    )


def test_y_complex_spans_both_faces_with_two_rings_each() -> None:
    """32 Y-complexes per NPC: 16 cytoplasmic + 16 nuclear, two rings per face."""
    slots = {
        demo.assign_chain("Nuclear pore complex protein Nup160", f"R{i}")[0::2]
        for i in range(4)
    }
    assert slots == {
        ("cytoplasmic_ring", "inner"),
        ("cytoplasmic_ring", "outer"),
        ("nuclear_ring", "inner"),
        ("nuclear_ring", "outer"),
    }


def test_elys_is_nuclear_only() -> None:
    """ELYS decorates the nuclear ring alone — the asymmetry is real biology."""
    for chain in ("T0", "T1"):
        module, slug, _ = demo.assign_chain("Protein ELYS", chain)
        assert module == "nuclear_ring"
        assert slug == "elys"


# =============================================================================
# Eight-fold assembly
# =============================================================================


def _c8_operators(centre=(10.0, 20.0, 0.0)):
    """Build eight exact C8 operators about ``centre``, as a deposition would."""
    operators = []
    axis = np.asarray(centre, dtype=np.float64)
    for i in range(8):
        angle = 2.0 * np.pi * i / 8.0
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        operators.append((rotation, axis - rotation @ axis))
    return operators


def test_symmetry_axis_recovers_the_rotation_centre() -> None:
    """The axis comes from the operators, not from the coordinate centroid.

    The protomer is a 45-degree wedge, so its own centroid is nowhere near the
    pore; solving ``(I - R) c = v`` is what makes the assembly the deposited one.
    """
    axis = demo.symmetry_axis(_c8_operators((993.6, 993.6, 0.0)))
    np.testing.assert_allclose(axis[:2], [993.6, 993.6], atol=1e-6)


def test_expand_eightfold_is_c8_symmetric_and_in_nanometres() -> None:
    """Eight copies, exact 45-degree symmetry, Angstrom -> nm, centred on axis."""
    rng = np.random.default_rng(0)
    protomer = rng.normal(size=(50, 3)) * 10.0 + np.array([993.6 + 400.0, 993.6, 0.0])
    ring = demo.expand_eightfold(protomer, _c8_operators((993.6, 993.6, 0.0)))

    assert ring.shape == (400, 3)
    # Centred: the ring's own centroid sits on the axis.
    np.testing.assert_allclose(ring[:, :2].mean(axis=0), [0.0, 0.0], atol=1e-4)
    # Scaled: a 400 A offset became 40 nm.
    radius = np.hypot(ring[:, 0], ring[:, 1])
    assert 30.0 < radius.mean() < 50.0
    # Symmetric: rotating protomer i by -45*i degrees lands on protomer 0.
    first = ring[:50]
    for i in range(1, 8):
        angle = -2.0 * np.pi * i / 8.0
        c, s = np.cos(angle), np.sin(angle)
        back = ring[i * 50 : (i + 1) * 50] @ np.array(
            [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        np.testing.assert_allclose(back, first, atol=1e-3)


def test_symmetrize_makes_a_scalar_exactly_eightfold() -> None:
    """The occlusion grid is axis-aligned; averaging mates restores C8 exactly."""
    values = np.arange(24, dtype=np.float64)
    out = demo.symmetrize(values, n_fold=8)
    per_protomer = out.reshape(8, 3)
    for i in range(1, 8):
        np.testing.assert_allclose(per_protomer[i], per_protomer[0])
    # Mass preserved.
    assert out.sum() == pytest.approx(values.sum())


# =============================================================================
# The state axis must be LAST (or the viewer discards every BSP tree)
# =============================================================================


def _fake_states(n_states: int, n_atoms: int = 12):
    """Build minimal shaded state dicts for the stacking helpers."""
    states = []
    for index in range(n_states):
        states.append(
            {
                "positions": np.full((n_atoms, 3), float(index), dtype=np.float32),
                "colors": np.full((n_atoms, 3), 0.5, dtype=np.float32),
                "radii": np.full(n_atoms, 0.17, dtype=np.float32),
                "keys": np.array(
                    ["m\x00a"] * (n_atoms // 2) + ["m\x00b"] * (n_atoms // 2)
                ),
                "modules": np.array(["inner_ring"] * n_atoms),
                "nups": np.array(["nup160"] * n_atoms),
                "elements": np.array(["C"] * n_atoms),
                "protomer": np.repeat(np.arange(2), n_atoms // 2),
                "n_fold": 2,
            }
        )
    return states


def test_state_column_is_last_so_bsp_splits_on_xyz() -> None:
    """x/y/z first, state last.

    ``spatial_bsp_tree`` always splits on positions columns 0-2 whatever they
    mean, and the viewer drops a ``bsp_tree`` whose split axis is not displayed.
    Putting the state axis first would leave displayDims == [1, 2, 3] and revert
    every partition to approximate centroid ordering, silently.
    """
    positions, _, _ = demo.stack_all(_fake_states(2))
    assert positions.shape[1] == 4
    # Column 3 holds the state index; columns 0-2 vary as coordinates.
    np.testing.assert_array_equal(np.unique(positions[:, 3]), [0.0, 1.0])


def test_single_state_stays_three_dimensional() -> None:
    """One state means no extra axis at all, so the scene stays plain 3D."""
    positions, _, _ = demo.stack_all(_fake_states(1))
    assert positions.shape[1] == 3


def test_scene_dimensions_put_state_last_and_hide_it() -> None:
    """The scene's own dimension order matches the positions layout."""
    dims = demo.scene_dimensions(2).dimensions
    assert [d.name for d in dims] == ["x", "y", "z", "state"]
    assert [d.display for d in dims] == [True, True, True, False]
    assert dims[3].spatial is False, "a spatial state axis renders a 1000-stop slider"
    assert list(dims[3].categories) == ["Constricted", "Dilated"]
    single = demo.scene_dimensions(1).dimensions
    assert [d.name for d in single] == ["x", "y", "z"]


def test_stack_node_selects_on_the_composite_key() -> None:
    """Selecting by key partitions the atoms; no atom is dropped or doubled."""
    states = _fake_states(2)
    total = 0
    for key in ("m\x00a", "m\x00b"):
        positions, colors, radii = demo.stack_node(states, key)
        assert len(positions) == len(colors) == len(radii)
        total += len(positions)
    assert total == sum(len(s["positions"]) for s in states)


# =============================================================================
# Appearance
# =============================================================================


def test_van_der_waals_radii_are_true_scale() -> None:
    """No visibility fudge factor: a scene unit really is a nanometre."""
    radii = demo.element_radii(np.array(["C", "N", "O", "S", "X"]))
    np.testing.assert_allclose(radii[:4], [0.170, 0.155, 0.152, 0.180], atol=1e-6)
    assert radii[4] == pytest.approx(demo.DEFAULT_VDW_NM)


@pytest.mark.parametrize("color_by", ["module", "nucleoporin", "element", "protomer"])
def test_all_color_schemes_cover_real_assignment_keys(color_by: str) -> None:
    """Every documented colour scheme accepts the arrays produced by a real state."""
    modules = np.array(["cytoplasmic_ring", "central_channel"])
    nucleoporins = np.array(["nup160", "nup58_p45"])
    elements = np.array(["C", "O"])
    protomer = np.array([0, 1])

    colors = demo.base_colors(color_by, modules, nucleoporins, elements, protomer)

    assert colors.shape == (2, 3)
    assert colors.dtype == np.float32
    assert np.isfinite(colors).all()
    assert not np.array_equal(colors[0], colors[1])


def test_shade_states_normalizes_across_states_and_restores_symmetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production shading shares one peak and makes symmetry mates identical."""
    base = np.array([0.7, 0.35, 0.15], dtype=np.float32)
    monkeypatch.setitem(demo.MODULE_COLORS, "inner_ring", tuple(base))
    raw_occlusions = iter(
        [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([2.0, 4.0, 6.0, 8.0]),
        ]
    )
    monkeypatch.setattr(
        demo, "bake_ambient_occlusion", lambda *args, **kwargs: next(raw_occlusions)
    )
    states = _fake_states(2, n_atoms=4)

    demo.shade_states(states, "module")

    first_scale = states[0]["colors"] / base[None, :]
    second_scale = states[1]["colors"] / base[None, :]
    np.testing.assert_allclose(first_scale[:, 0], [1 / 3, 1 / 2, 1 / 3, 1 / 2])
    np.testing.assert_allclose(second_scale[:, 0], [2 / 3, 1, 2 / 3, 1])
    np.testing.assert_allclose(
        first_scale[:, 1:], np.repeat(first_scale[:, :1], 2, axis=1)
    )
    np.testing.assert_allclose(
        second_scale[:, 1:], np.repeat(second_scale[:, :1], 2, axis=1)
    )


def test_module_colors_cover_every_assigned_module() -> None:
    """Every module a chain can land in has a colour, so nothing renders grey."""
    modules = {demo.assign_chain(d, c)[0] for d, c in _all_chains()}
    assert modules == set(demo.MODULE_COLORS)


def test_every_nucleoporin_has_a_slug() -> None:
    """Each nucleoporin the table can name resolves to a node-name slug."""
    for description, chain in _all_chains():
        _, slug, _ = demo.assign_chain(description, chain)
        assert slug in set(demo.NUP_SLUG.values())


# =============================================================================
# mmCIF reading
# =============================================================================

MINIMAL_CIF = """data_TEST
loop_
_entity.id
_entity.pdbx_description
1 'Protein ELYS'
2 'Nucleoporin NDC1'
#
loop_
_atom_site.group_PDB
_atom_site.label_entity_id
_atom_site.auth_asym_id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 T0 C CA 1.0 2.0 3.0
ATOM 1 T0 N N 4.0 5.0 6.0
ATOM 2 E0 O O 7.0 8.0 9.0
#
loop_
_pdbx_struct_oper_list.id
_pdbx_struct_oper_list.type
_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[1][2]
_pdbx_struct_oper_list.matrix[1][3]
_pdbx_struct_oper_list.vector[1]
_pdbx_struct_oper_list.matrix[2][1]
_pdbx_struct_oper_list.matrix[2][2]
_pdbx_struct_oper_list.matrix[2][3]
_pdbx_struct_oper_list.vector[2]
_pdbx_struct_oper_list.matrix[3][1]
_pdbx_struct_oper_list.matrix[3][2]
_pdbx_struct_oper_list.matrix[3][3]
_pdbx_struct_oper_list.vector[3]
1 'identity operation' 1 0 0 0 0 1 0 0 0 0 1 0
2 'point symmetry operation' 0 -1 0 0 1 0 0 0 0 0 1 0
#
"""


def test_reads_atoms_names_and_operators(tmp_path) -> None:
    """The hand-rolled reader gets coordinates, entity names and operators."""
    path = tmp_path / "mini.cif"
    path.write_text(MINIMAL_CIF)
    table = demo.read_protomer(path)

    np.testing.assert_allclose(table.positions[0], [1.0, 2.0, 3.0])
    assert list(table.elements) == ["C", "N", "O"]
    assert list(table.chain) == ["T0", "T0", "E0"]
    assert table.nucleoporin[0] == "Protein ELYS"
    assert table.nucleoporin[2] == "Nucleoporin NDC1"
    assert len(table.operators) == 2
    np.testing.assert_allclose(table.operators[0][0], np.eye(3))


def test_build_state_uses_the_deposited_symmetry_order(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every expanded array and the shading contract use the operator count."""
    path = tmp_path / "mini.cif"
    path.write_text(MINIMAL_CIF)
    monkeypatch.setattr(demo, "cached_download", lambda *args: path)

    state = demo.build_state("TEST", "all", "test")

    assert state["n_fold"] == 2
    assert len(state["positions"]) == 6
    assert list(state["protomer"]) == [0, 0, 0, 1, 1, 1]
    assert list(state["nups"]) == ["elys", "elys", "ndc1"] * 2


@pytest.mark.parametrize(
    "states,expected_entry",
    [([("Constricted", "7R5K")], "7R5K"), ([("Dilated", "7R5J")], "7R5J")],
)
def test_single_state_caption_names_the_selected_deposition(
    states, expected_entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-state scene never labels 7R5K coordinates as 7R5J."""

    class Scene:
        def add_text(self, *args, **kwargs) -> None:
            pass

    captured = {}
    monkeypatch.setattr(
        demo,
        "add_demo_caption",
        lambda scene, detail, citation: captured.setdefault("detail", detail),
    )

    demo._add_annotations(Scene(), states)

    assert captured["detail"].endswith(f"PDB {expected_entry}")


@pytest.mark.parametrize(
    "representation,expected",
    [("all", 3), ("backbone", 3), ("calpha", 1)],
)
def test_representation_filters(tmp_path, representation: str, expected: int) -> None:
    """Atom-name filtering selects the requested representation."""
    path = tmp_path / "mini.cif"
    path.write_text(MINIMAL_CIF)
    table = demo.read_protomer(path, atom_filter=representation)
    assert len(table.positions) == expected


def test_quoted_description_with_spaces_survives(tmp_path) -> None:
    """Entity descriptions are quoted and contain spaces; tokenizing must cope."""
    path = tmp_path / "mini.cif"
    path.write_text(MINIMAL_CIF)
    table = demo.read_protomer(path)
    assert " " in table.nucleoporin[0]


def test_unknown_representation_is_rejected() -> None:
    """A typo'd representation fails rather than silently keeping everything."""
    with pytest.raises(ValueError, match="unknown representation"):
        demo._atom_name_filter("sidechain")


def test_unknown_state_is_rejected() -> None:
    """A typo'd --state fails with the accepted values listed."""
    with pytest.raises(ValueError, match="unknown state"):
        demo._select_states("relaxed")


@pytest.mark.parametrize(
    "argument,expected",
    [("both", 2), ("dilated", 1), ("Constricted", 1)],
)
def test_state_selection(argument: str, expected: int) -> None:
    """``--state`` resolves to the right number of depositions, case-insensitively."""
    assert len(demo._select_states(argument)) == expected


def test_states_are_ordered_constricted_then_dilated() -> None:
    """Scrub order is the physical one, so ']' opens the pore."""
    assert [label for label, _ in demo.STATE_ENTRIES] == ["Constricted", "Dilated"]
    assert [entry for _, entry in demo.STATE_ENTRIES] == ["7R5K", "7R5J"]
