#!/usr/bin/env python3
"""Self-Contained Demo: the complete human Nuclear Pore Complex, all subunits.

Renders the whole 120 MDa human NPC scaffold at atomic resolution — **4,937,064
atoms, 808 protein chains, 25 distinct nucleoporins** — assembled from the PDB
deposition's *own* eight-fold symmetry operators, and scrubbable between the
**constricted** and **dilated** conformational states.

================================================================================
WHAT THIS IS, AND WHY IT IS BUILT THIS WAY
================================================================================

The NPC is the sole gateway between nucleus and cytoplasm: ~120 MDa, ~1000
nucleoporin copies drawn from ~30 distinct proteins, with eight-fold rotational
symmetry about the transport axis. It is the canonical "hard" structure to
depict, and the canonical way to get it wrong is to take one crystal fragment
and copy it eight times around an invented circle.

This demo instead uses the *reference model of the whole thing*:

    Mosalaganti, S. et al. (2022) "AI-based structure prediction empowers
    integrative structural analysis of human nuclear pores."
    Science 376: eabm9506.  doi:10.1126/science.abm9506

    PDB 7R5K — human NPC, CONSTRICTED (isolated nuclear envelopes)
    PDB 7R5J — human NPC, DILATED    (intact cells, the native state)

**The assembly is not invented — it is read off the deposition.** Each entry
deposits ONE eight-fold protomer (101 chains, 617,133 atoms) plus the eight C8
operators that generate the biological assembly, which the deposition itself
declares as ``808-meric``. This demo applies exactly those operators
(``_pdbx_struct_oper_list``): rotations of 0/45/.../315 degrees about the axis at
(993.6, 993.6, z) in the deposited frame. Nothing about the radius, the
orientation, or the spacing is chosen here. That is the whole point — the
placement of every subunit is the published one, so what you see is a real
structure and not a diagram of one.

Measured on the assembled result (not quoted from a paper):

    state         outer diameter   central channel   axial height
    constricted      149.9 nm          41.1 nm          72.2 nm
    dilated          159.7 nm          53.1 nm          76.5 nm

The outer diameter includes gp210's lumenal ring; the *scaffold* alone is
138.0 nm across in the dilated state. Atoms carry their TRUE van der Waals
radii (C 0.170 nm, N 0.155, O 0.152, S 0.180) — no fudge factor — so a scene
unit really is a nanometre at every scale.

WHAT IS PRESENT (six structural modules, every chain assigned)
--------------------------------------------------------------
Copy numbers and extents are measured from 7R5J after C8 expansion; +z is the
CYTOPLASM (RanBP2 is there), -z the NUCLEOPLASM (ELYS is there).

    module                 chains/NPC  atoms/NPC   radius nm      z nm
    cytoplasmic filaments      80        329,688   35.2-60.6   +15.0..+36.9
    cytoplasmic ring          176      1,101,040   39.4-65.1   +11.0..+35.3
    inner ring                144      1,242,528   30.2-55.2   -26.4..+21.8
    membrane ring              96      1,016,608   43.1-79.8   -12.9.. +6.9
    central channel (FG)      136        193,784   26.5-52.2   -22.3.. +8.6
    nuclear ring              176      1,053,416   38.7-69.0   -39.6..-14.2

* **Cytoplasmic filaments** — RanBP2/Nup358 (40 copies), the Nup214-Nup88-p62
  export platform. The fibrils that reach into the cytoplasm.
* **Cytoplasmic ring** and **nuclear ring** — the Y-complex (Nup107-160) coat.
  **32 Y-complexes in total: 16 cytoplasmic + 16 nuclear**, each face carrying
  TWO concentric rings of eight. The deposition encodes which is which in its
  chain names (Nup160 ``R0``/``R1`` = cytoplasmic inner/outer, ``R2``/``R3`` =
  nuclear inner/outer), so this demo reads the ring assignment rather than
  guessing it. ELYS joins the nuclear ring only — correct biology, and a thing
  a structural biologist checks first.
* **Inner ring** — Nup205, Nup188, Nup93, Nup155, Nup35 at the midplane. The
  module that dilates.
* **Membrane ring** — gp210/Nup210 (64 copies, reaching to r = 79.8 nm in the
  perinuclear lumen), plus the transmembrane nucleoporins NDC1 and ALADIN.
* **Central channel** — the FG nucleoporins p62/p54/p58-p45 and the Nup98
  anchors, lining the transport conduit at r = 26.5 nm inward.

WHAT IS ABSENT, AND WHY (stated rather than glossed)
-----------------------------------------------------
The model covers >90% of the NPC *scaffold*, which is not the same as the whole
NPC. Deliberately missing, because they are not resolved in this deposition:

* **The nuclear basket** — Tpr, Nup153, Nup50, ZC3HC1. Too flexible for
  subtomogram averaging; no full-basket coordinates exist to place honestly.
* **Most FG repeat regions.** The permeability barrier is intrinsically
  disordered; only its anchor points are modelled. The channel therefore looks
  emptier here than it is in a cell.
* **The nuclear envelope itself** — the membrane is not protein and is not in
  the coordinate file. gp210's lumenal ring marks where it runs.

Also worth knowing: this is a 50 A cryo-ET *integrative model* built by fitting
AlphaFold-predicted nucleoporins into subtomogram averages (EMD-14321/14322) —
not an experimental atomic structure. Atom positions within a domain are
predicted; the domain placements are the measurement.

================================================================================
HOW THE SCENE IS BUILT (and why the node layout looks the way it does)
================================================================================

**ONE node, partitioned by SPACE — not one node per protein.** This is the
scene's central design decision and it is forced, not stylistic.

The NPC's subunits are concave and they INTERPENETRATE: a Y-complex arm threads
past the inner ring, gp210 wraps the scaffold, the FG anchors sit inside the
inner-ring lumen. The viewer orders *whole objects* against each other
(bounding-sphere view-z plus a containment pass), and **two concave
interpenetrating objects have no valid whole-object draw order at all** — so
splitting the complex by protein cannot be composited correctly by any sorting
rule. Splitting it by space can: a BSP cell is a disjoint convex box, so a
back-to-front order always exists.

So the whole complex goes through a single ``add_points`` call with
``partition={"max_elements": 500_000}``. That makes the writer record the
recursive split planes (``bsp_tree``), which the viewer traverses back-to-front
(Fuchs-Kedem-Naylor) for an ordering that is *exact* for point BSP cells —
including with the camera inside the volume, which is what happens when you fly
down the central channel, and precisely where the per-object centroid fallback
degenerates. The wrapper is a single Layers row, so this costs no UI clutter.

The partition is also a hard capacity requirement, not just a quality one. One
Points node clamps at ``floor(4096/3) * maxTextureSize`` = **5,591,040**
elements on a 4096-class GPU; past that ``clampElementCapacity`` TRUNCATES with
one browser-console warning and no Python-side error. At 9,874,128 atoms an
unpartitioned node would silently drop ~4.3M of them.

``--split=nucleoporin`` opts into the alternative: ~60 nodes, one per
nucleoporin per ring copy, grouped under six module Groups, turning the Layers
panel into a nucleoporin browser (toggle gp210 off to see the scaffold; leave
only the Y-complexes on to see the coat). Useful for INSPECTING the
architecture, but it reintroduces exactly the cross-node ordering problem above,
so it is not the default and the docstring on that function says so.

One trap found while building this: node names are not unique on their own.
``nup160_inner`` is a legitimate child of *both* the cytoplasmic and the nuclear
ring (chains ``R0`` and ``R2``), so selecting atoms by node NAME rather than by
``(module, name)`` writes each Y-complex node twice — once under each module,
each copy holding both modules' atoms. Duplicated geometry, inflated counts, and
z-fighting between the copies. The build now asserts the written atom count
against the input, which is what catches that class of mistake.

**The `state` dimension is LAST, not first.** The house convention for a hidden
categorical axis puts it first (see ``demo_caida_as_topology``), but that would
silently disable the BSP ordering above: ``spatial_bsp_tree`` always splits on
positions columns 0-2 whatever they mean, and the viewer's ``bspAxisToComponent``
discards a tree whose split axis is not currently displayed. With ``state``
first, displayDims becomes [1,2,3], axis 0 is undisplayed, and every partition
quietly reverts to centroid ordering. With x/y/z first, ``displayDims ==
[0, 1, 2]`` — the one case where a stored split column IS the screen component.

The two states are atom-for-atom corresponding: both entries deposit the same
617,133 atoms in the same order (verified, not assumed), so scrubbing the state
axis is a genuine conformational morph of the same molecules and not a crossfade
between two unrelated point clouds.

Usage:
    luxar demo run nuclear_pore_complex

    --state=dilated|constricted|both   default both (adds the scrubbable axis)
    --color=module|nucleoporin|element|protomer     default module
    --representation=all|backbone|calpha            default all
    --split=none|nucleoporin           default none (one BSP-partitioned node)

Controls:
    - Top-down: the eight-fold ring and the real central channel
    - Side view: the three-ring stack, 76 nm tall, gp210 wings in the lumen
    - Press '1' then '[' / ']' to scrub constricted <-> dilated
    - Press L for the Layers panel
"""

DEMO_META = {
    "key": "nuclear_pore_complex",
    "title": "Nuclear Pore Complex",
    "description": (
        "The complete human NPC — 4.9M atoms, 808 chains, 25 nucleoporins "
        "(PDB 7R5J/7R5K) assembled with the deposition's own C8 operators; "
        "constricted↔dilated scrubbable in one BSP-partitioned layer; "
        "--split=nucleoporin adds per-nucleoporin layers."
    ),
    "category": "structural",
    "geometry": "points",
    "requirements": {
        "download_mb": 28,  # 7R5J.cif.gz + 7R5K.cif.gz, ~13.8 MB each
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["nuclear_pore_complex"],
    "outputs": ["nuclear_pore_complex"],
    "citation": {
        "short": "Mosalaganti et al. 2022",
        "doi": "10.1126/science.abm9506",
    },
}

import gzip
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import add_demo_caption, cached_download, launch_viewer
from luxar.demos._lod_policy import stream_ladder
from luxar.shading import bake_ambient_occlusion
from luxar.utils.paths import get_demos_output_dir

#: PDB entries for the two conformational states, in scrub order. Both deposit
#: the SAME 617,133 atoms in the SAME order, which is what makes the state axis
#: a morph rather than a crossfade.
STATE_ENTRIES: Tuple[Tuple[str, str], ...] = (
    ("Constricted", "7R5K"),
    ("Dilated", "7R5J"),
)

# =============================================================================
# Minimal mmCIF reading
# =============================================================================
#
# The full NPC is mmCIF-only — 617k atoms and 101 chains do not fit the legacy
# PDB format's fixed columns. Luxar has no structural-biology dependency, and
# adding gemmi to pull three loops would be a heavy dependency for a demo, so
# this reads the loops it needs directly. Only three categories are consumed:
# ``_entity`` (nucleoporin names), ``_atom_site`` (coordinates) and
# ``_pdbx_struct_oper_list`` (the C8 operators).


def _cif_tokens(line: str) -> List[str]:
    """Split one mmCIF data line, honouring single and double quoting.

    Args:
        line: A raw mmCIF line, without its trailing newline.

    Returns:
        The whitespace-separated tokens, with matched quotes stripped.
    """
    out: List[str] = []
    i, n = 0, len(line)
    while i < n:
        if line[i] in " \t":
            i += 1
            continue
        if line[i] in "'\"":
            quote = line[i]
            i += 1
            start = i
            while i < n and not (
                line[i] == quote and (i + 1 >= n or line[i + 1] in " \t")
            ):
                i += 1
            out.append(line[start:i])
            i += 1
        else:
            start = i
            while i < n and line[i] not in " \t":
                i += 1
            out.append(line[start:i])
    return out


def _read_semicolon_block(lines: Sequence[str], i: int) -> Tuple[str, int]:
    """Read an mmCIF multi-line (semicolon-delimited) text field.

    Args:
        lines: All lines of the file.
        i: Index of the opening ``;`` line.

    Returns:
        ``(value, next_index)``.
    """
    parts = [lines[i][1:]]
    i += 1
    while i < len(lines) and not lines[i].startswith(";"):
        parts.append(lines[i])
        i += 1
    return "\n".join(parts), i + 1


def _read_one_loop(
    lines: Sequence[str], i: int
) -> Tuple[str, List[str], List[List[str]], int]:
    """Read a single ``loop_`` block starting at ``lines[i] == 'loop_'``.

    Args:
        lines: All lines of the file.
        i: Index of the ``loop_`` keyword line.

    Returns:
        ``(category, tags, rows, next_index)`` where ``tags`` are the
        category-stripped tag names.
    """
    i += 1
    tags: List[str] = []
    while i < len(lines) and lines[i].strip().startswith("_"):
        tags.append(lines[i].strip().split()[0])
        i += 1
    category = tags[0].split(".")[0]
    names = [t.split(".", 1)[1] for t in tags]
    rows: List[List[str]] = []
    buf: List[str] = []
    while i < len(lines):
        line = lines[i]
        if line.startswith(("#", "loop_", "data_")) or (
            line.startswith("_") and not buf
        ):
            break
        if line.startswith(";"):
            value, i = _read_semicolon_block(lines, i)
            buf.append(value)
        elif line.strip():
            buf.extend(_cif_tokens(line))
            i += 1
        else:
            i += 1
        while len(buf) >= len(names):
            rows.append(buf[: len(names)])
            buf = buf[len(names) :]
    return category, names, rows, i


def read_cif_loops(
    path: Path, wanted: Iterable[str]
) -> Dict[str, Tuple[List[str], List[List[str]]]]:
    """Read the named ``loop_`` categories out of a (optionally gzipped) mmCIF.

    Args:
        path: Path to a ``.cif`` or ``.cif.gz`` file.
        wanted: Category names to keep, e.g. ``("_atom_site", "_entity")``.

    Returns:
        Mapping of category name to ``(tag_names, rows)``. Categories absent
        from the file are simply absent from the result.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:  # type: ignore[operator]
        lines = handle.read().split("\n")
    keep = set(wanted)
    found: Dict[str, Tuple[List[str], List[List[str]]]] = {}
    i = 0
    while i < len(lines):
        if lines[i].startswith("loop_"):
            category, names, rows, i = _read_one_loop(lines, i)
            if category in keep:
                found[category] = (names, rows)
            continue
        i += 1
    return found


class ProtomerTable:
    """The deposited asymmetric unit: one eight-fold protomer of the NPC.

    Attributes:
        positions: ``(N, 3)`` atom coordinates in Angstroms, deposited frame.
        elements: ``(N,)`` element symbols.
        nucleoporin: ``(N,)`` entity description (the nucleoporin name).
        chain: ``(N,)`` author chain id — which carries the RING assignment.
        atom_name: ``(N,)`` PDB atom name, for backbone/C-alpha filtering.
        operators: The eight ``(rotation, translation)`` C8 operators.
    """

    def __init__(
        self,
        positions: np.ndarray,
        elements: np.ndarray,
        nucleoporin: np.ndarray,
        chain: np.ndarray,
        atom_name: np.ndarray,
        operators: List[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.positions = positions
        self.elements = elements
        self.nucleoporin = nucleoporin
        self.chain = chain
        self.atom_name = atom_name
        self.operators = operators


def _parse_operators(
    names: List[str], rows: List[List[str]]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Turn ``_pdbx_struct_oper_list`` rows into rotation/translation pairs.

    Args:
        names: The loop's tag names.
        rows: The loop's rows.

    Returns:
        One ``(3x3 rotation, 3-vector translation)`` per operator, in file order.
    """
    operators = []
    for row in rows:
        field = dict(zip(names, row))
        rotation = np.array(
            [[float(field[f"matrix[{r}][{c}]"]) for c in (1, 2, 3)] for r in (1, 2, 3)],
            dtype=np.float64,
        )
        translation = np.array(
            [float(field[f"vector[{r}]"]) for r in (1, 2, 3)], dtype=np.float64
        )
        operators.append((rotation, translation))
    return operators


def read_protomer(path: Path, atom_filter: str = "all") -> ProtomerTable:
    """Read one NPC protomer plus its symmetry operators from an mmCIF file.

    Args:
        path: Path to ``7R5J.cif.gz`` / ``7R5K.cif.gz``.
        atom_filter: ``"all"``, ``"backbone"`` (N, CA, C, O) or ``"calpha"``.

    Returns:
        The populated :class:`ProtomerTable`.

    Raises:
        ValueError: If the file lacks atoms or the eight-fold operator list.
    """
    loops = read_cif_loops(path, ("_entity", "_atom_site", "_pdbx_struct_oper_list"))
    if "_atom_site" not in loops or "_pdbx_struct_oper_list" not in loops:
        raise ValueError(f"{path.name} is missing _atom_site or _pdbx_struct_oper_list")

    entity_names, entity_rows = loops["_entity"]
    e_id, e_desc = entity_names.index("id"), entity_names.index("pdbx_description")
    descriptions = {row[e_id]: row[e_desc] for row in entity_rows}

    names, rows = loops["_atom_site"]
    col = {
        n: names.index(n)
        for n in (
            "label_entity_id",
            "auth_asym_id",
            "type_symbol",
            "label_atom_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
        )
    }
    keep = _atom_name_filter(atom_filter)
    selected = [r for r in rows if keep is None or r[col["label_atom_id"]] in keep]
    if not selected:
        raise ValueError(f"no atoms selected from {path.name} (filter={atom_filter})")

    positions = np.array(
        [
            [
                float(r[col["Cartn_x"]]),
                float(r[col["Cartn_y"]]),
                float(r[col["Cartn_z"]]),
            ]
            for r in selected
        ],
        dtype=np.float64,
    )
    return ProtomerTable(
        positions=positions,
        elements=np.array([r[col["type_symbol"]] for r in selected]),
        nucleoporin=np.array(
            [descriptions[r[col["label_entity_id"]]] for r in selected]
        ),
        chain=np.array([r[col["auth_asym_id"]] for r in selected]),
        atom_name=np.array([r[col["label_atom_id"]] for r in selected]),
        operators=_parse_operators(*loops["_pdbx_struct_oper_list"]),
    )


def _atom_name_filter(atom_filter: str) -> Optional[frozenset]:
    """Map a representation name to the set of PDB atom names to keep.

    Args:
        atom_filter: ``"all"``, ``"backbone"`` or ``"calpha"``.

    Returns:
        ``None`` to keep everything, else the permitted atom names.

    Raises:
        ValueError: On an unknown representation.
    """
    if atom_filter == "all":
        return None
    if atom_filter == "backbone":
        return frozenset({"N", "CA", "C", "O"})
    if atom_filter == "calpha":
        return frozenset({"CA"})
    raise ValueError(f"unknown representation {atom_filter!r}")


# =============================================================================
# Eight-fold assembly, straight from the deposition
# =============================================================================


def symmetry_axis(operators: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Solve for the point the C8 operators rotate about.

    A rotation by ``R`` about a point ``c`` is ``x -> R(x - c) + c``, i.e.
    translation ``v = c - Rc``, so ``c`` solves ``(I - R) c = v``. Taking it from
    the operators rather than from the coordinate centroid is what keeps the
    result the deposited axis: the protomer is a 45-degree wedge and its own
    centroid is nowhere near the pore.

    Args:
        operators: The parsed operator list; the first non-identity one is used.

    Returns:
        The ``(3,)`` axis point in the deposited frame (z is left at 0).
    """
    for rotation, translation in operators:
        if not np.allclose(rotation, np.eye(3)):
            return np.linalg.lstsq(np.eye(3) - rotation, translation, rcond=None)[0]
    raise ValueError("operator list contains no rotation")


def expand_eightfold(
    positions: np.ndarray, operators: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Apply the deposited operators, then centre and convert to nanometres.

    Args:
        positions: ``(N, 3)`` protomer coordinates in Angstroms.
        operators: The eight C8 operators from ``_pdbx_struct_oper_list``.

    Returns:
        ``(8N, 3)`` float32 coordinates in nm, centred on the pore axis, with
        z = 0 at the mid-plane of the complex. Protomer ``i`` occupies rows
        ``i*N : (i+1)*N``, which :func:`symmetrize` relies on.
    """
    expanded = np.vstack([(rot @ positions.T).T + vec for rot, vec in operators])
    centre = symmetry_axis(operators)
    centre[2] = expanded[:, 2].mean()
    return ((expanded - centre) * 0.1).astype(np.float32)


def symmetrize(values: np.ndarray, n_fold: int) -> np.ndarray:
    """Average a per-atom scalar across the ``n_fold`` symmetry mates.

    The occlusion grid is axis-aligned and so is not itself eight-fold
    symmetric; averaging each atom with its mates makes the baked shading
    *exactly* C8-symmetric, as the geometry is.

    Args:
        values: ``(n_fold * N,)`` per-atom values, protomer-major.
        n_fold: The symmetry order.

    Returns:
        ``(n_fold * N,)`` values, identical across mates.
    """
    per_protomer = values.reshape(n_fold, -1).mean(axis=0)
    return np.tile(per_protomer, n_fold)


# =============================================================================
# NPC architecture: which chain belongs to which module and ring copy
# =============================================================================
#
# CURATED, not inferred. This is the table a structural biologist would check,
# so it is explicit, and `test_every_chain_is_assigned` fails the build if any
# chain of either entry falls through it.
#
# The keys are (a substring of the entity description, the set of author chain
# ids). The chain ids are the deposition's own, and they already encode the ring
# assignment — which is why this can be read rather than guessed.

#: The nine nucleoporins of the Y-complex (Nup107-160 coat). Their author chain
#: ids end in 0/1/2/3, and that digit IS the ring copy.
Y_COMPLEX_NUPS: Tuple[str, ...] = (
    "Nup160",
    "Nup133",
    "Nup107",
    "Nup96",
    "Nup85",
    "Nup43",
    "Nup37",
    "SEC13",
    "SEH1",
)

#: Trailing chain digit -> (module, ring copy) for every Y-complex nucleoporin.
#: Two concentric rings of eight on each face: 32 Y-complexes per NPC.
Y_RING_SLOT: Dict[str, Tuple[str, str]] = {
    "0": ("cytoplasmic_ring", "inner"),
    "1": ("cytoplasmic_ring", "outer"),
    "2": ("nuclear_ring", "inner"),
    "3": ("nuclear_ring", "outer"),
}

#: (description substring, chain ids or None for "every chain", module, slot).
#: Order matters only in that the first match wins.
CHAIN_ASSIGNMENTS: Tuple[Tuple[str, Optional[frozenset], str, str], ...] = (
    # --- cytoplasmic filaments / mRNA export platform (+z, reaching outward)
    (
        "RanBP2",
        frozenset({"00", "01", "02", "03", "04"}),
        "cytoplasmic_filaments",
        "filaments",
    ),
    ("Nup214", frozenset({"V0"}), "cytoplasmic_filaments", "export_platform"),
    ("Nup88", frozenset({"W0"}), "cytoplasmic_filaments", "export_platform"),
    ("p62", frozenset({"J4"}), "cytoplasmic_filaments", "export_platform"),
    ("Nup98", frozenset({"U0", "U1"}), "cytoplasmic_filaments", "export_platform"),
    # --- ring-associated linker nucleoporins (Nup93/Nup205 also serve the IR)
    ("Nup93", frozenset({"A4", "A5"}), "cytoplasmic_ring", "linker"),
    ("Nup205", frozenset({"C2", "C3"}), "cytoplasmic_ring", "linker"),
    ("Nup93", frozenset({"A6"}), "nuclear_ring", "linker"),
    ("Nup205", frozenset({"C4"}), "nuclear_ring", "linker"),
    # --- ELYS: nuclear ring ONLY. Its absence cytoplasmically is real biology.
    ("ELYS", frozenset({"T0"}), "nuclear_ring", "inner"),
    ("ELYS", frozenset({"T1"}), "nuclear_ring", "outer"),
    # --- inner ring, the module that dilates
    ("Nup205", frozenset({"C0", "C1"}), "inner_ring", "core"),
    ("NUP188", frozenset({"B0", "B1"}), "inner_ring", "core"),
    ("Nup93", frozenset({"A0", "A1", "A2", "A3"}), "inner_ring", "core"),
    ("Nup155", frozenset({"D0", "D1", "D2", "D3"}), "inner_ring", "core"),
    ("Nup155", frozenset({"D4", "D5"}), "inner_ring", "linker"),
    ("NUP35", frozenset({"F0", "F1", "F2", "F3"}), "inner_ring", "core"),
    # --- central transport channel: the FG nucleoporins' anchored domains
    ("p62", frozenset({"J0", "J1", "J2", "J3"}), "central_channel", "fg"),
    ("p54", frozenset({"H0", "H1", "H2", "H3"}), "central_channel", "fg"),
    ("p58/p45", frozenset({"I0", "I1", "I2", "I3"}), "central_channel", "fg"),
    ("Nup98", frozenset({"U2", "U3", "U4", "U5", "U6"}), "central_channel", "fg"),
    # --- membrane ring: gp210's lumenal ring plus the transmembrane nups
    ("glycoprotein 210", None, "membrane_ring", "lumenal"),
    ("NDC1", frozenset({"E0", "E1"}), "membrane_ring", "transmembrane"),
    ("Aladin", frozenset({"40", "41"}), "membrane_ring", "transmembrane"),
)

#: Entity description substring -> short slug used in node names.
NUP_SLUG: Dict[str, str] = {
    "Nup160": "nup160",
    "Nup133": "nup133",
    "Nup107": "nup107",
    "Nup96": "nup96",
    "Nup85": "nup85",
    "Nup43": "nup43",
    "Nup37": "nup37",
    "SEC13": "sec13",
    "SEH1": "seh1",
    "RanBP2": "nup358_ranbp2",
    "Nup214": "nup214",
    "Nup88": "nup88",
    "Nup98": "nup98",
    "Nup93": "nup93",
    "Nup205": "nup205",
    "NUP188": "nup188",
    "Nup155": "nup155",
    "NUP35": "nup35",
    "ELYS": "elys",
    "p62": "nup62",
    "p54": "nup54",
    "p58/p45": "nup58_p45",
    "glycoprotein 210": "gp210",
    "NDC1": "ndc1",
    "Aladin": "aladin",
}

#: Modules in cytoplasm -> nucleoplasm order, with their linear-light colours.
#: A categorical encoding, so a scalar occlusion multiplier keeps every hue.
MODULE_COLORS: Dict[str, Tuple[float, float, float]] = {
    "cytoplasmic_filaments": (1.00, 0.45, 0.20),
    "cytoplasmic_ring": (1.00, 0.78, 0.25),
    "inner_ring": (0.25, 0.70, 1.00),
    "membrane_ring": (0.55, 0.35, 0.85),
    "central_channel": (0.30, 0.95, 0.60),
    "nuclear_ring": (1.00, 0.35, 0.45),
}


def assign_chain(description: str, chain: str) -> Tuple[str, str, str]:
    """Map one chain to its structural module, nucleoporin slug and ring copy.

    Args:
        description: The ``_entity.pdbx_description`` for the chain.
        chain: The ``auth_asym_id``.

    Returns:
        ``(module, nucleoporin_slug, ring_copy)``.

    Raises:
        KeyError: If the chain matches nothing in the curated table. This is a
            hard failure by design — a silently unassigned chain would vanish
            from the scene, and a missing nucleoporin is exactly the defect
            this table exists to prevent.
    """
    for nup in Y_COMPLEX_NUPS:
        if nup in description and chain[-1] in Y_RING_SLOT:
            module, copy = Y_RING_SLOT[chain[-1]]
            return module, NUP_SLUG[nup], copy
    for substring, chains, module, copy in CHAIN_ASSIGNMENTS:
        if substring in description and (chains is None or chain in chains):
            return module, NUP_SLUG[substring], copy
    raise KeyError(f"unassigned chain: description={description!r} chain={chain!r}")


#: Separator for the composite ``module/node`` key. A node NAME alone is not
#: unique — ``nup160_inner`` is a legitimate child of both the cytoplasmic and
#: the nuclear ring — so every selection must key on the PAIR. Selecting on the
#: name alone silently writes each Y-complex node twice, once under each module,
#: each copy holding both modules' atoms: duplicated geometry, inflated counts,
#: and visible cross-node depth-order artefacts.
KEY_SEP = "\x00"


def node_keys(table: ProtomerTable) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-atom node keys and nucleoporin slugs.

    Args:
        table: The parsed protomer.

    Returns:
        Two ``(N,)`` arrays: composite ``module<sep>nucleoporin_copy`` keys and
        bare nucleoporin slugs.
    """
    cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
    keys = np.empty(len(table.positions), dtype=object)
    nucleoporins = np.empty(len(table.positions), dtype=object)
    for i, (desc, chain) in enumerate(zip(table.nucleoporin, table.chain)):
        pair = (desc, chain)
        if pair not in cache:
            module, slug, copy = assign_chain(desc, chain)
            cache[pair] = f"{module}{KEY_SEP}{slug}_{copy}", slug
        keys[i], nucleoporins[i] = cache[pair]
    return keys, nucleoporins


# =============================================================================
# Appearance
# =============================================================================

#: TRUE van der Waals radii in nm — no visibility fudge factor. At the scene's
#: 160 nm extent an atom is ~1 screen pixel at full-frame framing and resolves
#: into a solid molecular surface on zoom, so the honest number is also the one
#: that looks right. (Bonded C-C is 0.154 nm against a 0.170 nm radius, so
#: neighbouring spheres overlap heavily and the surface has no holes.)
VDW_RADII_NM: Dict[str, float] = {
    "C": 0.170,
    "N": 0.155,
    "O": 0.152,
    "S": 0.180,
    "P": 0.180,
}
DEFAULT_VDW_NM = 0.170

#: CPK element colours (linear light), for ``--color=element``.
CPK_COLORS: Dict[str, Tuple[float, float, float]] = {
    "C": (0.90, 0.90, 0.90),
    "N": (0.30, 0.50, 1.00),
    "O": (1.00, 0.30, 0.30),
    "S": (1.00, 0.90, 0.20),
    "P": (1.00, 0.50, 0.00),
}

#: Ambient-occlusion window, in nm, MEASURED against this structure rather than
#: assumed. Normalized contrast (std/mean) over the full 4.94M-atom dilated ring:
#:
#:   grid_cells   r=1.5   r=2.5   r=3.5   r=5.0 nm
#:   128          0.4215  0.4096  0.4025  0.4001
#:   192          0.4241  0.4046  0.3906  0.3782
#:   256          0.4397  0.4021  0.3784  0.3678
#:
#: Contrast falls as the window widens, because a window comparable to the
#: object stops reporting enclosure and starts reporting depth. 1.5 nm is both
#: the measured optimum and the biophysically meaningful scale — it is about one
#: hydration shell, so the term tracks an atom's BURIAL. Keep the radius
#: comfortably above the cell size (extent / grid_cells = 0.62 nm here) or the
#: window rounds to one cell and the term flattens.
AO_RADIUS_NM = 1.5

#: 256 measured best above and costs ~7 s for 4.94M atoms. The rotated grid is
#: the memory driver; drop to 192 (a 3.5% contrast loss) on a small machine.
AO_GRID_CELLS = 256

#: Fraction of the ambient illumination that is DIRECT, and so occludable. At
#: 1.0 the whole ambient is occludable, leaving no indirect floor — deliberately
#: strong. The colours are a CATEGORICAL encoding (per module or per element)
#: and a scalar multiplier preserves hue while changing only lightness, so
#: identity survives the darkening.
AO_STRENGTH = 1.0


def element_radii(elements: np.ndarray) -> np.ndarray:
    """Look up per-atom van der Waals radii in nanometres.

    Args:
        elements: ``(N,)`` element symbols.

    Returns:
        ``(N,)`` float32 radii in nm.
    """
    radii = np.full(len(elements), DEFAULT_VDW_NM, dtype=np.float32)
    for symbol, radius in VDW_RADII_NM.items():
        radii[elements == symbol] = radius
    return radii


def base_colors(
    color_by: str,
    modules: np.ndarray,
    nucleoporins: np.ndarray,
    elements: np.ndarray,
    protomer: np.ndarray,
) -> np.ndarray:
    """Build the unshaded per-atom colour array for a colouring scheme.

    Args:
        color_by: ``"module"``, ``"nucleoporin"``, ``"element"`` or ``"protomer"``.
        modules: ``(N,)`` structural-module names.
        nucleoporins: ``(N,)`` bare nucleoporin slugs.
        elements: ``(N,)`` element symbols.
        protomer: ``(N,)`` protomer index.

    Returns:
        ``(N, 3)`` float32 linear-light colours.
    """
    if color_by == "module":
        return _lookup_colors(modules, MODULE_COLORS, (0.7, 0.7, 0.7))
    if color_by == "element":
        return _lookup_colors(elements, CPK_COLORS, (0.9, 0.9, 0.9))
    if color_by == "protomer":
        return _hue_wheel(protomer.astype(np.int64), int(protomer.max()) + 1)
    if color_by == "nucleoporin":
        slugs = sorted(set(NUP_SLUG.values()))
        index = {s: i for i, s in enumerate(slugs)}
        return _hue_wheel(np.array([index[s] for s in nucleoporins]), len(slugs))
    raise ValueError(f"unknown colour scheme {color_by!r}")


def _lookup_colors(
    keys: np.ndarray,
    table: Dict[str, Tuple[float, float, float]],
    fallback: Tuple[float, float, float],
) -> np.ndarray:
    """Map string keys through a colour table.

    Args:
        keys: ``(N,)`` string keys.
        table: Key to linear-light RGB.
        fallback: Colour for keys absent from ``table``.

    Returns:
        ``(N, 3)`` float32 colours.
    """
    colors = np.tile(np.array(fallback, dtype=np.float32), (len(keys), 1))
    for key, rgb in table.items():
        colors[keys == key] = rgb
    return colors


def _hue_wheel(index: np.ndarray, n: int) -> np.ndarray:
    """Assign evenly spaced saturated hues to integer categories.

    Args:
        index: ``(N,)`` integer category indices.
        n: Total number of categories.

    Returns:
        ``(N, 3)`` float32 linear-light colours.
    """
    hue = (index.astype(np.float32) / max(n, 1)) * 6.0
    chroma = 1.0 - np.abs((hue % 2.0) - 1.0)
    sector = np.floor(hue).astype(np.int64) % 6
    ramp = np.stack(
        [
            np.choose(
                sector,
                [
                    np.ones_like(chroma),
                    chroma,
                    np.zeros_like(chroma),
                    np.zeros_like(chroma),
                    chroma,
                    np.ones_like(chroma),
                ],
            ),
            np.choose(
                sector,
                [
                    chroma,
                    np.ones_like(chroma),
                    np.ones_like(chroma),
                    chroma,
                    np.zeros_like(chroma),
                    np.zeros_like(chroma),
                ],
            ),
            np.choose(
                sector,
                [
                    np.zeros_like(chroma),
                    np.zeros_like(chroma),
                    chroma,
                    np.ones_like(chroma),
                    np.ones_like(chroma),
                    chroma,
                ],
            ),
        ],
        axis=1,
    )
    return ramp.astype(np.float32)


# =============================================================================
# Scene construction
# =============================================================================

#: Atoms per BSP part. This is the load-bearing number in the whole scene, for
#: two independent reasons.
#:
#: **Correctness.** The NPC's subunits are concave and INTERPENETRATE — a
#: Y-complex arm threads past the inner ring, gp210 wraps the scaffold. Two
#: concave interpenetrating objects have no valid whole-object draw order, and
#: the viewer's cross-node ordering is per-object (bounding-sphere view-z, with
#: a containment pass). So splitting the complex by PROTEIN cannot be ordered
#: correctly at all, no matter how the parts are sorted. Splitting it by SPACE
#: can: BSP cells are disjoint convex boxes, so a back-to-front order always
#: exists, and ``partition=`` records the split planes (``bsp_tree``) that let
#: the viewer traverse them exactly (Fuchs-Kedem-Naylor) — including with the
#: camera inside the volume, which is what happens flying down the channel.
#:
#: **Capacity** — and this one is tighter than the node total makes it look, in
#: both directions. One Points node clamps at ``floor(4096/3) * maxTextureSize``
#: = 5,591,040 elements on a 4096-class GPU; past that ``clampElementCapacity``
#: TRUNCATES, with one browser-console warning and no Python-side error.
#:
#: The two-state scene stores 9,874,128 atoms, but ``state`` is a HIDDEN axis, so
#: the viewer slices to one state and the resident set is **4,937,064** —
#: under the clamp, not over it. A single unpartitioned node would therefore
#: not silently drop ~4.3M atoms, as an earlier version of this note said; only
#: the resident slice is allocated (see ``demos/_lod_policy`` on why the node
#: total is the wrong number to compare against a ceiling, and note the compiler
#: warns on the total anyway).
#:
#: What is true is that 4,937,064 leaves only 12% headroom, which is no margin at
#: all for a demo that may gain atoms — and the correctness reason above is
#: decisive on its own, so the partition does not depend on this argument.
#:
#: 500k gives 32 parts for the two-state all-atom scene: few enough that first
#: paint stays request-cheap (each node is ~1 request), small enough for useful
#: frustum culling. Note ``partition=`` is EAGER — every part is fetched, and the
#: GPU only frustum-culls at draw time — which is why the node also carries an
#: additive ladder; without one, first paint was the whole resident state.
MAX_ELEMENTS_PER_PART = 500_000

#: Soft-edged but crisp atoms; a normalized [0, 1] knob, uniform so the writer
#: broadcasts it instead of storing 10M floats.
ATOM_SHARPNESS = 0.85


def build_state(entry: str, representation: str, cache_dir: str) -> Dict[str, Any]:
    """Download, parse and eight-fold-expand one conformational state.

    Args:
        entry: PDB id, e.g. ``"7R5J"``.
        representation: ``"all"``, ``"backbone"`` or ``"calpha"``.
        cache_dir: ``cached_download`` namespace.

    Returns:
        Dict with ``positions`` (nm), ``elements``, ``keys``, ``modules``,
        ``nups``, ``protomer`` and ``radii``, each covering the full ring, plus
        its integer ``n_fold`` symmetry order.
    """
    url = f"https://files.rcsb.org/download/{entry}.cif.gz"
    path = cached_download(url, cache_dir, f"{entry}.cif.gz")
    table = read_protomer(Path(path), atom_filter=representation)
    keys, nucleoporins = node_keys(table)
    n_pro = len(table.positions)
    n_fold = len(table.operators)
    aprint(
        f"✓ {entry}: {n_pro:,} atoms, {len(set(table.chain))} chains, "
        f"{len(set(table.nucleoporin))} nucleoporins in the protomer"
    )
    ring = expand_eightfold(table.positions, table.operators)
    return {
        "positions": ring,
        "elements": np.tile(table.elements, n_fold),
        "keys": np.tile(np.asarray(keys), n_fold),
        "modules": np.tile(
            np.array([key.split(KEY_SEP)[0] for key in keys], dtype=object), n_fold
        ),
        "nups": np.tile(nucleoporins, n_fold),
        "protomer": np.repeat(np.arange(n_fold, dtype=np.int32), n_pro),
        "radii": np.tile(element_radii(table.elements), n_fold),
        "n_fold": n_fold,
    }


def shade_states(states: List[Dict[str, Any]], color_by: str) -> None:
    """Bake burial shading into each state's colours, in place.

    Occlusion is computed per state — the two conformations must never occlude
    each other — but the normalization is SHARED across states, so scrubbing the
    state axis does not step the overall brightness.

    Args:
        states: The per-state dicts from :func:`build_state`.
        color_by: The colouring scheme to shade.
    """
    occlusions = []
    for state in states:
        raw = bake_ambient_occlusion(
            state["positions"],
            mass=(state["radii"].astype(np.float64) ** 3),
            radius=AO_RADIUS_NM,
            grid_cells=AO_GRID_CELLS,
            strength=AO_STRENGTH,
        )
        occlusions.append(symmetrize(raw, state["n_fold"]))
    peak = max(float(o.max()) for o in occlusions)
    for state, occlusion in zip(states, occlusions):
        shade = occlusion / max(peak, 1e-6)
        rgb = base_colors(
            color_by,
            state["modules"],
            state["nups"],
            state["elements"],
            state["protomer"],
        )
        state["colors"] = (rgb * shade[:, None]).astype(np.float32)
    aprint(
        f"✓ Burial shading baked per state (r={AO_RADIUS_NM} nm, "
        f"grid={AO_GRID_CELLS}), shared normalization, exactly C8-symmetric"
    )


def stack_node(
    states: List[Dict[str, Any]], key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather one node's atoms across every state into stacked arrays.

    Args:
        states: The per-state dicts, already shaded.
        key: The composite ``module<sep>node`` key to select. NOT the bare node
            name — see :data:`KEY_SEP`.

    Returns:
        ``(positions, colors, radii)`` where positions is ``(M, 3)`` when there
        is a single state and ``(M, 4)`` — x, y, z, state — when there are more.
        The state column is LAST so that the BSP splits on x/y/z; see the module
        docstring.
    """
    chunks, colors, radii = [], [], []
    for index, state in enumerate(states):
        mask = state["keys"] == key
        xyz = state["positions"][mask]
        if len(states) > 1:
            column = np.full((len(xyz), 1), float(index), dtype=np.float32)
            xyz = np.hstack([xyz, column])
        chunks.append(xyz)
        colors.append(state["colors"][mask])
        radii.append(state["radii"][mask])
    return np.vstack(chunks), np.vstack(colors), np.concatenate(radii)


def stack_all(
    states: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather EVERY atom of every state into one set of stacked arrays.

    Args:
        states: The per-state dicts, already shaded.

    Returns:
        ``(positions, colors, radii)``; positions is ``(M, 3)`` for a single
        state and ``(M, 4)`` — x, y, z, state — otherwise, with the state column
        LAST so the BSP splits on x/y/z. See the module docstring.
    """
    chunks, colors, radii = [], [], []
    for index, state in enumerate(states):
        xyz = state["positions"]
        if len(states) > 1:
            column = np.full((len(xyz), 1), float(index), dtype=np.float32)
            xyz = np.hstack([xyz, column])
        chunks.append(xyz)
        colors.append(state["colors"])
        radii.append(state["radii"])
    return np.vstack(chunks), np.vstack(colors), np.concatenate(radii)


def scene_dimensions(n_states: int) -> Dimensions:
    """Build the scene's dimensions, with the state axis last when present.

    Args:
        n_states: How many conformational states are in the scene.

    Returns:
        The :class:`Dimensions` for the scene.
    """
    dims = [
        Dimension("x", unit="nm", display=True),
        Dimension("y", unit="nm", display=True),
        Dimension("z", unit="nm", display=True),
    ]
    if n_states > 1:
        dims.append(
            Dimension(
                "state",
                unit="",
                categories=[label for label, _ in STATE_ENTRIES],
                display=False,
                spatial=False,
                description="Pore dilation (press '1' then '[' / ']')",
            )
        )
    return Dimensions(dims)


def add_npc_node(scene, states: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Add the whole complex as ONE BSP-partitioned Points node.

    This is the correct-by-default layout: see :data:`MAX_ELEMENTS_PER_PART` for
    why the complex is split by space rather than by protein. All atoms of all
    states go through a single ``add_points`` call, so there is exactly one
    order-dependent object in the scene and no cross-node ordering to get wrong.

    Args:
        scene: The scene to populate.
        states: The per-state dicts, already shaded.

    Returns:
        ``(n_atoms, n_nodes)`` where ``n_nodes`` is 1 (the partition wrapper,
        which is also a single Layers row).
    """
    positions, colors, radii = stack_all(states)
    scene.add_points(
        "nuclear_pore_complex",
        positions=positions,
        colors=colors,
        radii=radii,
        sharpness=ATOM_SHARPNESS,
        layer=True,
        blending_mode="normal",
        opacity=1.0,
        intensity=1.0,
        partition={"max_elements": MAX_ELEMENTS_PER_PART},
        # The partition stays (see MAX_ELEMENTS_PER_PART: it is load-bearing for
        # BOTH interpenetration ordering and capacity) but it did NOT stream. A
        # `partition=` alone is eager — every part is fetched and the GPU only
        # frustum-culls at draw time — so first paint was all 4,937,064 atoms of
        # the resident state across 32 parts. The ladder is resolved per part by
        # `_validate_counts`, which clamps a cumulative list to each part's own
        # count, so a ~308k part gets the geometric head and stops. Sized for the
        # PART, which is the unit the ladder is applied to, not the 9.87M node.
        additive_lod=stream_ladder(MAX_ELEMENTS_PER_PART),
    )
    return len(positions), 1


def add_nucleoporin_nodes(scene, states: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Add one Points node per nucleoporin copy, grouped by structural module.

    OPT-IN (``--split=nucleoporin``), and deliberately not the default. It turns
    the Layers panel into a nucleoporin browser — toggle gp210 off to see the
    scaffold, or leave only the Y-complexes on to see the coat — at the cost of
    correct compositing: ~60 sibling nodes whose concave bounds interpenetrate
    have no valid per-object draw order, so soft sprite fringes at module
    boundaries will occlude wrongly from some angles. Use it to INSPECT the
    architecture, not to judge how the complex looks.

    Args:
        scene: The scene to populate.
        states: The per-state dicts, already shaded.

    Returns:
        ``(n_atoms, n_nodes)`` actually written.
    """
    total_atoms = 0
    n_nodes = 0
    for module in MODULE_COLORS:
        keys = sorted({k for k in states[0]["keys"] if k.startswith(module + KEY_SEP)})
        group = scene.add_group(module)
        for key in keys:
            positions, colors, radii = stack_node(states, key)
            group.add_points(
                key.split(KEY_SEP)[1],
                positions=positions,
                colors=colors,
                radii=radii,
                sharpness=ATOM_SHARPNESS,
                layer=True,
                blending_mode="normal",
                opacity=1.0,
                intensity=1.0,
                partition={"max_elements": MAX_ELEMENTS_PER_PART},
            )
            total_atoms += len(positions)
            n_nodes += 1
        aprint(f"  {module:<24} {len(keys):3d} nodes")
    return total_atoms, n_nodes


def generate_nuclear_pore_complex(
    output_path: Path,
    which_states: str = "both",
    representation: str = "all",
    color_by: str = "module",
    split: str = "none",
) -> int:
    """Build the complete-NPC scene.

    Args:
        output_path: Where to write the ``.luxar.zarr``.
        which_states: ``"both"``, ``"dilated"`` or ``"constricted"``.
        representation: ``"all"``, ``"backbone"`` or ``"calpha"``.
        color_by: ``"module"``, ``"nucleoporin"``, ``"element"`` or ``"protomer"``.
        split: ``"none"`` for one BSP-partitioned node (correct compositing) or
            ``"nucleoporin"`` for one node per nucleoporin copy (inspectable
            Layers panel, approximate cross-node order).

    Returns:
        Total atoms written across all states.

    Raises:
        ValueError: On an unknown option, or if the written atom count does not
            match the input — which is how a node-key collision (writing the same
            atoms into two nodes) is caught rather than shipped.
    """
    wanted = _select_states(which_states)
    _atom_name_filter(representation)
    if color_by not in {"module", "nucleoporin", "element", "protomer"}:
        raise ValueError(f"unknown colour scheme {color_by!r}")
    if split not in {"none", "nucleoporin"}:
        raise ValueError(f"unknown split {split!r}; expected none|nucleoporin")

    with asection("Reading the deposited NPC protomers"):
        aprint("Mosalaganti et al. Science 2022 — the reference whole-NPC model")
        states = [
            build_state(entry, representation, "nuclear_pore_complex")
            for _, entry in wanted
        ]
    with asection("Applying the deposition's own eight-fold operators"):
        _report_geometry(states, wanted)
    with asection(f"Baking burial shading (colour by {color_by})"):
        shade_states(states, color_by)
    with asection("Creating Luxar scene"):
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=scene_dimensions(len(states)),
                citation=DEMO_META["citation"],
                viewer_config=ViewerConfig(cinematic_mode=True),
            )
            if split == "none":
                total, n_nodes = add_npc_node(scene, states)
            else:
                total, n_nodes = add_nucleoporin_nodes(scene, states)
            _add_annotations(scene, wanted, len(states[0]["positions"]))
        expected = sum(len(state["positions"]) for state in states)
        if total != expected:
            raise ValueError(
                f"wrote {total:,} atoms but the input has {expected:,} — a node "
                "selection is double-counting or dropping atoms"
            )
        aprint(f"✓ {total:,} atoms written across {n_nodes} node(s)")
        size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        aprint(f"  Dataset size: {size_mb / (1024 * 1024):.1f} MB")
    return total


def _select_states(which_states: str) -> List[Tuple[str, str]]:
    """Resolve the ``--state`` argument to entries to load.

    Args:
        which_states: ``"both"``, ``"dilated"`` or ``"constricted"``.

    Returns:
        The ``(label, pdb_id)`` pairs to load, in scrub order.

    Raises:
        ValueError: On an unknown state name.
    """
    if which_states == "both":
        return list(STATE_ENTRIES)
    matched = [e for e in STATE_ENTRIES if e[0].lower() == which_states.lower()]
    if not matched:
        names = "both, " + ", ".join(label.lower() for label, _ in STATE_ENTRIES)
        raise ValueError(f"unknown state {which_states!r}; expected one of: {names}")
    return matched


def _report_geometry(
    states: List[Dict[str, Any]], wanted: List[Tuple[str, str]]
) -> None:
    """Print the assembled dimensions, measured rather than quoted.

    Args:
        states: The per-state dicts.
        wanted: The matching ``(label, pdb_id)`` pairs.
    """
    for (label, entry), state in zip(wanted, states):
        pos = state["positions"]
        radial = np.hypot(pos[:, 0], pos[:, 1])
        aprint(
            f"✓ {label} ({entry}): {len(pos):,} atoms — "
            f"outer diameter {2 * radial.max():.1f} nm, "
            f"central channel {2 * radial.min():.1f} nm, "
            f"height {pos[:, 2].max() - pos[:, 2].min():.1f} nm"
        )


def _add_annotations(scene, states: Sequence[Tuple[str, str]], atom_count: int) -> None:
    """Add the title and caption overlays.

    Args:
        scene: The scene to annotate.
        states: Selected ``(label, pdb_id)`` pairs, which determine the caption.
        atom_count: Number of atoms in each selected conformational state.
    """
    scene.add_text(
        "Nuclear Pore Complex",
        position=(0.02, 0.02),
        font_size=0.055,
        anchor="top-left",
        color="rgba(255,255,255,0.6)",
        blend_mode="difference",
    )
    detail = (
        f"{atom_count:,} atoms • 808 chains • 25 nucleoporins • PDB 7R5J/7R5K"
        if len(states) > 1
        else f"{atom_count:,} atoms • 808 chains • 25 nucleoporins • PDB {states[0][1]}"
    )
    add_demo_caption(scene, detail, DEMO_META.get("citation"))


# =============================================================================
# Main Entry Point
# =============================================================================


def _parse_args(argv: Sequence[str]) -> Dict[str, str]:
    """Parse the demo's ``--key=value`` arguments.

    Args:
        argv: Argument list, excluding the program name.

    Returns:
        Mapping with ``state``, ``representation``, ``color`` and ``split`` keys.
    """
    options = {
        "state": "both",
        "representation": "all",
        "color": "module",
        "split": "none",
    }
    for arg in argv:
        for key in options:
            if arg.startswith(f"--{key}="):
                options[key] = arg.split("=", 1)[1]
    return options


def _print_banner(options: Dict[str, str]) -> None:
    """Print the representation-neutral startup banner."""
    aprint("=" * 70)
    aprint("NUCLEAR PORE COMPLEX — THE COMPLETE HUMAN NPC")
    aprint("=" * 70)
    aprint("")
    aprint("  808 chains · 25 nucleoporins · 6 modules")
    aprint("  PDB 7R5J (dilated) / 7R5K (constricted), Mosalaganti et al. 2022")
    aprint("  Assembled with the deposition's OWN C8 symmetry operators")
    aprint("")
    aprint(f"  state:          {options['state']}")
    aprint(f"  representation: {options['representation']}")
    aprint(f"  colour by:      {options['color']}")
    aprint(f"  node layout:    {options['split']}")
    aprint("")


def main() -> None:
    """Main demo entry point."""
    options = _parse_args(sys.argv[1:])
    _print_banner(options)

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "nuclear_pore_complex.luxar.zarr"
        _generate_or_exit(output_path, options)
        aprint(f"Dataset generated at {output_path}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_npc_") as tmpdir:
        output_path = Path(tmpdir) / "nuclear_pore_complex.luxar.zarr"
        _generate_or_exit(output_path, options)
        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("  - Top-down: the eight-fold ring and the real central channel")
        aprint("  - Side view: the three-ring stack, 76 nm tall")
        aprint("  - gp210's wings reach out into the perinuclear lumen")
        aprint("  - +z is the CYTOPLASM (RanBP2 fibrils); -z the NUCLEOPLASM (ELYS)")
        if options["state"] == "both":
            aprint("  - Press '1' then '[' / ']' to scrub constricted <-> dilated")
        if options["split"] == "nucleoporin":
            aprint("  - Press L for the Layers panel: one row per nucleoporin copy")
        else:
            aprint("  - Press L for the Layers panel: one BSP-partitioned NPC row")
            aprint("    Re-run with --split=nucleoporin for per-nucleoporin rows")
        aprint("")
        launch_viewer(output_path)
    aprint("Cleanup complete")


def _generate_or_exit(output_path: Path, options: Dict[str, str]) -> None:
    """Generate the scene, reporting a friendly diagnosis on failure.

    Args:
        output_path: Where to write the scene.
        options: Parsed command-line options.
    """
    try:
        generate_nuclear_pore_complex(
            output_path,
            which_states=options["state"],
            representation=options["representation"],
            color_by=options["color"],
            split=options["split"],
        )
    except Exception as error:  # noqa: BLE001 - demo-level diagnosis
        aprint(f"\n Error: {error}")
        aprint("\nPossible issues:")
        aprint("  - Network connection failed (needs ~28 MB from RCSB)")
        aprint("  - RCSB changed a deposition's chain naming")
        aprint("  - Not enough memory for the 4.9M-atom occlusion bake")
        sys.exit(1)


if __name__ == "__main__":
    main()
