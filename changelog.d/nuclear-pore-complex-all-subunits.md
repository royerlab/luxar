#### Nuclear Pore Complex: the whole complex, all 25 nucleoporins, both dilation states

The `nuclear_pore_complex` demo was depicting the wrong molecule, wrongly assembled. It
downloaded PDB **3I4R** and described it as the "Nup107-160 Y-complex (one spoke), 10 proteins
per complex". 3I4R is nothing of the kind: it is a two-chain crystal fragment — Nup107 residues
658-925 and the Nup133 helical domain, 5,161 atoms — from a paper about Nup133's evolutionary
kinship with Nup157/170. Two of the Y-complex's ten proteins, and only fragments of those two.

The eight-fold assembly was then geometrically impossible. `spoke_radius` was an invented
8.0 nm ("Scaled down for visualization") while the fragment's own radius from its centroid is
8.26 nm, so adjacent copies interpenetrated: minimum inter-spoke atom distance **0.30 Å**, with
9.3% of atoms holding a sub-3 Å contact in a neighbouring copy — 115x the clash rate of the real
deposition. Every dimension was wrong, and inconsistently so, since intra-spoke coordinates were
true-scale while only the ring radius was fabricated: 25.6 nm outer diameter against ~160 nm,
an 8.2 nm channel against ~50 nm, and a **9.6 nm axial height** against ~76 nm — a flat pancake
where the NPC's defining feature is a three-ring stack. The docstring stated the correct figures
(~120 nm, ~40 nm) directly above the code that ignored them. Missing entirely: the inner ring,
the membrane and transmembrane rings, the FG central channel, the cytoplasmic filaments, ELYS,
and the nuclear ring.

It now renders the reference model of the whole thing — **4,937,064 atoms, 808 protein chains,
25 distinct nucleoporins** — from PDB **7R5J** (dilated) and **7R5K** (constricted), Mosalaganti
et al., *Science* 2022. Crucially the assembly is no longer invented but read off the deposition:
each entry deposits one C8 protomer (101 chains, 617,133 atoms) plus the eight operators that
generate its self-declared `808-meric` biological assembly, and the demo applies exactly those
`_pdbx_struct_oper_list` rotations about the axis they define. Measured on the assembled result:
outer diameter 149.9/159.7 nm (constricted/dilated), central channel 41.1/53.1 nm, axial height
72.2/76.5 nm. Atoms carry true van der Waals radii, so a scene unit really is a nanometre.

Both entries deposit the same 617,133 atoms in the same order, so the two states become a hidden
categorical `state` axis and scrubbing it is a genuine conformational morph rather than a
crossfade between unrelated point clouds. Verified in-browser: the rendered channel wall moves
from 76 px to 98 px, a ratio of 1.289 against the coordinates' predicted 26.5/20.6 = 1.286.

Six structural modules, with every one of the 101 chains assigned by a curated table rather than
inferred — the deposition encodes ring identity in its chain names (Nup160 `R0`/`R1` are the
cytoplasmic inner/outer Y-complexes, `R2`/`R3` the nuclear ones), giving the published 32
Y-complexes as 16 cytoplasmic + 16 nuclear with ELYS nuclear-only. A test pins the full chain
inventory, so a chain that stops matching goes red instead of silently dropping a protein. What
the model does *not* contain is now stated rather than glossed: the nuclear basket
(Tpr/Nup153/Nup50/ZC3HC1), most FG repeat regions, and the membrane itself; and the docstring
says plainly that this is a 50 Å cryo-ET integrative model of AlphaFold-predicted nucleoporins,
not an experimental atomic structure.

Two rendering findings drove the scene layout, both worth knowing beyond this demo.

The complex is written as **one** BSP-partitioned Points node, not one node per protein. The
NPC's subunits are concave and interpenetrate — a Y-complex arm threads past the inner ring,
gp210 wraps the scaffold — and cross-node ordering is per-object, so two concave interpenetrating
objects have no valid whole-object draw order under any sorting rule. Splitting by *space* does:
`partition={"max_elements": 500_000}` records the recursive split planes, and the viewer's
Fuchs-Kedem-Naylor traversal of them is exact for point BSP cells, including with the camera
inside the central channel where the centroid fallback degenerates. The partition is also a hard
capacity requirement: one Points node clamps at `floor(4096/3) * maxTextureSize` = 5,591,040
elements and then *truncates* with only a browser-console warning, so an unpartitioned 9,874,128-
atom node would have silently dropped ~4.3M atoms. `--split=nucleoporin` opts into ~60
per-nucleoporin nodes for inspecting the architecture in the Layers panel, documented as trading
correct compositing for that.

The hidden `state` axis is placed **last**, against the house convention of putting a hidden
categorical dimension first. `spatial_bsp_tree` always splits on positions columns 0-2 whatever
they mean, and the viewer discards a `bsp_tree` whose split axis is not currently displayed — so
a state-first layout leaves `displayDims == [1, 2, 3]`, makes axis 0 undisplayed, and quietly
reverts every partition to approximate centroid ordering. With x/y/z first, `displayDims ==
[0, 1, 2]` is the one case where a stored split column *is* the screen component.

Baked ambient occlusion is kept as a burial cue but retuned for the real structure: the window
is 1.5 nm at `grid_cells=256`, which measured highest normalized contrast (0.4397 versus 0.4021
at 2.5 nm and 0.3678 at 5.0 nm) on the actual 4.94M-atom ring. Occlusion is computed per state,
so the two conformations never occlude each other, but normalized across both so scrubbing does
not step the brightness; each atom is then averaged with its seven symmetry mates, making the
shading exactly eight-fold symmetric as the geometry is.

The scene is 55.5 MB and builds in about 48 seconds.
