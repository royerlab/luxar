#### `luxar gsplat decimate` — reduce a fit to a target splat count

A fitted dataset is usually bigger than it needs to be, and until now there was
no way to say so: `cull` removes splats by a quality threshold (ratios, no target
count) and `lod` builds a multi-level structure, so reaching a single smaller
dataset meant calling `make_substitutive_lod` and digging a level out of the
result by hand. `luxar gsplat decimate` (and `luxar.gsplats.lod.decimate`) takes
a count — `--target 165000` — or a share — `--fraction 0.1` — and returns one
flat leaf.

Two families are exposed, and which one you use is worth more than the count.
Measured on a 1.65M-splat light-sheet fit of a zebrafish embryo, scored as
FOREGROUND PSNR against the source volume (a global PSNR over a 97.8%-empty
stack mostly measures how well a scheme reproduces black, and flatters every
variant by ~15 dB):

    kept   merge     prefix
    50%    44.5 dB   45.5 dB   <- prefix wins; little redundancy left to merge
    25%    41.7 dB   39.1 dB
    10%    38.3 dB   34.5 dB   <- ~10x smaller at a still-high fidelity
     1%    33.1 dB   29.6 dB   <- merge wins by 3.5 dB

`merge` clusters neighbours into representatives carrying their combined mass;
`prefix` keeps the first N of an additive ordering and discards the rest, so the
object loses mass and dims. Below roughly half, merging leads by 3-4 dB at equal
count — merge at 10% matches what a prefix needs ~40% to reach — and above it the
ranking inverts, because there is little redundancy left to summarise and the
ordering is already in the file. `auto` follows that crossover.

Quality falls smoothly at ~3-4 dB per halving with no knee, so the docs give the
curve rather than a single recommended number.

#### Content-tiled fits can be added to a scene again

`gsplat fit --tiling content` (and the content plans `batch-fit` builds) balances
its boxes by feature density, so it leaves `max_elements` — a per-part CAP that
only a capped splitter sets — at its default of 0. The graft path fed that
straight into a validator requiring `>= 1`, so every content-tiled result failed
with `max_elements must be an int >= 1, got 0`: the whole `batch-fit` -> scene
route was unusable. The graft now derives the honest cap (the largest part) when
the source declares none, and passes a declared one through untouched.
