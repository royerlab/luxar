#### MCFO neurons: converge the fit, and stop beading the axons

Thin axons in the FlyLight/FISBe demo rendered as chains of beads. Some of that
is real — MCFO axons have varicosities, and the original data dips below a
quarter of its local ridge on 11.9% of the filament skeleton — but the fit was
doubling it, to 22.2%.

The cause was not the data and not the splat budget. `fit_gaussian_splats`
defaults to `n_iters=1000`, which is *below* the CLI's own `draft` preset (2000)
and a fifth of `standard` (5000), so calling the Python API without a schedule
is not "default quality" but below the lowest preset the CLI offers. At 1000
iterations the splats never left their seed shape: edge seeding initialises them
isotropic at sigma = 1.0 voxel, and the fitted result measured sigma 0.87-1.16
voxels at a median axis ratio of 1.30. A one-voxel-wide axon rebuilt out of
one-voxel spheres spaced 2.5 sigma apart beads by construction.

Raising the iteration count alone is not enough, and briefly makes things worse.
Dynamic relocation periodically resets splats to isotropic sigma=0.5 with the
off-diagonals zeroed, undoing the shapes the extra iterations bought — a 5000
iteration fit with relocation on *lost* 1.1 dB. And `max_eccentricity` (default
10.0, an axis ratio of sqrt(10)) is inert at 1000 iterations but binds once
converged: 14.9% of splats pile against that ceiling, costing 2.65 dB. The demo
now sets all four knobs together in `NEURON_FIT_SCHEDULE`.

Measured on the annotated sample — skeleton points dipping below 25% of their
local ridge, and foreground PSNR against the raw data:

    shipped     1000 iters                      22.2%   22.94 dB
    +           5000 iters                      16.6%   21.86 dB
    +           enable_dynamic_ops=False        16.0%   23.54 dB
    +          10000 iters  (what ships)        16.5%   25.85 dB
                20000 iters (not used)          16.3%   26.06 dB

Beading converges by 5000 while fidelity keeps climbing to 10000 and then
flattens, so the schedule stops at 10000: +2.9 dB over what shipped and a third
of the beading gone, for a one-time 13-minute cached fit.

Three things that did NOT work, measured rather than assumed, so nobody retries
them: doubling the seeds to 2.4M made the geometry *worse* (splat spacing 2.45 ->
3.22 sigma, because finer subdivision shrinks sigma faster than it shrinks the
gaps) for 0.4 dB and 73% more data; dropping to 300k gave the tightest spacing
and the longest splats measured and still beaded more; and widening the render
truncation from 2.75 to 6.0 sigma moved the dips by 1.2 points, so the splats
genuinely do not reach each other.

`fit_cache_key` is bumped to v2 and now digests the schedule. It previously
covered seeds, floor and retention — none of which this change touches — so a
key blind to the schedule would have handed back the old beaded fit and made the
retune look like a no-op. The neuropil is deliberately left on the stock
defaults: it is a diffuse counterstain with no filaments to bead, and the solid
volume-rendered look depends on its current character.
