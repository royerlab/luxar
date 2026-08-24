#### Zebrafish timelapse: delete the shot noise instead of smoothing it

A third of this recording's energy at the early timepoints is shot noise, and
the fit was spending its budget on it. The noise is unusually literal: the
MEDIAN object in a frame is ONE voxel, and p90 is 1-2 at every timepoint. Each
timepoint now has its connected components below 4 voxels deleted before it is
fitted.

A size filter is used rather than a smoothing one because it matches that noise
model exactly, and because it cannot damage what it keeps. Measured against a
deliberately stricter cell definition than the threshold itself (components of
27 voxels or more, so no arm is scored against its own definition), every
threshold from 2 to 12 holds cell energy AND mean cell peak at exactly 1.0000.
There is no trade-off to tune; only how much noise goes, and that curve is flat
past 4 voxels. So 4 takes essentially all of the available reduction while
deleting nothing larger than 3 voxels.

Non-local means was tried first, at a strength picked by sweeping h and scoring
"the share of the noise voxels' energy the fit still reproduces". That metric
was wrong, and the way it was wrong is worth recording. NLM is a neighbourhood
average, so it SPREADS a spike rather than deleting it: energy that moved one
voxel out of its original spike left the mask the metric was watching and scored
as removed, while remaining plainly visible as a softer, wider blob. The metric
measured displacement and reported destruction. A second metric, energy outside
a dilated cell mask, failed the opposite way, counting NLM's own halo around
real cells as residual noise. The general lesson is not that one of these masks
is better: when a filter MOVES things, any mask fixed on the unfiltered data
measures the movement rather than the removal.

NLM also dimmed the cells it was meant to protect, to 0.78 of their raw peak at
the first timepoint. The component filter leaves every object at or above its
threshold bit-identical, so that cost is zero by construction rather than by
measurement -- which is also why the number does not have to be re-established
for another dataset.

The demo's module docstring records the method behind all of this rather than
just the numbers.
