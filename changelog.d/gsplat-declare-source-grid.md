#### A fit can be told what its source actually was

`gsplat fit` learned to record the volume its splats represent, but it recorded
the array it was *handed*. For the producers that matter that is not the same
thing: a demo pulls one channel out of a 5D OME-Zarr, downscales it and
normalizes it, and only then calls the fitter — so the recorded "source" was a
preprocessed float32 working copy, and every compression ratio computed from it
answered a question nobody asked.

`fit_gaussian_splats` now takes `source_shape` alongside `source_dtype`, so a
caller that preprocessed can declare the acquisition. Measured on the DAPI demo,
whose stored channel is 236x275x271 uint16 and which fits a 128³ normalized
copy:

```
before   compression: 83:1  (8.0 MB -> 98.5 KB)     # the working copy
after    compression: 349:1 (33.5 MB -> 98.3 KB)    # the acquisition
```

A factor of 4.2 — the downscale (8.4x fewer voxels) and the float32 cast (2x
more bytes) had been partly cancelling, which is exactly the kind of error that
looks plausible and survives review.

Three things keep the new number honest rather than merely bigger.

The declaration is marked. `source_declared: True` is stamped beside it, because
a stated denominator that is indistinguishable from a measured one is worse than
no denominator: nobody downstream could tell which they were reading.

The fitted grid is still reported separately, and `gsplat info` prints
`fitted at: 128 x 128 x 128 (downscaled before fitting)` under the source line.
A ratio against the acquisition necessarily folds in the decimation, and the
reader is entitled to see that rather than infer it.

A malformed declaration is refused at the call, not stored. Empty, zero,
negative and non-integer shapes all raise, because this number becomes the
denominator of a published figure and a bad one would otherwise surface much
later as a plausible ratio nobody can reproduce.
