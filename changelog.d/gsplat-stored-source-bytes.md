#### Compression is quoted against both the raw voxels and the file you download

The compression line compared unlike things. Its numerator, `source_bytes`, is
the DECODED array — voxels × itemsize — while its denominator is the splat store
as it sits on disk, which is compressed. So the ratio silently credited the
splats with whatever the source format's own codec was already achieving.

On the DAPI demo that is not a rounding difference:

```
349:1   against the decoded acquisition (33.5 MiB)
 10:1   against what is actually downloaded (~1 MB OME-Zarr)
```

Both are true and they answer different questions. "The splat representation is
349x smaller than the raw voxel array" is how volumetric compression is normally
quoted. "This dataset is 349x smaller than the data it came from" is what a
reader would infer, and that one would be wrong by 35x.

A fit can now be told `source_stored_bytes` — what the acquisition occupies, as
opposed to what it decodes to — and `gsplat info` prints both ratios, each
labelled with its basis:

```
Source volume: 96 x 128 x 128 uint16 (1,572,864 voxels) [declared by the producer]
  occupancy:   2.197% of voxels above 1% of the intensity range
  voxels/splat: 372
  stored source: 976.6 KB (as downloaded)
  compression: 172:1 vs raw voxels (3.0 MB -> 17.9 KB)
               55:1 vs the stored source (976.6 KB -> 17.9 KB)
```

The stored size is never inferred from the decoded one — the codec's factor is
exactly what is unknown — so a dataset that does not declare it prints one ratio
rather than a guessed second. Zero, negative, fractional, boolean and string
values are all refused at the call, since this becomes a published denominator.
