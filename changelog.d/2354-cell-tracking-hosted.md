#### Seven cell-tracking crops are pinned for publication

Seven of the nine annotated crops behind `gsplats_cell_tracking` are uploaded to
the `cc-by` draft and pinned in the demo-data manifest: per crop, the finished
4D Gaussian-splat volume with its LOD ladder plus the lineage-track geometry.
The record is not public yet, so the demo still takes its existing Kaggle path
for now. Once the record is published, these pins make the default six-crop path
need no credentials, no 4 GB download and no GPU.

The seven are the crops with a complete 100-timepoint fit cache at the demo's
default 60,000 seeds. They cover the default `--datasets 6` with one spare;
`--datasets 8` or `9` still falls back to the Kaggle path, which the loader
already treats as a supported partial-record state rather than a fault, and
announces by name.

Splat counts run 2.2M to 4.6M per crop over the 100 timepoints, 632.4 MB in
total. The archive characteristics record the declared 100 × 64 × 256 × 256
uint16 source grid, so the publication table can state compression honestly.
The archives carry no PSNR and the record's table says `not scored` rather than
guessing: the bundle builder loads its per-timepoint fits without statistics, so
none reach the combined 4D store. The quality choice behind them is documented
where it was measured — `DEFAULT_SEEDS` in the demo records a nuclei-masked K
sweep (37.41 dB at 12,126 splats rising to 40.29 dB at 120,000, about +1 dB per
doubling with no plateau) and explains why 60,000 is the balance point.

Pinning a hosted-only dataset means editing `data_manifest.json` itself: with no
in-repo directory to scan, the generator preserves the committed list, and a
`files=` in the generator's spec would be silently discarded. The spec now says
so, and the bundle builder now points at the committed manifest rather than the
generator spec.
