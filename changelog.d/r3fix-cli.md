#### `gsplat fit` refuses an existing output without `--overwrite`

`luxar gsplat fit` now stops with a usage error naming `--overwrite` when OUTPUT
already exists, as `gsplat lod` and `luxar export` do, instead of replacing it
silently. The internal `--tile k/M` and `--plan-box` workers that `-j N` and
`batch-fit` spawn are exempt, so batch resumption is unchanged. `--plan-only`
writes only a plan JSON (OUTPUT itself when it is not store-like, else
`<stem>.plan.json`) and is now refused unless the fit resolves to `--tiling
content`: elsewhere it was ignored and a full fit replaced OUTPUT.

#### Remote inputs are refused by name

`gsplat fit`, `cal`, `compare` and `info` read local paths only. A URL argument
(`https://`, `s3://`, `gs://`, ...) used to be reported as a missing file under a
slash-collapsed spelling; it now fails with a usage error that names the URL and
says to download the data first.

#### Tiled fits report their elapsed time

The closing `Done: N splats in Xs` line of a sequential tiled fit printed `0.0s`,
because a partition result carries no single fit time, and a `-j N` fit printed
no time at both. Both now report wall-clock time; a flat fit still reports its
optimization time.

#### Piped `--help` is no longer clipped to 80 columns

When stdout is not a terminal, `luxar ... --help` renders at 120 columns instead
of Rich's 80-column fallback, so agents and piped logs see whole option lines.
Only the console entry point does this, and only for a help request: importing
`luxar.cli` changes nothing and the environment is left alone. An explicit
`COLUMNS` or `TERMINAL_WIDTH` still wins.

#### `gsplat merge` flattens partition inputs itself

`gsplat merge` accepted only flat stores and told the user to `flatten` the
default output of a tiled fit first. It now flattens a partition or nested-LOD
input with the same streaming code `gsplat flatten` runs, staged in a temporary
store beside the output (or the system temp directory), and prints a line naming
the input it flattened. Flatness is decided from group attrs, so a flat input is
still read once.

#### `gsplat cal --pdf` checks for matplotlib before the sweep

A missing matplotlib was discovered only when the report was rendered, after the
whole K sweep. `cal --pdf` now checks first and exits with the install hint.

#### No range warning for values equal to a dimension's bounds

The scene's "values outside declared range" warning compared exactly, so a
coordinate that differs from a range bound only by float32 rounding (the same
number reached through float32 and float64) warned although both print as the
same value. The check now allows a few float32 ulps at each bound.
