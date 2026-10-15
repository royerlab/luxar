#### CLI progress streams when redirected; quiet runs still name their backend

Every `luxar` command now switches stdout and stderr to line buffering at startup. Python block-buffers a stdout that is not a terminal, so a long command run under `nohup`, `> log` or `| tee` used to show nothing until a buffer filled or the process exited — a `gsplat cal` log could stay empty for half an hour. Worker subprocesses (`fit --tile`, `--plan-box`, batch-fit tasks) are `luxar` invocations and get the same behavior.

`gsplat fit --quiet` (and the quiet inner fits of `cal` and content tiling) now prints a one-line `Fitting backend: …` summary naming the model class and kernel backend, once per process, so a fall back to the much slower PyTorch renderer is never silent.

Content-tiled fits now print one per-box progress stream in place of the per-box fit narration: a start line, iteration lines at 25/50/75% of each box, and a completion line with kept splats, the box PSNR on its padded crop, elapsed time and a budget-weighted ETA (`-j N` prints the completion line per finished worker). The inner box fits are quiet, and each distinct warning (such as the CPU-fallback ones) is shown once per run instead of once per box. The new `luxar.utils.verbosity.quiet_narration()` / `aprint_heartbeat()` pair provides this.

`gsplat compare -q` now silences the whole output tree rather than only the final table; failures still reach stderr.
