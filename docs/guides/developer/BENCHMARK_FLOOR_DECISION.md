# Why the manuscript benchmarks pin `--floor none`

**Decision 2026-07-12.** The manuscript's fit harnesses are pinned to
`--floor none`, while the shipped CLI default stays `--floor auto`. This page
exists because that looks like an inconsistency and is not one, and because
three places in the tree cite the reasoning:
`packages/luxar/src/luxar/tests/test_benchmark_floor_pin.py`,
`scripts/benchmark_progressive_psnr.py`, and `changelog.d/1184.md`.

> Relocated and reformatted from `TODO.md` when that file's release section was
> retired. The reasoning is unchanged.

## The short version

Keep the legacy no-floor numbers for the paper, for **pragmatic — not
scientific — reasons**, and document the choice. The initial call was a full
re-run with `floor=auto`; a quick measurement changed the *how*, not the
science.

## Scientifically, floor suppression is the right thing

The constant background pedestal is *not real signal*, so removing it before
fitting is principled. The *correct* way to score a floor-suppressed fit is
against a **floor-suppressed reference** — floor-recon vs floor-original.

## The measured −5.86 dB is a reference mismatch, not evidence floor is worse

kidney_dapi 31.77 → 25.91; blastocyst +0.01. That comparison scores a
background-free reconstruction against the *original, pedestal-bearing* volume,
penalising the fit for correctly dropping non-signal. Under the principled
floor-suppressed reference, floor would be fair — and appropriate.

## Why keep no-floor for the paper anyway

Adopting floor properly means switching the evaluation to background-relative
PSNR: a protocol *and* narrative change, plus re-checking the blind-spot
cross-validation story. That was not worth doing immediately before bioRxiv.
The committed legacy numbers **are** the `--floor none` numbers (committed
kidney 31.81 dB ≈ floor=none 31.77) and are internally consistent —
original-referenced throughout — so they stand.

## The tool default is confirmed correct and stays

For general CLI use the floor should always be removed by default, because
background is not signal. `gsplat fit` and `gsplat cal` keep `--floor auto`.

**Do not change the shipped default.** The `--floor none` pin is a *paper-only*
deviation for original-referenced comparability, not a statement about the tool.

## Resolution, without a GPU re-run

1. The numbers stand as-is.
2. **Pin the manuscript fit harnesses to `--floor none`** — `run_analysis`,
   `run_convergence`, `run_noise2self`, `progressive`, `loss_comparison` — so a
   future re-run stays reproducible instead of silently inheriting
   `floor=auto`.

   *Amended 2026-08 per #1184:* `run_noise_floor` was listed here in error. In
   the luxar-paper repo it only calls `estimate_noise_floor`, which never fits,
   so there is nothing to pin.
3. **A Methods paragraph** stating that the benchmarks use `--floor none` with
   original-referenced PSNR for comparability, while `floor=auto` — the shipped
   default — is the more principled fit for real use, and noting
   background-relative evaluation as appropriate future work.

The harness pin and Methods paragraph landed in `luxar-paper` PRs #8 and #7,
respectively, on 2026-07-12; this decision is closed.

## Deferred

Post-bioRxiv / journal / future work: the floor-suppressed-reference
evaluation, and the small open check of whether floor improves the blind-spot
CV / K\* selection (a `cal` sweep on a few datasets, not `run_all`).
