#### Splat fits are reproducible across platforms

The same CPU fit on the same data used to give different results on Linux and
macOS, by up to 1 dB on a small tile. Seeding ranked candidates with NumPy's
default `argsort`, whose order for equal values depends on the CPU's SIMD
backend, so the flat background and the symmetric peaks of a blob came out in
a different order on x86 and arm64. Seed deduplication, the grid top-up and the
per-tile residual peak search all depend on that order, so the two machines
started the fit from different seeds.

Every sort that decides seed or peak order is now stable: brightest first, with
equal values kept in array (C) order. `torch.topk` in the tiled peak search is
replaced the same way. Fits are now the same on every platform up to
floating-point rounding, which a short fit can amplify to a few tenths of a dB.

Refitting data whose seeds had ties now starts from a different seed order, on
Linux as well as macOS, so the result can change by up to about 1 dB on a short
fit. On the tiled haze fixture, individual tiles moved by +0.94 and -0.75 dB.
