#### The first blend-mode switch no longer links its shader on the click path (#1648)

Switching a layer's blending mode flips shader defines (`max`, `volumetric`,
gsplat `normal`, mesh `opaque` cutout), so the first click onto each variant was
the first time its program existed — and the link is synchronous. On a real
driver that is a hitch; on headless Chromium, which runs WebGL through
SwiftShader and exposes no `KHR_parallel_shader_compile`, it monopolized the main
thread for tens of seconds and starved every queued interaction behind it.

The viewer now pre-links each *distinct* reachable variant after a dataset load,
one compile per post-frame idle opportunity so visible rendering and input each
get a turn in between, and keeps the programs pinned with isolated keeper
materials until the source material is replaced or disposed. Variants are deduped
against an approximation of Three's own program cache key, so a mode that lands
on a program another mode already produced costs nothing, and a material whose
compile-time state moves for an unrelated reason (colormap enablement, the
element-texture width define, the mesh flat-normal variant) has its keepers
rebuilt rather than pinning programs that no longer match. The set is rebuilt
across scene replacement, material disposal, and WebGL context loss/restore.

Warming is WebGL-only — WebGPU/TSL sessions never configure it — and
`?noBlendWarmup` turns it off.

A dataset load waits for the warm-up before reporting readiness, so an
interaction cannot race the initial queue, but it waits on a *budget* rather than
on the queue emptying. The queue is not a finite set: every geometry commit
schedules the node it just populated, so a progressively-loading partition keeps
handing the drain new work for as long as it streams, and gating readiness on
"nothing left" would hold `window.__luxarDebug`, the `dataset-loaded` event and
`switchDataset`'s promise for the whole load. The same wait never returns at all
in a background tab, where `requestAnimationFrame` is not serviced and no turn
ever comes. After five seconds readiness is released and the remaining variants
keep warming behind it.
