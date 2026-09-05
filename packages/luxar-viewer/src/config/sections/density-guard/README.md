# density-guard

Projected-density guard configuration slice. Owns the cap on elements per
drawing-buffer pixel above which a data node counts as over-drawn, the floor
of the per-node keep-fraction ladder, and the hysteresis ratios around the cap.

Why it exists: frame cost on element-dense views tracks elements per pixel,
not pixels. A 1.5 M-point example framed into ~1 600 px rendered at 42 ms per
frame at DPR 1 and 83 ms at DPR 0.5, while the same points dollied 4× closer
rendered at 120 fps — so resolution scaling moves in the wrong direction on
exactly these scenes. The guard measures the density per node each frame
(`scene/projected-density.ts`), feeds it to `__luxarDebug.getPerf().density`,
thins blendable nodes on the shaders with brightness compensation, and stops
the refinement loop from admitting rungs the view cannot resolve.

`?no-density-guard` clears `enabled` for a session. At runtime the guard is
also a **Density Guard** toggle in the Performance rail popover (right-click
the gauge), persisted per scene as `renderingControls.defaults.densityGuardEnabled`
— the same split as `adaptiveDPR.enabled` vs `adaptiveDPREnabled`. Turning it
off releases every thinned node (keep back to 1, brightness restored) and lets
any refinement rung the gate was holding back load; the URL flag wins over the
stored choice for that session without overwriting it.

Two consumers read the cap. The shader keep-fraction ladder
(`scene/density-guard.ts`) thins blendable nodes down to `capElementsPerPixel`
with hysteresis (`enterRatio` / `leaveRatio`) and a floor (`minKeepFraction`).
The refinement rung gate (`data/scene-loader/progressive/density-gate.ts`)
defers the next additive rung of any node already denser than its cap on
screen: `capElementsPerPixel` for blendable nodes, the tighter
`nonBlendableCapElementsPerPixel` for `max` / `normal` / `opaque`, which cannot
be thinned. Both consumers exclude shaded triangle meshes. Deferred rungs
resume when the camera moves in.
