#### Frame pacing stops a slow scene from livelocking the whole main thread (#1724)

`performance_benchmark_example.luxar.zarr` (100 point-cloud nodes, 100k points,
8 MB) rendered a healthy ~59 fps for about thirty seconds while its nodes
arrived, then collapsed to ~1 fps and never recovered. It also wedged the
Playwright/CDP control channel: `getState()` costs 0.5 ms in-page but took 111 s
across the bridge, and a trivial `page.evaluate` timed out at 15 s. The dataset
was consequently parked out of the all-examples smoke test.

None of the usual suspects was responsible. Pinning `?dpr=0.25` — sixteen times
fewer pixels — left the frame at 925 ms against 940 ms, and a hundred-fold
covered-area sweep did not move it either, so this was not fill rate. Cost was
strictly linear in element count (1k points 21 ms, 100k points 1030 ms) and
100k points in _one_ node cost the same as in a hundred, so it was not draw-call
or node count. Adaptive DPR was not starved out: it ran, walked 0.99 → 0.89 →
0.80, and its U-shape probe correctly rejected further steps because there was
no fill to shed. Time inside WebGL calls was under 2 ms per frame and an
explicit `finish()` returned in 0 ms. What was busy was the GPU process (~1100 %
CPU over thirteen SwiftShader threads at nice 5), starving the renderer's main
thread while V8 sat 99.4 % idle.

The mechanism is the animation loop's scheduling. Every frame re-arms
`requestAnimationFrame` back-to-back, so at ~1 s per frame the main thread is at
a 100 % duty cycle of long tasks and _nothing_ else ever gets a slot — not
worker message delivery, not a CDP `Runtime.callFunctionOn`. That is a livelock
rather than merely a slow render: the depth-sort worker sorted each node in
0.1 ms, but its Comlink replies were dispatched at ~0.5/s (round trips climbing
from 4 ms to 60 s), and every reply that landed staged an ordering apply that
called `requestRender()`, re-arming the loop. The rendering starved the very
hand-off that would have let it stop — 200 dispatches, ~120 still outstanding
after 70 s, `animating=true` throughout.

`AnimationController` now paces itself. When a frame's own cost exceeds
`config.animation.pacing.slowFrameMs` (250 ms — 4 fps, already far past any
interactive threshold, so every healthy frame rate keeps the untouched fast path)
for _two consecutive_ frames, the next frame is scheduled after a cooldown of
`min(maxCooldownMs, 25 % of the cost)` instead of re-arming immediately. The
cooldown is a bounded _fraction_ rather than a constant because a flat gap cannot
track the cost across the band where the fraction is what decides it — 250 ms to
1 s, where a fixed 100 ms over-yields at the bottom and under-yields at the top
(9 % of wall-clock after a 1 s frame, against the fraction's 20 %). Past 1 s the
`maxCooldownMs` clamp makes the shipped behaviour flat as well, which is its job:
it bounds the added latency of an on-demand repaint. (The measured wedge, at
~1042 ms a frame, sits essentially at that clamp.)

The gap is armed from a zero-delay hop — `setTimeout(0)` →
`setTimeout(cooldown)` → `requestAnimationFrame` — and that indirection is the
whole of it. The scheduling decision is taken at the _top_ of the frame so the
loop survives an exception thrown by controls, a per-frame callback or the
render; but a slow frame spends its second on browser rendering work that runs
after the rAF callback returns and inside the same main-thread task (which is
why the `longtask` entries read ~1042 ms while V8 sits 99.4 % idle). A timer
armed at the frame's start is therefore always already overdue by the time the
thread frees: it fires immediately and inserts nothing. The hop runs at the
first event-loop turn _after_ that work, and only then is the real cooldown
armed. Both halves of the fix then hold — no animation-frame request is
outstanding while the frame is drawn, so the compositor stops driving main
frames back-to-back, and a genuine cooldown follows it.

The cost measured is the frame _period_ minus the gap we ourselves inserted: the
wedge's second is non-JS main-thread time, so a JS-body span reads ~2 ms and
would never fire, and not subtracting our own gap would make pacing latch on for
the rest of the session. The measurement resets on the stopped→running edge, so
an idle rest or a tab-hide is not read as one enormous frame. Being a _period_
also means a foreign main-thread task of that size is charged to the loop even
when the loop did not cause it: scene loading runs frames in that neighbourhood
(the repo's own `tests/e2e/render-ticks.ts` measures ~230 ms periods, just under
the threshold), so a heavier load crosses it and does pace — which is what you
want there, since the cooldown yields to the decode tasks.

That last property is why the trigger is a _streak_ rather than a single frame.
The wedge is sustained — every frame ~1042 ms, forever — so waiting for a second
consecutive slow frame delays the first cooldown by exactly one frame and gives
up nothing, while a lone outlier (a GC pause, a shader compile, one synchronous
chunk decode, a foreign task charged to the loop because the measurement is a
period) is precisely what must _not_ be paced. Pacing one of those fed adaptive
DPR a cooldown it read as a 3.75 fps frame rate off a freshly cleared two-sample
window — an unprobed 10 % DPR scale-down — and pushed a just-under-`gapResetMs`
interval just over it, inventing a stall gap-reset that had not happened. A
single fast frame drops the streak back to zero, so recovery is immediate; an
_alternating_ slow/fast cadence is deliberately never paced, because the fast
frames are proof the main thread is already getting the slots a cooldown would
have bought. Sustained foreign work — a heavy scene load — still paces, which is
the behaviour you want there: yielding to those decode tasks is the point.

Frames are delayed, never skipped: each one that runs still emits exactly one
`frame-start` / `frame-end` pair, and still calls `recordFrame()` whenever it
does GPU work of its own (that call was already gated on the context-lost and
render-skip predicates, and pacing does not change the gate) — both on the real
clock.
The achieved frame rate really is lower and neither the FPS readout nor the DPR
control loop is told otherwise — a steady paced cadence is absorbed by the stall
detector's median-based outlier test rather than read as a gap, and the
isolated-hiccup interactions above cannot arise, because an isolated slow frame
is never paced. Pacing is suspended outright for the whole
of a recording, where the loop's exact cadence belongs to someone else — the
real-time MediaRecorder path records the canvas this loop paints, and the
offline capture drives its own `await requestAnimationFrame` cadence with
one-shot per-frame orbit callbacks registered on the controller. (A plain
screenshot needs no suspension: it reads the canvas after its own awaited
frame.) A suspend predicate that throws is treated as "not suspended", because
this scheduling point is the loop's only re-arm and a lost re-arm freezes the
viewer with no way back.

With pacing in place the same scene settles (`animating=false`) at 15 s and
`page.evaluate` answers in 1–5 ms, so the benchmark example is back in
`all-examples-smoke-test.spec.ts`, joined by a focused `frame-pacing.spec.ts`
that asserts the scene really loaded, then both halves of the fix: the loop idles
within a bounded budget (stably, not just for one poll tick), and the control
channel stays responsive afterwards under individually bounded probes. Neither
the spec nor the un-parked example is in the smoke subset
(`pnpm test:e2e:smoke`), so both run only in the full E2E suite.
