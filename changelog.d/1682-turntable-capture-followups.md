#### Offline capture stops fighting the render loop

Waking the animation loop for the duration of a smooth turntable or EXR sequence —
which is what made the camera rotate at all — had a cost nobody had looked at: every
tick ended in a full post-processing pass whose result was thrown away, because each
captured frame runs its own independent pipeline pass into an offscreen target. The
readback of that pass is asynchronous, so the loop ticked many times inside every
captured frame. For an EXR sequence it was worse than wasted work: the capture holds
global raw-HDR shader flags (effects off, no tone mapping) across the await, and the
capture overlay's scrim is translucent, so the viewport visibly flickered between the
normal image and a blown-out one for the whole sequence.

The loop now skips only its own render while an offline capture is active. Controls
and every per-frame callback still run each tick — the depth-sort scheduler and the
LOD group selector have to follow the orbiting camera, which is why the loop must run
in the first place. The suppression is offline-only: the real-time recording path
records the canvas the loop paints, and silencing it there would produce an empty
video. The trade-off is stated plainly: the viewport behind the translucent capture
overlay goes dark for the duration, not merely static — bringing the capture
resolution up already resized the render target, which cleared the canvas, and now
nothing repaints it. Progress is the overlay's own preview canvas, which shows each
captured frame; an EXR sequence never sets that preview, so it shows the frame counter
alone. That is what the viewport looked like before the loop was woken for a capture
at all, minus the flicker.

The capture's teardown also guarantees a repaint now. Restoring the recording state
resizes the render target back, which clears the canvas, and by that point the
keep-alive callback is gone — so a loop that was still stopped, or that the idle timer
halted in the gap right after the resize, left the viewer blank until the next mouse
move.

The confirmation dialog's promised frame count was rounded before it was multiplied,
while the capture loop and the panel's own Output field both use the unrounded
duration. At 7°/s and 30 FPS the dialog promised 1530 frames and a 51-second video
against the 1543 frames the capture actually produces; all three now agree.

Finally, the shared animation-controller test double never ran per-frame callbacks at
all — it was four bare stubs with no loop state — which made it structurally blind to
the class of bug that caused all of this: deleting the loop wake-up from the real-time
recording path failed none of the recording-panel tests. The double now models the real
stopped-loop semantics — only `startAnimation()` starts the loop, registering a
callback never does, and callbacks run only while it is running — and records the
`continuous` option, so the keep-alive that holds the loop open past the two-second
idle timer is asserted rather than assumed.
