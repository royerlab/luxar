#### Turntable recording rotates again

Smooth (offline) turntable recording produced a video in which the camera never
moved — every frame was the opening pose. The turntable advances the camera from a
per-frame callback, and those only run while the rAF loop is animating; the viewer
idle-stops that loop after ~2 s of no interaction, which is exactly the state it is
in by the time you have read the Recording panel and confirmed the dialog. The
offline loop registered a `continuous` keep-alive callback, but that only keeps a
*running* loop alive — it never restarts a stopped one. Because the capture path
renders its own pipeline pass, frames were still produced and encoded normally, so
the failure was silent: a correct-looking file with no rotation in it.

The offline loop now wakes the animation loop before registering its keep-alive, the
same way the real-time recording path already did.
