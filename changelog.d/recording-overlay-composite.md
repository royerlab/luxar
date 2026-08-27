#### Overlays now appear in real-time video recordings

Recording in Video mode produced a file with no overlays in it, however
"Include Overlays" was set. The real-time path fed `MediaRecorder` from
`canvas.captureStream()` on the WebGL canvas, and overlays are DOM
elements layered over that canvas — so there was nothing for the flag to
do, and it was never read. Only the screenshot and offline (turntable /
mp4 / image-sequence) paths composited them. Video mode is always
real-time WebM and the offline loop is turntable-only, so a Video-mode
recording had no way to include an overlay at all.

The real-time recorder now captures a mirror 2D canvas that is refreshed
once per rendered frame — a blit of the WebGL canvas plus the same
`compositeOverlays()` the screenshot path already used. Scenes with no
visible overlays, or with the flag off, still capture the WebGL canvas
directly and pay nothing. The refresh runs on the `frame-end` hook rather
than a `requestAnimationFrame` callback because the renderer keeps
`preserveDrawingBuffer: false`: reading the canvas one task later yields
a black frame. Measured cost is 0.1 ms median (0.3 ms max) per frame on a
4.7-megapixel canvas.

The confirmation dialog no longer warns about real-time WebM dropping
overlays, since it no longer does. EXR sequences keep the warning — they
carry the raw pre-grade HDR buffer, where a display-space overlay has no
meaning.
