#### Recorded overlays keep the size they have on screen

Overlays composited into a recording were sized against the capture frame, but
they are authored — and laid out on screen — in viewport units: the overlay manager
writes `font-size` in `vh`, `width`/`size` in `vw`/`vh`, and an image with no
configured size takes its natural CSS-pixel size. Those only coincide when the canvas
fills the window and the capture is exactly canvas-sized, which recording never is.
A 64-pixel logo therefore stayed 64 pixels no matter how tall the frame was: it
occupied 8.9% of the viewer on screen but 6.0% of a 1080p recording and 3.0% of a 4K
one, shrinking as you asked for more resolution. In an embedded viewer, whose canvas
is a fraction of the window, text came out at half its on-screen size too.

Overlay sizes now convert through the viewport and the canvas's CSS box — the same
mapping the HTML-overlay branch already used — so a capture matches the screen at
Native, 1080p, 1440p and 4K alike, embedded or not. Text with a configured `width`
also wraps in the capture the way it wraps on screen, instead of running one long
line off the side of the frame.

Three related fixes came out of the same pass. "Native" resolution now records at the
canvas's own size instead of silently forcing 1080 (the tooltip has always said
"current canvas size", and the real-time path always honoured it). The confirmation
dialog's frame count and duration now agree with the panel's own Output field and
with the capture loop — at 7 °/s it promised 3060 frames and 51 s for a capture that
runs 3086 frames and 51.4 s. And when overlays are visible but the chosen path cannot
composite them — real-time WebM records the canvas alone, EXR writes the raw HDR
buffer — the dialog says so rather than quietly producing a file without them.
