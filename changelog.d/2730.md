#### Playback detail: pin the ladder depth for smooth, consistent time-lapse playback

Dimension playback streamed each frame under a per-tick time budget: a frame showed
whatever additive-ladder rungs were resident within its tick and the background prefetch
deepened the next frame by roughly one rung per tick. On a heavy time-lapse -- millions
of splats per timepoint -- most frames therefore landed at their first one or two rungs
and quality flickered from frame to frame. The dimension context menu gains a **Detail**
section (Auto, a rung count, or All) that pins the ladder depth: every frame is drawn at
exactly that many rungs, cold or not, and the tick waits for the data, so quality stays
constant and the frame rate adapts instead. The pin rides the same per-pass directive
path as the frame budget (`ViewState.ladderDepth`), reaches the t+1 shadow prefetch so
the next frame's cache entry already carries the pinned prefix, and is available
programmatically as `play(dim, { ladderDepth })` / `setLadderDepth`. Applies to all
four geometry loaders.
