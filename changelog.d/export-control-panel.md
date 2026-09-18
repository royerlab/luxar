#### An exported folder that knows it is a kiosk

`luxar export` produced a folder that could host the touch-panel relay and
never mentioned whether the scene it contained had a panel to drive. The
relay was off by default, the README documented `--control` identically for
every scene, and an operator got a URL to type into a tablet by hand.

The exporter now reads the scene's own attributes, through the bi-format
`read_node_attrs` rather than by naming a metadata document — the attributes
live in `.zattrs` at zarr format 2 and under `attributes` in `zarr.json` at
format 3, and a literal filename would have answered "no control panel" for
half the stores in existence while looking like it worked.

A scene that declares a `viewer_config.control_panel` gets a `serve.py` whose
relay and network binding are **on by default**, because the alternative is an
exhibit operator reading a README to make the tablet work. A scene without one
behaves exactly as before: loopback, no relay, since a folder you
double-click should not start listening for anything that wants to drive it.
`--no-control` and `--host 127.0.0.1` override either way.

On start the script now prints the two URLs as a labelled pair — *Display*
for the big screen, *Control panel* for the tablet — followed by a QR of the
panel URL as terminal half-blocks, and writes the same code to
`control-qr.png` for printing or a second screen. Verified end to end on the
protein-universe scene: the exported folder runs under `/usr/bin/python3` with
nothing installed, serves both pages, and the PNG decodes to exactly the URL
printed above it.

The token default is unchanged, deliberately. With the relay on by default
that means an exhibit scene listens on the LAN without a secret, so the banner
now says in one line what that permits: anyone who can reach the machine can
drive the display, a foreign web page cannot (the relay checks the request
origin on the WebSocket handshake), and `--control-token` is there for a
network you do not control.

The README gained an *About this scene* section built from the attributes —
title, which dimensions are displayed and which are steppable, how many stops
the panel names, the citation — so the folder explains the thing it contains
and not only how to serve it. Getting that lookup wrong is silent, and did
happen: the dimensions live under `scene_dimensions.dimensions`, a top-level
`dimensions` finds nothing, and the README simply omitted those lines rather
than complaining. The shape is asserted in the tests now.

Native app bundles are deliberately untouched: a `.app` with an embedded
WebView has no console to print a URL to and needs its own pairing surface,
which is a separate design.

Four things the README got wrong, found by reading the shipped file rather
than the template. Quick Start said `python serve.py`, and a stock macOS has
no `python` at all -- it worked on the author's machine only because
miniconda was on the PATH, which is exactly the audience this folder is not
for. The Options block listed `--control` and `--host` as things to turn on
while the kiosk section above it said they were already the default, so the
two contradicted each other; those options now belong to the kiosk section
alone. The folder listing hard-coded its column padding for a data directory
called "data" and went crooked for any other name. And "stdlib only" was
nearly true rather than true: it now says what `luxar_qr.py` is and that
deleting it costs you the QR and nothing else.

The floor is tested, not asserted: the exported folder was run end to end
under `/usr/bin/python3`, which on this machine is Python 3.9.6, the same
version the README claims.

A kiosk export also gets a `TESTING.txt`, written only when the scene declares
a panel: a plain export has nothing non-obvious to check, whereas pairing a
tablet has an order to it and failure modes that look like software faults and
are not. It states what has NOT been verified as plainly as what has.

Cross-device reachability was then tested rather than assumed, by running the
exported folder on the Linux box and driving it from this Mac over the
network: both pages and the zarr data served, the relay completed its
handshake, the panel built its 21 tiles, and a tap landed the display on the
right stop. Two things came out of that which the code reading had got wrong.

The origin check does NOT reject a foreign origin during the handshake -- it
completes the handshake and then closes with 1008, which is the conforming
way to do it. A first probe read only the HTTP status line, saw `101` for
`https://evil.example.com` and looked like a security hole. Reading the frame
after the handshake shows `cross-origin handshake`, and an origin matching the
loaded URL is accepted, so the claim in the README holds -- but it held for a
reason the first test could not see.

And the panel showed ZERO tiles over the network while showing 21 on
loopback, which looked like a latency bug in the relay. It is not: the panel
asks the display for its stops and gives up after 30 seconds, and the harness
had both pages in one browser, where the display becomes a throttled
background tab. The same two pages in two separate browser processes work over
the same network. That is now in TESTING.txt, because anyone testing a kiosk
on one laptop with two tabs will hit it and report it as broken.

One more found by running it rather than testing it: a scene load printed
dozens of `BrokenPipeError` tracebacks. The viewer cancels in-flight chunk
fetches on every camera move and every level-of-detail decision, each
cancellation aborts a response the server is still writing, and
`socketserver` treats that as a handler crash. Nothing was broken -- the scene
worked throughout -- but the two URLs and the QR an operator needs were buried
under stack traces that read as the server falling over.

`ControlServer.handle_error` now swallows `BrokenPipeError`,
`ConnectionResetError` and `ConnectionAbortedError` and passes everything else
to the base implementation, so a real handler fault stays loud. The narrowness
is asserted in the tests, because the tempting fix -- a bare `except` around
`do_GET` -- would have hidden those too.

Worth recording how this was missed: every earlier check looked at HTTP status
codes and browser state and never at the server's own stderr. Forty
deliberately aborted mid-response requests now produce zero tracebacks, and a
full browser load with six story steps leaves the server output at its
27-line banner.
