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
