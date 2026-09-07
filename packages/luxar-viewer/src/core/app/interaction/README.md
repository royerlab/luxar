# Element Interaction

> What happens when you **click** a picked element, as opposed to merely
> hovering it: open the author-defined link, or offer a right-click menu with
> `Copy`, `Open link in new tab` and `Copy link address`. Issue #1917.

## Overview

Before this folder existed, nothing in the viewer listened for `click`,
`contextmenu`, `mousedown` or `pointerdown` on the 3D canvas — both gestures
were swallowed by the camera controls (which `preventDefault` the native menu
in orbit and fly mode alike) and replaced by nothing.

An author opts in per **node**, with three plain `.zattrs` values that ride the
existing `**attrs` pass-through and need no adder signature:

```python
scene.add_gsplats(
    "proteins", centers=..., labels=protein_names, keys=accessions,
    link="https://www.uniprot.org/uniprotkb/{hover_key}/entry",
    copy="{hover_key}",             # optional; without a template, Copy uses the label
    link_target="_blank",           # optional; the default
)
```

They land in `userData.attrs` on each geometry leaf, which is what a pick hits.

## Gesture ownership

`canvas-gesture-ownership.ts` — `installCanvasGestureOwnership(canvas, events)`,
called from `LuxarApp.init()` before the init pipeline. Stamps `touch-action: none`
when the canvas's computed value is still `auto`, so an embedder's explicit choice
is respected; callout and text-selection suppression are stamped unconditionally.
It also cancels Safari's proprietary `gesturestart/gesturechange/gestureend` pinch
events on devices that report touch points. Without it the browser claims a
two-finger pinch as page zoom and the orbit controls' touch handlers never run. The
standalone page declares the same in `styles/base/layout.css` (`#app`); a
`LuxarLayer` host owns its canvas and sets `touch-action` itself.

## File Structure

```
interaction/
├── picked-element-cache.ts   # the settled pick + its staleness guard
├── element-actions.ts        # attrs → safe URL + copy string (pure)
└── canvas-actions.ts         # pointer/keyboard listeners, menu, clipboard, cursor
```

`initPicking` (`../picking/init-picking.ts`) constructs all three and registers
the listeners through the picking session's `EventGroup`, so one
`pickingEvents.dispose()` tears everything down together.

## `picked-element-cache.ts`

A click acts on the pick the tooltip is **already** showing, not on a fresh one.
Two reasons, and the second is the load-bearing one:

1. It is the element the user read the name of and decided to click.
2. It is synchronous. There is no "pick at (x, y)" — picking is hover-driven
   and settles asynchronously through a GPU readback — and an `await` risks
   spending the browser's _transient user activation_, which turns "open the
   link" into "popup blocked".

Staleness is handled entirely by counters `PickingSystem` already maintains, so
this folder invents no invalidation scheme of its own:

| signal             | advances on                                                                                                                                               |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pickGeneration`   | `markDirty()` — camera move, resize, persp↔ortho, FOV edit (#1916), layers-panel invalidator — plus any mousemove, mouseleave, dispose, and each new pick |
| `visibleSignature` | a layer being hidden or shown, which deliberately does **not** dirty the buffer                                                                           |
| cursor position    | belt-and-braces: real movement fires `mousemove`, which already advances the generation                                                                   |

**`suppress()` deliberately does not advance `pickGeneration`.** pointerdown →
controls `start` → `suppress(true)` is the first half of an ordinary click; if
suppression invalidated the cache, every click would refuse itself and the
feature would be dead on arrival with no obvious cause. A test pins this.

`peek()` skips the position check, for the two consumers that have no cursor
event: the pointer-cursor affordance and the Shift+F10 keyboard path.

## `element-actions.ts`

Pure. Turns a node's templates into `{ url, target, copyText }`.

Both halves of a link are untrusted — the template comes from `.zattrs`, the
substituted values from per-element data — so the guarantees are established in
order:

1. Substituted values are `encodeURIComponent`-escaped
   (`../../../utils/hover-template.ts`), so a label contributes _content_ but
   never _structure_: no injected path segment, query, fragment or authority.
2. `new URL(built)` with **no base**, so a relative template throws instead of
   resolving against the viewer's own origin (a third-party store must not be
   able to aim a click at an embedder's site).
3. Scheme must be `http:`/`https:` — an **allowlist**, deliberately stricter
   than the denylist `OverlayManager.sanitizeHtml` uses for rendered markup.
   That decides what to _display_; this decides where to _navigate_.
4. Length capped.

**An empty substitution suppresses the action.** `https://uniprot.org/{hover_label}`
on an unlabelled element would otherwise become `https://uniprot.org/` — a
valid URL to the wrong place. Not hypothetical: under `substitutive_lod=` the
Python adders copy non-compositing attrs onto the coarse children too, and
those synthesised gsplat levels carry no labels, so it is the _normal_ case at
coarse LOD.

Templates are read from the hit leaf, then from the nearest ancestor carrying
any. Both placements occur — `COMPOSITING_ATTRS` in
`packages/luxar/src/luxar/core/group/compositing.py` sends a non-compositing attr to every
`part_<i>` leaf and a compositing one to the wrapper alone — and the walk makes
the viewer indifferent to that choice. The nearest level wins outright;
templates are never merged across levels, so a leaf declaring only `copy` does
not silently inherit an ancestor's `link`.

## `canvas-actions.ts`

**Click vs camera gesture is decided by movement, never by button.** Which
button orbits and which pans is platform-dependent (`naturalDrag` defaults to
`isMacPlatform()`, and the ortho controls map RIGHT to nothing), so any
button-based rule would be wrong on half the installs. A press that moves no
further than `CLICK_SLOP_PX` is a click; anything further is left to the
controls. A second pointer (pinch) suppresses the gesture entirely.

The menu opens on `pointerup`, **not** on the `contextmenu` event: on macOS
that event fires at press time, so a right-drag to rotate would pop a menu the
instant the drag began. macOS Ctrl+primary-click is also treated as secondary,
matching every native app.

`window.open`, the clipboard, the toast and the menu are all injected ports.
`window.open` is called nowhere else in the viewer and is stubbed in no other
test, so injection is the only way to assert the URL a click would navigate to
without opening a real popup.

The keyboard path arrives as a `luxar-open-element-menu` window event, because
the Shift+F10 / ContextMenu binding is registered once for the app's lifetime
while these listeners are rebuilt on every dataset load — the same decoupling
`open-dataset-browser` uses.

## The kill switch

`allowLinks: false` (option) / `?no-links` (URL) suppresses navigation, both
link menu items and the pointer cursor, while leaving `Copy` working — the
clipboard is not navigation. The `element-click` / `element-contextmenu`
embedder events still fire, with `link: null`, so a host can implement its own
behaviour instead. This is what an embedder showing scenes it did not author
needs in order to guarantee that no navigation can originate in data.

Like `selection`, an `element-click` / `element-contextmenu` listener present
at dataset-load time provisions the picking pipeline — so a host can drive its
own behaviour on a scene that declares neither labels nor templates, as long as
it subscribes before `init()` / `switchDataset()`.

## See Also

- [`../picking/README.md`](../picking/README.md) — the hover pipeline this builds on
- `../../../utils/hover-template.ts` — the `{hover_*}` vocabulary, shared with the tooltip
- `../../../ui/overlay-widgets/context-menu.ts` — the shared menu widget
- `docs/guides/user/LUXAR_ZARR_FORMAT.md` — the authoring contract
- `docs/guides/developer/UI_DESIGN_GUIDE.md` §7.8 — context-menu rules
