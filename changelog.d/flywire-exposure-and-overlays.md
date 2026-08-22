#### FlyWire dims down, and its overlays stop fighting the brain

139K cell bodies summed into a wash that blew out the centre of the brain. The
fix is exposure rather than fewer neurons: the per-super-class Points layers drop
from `intensity=0.1` to `1/270.91`.

That reciprocal is not a magic number. The Layers panel's display-range slider IS
the `intensity` attr — the viewer recovers its window as
`[-offset/intensity, (1-offset)/intensity]` and only falls back to the data range
when `intensity` is exactly 1.0 (`computeDisplayRange`, viewer
`ui/layers/layer-state.ts`). So a window top of 270.91 is authored as
`intensity = 1/270.91`, and authoring it is what makes it survive a rebuild
instead of having to be re-dragged. The layers also move from `luminous` to
`additive`: at this exposure luminous's per-point falloff buys nothing visible,
and additive is order-independent, so the super-class layers composite the same
however they are toggled. The connection layers are deliberately untouched —
their own 0.08/0.08 was measured separately and the panel reading came from a
neuron layer (opacity 0.79).

The hover readout moves from mid-left to the top-right corner. Mid-left put it in
the middle of the frame's empty side, where it read as a caption for the brain
rather than as a response to the cursor, and it sat directly across from the NT
legend. Top-right is clear of the title, the legends and the footer.

Both legends now render as ONE bottom-left overlay holding them side by side,
rather than one bottom-left and one floating at mid-right. They are laid out with
a flex row instead of a second hand-picked position: the super-class legend's
width depends on its longest label and on how many digits its counts run to, so
any x-offset chosen for a separate overlay is a guess that breaks the moment a
class is renamed or the counts gain a digit.
