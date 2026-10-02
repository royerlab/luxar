#### Control panel tiles fit their text on small screens

On a phone, a tour with twenty chapters gives each tile a fingertip-sized
cell, and the tile text overflowed it. The top of the title was clipped along
with the end of the sublabel, and the tour numeral sat on top of two-line
titles.

The panel now measures its tiles after layout and gives text up in a fixed
order until every tile fits. It switches to the authored short sublabel, then
limits the sublabel to fewer lines, then drops it. Next it drops the numeral
and switches to the authored short label. Only then does it shrink the label,
and as a last resort it breaks a word. One step applies to the whole grid, so
the tiles stay alike. The grid shape is chosen with the text in mind: when
the best-shaped grid would cost text, the next two shapes are also tried, and
the one that keeps the most text wins. The numeral now has its own space at
the top of the tile, and any text that still overflows is clipped at the
bottom, never the top. Tablets and wall displays, where everything already
fit, look exactly as before.

`Chapter` gains `short_label` and `short_sublabel` for those shorter texts.
