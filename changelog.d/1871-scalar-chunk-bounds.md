#### Points and lines chunk bounds include decoded radii and widths

Point radii and line widths may be quantized as positive scalars, so their
decoded values can be slightly larger than the authored values used to build
spatial chunk bounds. The encoder now reports one conservative round-trip slack
for a positive-scalar array, and the points/lines ordering writers add it to
spatial radius/width pads. This keeps decoded geometry inside the bounds the
viewer uses to decide which chunks to fetch; categorical barrier dimensions
remain unexpanded.
