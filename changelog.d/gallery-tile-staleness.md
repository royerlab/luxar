#### Gallery tiles now report when their render inputs moved on

`make check-gallery-staleness` compares each committed README still/video pair
with the gallery capture policies, its demo generator, directly imported
private demo helpers, applicable `luxar.shading` production code, and its own
manifest entry. Stale or unknown media are reported without gating a reviewed
gallery refresh.
