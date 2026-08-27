#### Gallery tiles now report when their render inputs moved on

`make check-gallery-staleness` compares each committed README tile with its
demo generator, `luxar.shading` production code, and its own manifest entry.
Stale media are printed with the newer inputs that triggered the report, but
remain non-gating so a gallery refresh stays an explicit reviewed batch update.
