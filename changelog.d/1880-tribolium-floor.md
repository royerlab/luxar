#### Tribolium demos no longer fit the specimen's background haze

The four demos that share the tribolium fit (`gsplats_3d_tribolium_embryo`,
`gsplats_lod_tribolium`, `gsplats_recipes_tribolium`, `gsplats_lod_embryo_line`) were
spending most of their splat budget reconstructing background rather than nuclei.

The stack carries two background levels: a ~204-count detector offset in the medium
outside the embryo, and the specimen's own ~675-count autofluorescence inside it. The
default `floor="auto"` takes the histogram mode over the whole box, and since roughly
45% of this field of view is empty medium, the narrow 204-count peak is the tallest bin
— so `auto` stripped the detector offset and stopped, leaving 75.6% of the embryo's
interior mass as haze to be fitted. The median cap could not correct it either, because
the mode sat below the median.

The fit now subtracts the measured specimen level. Background reconstruction drops by
~4700x, nuclei-to-background contrast goes from 4x to over 12000x, and the dim nuclei
that were previously indistinguishable from haze (1.7x above it) now stand 784x clear of
it. Two separate measurements, since they answer different questions: the scene's total
emitted mass falls to 16% of its former value (so about 84% of what it emitted was
background), and independently, the share of the reconstruction landing in voxels the
source calls pure background drops from 54.6% to 0.06% while the share on nuclei rises
from 29.3% to 94.2%. Nuclei brightness is essentially unchanged, but the volumetric
absorption/opacity pairing was dialled on the pre-floor fit and still needs a live A/B
against the replacement.

The floor is measured in camera counts, where it can be checked against a histogram, and
converted to the fit's normalised space at the point of use. The value is documented
in-place alongside the sweep that chose it, including the arm that over-floors and
destroys the dim band, so it cannot be raised casually.

A fit now records the floor it used, so the embryo demo checks the cache on load and
warns when it finds a pre-floor one rather than quietly rendering the haze. Run any of
these demos once with `--recompute` to replace it; the downloaded archive and extracted
TIFF are reused when already present.
