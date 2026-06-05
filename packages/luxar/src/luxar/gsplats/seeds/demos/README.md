# luxar.gsplats.seeds.demos

Runnable demo scripts for seed generation methods.

## Demos

| Script | Description |
|--------|-------------|
| `demo_decomp_seeds_mitosis.py` | Compares grid, decomposition, edge, and combined seeding methods on the scikit-image human mitosis dataset. Loads each method's seeds as a separate, toggleable napari points layer over the original image for visual comparison. The combined method (`combine_seeds(decomp, edges, grid)`) mirrors what `fit_gaussian_splats` uses with `method='auto'`. |

Uses `seed_from_grid`, `seed_from_decomposition`, `seed_from_edges`, and
`combine_seeds` from `luxar.gsplats.seeds`. Each seeding method returns
`GSplatData` with scale-informed Gaussian shapes; the demo reads `.centers` for
display.

## Running

```bash
hatch run python packages/luxar/src/luxar/gsplats/seeds/demos/demo_decomp_seeds_mitosis.py

# Run all computations without launching the napari viewer (CI / headless)
hatch run python packages/luxar/src/luxar/gsplats/seeds/demos/demo_decomp_seeds_mitosis.py --no-napari
```

Requires napari and scikit-image.
