"""Shared help text for batch fitting commands."""

BATCH_PROGRESSIVE_HELP = (
    "Optimize each tile in several passes against residuals. This is an "
    "optimization schedule and returns one flat splat set per tile, not a LOD "
    "ladder. Not supported with --tiling content. Build a streaming ladder "
    "with --merge-recipe stream, or run `luxar gsplat additive` on the merged store."
)
