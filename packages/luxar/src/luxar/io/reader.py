"""luxar._io – Provides default compressor for Zarr datasets in Luxar scenes."""

from numcodecs import Blosc

# default compressor: fast, bit‑shuffle‑friendly
DEFAULT_COMP = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
