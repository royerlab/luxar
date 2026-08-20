#!/usr/bin/env python3
"""Compatibility CLI for the packaged Gaia DR3 catalog builder."""

import sys

from luxar.demos._gaia_catalog import main

if __name__ == "__main__":
    sys.exit(main())
