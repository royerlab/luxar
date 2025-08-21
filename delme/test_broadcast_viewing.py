#!/usr/bin/env python3
"""
Test script to verify broadcast group behavior.
"""

import zarr
import numpy as np

# Check the broadcast example
z = zarr.open('examples/broadcast_clarification_example.zarr', 'r')

print("=== Checking broadcast metadata ===")
for group_name in ['ManualReplication', 'BroadcastReplication', 'PartialCoverage']:
    print(f"\n{group_name}:")
    attrs = dict(z[group_name].attrs)
    print(f"  Attributes: {attrs}")
    if 'broadcast_dims' in attrs:
        print(f"  📡 Broadcasting across: {attrs['broadcast_dims']}")
    else:
        print(f"  No broadcasting")
    
    pos = z[f'{group_name}/positions'][:]
    print(f"  Position shape: {pos.shape}")
    print(f"  Unique Time values: {np.unique(pos[:, 3])}")

print("\n=== Expected behavior ===")
print("ManualReplication: 9 points, Time=[0, 1, 2] - should show 3 points per time")
print("BroadcastReplication: 3 points, Time=[0] - should show 3 points at ALL times")
print("PartialCoverage: 2 points, Time=[0, 1] - should show 1 point at t=0, 1 at t=1, none at t=2")