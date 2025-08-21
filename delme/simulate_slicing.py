#!/usr/bin/env python3
"""
Simulate what the viewer should show at each time slice.
"""

import zarr
import numpy as np

z = zarr.open('examples/broadcast_clarification_example.zarr', 'r')

# Scene has Time dimension with 3 values: 0, 1, 2
time_values = [0, 1, 2]

print("=== Simulating viewer behavior ===\n")

for t in time_values:
    print(f"Time = {t}:")
    print("-" * 40)
    
    # ManualReplication - regular slicing
    manual_pos = z['ManualReplication/positions'][:]
    manual_at_t = manual_pos[manual_pos[:, 3] == t]
    print(f"  ManualReplication: {len(manual_at_t)} points")
    if len(manual_at_t) > 0:
        for i, p in enumerate(manual_at_t):
            print(f"    Point {i}: X={p[0]:5.1f}, Y={p[1]:5.1f}")
    
    # BroadcastReplication - should show ALL points at ALL times
    broadcast_pos = z['BroadcastReplication/positions'][:]
    print(f"  BroadcastReplication: {len(broadcast_pos)} points (broadcasted)")
    for i, p in enumerate(broadcast_pos):
        print(f"    Point {i}: X={p[0]:5.1f}, Y={p[1]:5.1f}")
    
    # PartialCoverage - regular slicing
    partial_pos = z['PartialCoverage/positions'][:]
    partial_at_t = partial_pos[partial_pos[:, 3] == t]
    print(f"  PartialCoverage: {len(partial_at_t)} points")
    if len(partial_at_t) > 0:
        for i, p in enumerate(partial_at_t):
            print(f"    Point {i}: X={p[0]:5.1f}, Y={p[1]:5.1f}")
    
    print()

print("=== Visual Summary ===")
print("At Y=0 (red): 3 points at all times")
print("At Y=5 (green): 3 points at all times (broadcasted)")
print("At Y=-5 (blue): 1 point at t=0, 1 point at t=1, none at t=2")