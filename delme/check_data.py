#!/usr/bin/env python3

import zarr
import numpy as np

# Check each group's data
z = zarr.open('examples/broadcast_clarification_example.zarr', 'r')

print('=== ManualReplication ===')
pos = z['ManualReplication/positions'][:]
print(f'Shape: {pos.shape}')
print('All positions:')
for i, p in enumerate(pos):
    print(f'  Point {i}: X={p[0]:6.2f}, Y={p[1]:6.2f}, Z={p[2]:6.2f}, Time={p[3]:6.2f}')

print('\n=== PartialCoverage ===')
pos = z['PartialCoverage/positions'][:]
print(f'Shape: {pos.shape}')
print('All positions:')
for i, p in enumerate(pos):
    print(f'  Point {i}: X={p[0]:6.2f}, Y={p[1]:6.2f}, Z={p[2]:6.2f}, Time={p[3]:6.2f}')

print('\n=== BroadcastReplication ===')
pos = z['BroadcastReplication/positions'][:]
print(f'Shape: {pos.shape}')
print('All positions:')
for i, p in enumerate(pos):
    print(f'  Point {i}: X={p[0]:6.2f}, Y={p[1]:6.2f}, Z={p[2]:6.2f}, Time={p[3]:6.2f}')