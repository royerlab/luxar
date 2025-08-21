#!/usr/bin/env python3
import zarr
import numpy as np

z = zarr.open('examples/broadcast_clarification_example.zarr', 'r')

print('=== PartialCoverage Zarr Data ===')
pos_arr = z['PartialCoverage/positions']
print(f'Position array shape: {pos_arr.shape}')
print(f'Position array chunks: {pos_arr.chunks}')
print()

print('Full data:')
full_data = pos_arr[:]
for i in range(len(full_data)):
    print(f'  Index {i}: {full_data[i]}')

print()
print('Slice [1:2] (what viewer loads for Time=1):')
sliced = pos_arr[1:2]
print(f'  Result: {sliced}')
print(f'  First point: X={sliced[0][0]}, Y={sliced[0][1]}, Z={sliced[0][2]}, Time={sliced[0][3]}')

print()
print('The viewer is somehow getting Y=5 instead of Y=-5!')
print('This suggests the zarr data might be corrupted OR')
print('The lazy loading system is reading from the wrong location!')

# Check the raw chunks
print()
print('=== Raw Chunk Data ===')
import json
# Read the chunk directly
chunk_data = z.store['PartialCoverage/positions/0']
print(f'Chunk 0 size: {len(chunk_data)} bytes')

# Also check BroadcastReplication 
print()
print('=== BroadcastReplication Zarr Data ===')
br_pos = z['BroadcastReplication/positions'][:]
print('BroadcastReplication positions:')
for i in range(len(br_pos)):
    print(f'  Index {i}: {br_pos[i]}')