#!/usr/bin/env python3
import zarr
import numpy as np

z = zarr.open('examples/broadcast_clarification_example.zarr', 'r')

print('=== RAW DATA CHECK ===')
print()
print('PartialCoverage positions:')
pos = z['PartialCoverage/positions'][:]
for i in range(len(pos)):
    print(f'  Index {i}: {pos[i]}')

print()
print('When loading slice [1-2] (index 1), we get:')
sliced = pos[1:2]
print(f'  {sliced}')
print(f'  X={sliced[0][0]}, Y={sliced[0][1]}, Z={sliced[0][2]}, Time={sliced[0][3]}')

print()
print('The viewer reports: "First position: (2.00, 5.00, 0.00)"')
print('But it should be: X=2.00, Y=-5.00, Z=0.00')
print()
print('ERROR: The viewer is extracting the wrong Y value!')