# Luxar TODO List

This file tracks known issues, bugs, and improvements needed in the Luxar project.

## TODO List:

0- ✅ FIXED: The spatial index-based loading system now correctly handles point radius for slicing. The effective radius calculation takes into account the point radius when determining which points are visible in the current slice. This was addressed by the new SpatialIndexLoader and effective-radius-calculator modules that replaced the lazy loading system.
   - **Status**: RESOLVED
   - **Priority**: High

1- ✅ FIXED: Coverage files are now centralized in a `coverage/` folder at the root. Python coverage goes to `coverage/python/` and TypeScript coverage goes to `coverage/typescript/`. The folder is properly configured in `.gitignore` and both test configurations have been updated accordingly.
   - **Status**: RESOLVED
   - **Priority**: Medium

2- ✅ COMPLETED: Example sharpness_compensation_example.zarr is very dim when opened, then I touch the HDR intensity slider in rendering control, and it gets much brighter although the effective value has barely changed!
   - **Status**: Open
   - **Priority**: Medium

3- ✅ COMPLETED: Added support for lower bit-depth data types in zarr with automatic conversion for Three.js compatibility. Implementation includes:
   - Python: DataTypeConfig with AUTO, PRECISION, MEMORY, and CUSTOM modes
   - Intelligent dtype selection based on data ranges (uint8 for SDR colors, float32 for HDR)
   - TypeScript: Support for Uint8Array and Uint16Array with WebGL normalization
   - Memory savings of 30-65% demonstrated in memory_optimization_example.py
   - Full backward compatibility (float32 remains default)
   - **Status**: COMPLETED
   - **Priority**: Medium

4- ✅ COMPLETED: Improve CLI commands so that: (i) there is a command to generate examples, serve them and open them in a browser, (ii) there is a command to serve a folder and its content, (iii) there is a command to serve the viewer itself, (iv) there is a command to serve a zarr file and serve the viewer with it shown, with the option, by default on to open the browser (v) there is command to provide detailed information and stats about a luxar zarr and display its contents (and info and stats of each object) as a tree. Make sure all CLI commands are tested and have tests that cover them.
   - **Status**: Open
   - **Priority**: Medium

5- ✅ COMPLETED: Remove 'substraction' blending mode from the Python API (completely, do not leave dead code!), it is not useful and does not work well with the current implementation of the viewer. Same for minimum and maximum blending modes. Make sure to remove all three modes from all tests and examples.
    - **Status**: RESOLVED
    - **Priority**: Medium

7- Additional post-processing effects and better effect management using pmndrs: https://github.com/pmndrs/postprocessing
    - **Status**: Open
    - **Priority**: LOW (do not fix yet!)

8- Cleanup of AA modes, we need to decide what we keep and what we trash...
    - **Status**: Open
    - **Priority**: LOW (do not fix yet!)

9- The Depth of Focus effect is not working. 
    - **Status**: Open
    - **Priority**: LOW (do not fix yet!)

10- Implement Ray Casting: specific strings can be associated to objects, when picking an object the associated string is displayed on the screen at a fixed position. This is useful for providing visualisation context.
    - **Status**: Open
    - **Priority**: LOW (do not fix yet!)

11- Introduce the notion of "scene domain". The main domain is the 'main' domain that of nD space visualised in a 3D 'slice', another domain is 'overlay' which is that for a specific set of non-visible dimensions from main, we can associate a scene that is rendered as a transparent overlay on top of the main rendering, this overlay is fixed, and its frame of reference is in normalised canvas coordinates ([0, 1]x[0, 1]). The viewer controls do not affect that scene since it is fixed and not in the main domain. Finally another important domain is that of 'sound' which allows to associate sound to a scene, and have it played when the scene is loaded. This is useful for providing context to the visualisation.
    - **Status**: Open
    - **Priority**: LOW (do not fix yet!) 

12- VR/AR Add the possibility to activate VR/AR mode. 
    - **Status**: Open
    - **Priority**: LOW (do not fix yet!)

13- Rename luxar-player to luxar-viewer everywhere. 
    - **Status**: Open
    - **Priority**: LOW (do not fix yet!)

14- In the viewer, do not use 'point cloud' terminology, use 'points' instead. A 'Point Cloud' is a 'Points' object in Luxar terminology.  
    - **Status**: Open
    - **Priority**: LOW (do not fix yet!)

15- I noticed something strange when trying example 'rainbow_sphere_4d_example.zarr', the points that are not visible because they do not intersect the 3D hyperplane ar still visible as ultrathin points. I think what is going on is that the shader is rendering points of radius zero, and instead of discarding entirely these points, it is rendering them as very small points. This is not the desired behaviour, we want these points to be completely invisible and not waste shader render time on them. Please take this interpretation of the bug with a grain of salt, and VERY CAREFULLY READ ALL RELEVANT CODE to determine the true cause of the issue. 
    - **Status**: Open
    - **Priority**: HIGH

## Notes

- This TODO list should be reviewed and updated regularly
- Issues marked as "Critical" should be addressed first
- Consider creating GitHub issues for tracking progress
- Update CLAUDE.md when implementing significant changes