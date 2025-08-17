# Luxar TODO List

This file tracks known issues, bugs, and improvements needed in the Luxar project.
Last updated from comprehensive codebase scan.

## TODO List:

1- <ESC> key behavior:
We need to change what happens when we press the <ESC> key: in fullscreen mode, it should first close all panels.
If no panels are open, either they were never opened in the first place or we pressed already once <ESC> then pressing <ESC>
again should kick us out of fullscreen.
If not in fullscreen mode then pressing <ESC> should close all the panels.

2- The keys used for toggling the panels should always 'work'. For example, if I toggle the rendering control panel (R) and then the dimension sliders panel (N), then pressing R does not toggle the rendering control panel.
This is not great. We need to make sure that these toggling keys for all panels (R, H, N, P, ctrl+L, ctrl+M, ...) always work, independently of what panels we have previously interacted with.

3- Please rename the 'Lazy Loading Monitor' on the UI to 'Data Loading and Caching Monitor'. No need to rename classes or functions, just the user-facing panel title.

4- In the 'Data Loading and Caching Monitor', what is shown in the Data Slice Status does not make much sense: the number of slices sems to be stuck at 100,
if we have multiple objects (layers, for now just point clouds) it is ot clear what is shown there. It would probably be better to have a drop-down menu to let one
pick which array to inspect? In that case the 'slices' caching status map would ake sense as we would be considering each array separately. 

5- The whole caching and lazy loading, as well as corresponding panel, should be reset when a new scene is loaded (navigation panel via 'O' key).

6- In 'fly control mode' there is a 'sign' problem when useing mouse dragging to tilt the camera: dragging left moves the camera right, and dragging right moves the camera left.
Dragging up or down has the correct and expected behavior.

6- Better OrbitMode that has no limits like ArcBall:
I am not a big fan of the current ‘OrbitMode’, because there are limits to the amount of rotation that one can apply! I would much rater have something like ArcBall, that always allows me to rotate, in any direction, and as much as I want. Please analyze why we are using Orbit control, what key feature are we using. Could we instead just implement our own version of Arcball (by reviewing the original ArcBall code, or better deriving from the class perhaps) and just add the features from Orbit control that we are needing and using?



## Notes

- This TODO list should be reviewed and updated regularly
- Issues marked as "Critical" should be addressed first
- Consider creating GitHub issues for tracking progress
- Update CLAUDE.md when implementing significant changes