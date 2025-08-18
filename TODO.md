# Luxar TODO List

This file tracks known issues, bugs, and improvements needed in the Luxar project.
Last updated from comprehensive codebase scan.

## TODO List:

1- <ESC> key behavior:
We need to change what happens when we press the <ESC> key: in fullscreen mode, it should first close all panels.
If no panels are open, either they were never opened in the first place or we pressed already once <ESC>. 
Pressing <ESC> again should kick us out of fullscreen.
If we are not in fullscreen mode then pressing <ESC> should close all the panels.
--> not fixed yet!

6- In 'fly control mode' there is a 'sign' problem when using mouse dragging to tilt the camera: dragging left moves the camera right, and dragging right moves the camera left.
Dragging up or down has the correct and expected behavior.
--> not fixed yet!

6- Better OrbitMode that has no limits like ArcBall:
I am not a big fan of the current ‘OrbitMode’, because there are limits to the amount of rotation that one can apply! I would much rater have something like ArcBall, that always allows me to rotate, in any direction, and as much as I want. Please analyze why we are using Orbit control, what key feature are we using. Could we instead just implement our own version of Arcball (by reviewing the original ArcBall code, or better deriving from the class perhaps) and just add the features from Orbit control that we are needing and using?
--> hold for now! do not fix.


## Notes

- This TODO list should be reviewed and updated regularly
- Issues marked as "Critical" should be addressed first
- Consider creating GitHub issues for tracking progress
- Update CLAUDE.md when implementing significant changes