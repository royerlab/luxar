#### The control panel follows a display that reloads

The panel's subscription to story changes lives in the display page. When the
display reloaded, it came back with no subscribers and the hub did not tell the
panel, so taps still moved the display but the highlighted tile froze on the
last story the panel had heard about, and stayed wrong until the panel itself
was reloaded. A loaded panel now re-reads the display's position and renews its
subscription every five seconds while it is visible, so it corrects itself
within one interval of any reload.
