#### Size automatic line primitives by total scene load

The viewer now applies the automatic capsule-to-quad threshold to the
concurrent effective line load of the whole scene instead of each line node in
isolation. Large scenes split across many smaller line nodes therefore choose
one consistent primitive and stay within the measured frame budget.

This changes the bundled dMRI tractography demo from capsules to quads: its
scene totals several million effective segments and clears the 2 million
threshold even though its largest line node contains only about 162 thousand.
