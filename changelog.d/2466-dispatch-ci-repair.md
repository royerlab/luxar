#### Dispatch promotion repair windows on dev

Promotion repair windows are now requested explicitly on `dev` instead of running from
default-branch cron events. Each dispatch tests the exact dev commit receiving its check
contexts, defaults to the required Python 3.12 leg, and repairs cancelled required checks
from superseded dev push runs. Merge pushes continue to exercise the full supported
Python matrix.
