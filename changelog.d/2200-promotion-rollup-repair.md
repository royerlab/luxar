#### Repair cancelled push checks after green promotion windows

Scheduled CI now reruns cancelled `python-tests (3.12)` and `typescript-tests` jobs
from the push run on the same commit after all five promotion-required scheduled
contexts pass. This makes GitHub's protected-branch rollup converge to the successful
verdict instead of retaining cancelled duplicate checks that block `dev` to `main`
promotion.
