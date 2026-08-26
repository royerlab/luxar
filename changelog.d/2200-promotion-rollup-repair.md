#### Repair cancelled push checks after green promotion windows

Scheduled CI now reruns cancelled jobs for any of the five promotion-required contexts
from the push run on the same commit after those scheduled contexts pass. A rejected
rerun no longer prevents the remaining jobs from being attempted. This makes GitHub's
protected-branch rollup converge to the successful verdict instead of retaining
cancelled duplicate checks that block `dev` to `main` promotion.
