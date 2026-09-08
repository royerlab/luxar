#### Gallery auto-exposure uses its full guard range

The gallery capture harness now keeps lowering exposure past the former eight-step clip-guard budget until the frame clears its clipping and background limits or reaches the configured -6-stop floor. Captures that still violate either limit at the floor are marked in the per-demo log instead of silently returning an unverified exposure.
