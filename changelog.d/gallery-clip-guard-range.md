#### Gallery auto-exposure uses its full guard range

The gallery capture harness now keeps lowering exposure past the former eight-step clip-guard budget until the frame is no longer blown or the configured -6-stop floor is reached. Captures that remain blown at the floor are marked in the per-demo log instead of silently returning an unverified exposure.
