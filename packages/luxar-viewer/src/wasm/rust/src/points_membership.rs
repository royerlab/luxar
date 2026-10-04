//! Per-point slice membership when effective radii are unavailable.

use wasm_bindgen::prelude::*;

use crate::common::{validate_ndim, MAX_SUPPORTED_DIMS};

#[wasm_bindgen]
pub fn points_slice_membership(
    positions: &[f32],
    display_dims: &[u32],
    slice_position: &[f32],
    tolerance: &[f32],
    discrete_dims: &[u8],
    ndim: usize,
    num_points: usize,
    output: &mut [u8],
) -> u32 {
    validate_ndim(ndim, "points_slice_membership");
    let mut is_display_dim = [false; MAX_SUPPORTED_DIMS];
    for &d in display_dims {
        if (d as usize) < ndim {
            is_display_dim[d as usize] = true;
        }
    }
    let mut visible = 0;
    for i in 0..num_points {
        let mut matches = true;
        for d in 0..ndim {
            if is_display_dim[d] || tolerance[d] >= 1e9 {
                continue;
            }
            let delta = (positions[i * ndim + d] - slice_position[d]).abs();
            let reach = if discrete_dims[d] != 0 {
                0.5
            } else {
                tolerance[d]
            };
            if !(delta <= reach) {
                matches = false;
                break;
            }
        }
        output[i] = u8::from(matches);
        if matches {
            visible += 1;
        }
    }
    visible
}

#[cfg(test)]
mod tests {
    use super::points_slice_membership;

    #[test]
    fn discrete_boundary_and_continuous_reach() {
        let positions = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.7];
        let mut mask = [0; 2];
        let count = points_slice_membership(
            &positions,
            &[0, 1, 2],
            &[0.0, 0.0, 0.0, 1.0],
            &[0.0, 0.0, 0.0, 0.1],
            &[0, 0, 0, 1],
            4,
            2,
            &mut mask,
        );
        assert_eq!(count, 1);
        assert_eq!(mask, [1, 0]);

        let count = points_slice_membership(
            &positions,
            &[0, 1, 2],
            &[0.0, 0.0, 0.0, 1.45],
            &[0.0, 0.0, 0.0, 0.1],
            &[0, 0, 0, 0],
            4,
            2,
            &mut mask,
        );
        assert_eq!(count, 1);
        assert_eq!(mask, [1, 0]);
    }
}
