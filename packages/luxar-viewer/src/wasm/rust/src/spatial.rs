//! Spatial index queries and bounding box intersection tests.
//!
//! This module provides fast AABB (Axis-Aligned Bounding Box) intersection
//! tests for chunk-based spatial indexing in nD space.

use wasm_bindgen::prelude::*;

/// Query chunks whose bounding boxes intersect the nD slice.
///
/// # Arguments
/// * `chunk_bounds` - Flattened chunk bounds [numChunks * ndim * 2] (min/max pairs)
/// * `slice_position` - Current slice position in nD space [ndim]
/// * `tolerance` - Tolerance per dimension [ndim]
/// * `ndim` - Number of dimensions
/// * `num_chunks` - Total number of chunks
/// * `output` - Output buffer for matching chunk indices [numChunks]
///
/// # Returns
/// Number of matching chunks (indices stored in output buffer)
///
/// # Algorithm
/// For each chunk, test if its bounding box intersects the query hypercube
/// defined by slice_position +/- tolerance in each dimension.
#[wasm_bindgen]
pub fn query_chunks_for_view(
    chunk_bounds: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    ndim: usize,
    num_chunks: usize,
    output: &mut [u32],
) -> u32 {
    debug_assert!(
        output.len() >= num_chunks,
        "output too small: {} < {}",
        output.len(),
        num_chunks
    );

    let mut match_count = 0;
    let stride = ndim * 2; // Hoisted: f32 count per chunk in bounds array (min/max pair per dim)

    for chunk_idx in 0..num_chunks {
        let chunk_base = chunk_idx * stride; // Hoisted outside inner loop
        let mut intersects = true;

        for dim in 0..ndim {
            let bounds_offset = chunk_base + dim * 2;
            let chunk_min = chunk_bounds[bounds_offset];
            let chunk_max = chunk_bounds[bounds_offset + 1];

            let query_min = slice_position[dim] - tolerance[dim];
            let query_max = slice_position[dim] + tolerance[dim];

            // No intersection if chunk is completely outside query range
            if chunk_max < query_min || chunk_min > query_max {
                intersects = false;
                break;
            }
        }

        if intersects {
            output[match_count as usize] = chunk_idx as u32;
            match_count += 1;
        }
    }

    match_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_chunks_basic_3d() {
        // 3 chunks in 3D space
        let chunk_bounds = vec![
            // Chunk 0: [0,0,0] to [1,1,1]
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // Chunk 1: [1,1,1] to [2,2,2]
            1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // Chunk 2: [5,5,5] to [6,6,6] (far away)
            5.0, 6.0, 5.0, 6.0, 5.0, 6.0,
        ];

        let slice_pos = vec![0.5, 0.5, 0.5];
        let tolerance = vec![0.6, 0.6, 0.6]; // Covers [-0.1, 1.1]

        let mut output = vec![0u32; 3];
        let count = query_chunks_for_view(&chunk_bounds, &slice_pos, &tolerance, 3, 3, &mut output);

        assert_eq!(count, 2, "Should match chunks 0 and 1");
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 1);
    }

    #[test]
    fn test_edge_case_empty_input() {
        let mut output = vec![0u32; 0];
        let count = query_chunks_for_view(&[], &[0.0, 0.0], &[1.0, 1.0], 2, 0, &mut output);
        assert_eq!(count, 0, "Should handle empty input");
    }

    #[test]
    fn test_query_chunks_2d() {
        // 2 chunks in 2D space
        let chunk_bounds = vec![
            // Chunk 0: [0,0] to [1,1]
            0.0, 1.0, 0.0, 1.0, // Chunk 1: [3,3] to [4,4] (far away)
            3.0, 4.0, 3.0, 4.0,
        ];

        let slice_pos = vec![0.5, 0.5];
        let tolerance = vec![0.6, 0.6]; // Covers [-0.1, 1.1]

        let mut output = vec![0u32; 2];
        let count = query_chunks_for_view(&chunk_bounds, &slice_pos, &tolerance, 2, 2, &mut output);

        assert_eq!(count, 1, "Should match only chunk 0 in 2D");
        assert_eq!(output[0], 0);
    }

    #[test]
    fn test_query_chunks_single_chunk() {
        // 1 chunk that covers a large region
        let chunk_bounds = vec![
            // Chunk 0: [-100, -100, -100] to [100, 100, 100]
            -100.0, 100.0, -100.0, 100.0, -100.0, 100.0,
        ];

        let slice_pos = vec![50.0, -30.0, 0.0];
        let tolerance = vec![1.0, 1.0, 1.0];

        let mut output = vec![0u32; 1];
        let count = query_chunks_for_view(&chunk_bounds, &slice_pos, &tolerance, 3, 1, &mut output);

        assert_eq!(
            count, 1,
            "Single all-encompassing chunk should always match"
        );
        assert_eq!(output[0], 0);
    }

    #[test]
    fn test_query_chunks_no_match() {
        // 2 chunks, query position far outside both
        let chunk_bounds = vec![
            // Chunk 0: [0,0,0] to [1,1,1]
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // Chunk 1: [2,2,2] to [3,3,3]
            2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
        ];

        let slice_pos = vec![100.0, 100.0, 100.0];
        let tolerance = vec![0.5, 0.5, 0.5];

        let mut output = vec![0u32; 2];
        let count = query_chunks_for_view(&chunk_bounds, &slice_pos, &tolerance, 3, 2, &mut output);

        assert_eq!(
            count, 0,
            "No chunks should match when position is far outside"
        );
    }

    #[test]
    fn test_query_chunks_boundary() {
        // Position exactly on chunk boundary
        let chunk_bounds = vec![
            // Chunk 0: [0,0,0] to [1,1,1]
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];

        // Position at exact max boundary of chunk
        let slice_pos = vec![1.0, 1.0, 1.0];
        let tolerance = vec![0.0, 0.0, 0.0];

        let mut output = vec![0u32; 1];
        let count = query_chunks_for_view(&chunk_bounds, &slice_pos, &tolerance, 3, 1, &mut output);

        // query_min = query_max = 1.0, chunk_max = 1.0
        // Check: chunk_max < query_min? 1.0 < 1.0 = false
        // Check: chunk_min > query_max? 0.0 > 1.0 = false
        // So it intersects
        assert_eq!(
            count, 1,
            "Chunk should match when position is exactly on boundary"
        );
        assert_eq!(output[0], 0);
    }
}
