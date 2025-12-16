use cuda_std::prelude::*;

/// Parallel reduction: sum all elements in array
/// Uses shared memory for efficient block-level reduction
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn reduction_sum(input: &[f32], output: *mut f32, n: usize) {
    let tid = thread::index_x() as usize;
    let bid = block::index_x() as usize;
    let block_size = block::dim_x() as usize;
    let idx = bid * block_size + tid;

    // Shared memory for this block
    let mut shared = shared_array![f32; 1024]; // Max block size

    // Load data into shared memory
    if idx < n {
        shared[tid] = input[idx];
    } else {
        shared[tid] = 0.0;
    }

    block::sync_threads();

    // Reduction in shared memory
    let mut stride = block_size / 2;
    while stride > 0 {
        if tid < stride {
            shared[tid] += shared[tid + stride];
        }
        block::sync_threads();
        stride /= 2;
    }

    // First thread writes block result
    if tid == 0 {
        unsafe {
            *output.add(bid) = shared[0];
        }
    }
}
