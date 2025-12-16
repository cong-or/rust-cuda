use cuda_std::prelude::*;

/// Naive GEMM: C = A * B
/// A: M x K, B: K x N, C: M x N
///
/// This is a simple, non-optimized implementation.
/// Real GEMM would use shared memory tiling for better performance.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn gemm_naive(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],  // M x K
    b: &[f32],  // K x N
    c: *mut f32, // M x N
) {
    let row = (block::index_x() * block::dim_x() + thread::index_x()) as usize;
    let col = (block::index_y() * block::dim_y() + thread::index_y()) as usize;

    if row < m && col < n {
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += a[row * k + i] * b[i * n + col];
        }
        unsafe {
            *c.add(row * n + col) = sum;
        }
    }
}

/// Optimized GEMM with shared memory tiling
/// Block size should be (16, 16) for optimal performance
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn gemm_tiled(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: *mut f32,
) {
    const TILE_SIZE: usize = 16;

    // Shared memory for tiles of A and B
    let mut tile_a = shared_array![f32; 256]; // 16x16
    let mut tile_b = shared_array![f32; 256]; // 16x16

    let row = (block::index_x() * TILE_SIZE + thread::index_x()) as usize;
    let col = (block::index_y() * TILE_SIZE + thread::index_y()) as usize;

    let tx = thread::index_x() as usize;
    let ty = thread::index_y() as usize;

    let mut sum = 0.0f32;

    // Loop over tiles
    let num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    for tile in 0..num_tiles {
        // Load tile of A into shared memory
        let a_col = tile * TILE_SIZE + ty;
        if row < m && a_col < k {
            tile_a[tx * TILE_SIZE + ty] = a[row * k + a_col];
        } else {
            tile_a[tx * TILE_SIZE + ty] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = tile * TILE_SIZE + tx;
        if b_row < k && col < n {
            tile_b[tx * TILE_SIZE + ty] = b[b_row * n + col];
        } else {
            tile_b[tx * TILE_SIZE + ty] = 0.0;
        }

        // Synchronize to ensure tiles are loaded
        block::sync_threads();

        // Compute partial dot product
        for i in 0..TILE_SIZE {
            sum += tile_a[tx * TILE_SIZE + i] * tile_b[i * TILE_SIZE + ty];
        }

        // Synchronize before loading next tiles
        block::sync_threads();
    }

    // Write result
    if row < m && col < n {
        unsafe {
            *c.add(row * n + col) = sum;
        }
    }
}
