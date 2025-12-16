use cuda_std::prelude::*;

/// SAXPY: Y[i] = a * X[i] + Y[i]
/// Single-precision A*X Plus Y - classic GPU benchmark
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn saxpy(n: usize, a: f32, x: &[f32], y: *mut f32) {
    let idx = thread::index_1d() as usize;

    if idx < n {
        let y_ptr = unsafe { &mut *y.add(idx) };
        *y_ptr = a * x[idx] + *y_ptr;
    }
}
