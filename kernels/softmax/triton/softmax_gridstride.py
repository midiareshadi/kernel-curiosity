"""
softmax_gridstride.py — rung 3: a grid-stride (persistent) kernel.

The naive kernel launches one program per row. For many small rows that is a lot
of launch/scheduling overhead for tiny per-program work, so small shapes stall
around 35-47% of bandwidth while torch.softmax reaches ~85%.

Here we launch a fixed, modest number of programs and each one loops over rows in
a grid-stride pattern (row = pid, pid + num_programs, pid + 2*num_programs, ...).
Launch overhead is amortized across many rows, and the GPU stays busy. Each row
still uses the simple, efficient 1D safe-softmax reduction.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_gridstride_kernel(
    x_ptr, out_ptr,
    x_row_stride, out_row_stride,
    n_rows, n_cols,
    num_programs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # grid-stride loop over rows: this program handles rows pid, pid+P, pid+2P, ...
    row = pid
    while row < n_rows:
        x_row_ptr = x_ptr + row * x_row_stride
        out_row_ptr = out_ptr + row * out_row_stride

        x = tl.load(x_row_ptr + col_offsets, mask=mask, other=-float("inf"))
        m = tl.max(x, axis=0)
        e = tl.exp(x - m)
        s = tl.sum(e, axis=0)
        out = e / s
        tl.store(out_row_ptr + col_offsets, out, mask=mask)

        row += num_programs


def softmax(x: torch.Tensor, programs_per_sm: int = 4, num_warps: int = 4) -> torch.Tensor:
    assert x.dim() == 2, "expects a 2D (rows, cols) tensor"
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    # launch ~ (SMs * programs_per_sm) programs, capped at M (no point launching
    # more programs than rows). Each loops over its strided share of rows.
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    num_programs = min(num_sms * programs_per_sm, M)

    _softmax_gridstride_kernel[(num_programs,)](
        x, out,
        x.stride(0), out.stride(0),
        M, N,
        num_programs,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out
