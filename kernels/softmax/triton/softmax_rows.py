"""
softmax_rows.py — rung 2: process ROWS_PER_BLOCK rows per program.

The naive kernel launches one program per row. For small tensors (few rows)
that leaves the GPU under-occupied: too few programs to hide memory latency,
so we only reach ~35% of bandwidth on 1024x1024.

Here each program handles a small tile of rows. More programs are resident per
SM, latency is hidden better, and small-shape bandwidth improves. Each row still
fits in BLOCK_N (rounded up to a power of two), loaded whole for a safe softmax.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_rows_kernel(
    x_ptr, out_ptr,
    x_row_stride, out_row_stride,
    n_rows, n_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # this program handles BLOCK_ROWS consecutive rows
    row_start = tl.program_id(0) * BLOCK_ROWS
    row_offsets = row_start + tl.arange(0, BLOCK_ROWS)      # (BLOCK_ROWS,)
    col_offsets = tl.arange(0, BLOCK_N)                     # (BLOCK_N,)

    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols

    # 2D block of pointers: (BLOCK_ROWS, BLOCK_N)
    x_ptrs = x_ptr + row_offsets[:, None] * x_row_stride + col_offsets[None, :]
    out_ptrs = out_ptr + row_offsets[:, None] * out_row_stride + col_offsets[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))    # (BLOCK_ROWS, BLOCK_N)

    # safe softmax along the column axis (axis=1), per row
    m = tl.max(x, axis=1)[:, None]
    e = tl.exp(x - m)
    s = tl.sum(e, axis=1)[:, None]
    out = e / s

    tl.store(out_ptrs, out, mask=mask)


def softmax(x: torch.Tensor, block_rows: int = 8) -> torch.Tensor:
    assert x.dim() == 2, "expects a 2D (rows, cols) tensor"
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    grid = (triton.cdiv(M, block_rows),)
    _softmax_rows_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        M, N,
        BLOCK_ROWS=block_rows,
        BLOCK_N=BLOCK_N,
    )
    return out
