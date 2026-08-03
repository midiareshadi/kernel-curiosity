"""
softmax_naive.py — the baseline. One Triton program per row, whole row in one block.

This is "safe softmax" (numerically stable): subtract the row max before exp, so
we never exp a large number and overflow.

    m = max(x_row)
    e = exp(x_row - m)
    out = e / sum(e)

Naive here means: simplest correct thing. One program handles one row. The row
must fit in BLOCK_SIZE (we round N up to a power of two). No tiling, no vectorized
loads, no fusion cleverness. This establishes the floor we climb from.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_naive_kernel(
    x_ptr, out_ptr,
    x_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * x_row_stride
    out_row_ptr = out_ptr + row * out_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # load the row (masked lanes get -inf so they never win the max / contribute to sum)
    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=-float("inf"))

    # safe softmax
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(out_row_ptr + col_offsets, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "expects a 2D (rows, cols) tensor"
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _softmax_naive_kernel[(M,)](
        x, out,
        x.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
