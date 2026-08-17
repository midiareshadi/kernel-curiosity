"""
reread_1pass.py — baseline: read each row once, write once.

Part of the cache-cliff experiment. The three kernels (1/2/3-pass) do the same
kind of trivial, memory-bound work but re-read each row a different number of
times. If the L4 cache cliff is caused by needing the row resident across
multiple passes, the cliff should move to a smaller tensor as pass count grows.
This 1-pass kernel is the control: minimal reuse, so its cliff should sit at the
largest size (closest to the full cache).

One program per row — same launch config as the softmax kernels, so occupancy
does not differ between the three variants.
"""
import triton
import triton.language as tl
import torch


@triton.jit
def _reread_1pass_kernel(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * n_cols
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < n_cols

    # single pass: read the row once, scale, write once
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    out = x * 2.0
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


def reread_1pass(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(n_cols)
    _reread_1pass_kernel[(n_rows,)](x, out, n_cols, BLOCK_N=BLOCK_N)
    return out
