"""
read_only.py — read each row, write only a scalar per row (no full-size output).

Tests whether the L4 cache cliff is about input AND output competing for cache.
The normal softmax/copy kernel reads a full tensor and writes a full tensor, so
both share the ~48 MB cache. This kernel reads the full row but writes only one
number per row (the sum) — an Nx1 output, negligible in cache. So only the input
uses cache.

Prediction: if the cliff is caused by input+output sharing cache, this read-only
kernel should cliff at a LARGER input size (near the full 48 MB cache) than the
read-write kernel (which cliffs near half the cache, ~24 MB), because it leaves
the whole cache to the input.
"""
import triton
import triton.language as tl
import torch


@triton.jit
def _read_only_kernel(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * n_cols
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < n_cols

    # read the full row, reduce to one scalar, write only that scalar
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    row_sum = tl.sum(x, axis=0)
    tl.store(out_ptr + row, row_sum)


def read_only(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    out = torch.empty(n_rows, device=x.device, dtype=x.dtype)  # Nx1, tiny
    BLOCK_N = triton.next_power_of_2(n_cols)
    _read_only_kernel[(n_rows,)](x, out, n_cols, BLOCK_N=BLOCK_N)
    return out
