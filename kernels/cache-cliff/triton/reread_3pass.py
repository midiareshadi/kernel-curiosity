"""
reread_3pass.py — read each row three times before writing.

Pass 1 reduces the row to a sum. Pass 2 re-reads and reduces to a max (of the
sum-scaled row, so pass 2 depends on pass 1). Pass 3 re-reads and uses both.
Each pass depends on the previous, so the compiler cannot fold the three reads
into fewer — the row is genuinely read three times.

If the cache cliff is about staying resident across passes, this kernel's cliff
should sit at a smaller size than the 2-pass kernel: more reuse, so the tensor
must be smaller to stay cache-resident across all three passes.
"""
import triton
import triton.language as tl
import torch


@triton.jit
def _reread_3pass_kernel(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * n_cols
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < n_cols

    # pass 1: read, reduce to sum
    x1 = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0,
                 eviction_policy="evict_last")
    row_sum = tl.sum(x1, axis=0)
    tl.debug_barrier()

    # pass 2: re-read, reduce to max of the sum-scaled row (depends on pass 1)
    x2 = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0,
                 eviction_policy="evict_last")
    row_max = tl.max(x2 / row_sum, axis=0)
    tl.debug_barrier()

    # pass 3: re-read, use both prior results (depends on passes 1 and 2)
    x3 = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0,
                 eviction_policy="evict_last")
    out = (x3 / row_sum) - row_max
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


def reread_3pass(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(n_cols)
    _reread_3pass_kernel[(n_rows,)](x, out, n_cols, BLOCK_N=BLOCK_N)
    return out
