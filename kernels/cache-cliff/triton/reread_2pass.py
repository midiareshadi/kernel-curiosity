"""
reread_2pass.py — read each row twice before writing.

Pass 1 reads the row and reduces it to a single value (the sum). Pass 2 re-reads
the SAME row and uses that sum. Pass 2 depends on pass 1's result, so the
compiler cannot fold the two reads into one — the row genuinely has to be read
twice. This mirrors real safe softmax (max pass, then normalise pass).

If the cache cliff is about staying resident across passes, this kernel's cliff
should sit at a smaller tensor size than the 1-pass baseline.
"""
import triton
import triton.language as tl
import torch


@triton.jit
def _reread_2pass_kernel(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * n_cols
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < n_cols

    # pass 1: read the row, reduce to a scalar (sum)
    x1 = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0,
                 eviction_policy="evict_last")
    row_sum = tl.sum(x1, axis=0)

    # barrier stops the compiler merging the two loads into one
    tl.debug_barrier()

    # pass 2: re-read the SAME row, use pass 1's result so the read is real
    x2 = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0,
                 eviction_policy="evict_last")
    out = x2 / row_sum
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


def reread_2pass(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(n_cols)
    _reread_2pass_kernel[(n_rows,)](x, out, n_cols, BLOCK_N=BLOCK_N)
    return out
