"""
verify_passes.py — the validity gate for the cache-cliff experiment.

The whole experiment assumes the 1/2/3-pass kernels genuinely read each row
1, 2, 3 times. The compiler could defeat this by eliminating redundant reads.
This harness measures the effective read bandwidth of each kernel on a
memory-bound (large, out-of-cache) tensor. If the reads are real, the 2-pass
kernel should take ~2x as long as 1-pass, and 3-pass ~3x (they move that much
more data). If the times are equal, the compiler collapsed the reads and the
kernels must be fixed before the experiment is meaningful.

Run on the L4:
    python3 verify_passes.py
"""
import sys
import torch
import triton

sys.path.insert(0, "triton")
from reread_1pass import reread_1pass
from reread_2pass import reread_2pass
from reread_3pass import reread_3pass


def time_kernel(fn, x, warmup=10, iters=50):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per call


def main():
    assert torch.cuda.is_available(), "need a CUDA GPU"
    print("GPU:", torch.cuda.get_device_name(0))

    # a large, out-of-cache tensor so we measure real HBM traffic, not cache
    x = torch.randn(262144, 512, device="cuda", dtype=torch.float16)  # 268 MB
    mb = x.numel() * x.element_size() / 1e6
    print(f"tensor: {x.shape} = {mb:.0f} MB (out of cache)\n")

    kernels = [("1-pass", reread_1pass), ("2-pass", reread_2pass),
               ("3-pass", reread_3pass)]

    t1 = None
    print(f"{'kernel':>8} {'ms':>8} {'vs 1-pass':>10}")
    print("-" * 30)
    for name, fn in kernels:
        ms = time_kernel(fn, x)
        if t1 is None:
            t1 = ms
        ratio = ms / t1
        print(f"{name:>8} {ms:>8.4f} {ratio:>9.2f}x")

    print("\nExpected if reads are real: ~1.0x, ~2.0x, ~3.0x")
    print("If all ~1.0x: compiler eliminated the extra reads (must fix).")


if __name__ == "__main__":
    main()
