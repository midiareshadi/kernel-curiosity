"""
cliff_sweep.py — locate the cache cliff for read-only vs read-write kernels.

Runs two kernels across a dense range of tensor sizes on the L4:
  - read-write: reads the full row, writes a full row (1-pass). Input+output
    both compete for cache.
  - read-only:  reads the full row, writes only a scalar per row. Only input
    uses cache.

For each size it reports effective INPUT bandwidth as % of a measured copy peak,
averaged over several repeats. The "cliff" is the size where that efficiency
collapses. If input+output sharing cache causes the cliff, read-only should
cliff at a LARGER input size than read-write (roughly double: full cache vs
half cache).

Writes results/<hw>/cache-cliff/readonly_vs_readwrite_<dtype>.csv

Run on the L4:
    python3 cliff_sweep.py --peak-gbps 230 --hw L4-dev
"""
import os
import sys
import argparse
import statistics
import torch
import triton

sys.path.insert(0, "triton")
from reread_1pass import reread_1pass   # read-write baseline
from read_only import read_only


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
    return start.elapsed_time(end) / iters  # ms


def best_pct(fn, x, in_bytes, moved_bytes, peak_gbps, repeats=3):
    # take the best (fastest) of a few repeats to reduce noise
    best_ms = min(time_kernel(fn, x) for _ in range(repeats))
    gbps = moved_bytes / (best_ms * 1e-3) / 1e9
    return 100 * gbps / peak_gbps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-gbps", type=float, required=True)
    ap.add_argument("--hw", type=str, default="L4-dev")
    ap.add_argument("--dtype", type=str, default="fp16")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need a CUDA GPU"
    print("GPU:", torch.cuda.get_device_name(0), "| peak", args.peak_gbps, "GB/s")

    # dense sweep, N=1024 so MB = rows * 1024 * 2 / 1e6; extra points around
    # both expected cliffs (~30 MB for RW, ~60 MB for RO)
    rows_list = [2048, 4096, 8192, 12288, 14336, 16384, 18432, 20480,
                 24576, 28672, 32768, 40960, 49152, 57344, 65536, 98304,
                 131072, 262144]
    N = 1024

    rows_out = []
    print(f"\n{'rows':>8} {'in_MB':>8} {'RW in%':>8} {'RO in%':>8}")
    print("-" * 38)
    for rows in rows_list:
        x = torch.randn(rows, N, device="cuda", dtype=torch.float16)
        in_bytes = x.numel() * x.element_size()
        in_mb = in_bytes / 1e6

        pct_rw = best_pct(reread_1pass, x, in_bytes, 2 * in_bytes, args.peak_gbps)
        pct_ro = best_pct(read_only, x, in_bytes, in_bytes, args.peak_gbps)

        print(f"{rows:>8} {in_mb:>8.1f} {pct_rw:>7.1f}% {pct_ro:>7.1f}%")
        rows_out.append((rows, N, round(in_mb, 1), round(pct_rw, 1), round(pct_ro, 1)))

    out_dir = os.path.join("..", "..", "results", args.hw, "cache-cliff")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"readonly_vs_readwrite_{args.dtype}.csv")
    with open(out_csv, "w") as f:
        f.write("rows,N,in_mb,rw_pct,ro_pct\n")
        for r in rows_out:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"\nwrote {out_csv}")
    print("RW cliffs where its % drops to ~100; RO should cliff at ~2x that size.")


if __name__ == "__main__":
    main()
