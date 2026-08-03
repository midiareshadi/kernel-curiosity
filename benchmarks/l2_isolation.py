"""
l2_isolation.py — the experiment that explains the softmax "gap".

A memory-bound kernel's GB/s only means HBM bandwidth if the data lives in HBM.
Small tensors fit in L2 cache, so their apparent bandwidth is L2 bandwidth and
can exceed the HBM roofline. This sweeps softmax from L2-resident to HBM-bound,
comparing naive Triton vs torch.softmax; the gap vanishes once data spills L2.

Usage: python l2_isolation.py --peak-gbps 230 --l2-mb 48
"""
import argparse, csv, os, statistics, sys, torch, triton
sys.path.insert(0, "../kernels/softmax/triton")
import softmax_naive as sn


def time_ms(fn, x, warmup=50, iters=200):
    for _ in range(warmup): fn(x)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(x); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def naive(x, num_warps=4):
    M, N = x.shape
    out = torch.empty_like(x)
    B = triton.next_power_of_2(N)
    sn._softmax_naive_kernel[(M,)](x, out, x.stride(0), out.stride(0), N,
                                   BLOCK_SIZE=B, num_warps=num_warps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-gbps", type=float, required=True)
    ap.add_argument("--l2-mb", type=float, default=48.0,
                    help="approx L2/cache size in MB, to label which shapes are cache-resident")
    ap.add_argument("--hw", required=True, help="hardware label, e.g. L4-dev or MI300X")
    ap.add_argument("--big", action="store_true",
                    help="use larger shapes (for big-cache GPUs like MI300X)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    db = {"fp16": 2, "bf16": 2, "fp32": 4}[args.dtype]
    dev = "cuda"
    print("GPU:", torch.cuda.get_device_name(0), "| peak", args.peak_gbps, "GB/s | L2~", args.l2_mb, "MB")
    print(f"\n{'shape':>13} {'MB':>8} {'cache?':>7} {'naive%':>8} {'torch%':>8} {'t/n':>6}")
    print("-" * 58)

    rows = []
    if args.big:
        # big-cache GPUs (e.g. MI300X, 256MB Infinity Cache) need multi-GB
        # tensors before the numbers are honestly HBM-bound
        shapes = [(1024, 1024), (16384, 1024), (32768, 1024), (49152, 1024),
                  (65536, 1024), (98304, 1024), (131072, 1024), (196608, 1024),
                  (262144, 512), (524288, 512), (1048576, 512), (2097152, 512)]
    else:
        shapes = [(1024, 1024), (4096, 1024), (8192, 1024), (12288, 1024),
                  (16384, 1024), (24576, 1024), (49152, 1024), (65536, 1024),
                  (131072, 512), (262144, 512)]
    for (M, N) in shapes:
        x = torch.randn(M, N, device=dev, dtype=dt)
        mb = M * N * db / 1e6
        resident = "L2" if mb < args.l2_mb else "HBM"
        tn = time_ms(naive, x)
        tt = time_ms(lambda z: torch.softmax(z, dim=-1), x)
        gn = 2 * M * N * db / (tn * 1e-3) / 1e9
        gt = 2 * M * N * db / (tt * 1e-3) / 1e9
        print(f"{M:>6}x{N:<6}{mb:>8.1f} {resident:>7} "
              f"{100*gn/args.peak_gbps:>7.1f}% {100*gt/args.peak_gbps:>7.1f}% {gt/gn:>5.2f}x")
        rows.append(dict(M=M, N=N, mb=round(mb, 1), resident=resident,
                         naive_gbps=round(gn, 1), torch_gbps=round(gt, 1),
                         naive_pct=round(100*gn/args.peak_gbps, 1),
                         torch_pct=round(100*gt/args.peak_gbps, 1),
                         ratio=round(gt/gn, 3), peak_gbps=args.peak_gbps))


    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "results", args.hw)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"l2_isolation_{args.dtype}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
