"""
bench_softmax.py — measure row-wise softmax throughput in GB/s and % of HBM peak.

Softmax is memory-bound: read the input row, write the output row, a little math
between. Honest metric = achieved memory bandwidth vs the GPU's peak (roofline).

Bytes moved for an (M, N) tensor: read + write = 2 * M * N * dtype_bytes.

Checks correctness vs torch.softmax first, warms up, times with CUDA events
(median of runs), reports ms / GB/s / % peak, writes a CSV under results/<hw>/.
"""
import argparse, csv, importlib, os, statistics, sys
import torch

DEFAULT_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 2048),
    (16384, 1024),
]

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}


def bytes_moved(M, N, dtype_bytes):
    return 2 * M * N * dtype_bytes

def get_softmax_fn(impl):
    if impl == "torch":
        return lambda x: torch.softmax(x, dim=-1)
    mod = importlib.import_module(f"softmax_{impl}")
    return mod.softmax


def check_correct(fn, dtype, device):
    torch.manual_seed(0)
    x = torch.randn(2048, 2048, device=device, dtype=dtype)
    ref = torch.softmax(x, dim=-1)
    got = fn(x)
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=0)
    max_err = (got.float() - ref.float()).abs().max().item()
    return ok, max_err

def time_ms(fn, x, warmup=25, iters=100):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--hw", required=True)
    ap.add_argument("--peak-gbps", type=float, required=True)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no GPU visible"
    device = "cuda"
    dtype = DTYPES[args.dtype]
    dtype_bytes = DTYPE_BYTES[args.dtype]

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "kernels", "softmax", "triton"))
    fn = get_softmax_fn(args.impl)

    ok, max_err = check_correct(fn, dtype, device)
    print(f"correctness vs torch.softmax: {'PASS' if ok else 'FAIL'} (max_err={max_err:.2e})")
    if not ok:
        print("refusing to benchmark an incorrect kernel.")
        raise SystemExit(1)

    print(f"\n{'shape':>14} {'ms':>10} {'GB/s':>10} {'% peak':>8}")
    print("-" * 46)
    rows = []
    for (M, N) in DEFAULT_SHAPES:
        x = torch.randn(M, N, device=device, dtype=dtype)
        ms = time_ms(fn, x, iters=args.iters)
        gbps = bytes_moved(M, N, dtype_bytes) / (ms * 1e-3) / 1e9
        pct = 100.0 * gbps / args.peak_gbps
        print(f"{M:>6}x{N:<7} {ms:>10.4f} {gbps:>10.1f} {pct:>7.1f}%")
        rows.append(dict(impl=args.impl, dtype=args.dtype, M=M, N=N,
                         ms=round(ms, 5), gbps=round(gbps, 1), pct_peak=round(pct, 1)))

    out_dir = os.path.join(here, "..", "results", args.hw)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"softmax_{args.impl}_{args.dtype}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
