"""
peak_bw.py — measure the GPU's achievable HBM bandwidth with a plain copy.
A big contiguous copy (read N, write N) is the simplest bandwidth-bound kernel.
Its GB/s is the honest roofline ceiling to compare softmax against.
"""
import statistics, torch

def time_ms(fn, warmup=25, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def main():
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0))
    for nbytes_mb in [64, 256, 1024]:
        n = nbytes_mb * 1024 * 1024 // 2
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        out = torch.empty_like(x)
        ms = time_ms(lambda: out.copy_(x))
        moved = 2 * n * 2
        gbps = moved / (ms*1e-3) / 1e9
        print(f"  copy {nbytes_mb:>5} MB : {ms:8.4f} ms  {gbps:8.1f} GB/s")

if __name__ == "__main__":
    main()
