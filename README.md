# kernel-curiosity

> Triton GPU kernels, measured honestly — against a tuned PyTorch baseline and the hardware roofline. Not just *how fast*, but *how close to the limit*.

---

## What is this?

`kernel-curiosity` is a structured, reproducible repository for GPU kernel work. It implements common operations in Triton — with a focus on LLM-related operators and core GPU primitives — and measures them against a tuned baseline (PyTorch) and the memory roofline, going beyond raw timing to explain the hardware-level reason for each result.

The goal is to answer two questions for every kernel comparison:

- **What?** — Which implementation is faster, and by how much, across different input shapes and data types?
- **Why?** — What do the measurements (memory throughput, roofline position, cache behaviour) tell us about the reason?

This repository is built incrementally. Early stages focus on correctness and clean benchmarks. Profiling depth and hardware coverage grow over time.

---

## Featured study: how close to peak is a simple softmax?

The first full investigation in this repo. Softmax is memory-bound, so the
question is what fraction of memory bandwidth it actually reaches. Testing a
simple Triton kernel against `torch.softmax` across a wide range of tensor sizes
on two GPUs turned up more about *measurement* than about kernels:

- On the **NVIDIA L4**, the simple kernel already hits 95-99% of measured
  bandwidth on memory-bound shapes. Three optimisation attempts won nothing,
  because nothing was there to win.
- The apparent "gap" on small tensors was a **cache mirage** — small tensors
  live in cache, so their bandwidth readings describe the cache and can exceed
  100% of the HBM limit (the L4 peaked at 339%).
- On the **AMD MI300X**, the same kernel plateaus at ~50% and shows no mirage at
  all: a thin-row softmax is too light to fill 304 CUs, so cache never gets the
  chance to inflate the numbers.

Details, kernels, and reproducible benchmarks: [`kernels/softmax/`](kernels/softmax/).
Full write-up: [How close to peak is a simple softmax? Two GPUs, one measurement trap](https://midiareshadi.github.io/blog/softmax-cache-mirage/).

---

## Hardware Targets

| GPU | Vendor | Architecture | Notes |
|-----|--------|--------------|-------|
| NVIDIA L4 | NVIDIA | Ada Lovelace | ~230 GB/s measured, ~48 MB L2 |
| AMD MI300X | AMD | CDNA3 | ~3,880 GB/s measured, ~256 MB last-level cache |

Results are stored per hardware target so that numbers from different machines are never mixed. Community contributions of results on other hardware are welcome.


---

## Repository Structure

```
kernel-curiosity/
- README.md
- LICENSE
- environment/requirements.txt
- kernels/softmax/            completed study
    - triton/                 softmax_naive, softmax_rows, softmax_gridstride
    - README.md               finding, failed attempts, how to reproduce
  (gemm, attention, layernorm, elementwise: planned)
- benchmarks/
    - peak_bw.py              measure real copy bandwidth
    - bench_softmax.py        correctness + GB/s per shape
    - l2_isolation.py         the sweep that exposed the cache effect
- plots/make_softmax_figures.py   regenerates figures from CSVs
- results/                    CSVs, one folder per GPU (L4-dev, MI300X)
- docs/figures/               generated SVGs
```

---

## Profiling Philosophy

Raw benchmark numbers answer *what*. Hardware profiling answers *why*.

For each operation, the goal is a short analysis that includes:

- **Roofline model position** — is the kernel compute-bound or memory-bandwidth-bound?
- **Memory throughput** — how close to theoretical peak (HBM bandwidth) does it get?
- **Occupancy** — what fraction of available warps/wavefronts are active?
- **Any notable bottleneck** — e.g. shared memory pressure, register spilling, pipeline stalls

On the cloud L4 and MI300X used here, hardware performance counters were locked (`ERR_NVGPUCTRPERM` on NVIDIA), so the analysis infers cache and roofline behaviour from carefully designed bandwidth sweeps rather than counter reads (Nsight Compute, `rocprof`). Counter-level profiling is a goal for machines where it is available.

---

## Reproducibility

Every result is tied to a specific GPU and software stack. Core dependencies (PyTorch, Triton, matplotlib, pandas) are pinned in [`environment/requirements.txt`](environment/requirements.txt):

```bash
pip install -r environment/requirements.txt
```

Each results folder records the GPU it was run on; numbers from different machines are never mixed.

---

## Results

Benchmark outputs live in `results/<hardware>/` as CSV files recording input shapes, data types, and measured bandwidth per shape.

If you run benchmarks on different hardware, contributions and results are welcome.

---

## Contributing

Contributions are welcome, particularly:

- Benchmark results on hardware not yet covered
- Alternative kernel implementations (e.g. different tiling strategies)

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.
