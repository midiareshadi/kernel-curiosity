# softmax

**What I found:** a simple Triton softmax already runs at 95-99% of the L4's
measured memory speed on memory-bound shapes. Three attempts to make it faster
won nothing — there was nothing left to win. The apparent "gap" I first chased
was a cache mirage. On an MI300X the same kernel plateaus at ~50% of peak and
shows no mirage at all, because a thin-row softmax is too light to fill 304 CUs.

Full write-up:
https://midiareshadi.github.io/blog/softmax-cache-mirage/

## The measurement traps

**1. The spec sheet is not the limit.** A plain copy reaches 230 GB/s on an L4
(spec: 300) and ~3,880 GB/s on an MI300X (spec: 5,300). Measure your own
ceiling with `benchmarks/peak_bw.py`; do not divide by the datasheet number.

**2. Small tensors never touch memory.** They fit in cache, so their apparent
bandwidth is cache bandwidth, not memory bandwidth. On the L4 the inflation
peaks at 339% of the HBM limit around 16 MB — impossible for real memory, and a
clear sign the test is measuring cache.

## Results

Numbers are % of that GPU's **measured** peak bandwidth (not the spec sheet).

### NVIDIA L4 — measured peak 230 GB/s, cache ~48 MB

| size | simple Triton | torch |
|---|---|---|
| 2 MB | 33.0% | 80.9% |
| 8 MB | 80.0% | 229.8% |
| 17 MB | 169.6% | 339.2% |
| 25 MB | 226.1% | 242.8% |
| 34 MB | 91.0% | 91.6% |
| 50 MB | 92.2% | 96.9% |
| 134 MB | 98.7% | 97.7% |
| 268 MB | 100.9% | 97.7% |

The inflation is not a gentle fade. It climbs, peaks near 16-17 MB, holds
through 25 MB, then cliffs between 25 and 34 MB — a collapse that lands *below*
the 48 MB cache size, because softmax reads each row about twice and the tensor
must stay resident across both passes. Past ~30 MB both kernels land at 95-99%
of the real ceiling and agree with each other. The small-tensor "gap" was cache,
not code.

### AMD MI300X — measured peak ~3,880 GB/s, cache ~256 MB

| size | simple Triton | torch |
|---|---|---|
| 2 MB | 4.5% | 9.4% |
| 34 MB | 38.7% | 46.2% |
| 67 MB | 49.7% | 58.5% |
| 134 MB | 67.8% | 70.0% |
| 268 MB | 61.5% | 55.3% |
| 537 MB | 50.0% | 53.0% |
| 2.1 GB | 50.2% | 52.1% |

No mirage here: nothing crosses 100%, even sampled densely across the 256 MB
cache. The number humps to ~70% near 134 MB, then plateaus at ~50% out to 2 GB.
A thin-row softmax is too light to fill 304 CUs, so starvation dominates cache
at every size and the cache never gets to inflate the reading. torch hits the
same ~50% wall, so this is not a kernel-quality gap — it is the operation
under-using a very wide GPU.

## What I tried, and why none of it helped

The small-tensor numbers looked terrible (33% of peak), so I tried three things
before realising I was measuring cache.

**1. More rows per program** (`softmax_rows.py`). One program handles a tile of
rows instead of one, for better occupancy. It did not help: a tile of rows makes
each program hold a larger block in registers, which lowers how many can be
resident. Sweeping `block_rows` over 1, 2, 4, 8, 16 stayed within a point or two
of the baseline.

**2. A grid-stride loop** (`softmax_gridstride.py`). A fixed, smaller number of
programs, each looping over its share of rows, to amortise launch overhead. Also
no help, and it cost a little on large shapes.

**3. Tuning `num_warps`.** This looked like a win at first — the small tensor
seemed to move from 35% to 47%. But the full sweep had no trend: every warp
count from 1 to 16 landed between 44% and 47%. The "35%" came from a different
run, and small-tensor timings wander ten points between sessions. It was noise,
not a gain.

The gap never closed because it was not a real gap. Growing the tensor past the
cache closed it — see `benchmarks/l2_isolation.py`.

## Files

    triton/softmax_naive.py         one program per row, safe softmax ("simple")
    triton/softmax_rows.py          attempt 1: a tile of rows per program
    triton/softmax_gridstride.py    attempt 2: grid-stride loop over rows

    ../../benchmarks/peak_bw.py        measure the GPU's real copy bandwidth
    ../../benchmarks/bench_softmax.py  correctness + GB/s per shape, writes CSV
    ../../benchmarks/l2_isolation.py   the sweep that exposed the cache effect
    ../../results/<hw>/                CSVs, one folder per GPU
    ../../plots/make_softmax_figures.py  regenerates figures from those CSVs

## Reproduce

Find your GPU's real ceiling first:

    cd benchmarks
    python3 peak_bw.py

Then run the sweep with that number:

    # NVIDIA L4
    python3 l2_isolation.py --peak-gbps 230 --l2-mb 48 --hw L4-dev

    # AMD MI300X (bigger cache needs bigger tensors)
    python3 l2_isolation.py --peak-gbps 3880 --l2-mb 256 --hw MI300X --big

Per-shape detail for one implementation:

    python3 bench_softmax.py --impl naive --hw L4-dev --peak-gbps 230 --dtype fp16

`--impl` accepts `torch`, `naive`, `rows`, or `gridstride`.

## The takeaway

Before asking "is my kernel fast?", ask two other questions: what is this GPU's
real bandwidth, and is my test tensor big enough to actually touch memory? Get
those wrong and you will spend a day optimising something that was never slow.
