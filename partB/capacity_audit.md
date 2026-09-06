# Part B: Capacity Reconciliation & Serving Audit

## B1: KV-Cache Memory & Sequence Capacity Derivation (7 pts)

### (a) Exact KV-Cache Bytes per Token

For FLM-4B-Instruct using Grouped-Query Attention (GQA):
* Layers ($L$) = 28
* KV Heads ($H_{KV}$) = 8
* Head Dimension ($d_{\text{head}}$) = 128
* Precision = fp16 (2 bytes per scalar element)

For **one layer**, storing both Key ($K$) and Value ($V$) tensors per token requires:

$$\text{KV}_{\text{bytes/layer}} = 2 \times H_{KV} \times d_{\text{head}} \times \text{bytes}_{\text{fp16}} = 2 \times 8 \times 128 \times 2 = 4,096 \text{ bytes/layer}$$

For **all 28 layers**, the exact KV-cache memory per token is:

$$\text{KV}_{\text{bytes/token}} = 28 \times 4,096 \text{ bytes} = \mathbf{114,688 \text{ bytes/token}} = \mathbf{112 \text{ KiB/token}} = 0.114688 \text{ MB/token}$$

---

### (b) Theoretical Maximum 4096-Token Sequence Capacity

Using the 1× NVIDIA L4 GPU (24 GB VRAM) specification:
* Total VRAM = $24.00 \text{ GB}$ ($24,000,000,000 \text{ bytes}$)
* `gpu_memory_utilization` limit = $0.92 \implies 22.08 \text{ GB}$ usable VRAM
* Model Weights (4.2B params in fp16) = $4.2 \times 2 = 8.40 \text{ GB}$
* Non-KV Runtime Overhead (activations, CUDA graphs) = $1.60 \text{ GB}$

#### Available KV Cache Memory Calculation:

$$M_{\text{KV}} = 22.08 \text{ GB} - 8.40 \text{ GB} - 1.60 \text{ GB} = \mathbf{12.08 \text{ GB}} = 12,080,000,000 \text{ bytes}$$

#### Memory Required per 4096-Token Sequence:

$$\text{Memory}_{\text{seq}} = 4096 \text{ tokens} \times 114,688 \text{ bytes/token} = \mathbf{469,762,048 \text{ bytes}} \approx \mathbf{0.46976 \text{ GB}} \quad (448.00 \text{ MiB})$$

#### Maximum Concurrent 4096-Token Sequences:

$$N_{\text{max\_seqs}} = \frac{M_{\text{KV}}}{\text{Memory}_{\text{seq}}} = \frac{12,080,000,000}{469,762,048} = \mathbf{25.71 \text{ sequences}} \implies \mathbf{25 \text{ sequences}}$$

---

### Empirical Verification Against `bench_log.csv`

Look at the long-prompt benchmark log (`prompt_len = 3584, gen_len = 512`, total sequence length = $3584 + 512 = 4096$ tokens):

| `batch_size` | `wall_clock_s` | `reported_tok_s` | `preempted_seqs` | `kv_cache_util` | Benchmark Status |
|---|---|---|---|---|---|
| **16** | 49.97 | 1311.4 | **0** | 0.62 (62%) | Clean |
| **24** | 61.16 | **1607.4** | **0** | **0.93 (93%)** | **Best Observed Long-Context Point** |
| **32** | 94.71 | 1384.0 | **7** | 0.97 (97%) | Exceeds 25 Limit (7 Preempted) |
| **48** | 151.41 | 1298.5 | **23** | 0.97 (97%) | Severe Thrashing (23 Preempted) |

> [!IMPORTANT]
> **Log Reconciliation**: The theoretical limit of **25 sequences** matches `bench_log.csv` exactly. At batch size 24 (**the best observed long-context operating point in the benchmark**), KV cache utilization reaches **93%** with 0 preemptions. At batch size 32 (>25 limit), the cache runs out of VRAM, forcing vLLM to preempt 7 sequences!

---

## B2: Long-Context Throughput Anomaly & Mechanism (6 pts)

### Measured Anomaly Identification
In the `prompt_len = 3584` sweep, naive expectations predict throughput will increase linearly with batch size. However:
1. Measured `reported_tok_s` peaks at batch 24 (**1,607.4 tok/s**) and then **drops to 1,384.0 tok/s at batch 32** (-13.9%) and **1,298.5 tok/s at batch 48** (-19.2%).
2. Wall clock latency jumps disproportionately from **61.16s at batch 24** to **94.71s at batch 32** (+54.8% latency for only +33% requests) and **151.41s at batch 48** (+147.5% latency).

### Physical Mechanism
* **VRAM Exhaustion**: Batch 32 requires $32 \times 0.46976 \text{ GB} = 15.03 \text{ GB}$ of KV cache, exceeding the available $12.08 \text{ GB}$ VRAM limit.
* **Preemption Thrashing**: vLLM's scheduler cannot allocate KV blocks for all 32 active requests simultaneously. It is forced to **preempt** sequences (evicting KV blocks and re-executing prompt prefill when re-scheduled).
* Column `preempted_seqs` confirms **7 preemptions at batch 32** and **23 preemptions at batch 48**. Wasted prefill re-computation degrades throughput and explodes latency.

### Proposed Candidate Mitigation & Theoretical Prediction
* **Proposed Config Change**: Propose FP8 KV Cache (`kv_cache_dtype="fp8"`) as a candidate mitigation.
* **Theoretical Prediction**: Halving KV-cache bytes/token from 112 KiB to 56 KiB approximately doubles theoretical sequence capacity from 25 to **51 concurrent sequences**, predicting that batch 32 and 48 would fit in VRAM with 0 preemptions (subject to kernel overhead and precision loss).

---

## B3: Misread Column & Derivation of Honest Goodput (4 pts)

### Misread Column
The intern misread `reported_tok_s`. `reported_tok_s` computes:

$$\text{reported\_tok\_s} = \frac{\text{batch\_size} \times (\text{prompt\_len} + \text{gen\_len})}{\text{wall\_clock\_s}}$$

For long prompts (3584 tokens), `reported_tok_s` counts prompt prefill tokens processed during initial parallel matrix multiplies. It measures **prefill compute rate**, NOT **text generation speed** (goodput).

### Honest Goodput Derivation for Batch 24 (`prompt 3584, gen 512`) in Two Independent Ways

* Batch 24 parameters: $B = 24$, $\text{prompt\_len} = 3584$, $\text{gen\_len} = 512$, $\text{wall\_clock} = 61.16\text{ s}$, $\text{reported\_tok\_s} = 1607.4$.
* Total generated output tokens = $24 \times 512 = 12,288 \text{ tokens}$.

#### Method 1: Direct Wall-Clock Output Generation Rate
$$\text{Goodput}_{\text{Method 1}} = \frac{\text{Total Generated Tokens}}{\text{Wall Clock Time}} = \frac{12,288 \text{ tokens}}{61.16 \text{ seconds}} = \mathbf{200.92 \text{ gen tok/s}}$$

#### Method 2: Correcting `reported_tok_s` via Generation Token Fraction
$$\text{Generation Fraction} = \frac{\text{gen\_len}}{\text{prompt\_len} + \text{gen\_len}} = \frac{512}{3584 + 512} = \frac{512}{4096} = 0.1250$$

$$\text{Goodput}_{\text{Method 2}} = \text{reported\_tok\_s} \times \text{Generation Fraction} = 1607.4 \times 0.1250 = \mathbf{200.925 \text{ gen tok/s}}$$

> [!IMPORTANT]
> **Exact Agreement**: Both independent methods yield **200.92 gen tok/s** (matching to rounding).

### Measured Comparison & Correct Conclusion
For Short Prompt Batch 16 (`512 prompt, 256 gen`), generation goodput is:

$$\text{Goodput}_{\text{short}} = \frac{16 \times 256 \text{ tokens}}{13.91 \text{ seconds}} = \mathbf{294.46 \text{ gen tok/s}}$$

$$\text{Relative Goodput Reduction} = 1 - \frac{200.92}{294.46} = 1 - 0.6823 = \mathbf{31.77\% \text{ slower}}$$

> [!WARNING]
> **What the Report Should Have Said**: Longer prompts make output text generation **31.77% slower** (200.92 gen tok/s vs 294.46 gen tok/s) due to prompt prefill delay (TTFT 500.5ms vs 71.7ms) and higher attention computation. Extrapolating batch 48 to ~3200 tok/s ignored preemption thrashing.

---

## B4: Serving Stack Monitoring Counter (3 pts)

To confirm the preemption thrashing mechanism, pull:

$$\text{Counter Metric: } \mathbf{vllm:num\_preemptions\_total}$$

* **Expected Value**: Must show **0** for batch $\le 24$, **7** at batch 32, and **23** at batch 48.
