# The Audit — LLM Tokenizer, Serving Capacity & Architectural Audit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![FLORES-200](https://img.shields.io/badge/Benchmark-FLORES--200-orange.svg)](https://github.com/facebookresearch/flores)

An evidence-first technical audit evaluating multilingual LLM tokenization efficiency, GPU memory capacity limits, and architecture trade-offs for Indic language deployment.

---

## Deliverables & Repository Structure

| File / Directory | Description |
|---|---|
| 📖 [**`NOTEBOOK.md`**](NOTEBOOK.md) | Chronological lab notebook documenting hypothesis → experiment → result → revision trajectory and dead ends |
| 🤖 [**`AI_USAGE.md`**](AI_USAGE.md) | Honest disclosure of AI assistant usage, script automation, and human course corrections |
| 📊 [**`partA/`**](partA/) | 4-language FLORES-200 corpus, isolated evidence runner (`fertility_audit.py`), fixed analyzer (`fertility_fixed.py`), and report (`analysis.md`) |
| ⚡ [**`partB/`**](partB/) | KV cache capacity math, two-method goodput derivation script (`bench_analysis.py`), and capacity report (`capacity_audit.md`) |
| 🏗️ [**`partC/memo.md`**](partC/memo.md) | Casualization architecture trade-off decision memo under hardware and reviewer constraints |
| 📦 [**`submission.zip`**](submission.zip) | Complete packaged submission zip archive |

---

## Key Audit Findings

### 1. Tokenizer Audit (Part A)
* **v0 Report Claim Sensitivity**: The original 5.89× result in `REPORT_v0.md` is highly sensitive to tokenizer choice and to the metric/denominator used.
* **Measured Evidence**:
  - **Tokenizer Sensitivity**: GPT-2's byte-level BPE produces substantially higher token counts for the Indic corpus than XLM-R in this benchmark. On an Indic-aware tokenizer (`xlm-roberta-base`), Hindi token overhead relative to English is **1.28× (+28% cost)**, Tamil is **1.37×**, and Kannada is **1.38×**.
  - **Script Encoding Impact**: Because UTF-8 byte length differs substantially from Unicode code-point count across scripts, the apparent Hindi/English ratio changes from **7.21× under code points** to **2.82× under bytes**.
  - **Segmentation & Aggregation**: Literal-space splitting is sensitive to repeated spaces and tabs, producing an **88.07% difference** in the observed fertility calculation on the benchmark corpus relative to the comparison word-count method, while macro-averaging introduced a **~2.0% difference** relative to micro-averaging.
  - **Normalization Check**: NFC normalization produced **0 token-count change** on the supplied clean corpus; therefore, it did not materially affect this benchmark.

### 2. Capacity Reconciliation & Serving (Part B)
* **KV-Cache Footprint**: Exact math yields **114,688 bytes/token** (**112 KiB/token** across 28 layers).
* **Max Concurrency**: Theoretical limit on 1× L4 GPU (24 GB VRAM) = **25 concurrent 4096-token sequences**, matching `bench_log.csv` (Batch 24 is the best observed long-context point at 93% utilization with 0 preemptions; Batch 32 triggers 7 preemptions).
* **Preemption Thrashing**: Batch 48 suffers severe preemption thrashing (`preempted_seqs = 23`), causing reported throughput to drop to 1,298.5 tok/s. Proposed FP8 KV cache as candidate mitigation to double capacity to 51 sequences.
* **Two-Method Goodput Derivation**: Batch-24 long-prompt generation goodput was derived in two independent ways (**200.92 gen tok/s** via wall clock vs **200.93 gen tok/s** via token fraction correction), proving long prompts generate output text **31.77% slower** than short prompts ($294.46 \text{ gen tok/s}$).

### 3. Architectural Decision Memo (Part C)
* **Trade-Off Evaluation**: Evaluated SFT vs Rewriter Model vs Prompt Engineering under 1× A100 GPU ($336\text{h}$) and 1 native reviewer ($30\text{h} = 900$ theoretical evaluated pairs max capacity assuming 2 min/pair for Hindi + Kannada).
* **Recommendation**: Selected **SFT on FLM-4B using QLoRA**. SFT embeds casual phrasing into model weights with 0 serving VRAM overhead and 0 decode latency penalty (unlike the Rewriter Model which steals an estimated ~2GB VRAM and adds an estimated +35–45ms latency).
* **Day-1 Pilot Experiment**: Designed a **300–500 sample pilot experiment** (Hindi + Kannada) for initial reviewer scoring before scaling synthetic dataset generation.

---

## Live Defense Command Palette

To reproduce all experimental results and capacity derivations live:

```bash
# 1. Run Part A isolated evidence audit
python3 partA/fertility_audit.py

# 2. Run Part A corrected multilingual benchmark
python3 partA/fertility_fixed.py --corpus eng=partA/corpus/eng.txt \
                                 --corpus hin=partA/corpus/hin.txt \
                                 --corpus tam=partA/corpus/tam.txt \
                                 --corpus kan=partA/corpus/kan.txt \
                                 --tokenizer xlm-roberta-base

# 3. Run Part B capacity math and two-method goodput verification
python3 partB/bench_analysis.py
```
