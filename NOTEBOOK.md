# Chronological Audit Lab Notebook (`NOTEBOOK.md`)

*Project:* "The Audit" — Tokenizer, GPU Capacity, and Architectural Audit  
*Author:* Bhavya  
*Environment:* macOS Apple Silicon, Python 3.11, PyTorch 2.8.0, Transformers 4.56.2, Tiktoken 0.14.0  

---

## Chronological Trajectory: Hypothesis → Experiment → Result → Revision

### Day 1: Environment Setup & Baseline Reproduction
* **Hypothesis**: `fertility.py` from starter kit should reproduce the baseline numbers reported in `REPORT_v0.md` (Hindi fertility 7.45, 5.89x English).
* **Experiment**: Executed `python3 fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2`.
* **Result**: Initial crash due to missing `tiktoken`. Installed `tiktoken` (0.14.0) via pip with network access. Re-ran script.
* **Output**:
  ```
  eng: fertility 1.27, tok/char 0.226
  hin: fertility 7.45, tok/char 1.579
  hin is 5.89x the fertility of eng
  ```
* **Revision**: Baseline numbers in `REPORT_v0.md` confirmed. Proceeding to systematic code and metric audit.

---

### Day 2: Unicode Normalization Check (Dead End / Harmless Feature)
* **Hypothesis**: Could `unicodedata.normalize("NFC", line)` in `read_lines()` be corrupting characters or splitting tokens?
* **Experiment**: Tested token count on raw Hindi FLORES text vs NFC normalized text.
* **Result**:
  - Raw tokens: 20,443
  - NFC tokens: 20,443 (Delta = 0)
* **Surprise / Revision**: **NFC normalization produced zero token-count change on clean text (0 delta)**. It is harmless on clean text and valid best practice to prevent decomposed diacritics from fragmenting into extra tokens.

---

### Day 3: Word Boundary Segmentation Audit (`line.split(" ")`)
* **Hypothesis**: `line.split(" ")` fails on double spaces, tabs, and attached punctuation.
* **Controlled Isolation Test**:
  - `"hello world"` (single space) $\rightarrow$ `split(" ")` count = 2, `regex` count = 2.
  - `"hello  world"` (double space) $\rightarrow$ `split(" ")` count = 3 (includes empty `""`), `regex` count = 2.
  - `"hello\tworld"` (tab space) $\rightarrow$ `split(" ")` count = 1 (words stay merged), `regex` count = 2.
* **Corpus Result**:
  - `split(" ")` word count: 2,640 words $\implies$ Fertility = 7.74 tok/word.
  - `regex \w+` word count: 4,965 words $\implies$ Fertility = 4.12 tok/word.
* **Delta**: Observed fertility difference is **+3.63 tok/word (+88.07%)**.

---

### Day 4: Lowercasing Normalization Bias (`line.lower()`)
* **Hypothesis**: Lowercasing Latin text alters GPT-2 subword tokenization while leaving Devanagari (caseless) unchanged.
* **Experiment**: Measured token counts on raw vs lowercased text for English and Hindi.
* **Result**:
  - English raw: 2,796 tokens vs lowercased: 2,928 tokens (**+132 tokens, +4.72% increase**).
  - Hindi raw: 20,443 tokens vs lowercased: 20,445 tokens (**+2 tokens, +0.01% increase**).
* **Finding**: Lowercasing increased English GPT-2 tokenization by +4.72% while Hindi changed by only 2 tokens. Lowercasing alters the English baseline, shifting the relative ratio from **7.31x to 6.98x (-4.50% reduction)**.

---

### Day 5: Macro vs Micro Aggregation Audit
* **Hypothesis**: `sum(per_line_ratios) / N` (macro-average) over-weights short outlier lines compared to global corpus ratio `sum(tokens)/sum(words)` (micro-average).
* **Experiment**: Evaluated macro vs micro averages on English and Hindi FLORES sets.
* **Result**:
  - English: Macro = 1.2445 vs Micro = 1.2269.
  - Hindi: Macro = 4.0955 vs Micro = 4.1174.
* **Delta**: Macro aggregation shifts the relative HIN/ENG fertility ratio from **3.36x (micro)** to **3.29x (macro)** (a **~2.0% difference**).

---

### Day 6: Code Point (`len(line)`) vs Byte Denominator Audit
* **Hypothesis**: Python `len(line)` measures Unicode code points. Because Devanagari uses ~3 UTF-8 bytes per code point, `tok/char` creates a different ratio relative to `tok/byte`.
* **Experiment**: Measured `tok/char` vs `tok/byte` across English and Hindi.
* **Result**:
  - English: `tok/char` = 0.2103, `tok/byte` = 0.2101 (1 byte/char).
  - Hindi: `tok/char` = 1.5159, `tok/byte` = 0.5924 (~3 bytes/code point).
  - Relative `tok/char` ratio (HIN/ENG) = **7.21x**.
  - Relative `tok/byte` ratio (HIN/ENG) = **2.82x**.
* **Finding**: Code point and UTF-8 byte denominators answer different operational questions.

---

### Day 7: Tokenizer Comparison & Parallel Corpus Construction
* **Hypothesis**: GPT-2 (50k English-only vocab) lacks Devanagari merges. An Indic-aware multilingual tokenizer (`xlm-roberta-base`) will show a different relative ratio.
* **Experiment**: Built a 100-sentence parallel FLORES-200 corpus (`eng`, `hin`, `tam`, `kan`) via `partA/build_corpus.py`. Evaluated `gpt2` vs `xlm-roberta-base`.
* **Result**:
  - `gpt2`: English = 2,796 tok, Hindi = 20,443 tok (**7.31x English**), Tamil = 42,141 tok (**15.07x**), Kannada = 36,957 tok (**13.22x**).
  - `xlm-roberta-base`: English = 3,120 tok, Hindi = 3,989 tok (**1.28x English**), Tamil = 4,277 tok (**1.37x**), Kannada = 4,310 tok (**1.38x**).
* **Revision**: The 6x-class overhead is not tokenizer-independent. Tokenizer choice is a major determinant of observed cross-language token overhead.

---

### Day 8: Capacity Reconciliation & Two-Method Goodput Derivation (Part B)
* **Hypothesis**: Theoretical KV cache limit on 1x L4 (24 GB VRAM) will match the preemption threshold in `bench_log.csv`.
* **Experiment**:
  - KV bytes/token = $28 \times (2 \times 8 \times 128 \times 2) = \mathbf{114,688 \text{ bytes/token}} = \mathbf{112 \text{ KiB/token}}$.
  - Usable KV VRAM = $24.00 \times 0.92 - 8.40 - 1.60 = \mathbf{12.08 \text{ GB}}$.
  - Capacity = $12.08 \text{ GB} / (4096 \times 114.688 \text{ KB}) = \mathbf{25.71 \implies 25 \text{ sequences}}$.
  - Reconciles with `bench_log.csv` (Batch 24 is the **best observed long-context point**, 93% util, 0 preemptions; Batch 32 = 7 preemptions).
* **Two Independent Goodput Derivations (Batch 24, Prompt 3584, Gen 512)**:
  - *Method 1 (Wall clock)*: $12,288 \text{ gen tokens} / 61.16\text{s} = \mathbf{200.92 \text{ gen tok/s}}$.
  - *Method 2 (Corrected Reported Throughput)*: $1607.4 \times (512 / 4096) = \mathbf{200.925 \text{ gen tok/s}}$.
  - Short prompt batch 16 generation speed = $16 \times 256 / 13.91 = \mathbf{294.46 \text{ gen tok/s}}$.
  - Relative goodput reduction: $1 - 200.92 / 294.46 = \mathbf{31.77\% \text{ slower}}$.

---

### Day 9: Casualization Decision Memo & Pilot Experiment Design (Part C)
* **Hypothesis**: Evaluate SFT vs Rewriter vs Prompt Engineering neutrally under constraints (1x A100, 30 reviewer hours = 900 evaluated pairs max capacity for Hindi + Kannada).
* **Trade-Off**:
  - Rewriter Model adds ~2GB VRAM overhead on L4 (reducing max KV batch size from 25 to 18 sequences) and +35-45ms latency.
  - Prompt Engineering adds ~200 prompt tokens per request, increasing TTFT.
  - SFT merges casual tone into weights (0 extra VRAM, 0 extra latency).
* **Outcome**: Selected SFT on FLM-4B. Designed Day-1 pilot (300-500 pairs) to evaluate reviewer feedback before scaling. Authored `partC/memo.md`.

---

## Defense Reproduction Command Palette

```bash
# 1. Reproduce Part A isolated bug evidence
python3 partA/fertility_audit.py

# 2. Run corrected multi-language fertility analysis
python3 partA/fertility_fixed.py --corpus eng=partA/corpus/eng.txt \
                                 --corpus hin=partA/corpus/hin.txt \
                                 --corpus tam=partA/corpus/tam.txt \
                                 --corpus kan=partA/corpus/kan.txt \
                                 --tokenizer xlm-roberta-base

# 3. Verify Part B capacity math and two-method goodput derivations
python3 partB/bench_analysis.py
```
