# Part A: Tokenizer Audit & Metric Reconciliation

## Executive Summary & Corrected Headline Numbers

The conclusion in `REPORT_v0.md` that Hindi serving costs ~6× more than English due to an inherent property of the script is **not supported by empirical measurement on Indic-aware tokenizers**. That 5.89× figure resulted from benchmarking a 2019 English-centric tokenizer (`gpt2`) combined with mathematical artifacts in word segmentation and character counting.

When evaluated on a 100-sentence parallel FLORES-200 benchmark using an Indic-aware multilingual tokenizer (`xlm-roberta-base`), the experimentally measured token overhead relative to English is:

| Language | Script | GPT-2 Ratio (v0 Report) | XLM-R Parallel Ratio (Measured) | Measured Cost Overhead |
|---|---|---|---|---|
| **English (eng)** | Latin | 1.00× | **1.00×** (31.2 tok/sent) | Baseline |
| **Hindi (hin)** | Devanagari | 5.89× (v0 claim) | **1.28×** (39.9 tok/sent) | **+28%** |
| **Tamil (tam)** | Dravidian | 15.07× | **1.37×** (42.8 tok/sent) | **+37%** |
| **Kannada (kan)** | Dravidian | 13.22× | **1.38×** (43.1 tok/sent) | **+38%** |

> [!IMPORTANT]
> **Corrected Routing Recommendation**: Do NOT budget 6× serving cost for Indic languages or route traffic to a separate 6× pricing tier. With an Indic-aware model/tokenizer, budget only **1.3× serving cost** (28%–38% extra token throughput) for Indic languages.

---

## 1. Audit of `fertility.py` Code Behaviors & Metrics (Evidence Rule Applied)

### Item 1: Whitespace-Split Word Segmentation (`line.split(" ")`)
* **Behavior**: `line.split(" ")` splits only on literal single spaces, producing empty string elements `""` on double spaces and leaving tab-separated words merged.
* **Controlled Isolation Test**:
  - `"hello world"` (single space) $\rightarrow$ `split(" ")` count = 2, `regex` count = 2.
  - `"hello  world"` (double space) $\rightarrow$ `split(" ")` count = 3 (includes empty `""`), `regex` count = 2.
  - `"hello\tworld"` (tab space) $\rightarrow$ `split(" ")` count = 1 (words stay merged), `regex` count = 2.
* **Corpus Benchmark Evidence**: On the 100-sentence Hindi corpus, `split(" ")` yields 2,640 word tokens vs 4,965 words identified by regex `\w+`. Total GPT-2 tokens = 20,443.
* **Quantitative Delta**: `split(" ")` gives a fertility of `7.74` tok/word, whereas `regex \w+` gives `4.12` tok/word (a **+3.63 tok/word (+88.07%) shift**).

### Item 2: Lowercasing Normalization (`line.lower()`)
* **Behavior**: Lowercases input strings. In `gpt2`, casing alters tokenization for Latin scripts (e.g. `"The"` vs `"the"` or capitalized proper nouns like `"Bengaluru"`).
* **Measured Evidence**: On English FLORES text, raw tokens = 2,796 vs lowercased = 2,928 (**+132 tokens, +4.72% increase**). On Hindi text, raw tokens = 20,443 vs lowercased = 20,445 (**+2 tokens, +0.01% increase**).
* **Quantitative Delta**: Lowercasing increases English GPT-2 tokenization by +4.72% while Hindi changes by only 2 tokens. Lowercasing alters the English baseline, shifting the apparent Hindi/English relative ratio from **7.31× to 6.98× (-4.50% reduction)**.

### Item 3: Macro (Per-Line Mean) vs Micro (Corpus Ratio) Aggregation
* **Behavior**: `fertility.py` computes `sum(per_line_fertility) / N` (macro-average) rather than total corpus tokens divided by total corpus words (`sum(tokens)/sum(words)`, micro-average).
* **Measured Evidence**: On FLORES English, Macro = 1.2445 vs Micro = 1.2269 (+0.0176 delta). On Hindi, Macro = 4.0955 vs Micro = 4.1174 (-0.0219 delta).
* **Quantitative Delta**: Macro aggregation shifts the relative HIN/ENG fertility ratio from **3.36× (micro)** to **3.29× (macro)** (a **~2.0% difference**). Micro aggregation is preferable because it weights observations according to their denominator.

### Item 4: Unicode Code Points (`len(line)`) vs UTF-8 Byte Denominator
* **Behavior**: Python `len(line)` counts Unicode code points, not UTF-8 bytes.
* **Measured Evidence**: On English text, `tok/char` = 0.2103 and `tok/byte` = 0.2101 (1 byte/char). On Hindi text, `tok/char` = 1.5159 and `tok/byte` = 0.5924 (~3 bytes/code point).
* **Quantitative Delta**: Code-point and UTF-8 byte denominators answer different questions. Comparing `tok/char` across English and Hindi yields a **7.21× ratio**, whereas `tok/byte` yields a **2.82× ratio**. The apparent fertility ratio changes substantially depending on the denominator.

### Item 5: Tokenizer Choice Mismatch
* **Behavior**: Benchmarking Indic scripts using `gpt2` (50k English-centric vocabulary).
* **Measured Evidence**: Under GPT-2, Hindi requires 20,443 tokens (7.31× English). Under XLM-RoBERTa, Hindi requires only 3,989 tokens (**1.28× English**).
* **Finding**: The 6×-class overhead is not tokenizer-independent. Tokenizer choice is a major determinant of observed cross-language token overhead.

### Item 6 (Suspicious-but-Valid Feature Check): Unicode NFC Normalization
* **Behavior**: `unicodedata.normalize("NFC", line)` normalizes strings to Canonical Composition.
* **Measured Evidence**: On clean Hindi text, raw tokens = 20,443 and NFC tokens = 20,443 (**0 delta**).
* **Conclusion**: **NFC normalization produced no token-count change on the clean FLORES corpus**, indicating the input was already composed. The existing NFC step is harmless on clean text and valid best practice to prevent diacritic fragmentation.

---

## 2. Denominator Selection & Routing Metric Reasoning

We evaluated four candidate denominators:

| Denominator | What it Holds Constant | Strengths & Weaknesses | Recommendation |
|---|---|---|---|
| **Whitespace Words** | Word count | Fails across scripts; agglutinative Dravidian languages bundle multiple morphemes into single words. | Rejected |
| **Unicode Code Points** | Character count | Python `len(line)`; script-dependent (Devanagari = 3 bytes/code point vs Latin = 1 byte). | Rejected |
| **UTF-8 Bytes** | Storage bytes | Measures byte compression efficiency, but raw bytes per concept differ by script. | Secondary Diagnostic |
| **Parallel Sentence** | **Semantic intent** | **Holds information content constant across languages.** Measures GPU token payload per prompt. | **PRIMARY ROUTING METRIC** |

$$\text{Primary Routing Metric} = \frac{\text{Total Tokens(Indic)}}{\text{Total Tokens(English)}} \quad \text{on Parallel Benchmark Sentences}$$

---

## 3. Production Monitoring Metric & Caveats

### Biggest Caveat
FLORES-200 sentence benchmark measures formal domain text (news/wikipedia). Production user queries with mixed code-switching (Hinglish/Tanglish) may exhibit slightly higher token counts depending on tokenizer vocabulary coverage.

### Production Alert Metric
To catch this analysis being wrong in production, monitor:

$$\text{Indic Output Token Ratio} = \frac{\text{Mean Output Tokens per Indic Request}}{\text{Mean Output Tokens per English Request (for same prompt task)}} \quad \text{in vLLM Serving Logs}$$

* **Alert Threshold**: If production `Indic Output Token Ratio` exceeds **1.50×**, alert the serving team to potential tokenizer fallback or prompt verbosity regressions.
