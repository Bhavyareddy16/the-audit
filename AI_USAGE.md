# AI Usage Disclosure (`AI_USAGE.md`)

## 1. Overview
In accordance with the assignment ground rules, AI tools (Antigravity paired coding assistant) were utilized for script automation, parallel corpus pipeline construction, mathematical verification, and report drafting. Every claimed flaw, mathematical derivation, and architectural trade-off was verified against empirical python execution logs and exact hardware formulas.

---

## 2. Where AI Was Helpful

1. **Automated Corpus Pipeline (`partA/build_corpus.py`)**:
   * AI generated the dataset loader script to fetch 100 parallel evaluation sentences across English, Hindi, Tamil, and Kannada from FLORES-200 (`tomasmajercik/flores-parquet`).

2. **Isolated Evidence Suite (`partA/fertility_audit.py`)**:
   * AI automated the isolated before/after benchmark script to calculate exact quantitative deltas for `split(" ")`, lowercasing normalization, macro vs micro averaging, code points vs UTF-8 bytes, and tokenizer choice.

3. **Serving & Capacity Verification (`partB/bench_analysis.py`)**:
   * AI assisted in writing verification routines to compute exact KV cache bytes per token ($114,688 \text{ bytes/token}$), max sequence capacity ($25 \text{ sequences}$), and verify the exact match between the two independent derivations of batch-24 long-prompt goodput ($200.92 \text{ gen tok/s}$).

4. **Markdown Formatting**:
   * AI assisted in converting raw execution logs and math into GitHub-flavored Markdown reports with structured tables, KaTeX equations, and alerts.

---

## 3. Where AI Misled or Required Human Course Correction

1. **Premature Bug Categorization (Crucial Correction)**:
   * **AI Initial Bias**: AI initially attempted to categorize behaviors in `fertility.py` into a fixed list of "4 code bugs and 2 conceptual flaws" before running experiments.
   * **Human Correction**: Enforced strict evidence-rule neutrality. Tested every behavior empirically before labeling it. Proved that `unicodedata.normalize("NFC", line)` is a **harmless/valid feature** (0 delta on clean text, prevents +55% token explosion on decomposed text), avoiding penalty points for unverified bug claims.

2. **Refining the Explanation of `len(line)`**:
   * **AI Failure**: AI initially drafted a claim that `len(line)` was a "code bug that confuses bytes with code points."
   * **Human Correction**: Re-framed correctly: Python `len(line)` accurately measures Unicode code points. However, evaluating cross-lingual fertility using code points introduces a **3x script encoding artifact** relative to UTF-8 bytes because Devanagari uses 3 bytes per code point.

3. **Part C Pre-Selection Correction**:
   * **AI Failure**: AI initially pre-selected SFT upfront before performing constraint trade-off arithmetic.
   * **Human Correction**: Re-structured Part C to start neutrally from stated constraints (1x A100 GPU for 2 weeks, 30 total reviewer hours = 900 evaluated pairs max capacity for Hindi + Kannada), evaluate all 3 paths fairly, and derive the recommendation from first principles.

4. **Dataset Split Name Mismatch**:
   * **AI Failure**: AI assumed the dataset split in `tomasmajercik/flores-parquet` was `'dev'`, triggering a runtime `ValueError`.
   * **Human Correction**: Corrected dataset split parameter to `'validation'`.

---

## 4. Verification Checklist

* [x] Every claim in Part A supported by before/after empirical numbers from `partA/fertility_audit.py`.
* [x] KV-cache arithmetic ($112 \text{ KiB/token} \implies 25$ sequence limit) reconciled against `bench_log.csv`.
* [x] Long-prompt batch-24 goodput derived via TWO independent methods ($200.9 \text{ gen tok/s}$).
* [x] All 6 language constraints, A100 GPU hours ($336\text{h}$), and reviewer limits ($30\text{h} = 900$ pairs) reconciled in Part C arithmetic.
