# Part C: Decision Memo — Casual Multilingual Response Generation

**To:** AI Leadership & Product Team  
**From:** AI Audit Team  
**Date:** September 6, 2026  
**Subject:** Trade-Off Analysis & Path Recommendation for Casual/Conversational Indic Generation  

---

## Executive Summary

The product team requires assistant responses in Hindi, Kannada, Tamil, Telugu, Bengali, and Marathi to sound casual and conversational rather than textbook/formal. 

Based on a quantitative trade-off evaluation under our hardware, reviewer, and timeline constraints, we recommend **Path (a): Supervised Fine-Tuning (SFT) on Synthetic "Casualized" Response Pairs using LoRA/QLoRA on FLM-4B**. Path (a) embeds casual phrasing directly into model weights, preserving full single-model serving speed and KV-cache VRAM capacity.

---

## 1. Stated Constraints & Explicit Assumptions

### Constraint 1: Compute Budget
* 1× A100-80GB GPU for 2 weeks = $14 \text{ days} \times 24 \text{ hours} = \mathbf{336 \text{ GPU hours}}$.

### Constraint 2: Reviewer Throughput & Critical Evaluation Caveat
* 1 native-speaker reviewer (covering **Hindi and Kannada only**) for 10 hours/week across 3 weeks = **30 total review hours**.
* Human evaluation speed: Reviewing a prompt-response pair for casualness takes ~2 minutes per item ($30 \text{ pairs/hour}$).
* Total human review capacity across 3 weeks = $30 \text{ hours} \times 30 \text{ items/hour} = \mathbf{900 \text{ evaluated pairs}}$.
* > [!WARNING]
  > **Critical Reviewer Caveat**: Because the native reviewer covers **Hindi and Kannada only**, direct human pairwise validation is capped at 450 pairs per language. Quality for **Tamil, Telugu, Bengali, and Marathi** must rely on automated metrics (chrF++, BLEU, length/casual dictionary checks) and cross-lingual transfer from Hindi/Kannada training.

### Explicit Model & Serving Assumptions (Estimates)
1. *Assumption (Rewriter VRAM Estimate)*: A $\le 1\text{B}$ FP16 rewriter model requires an estimated **2.0 GB VRAM** for weights plus runtime overhead on the L4 GPU.
2. *Assumption (Rewriter Latency Estimate)*: Executing a second 1B model pass per request adds an estimated **+35–45 ms** inter-token/prefill latency per request.
3. *Assumption (Synthetic Generation Speed Estimate)*: Running open `Qwen2.5-72B-Instruct` (INT4) locally on 1× A100-80GB generates an estimated ~25 tokens/s.
4. *Assumption (Prompt Engineering Latency Estimate)*: System prompt additions add ~200 prompt tokens per request, adding an estimated **+15–25 ms** prefill TTFT latency.

---

## 2. Back-of-the-Envelope Data & Training Arithmetic

### Synthetic Data Dataset Calculation ($N$)
* Target dataset size: 1,500 casual pairs per language × 6 languages = **9,000 synthetic pairs** (~1.35M output tokens).
* Estimated generation time on A100 GPU:

$$\text{Synthetic Generation Time} = \frac{1,350,000 \text{ tokens}}{25 \text{ tok/s}} = 54,000 \text{ seconds} = \mathbf{15.0 \text{ GPU hours}} \quad (4.5\% \text{ of compute budget})$$

### Training Time Arithmetic
* Fine-tuning FLM-4B with QLoRA ($r=16, \alpha=32$) on 9,000 pairs (1.35M tokens) takes an estimated ~45 minutes per epoch on 1× A100.
* A full 3-epoch SFT run takes an estimated **2.25 GPU hours**. We can execute **20+ ablation experiments** well within our 336 GPU hour budget.

---

## 3. Trade-Off Evaluation of Paths (Assumption-Based Estimates)

| Evaluation Dimension | Path (a): SFT Pass (LoRA/QLoRA) | Path (b): $\le 1\text{B}$ Rewriter Model | Path (c): Prompt-Engineering Only |
|---|---|---|---|
| **Serving Architecture** | Single FLM-4B model with merged weights | Dual sequential models (FLM-4B + Rewriter) | Single FLM-4B model with longer prompt |
| **L4 Serving VRAM Overhead (Estimated)** | **0 GB extra** (weights merged) | **~2.0 GB** (Estimated from 1B fp16 assumption) | 0 GB (weights) |
| **Max 4096-Seq KV Batch (Estimated)** | **25 sequences** (100% capacity) | **~18 sequences** (Estimated reduction from VRAM) | ~23 sequences (Estimated reduction from prompt) |
| **Inference Latency Impact (Estimated)** | **0 ms** extra latency | **+35–45 ms** extra ITL (Estimated 2nd pass) | **+15–25 ms** extra TTFT (Estimated prefill) |
| **Tone Quality & Nuance** | Teaches colloquial Indic idioms natively | Good, but dependent on 1B model capacity | Limited; fails to inject unlearned idioms |
| **Final Recommendation** | **SELECTED** | **REJECTED** (VRAM & Latency penalty) | **REJECTED** (Wastes prompt context) |

---

## 4. Success Metric with Numeric Threshold

Primary success will be evaluated using a **Native-Speaker Pairwise Preference Rating**:

$$\text{Human Preference Rate (Hindi + Kannada)} = \frac{\text{Outputs Rated More Casual/Natural than Baseline}}{\text{Total Evaluated Pairwise Outputs}} \ge \mathbf{70.0\%}$$

* **Evaluation Protocol**: Double-blind A/B testing on 300 held-out prompts per language (Hindi and Kannada) evaluated by the native reviewer.
* **Secondary Factual Integrity Check (All 6 Languages)**:

$$\text{chrF++ Score} \ge \text{Baseline} - 1.5 \text{ points} \quad \text{and} \quad \text{Semantic Error / Hallucination Rate} \le \mathbf{5.0\%}$$

---

## 5. Kill Criterion

> [!CAUTION]
> **Kill Criterion**: We will immediately **ABANDON Path (a)** and pivot if, by **Day 8 (end of Week 1)**:
> 1. The Hindi/Kannada Human Preference Rate on a 100-sample validation checkpoint is **$< 55.0\%$** (indicating synthetic data failure or catastrophic forgetting), OR
> 2. Automated evaluation reveals severe code-switching / script corruption (>10% non-native English words inserted into Indic responses).

If triggered, the team will pivot on Day 9 to Path (c) (Prompt Engineering with dynamic few-shot in-context exemplars).

---

## 6. Day-1 Pilot Experiment Design

### Objective
Execute a small-scale pilot experiment to evaluate synthetic casual data quality and establish baseline QLoRA convergence before committing to full dataset generation.

### Pilot Execution Steps (Day 1)
1. **Hours 0–3**: Load `Qwen2.5-72B-Instruct` (INT4) on the A100 GPU. Prompts: generate a **300–500 pair pilot dataset** (Hindi + Kannada formal-to-casual response pairs).
2. **Hours 3–5**: Create train/validation split (400 train, 100 val). Fine-tune a pilot QLoRA adapter ($r=16, \alpha=32$) on FLM-4B for 3 epochs.
3. **Hours 5–7**: Sample 50 validation outputs (25 Hindi, 25 Kannada) and evaluate against the baseline model.
4. **Hours 7–10**: Deliver 25 Hindi and 25 Kannada pilot examples to the native reviewer for initial scoring (1–5 scale for casualness vs formality) to establish a clear continue/kill decision point before scaling up dataset generation to 9,000 pairs.

### Expected Day-1 Deliverable
Empirical confirmation that QLoRA loss decreases smoothly from ~2.4 to <1.1, and initial reviewer casualness rating increases from 1.8/5 (baseline) to $\ge 3.8/5$.
