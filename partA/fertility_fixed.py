#!/usr/bin/env python3
"""
fertility_fixed.py -- Corrected multilingual tokenizer benchmark.

Computes tokenizer fertility and compression ratios using robust tokenizers,
micro-averaged totals, regex word boundaries, and standardized denominators.

Usage:
    python fertility_fixed.py --corpus eng=corpus/eng.txt \
                              --corpus hin=corpus/hin.txt \
                              --corpus tam=corpus/tam.txt \
                              --corpus kan=corpus/kan.txt \
                              --tokenizer xlm-roberta-base
"""

import argparse
import re
import sys
import unicodedata
from transformers import AutoTokenizer

def load_tokenizer(spec: str):
    if spec == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return lambda s: enc.encode(s)
    elif spec.startswith("hf:"):
        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    else:
        tok = AutoTokenizer.from_pretrained(spec)
        return lambda s: tok.encode(s, add_special_tokens=False)

def read_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines

def analyze_corpus(lines, encode):
    """
    Computes corpus-wide micro-averaged metrics:
    - total_tokens
    - tok_per_word (regex words)
    - tok_per_char (Unicode code points)
    - tok_per_byte (UTF-8 bytes)
    - tok_per_sentence (parallel sentences)
    """
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0
    num_sentences = len(lines)

    for line in lines:
        tokens = encode(line)
        words = re.findall(r'\w+', line, re.UNICODE)
        chars = len(line)
        bytes_cnt = len(line.encode('utf-8'))

        total_tokens += len(tokens)
        total_words += len(words)
        total_chars += chars
        total_bytes += bytes_cnt

    return {
        "sentences": num_sentences,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_bytes": total_bytes,
        "tok_per_sentence": total_tokens / num_sentences if num_sentences else 0,
        "tok_per_word": total_tokens / total_words if total_words else 0,
        "tok_per_char": total_tokens / total_chars if total_chars else 0,
        "tok_per_byte": total_tokens / total_bytes if total_bytes else 0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="language code and path, e.g. eng=corpus/eng.txt (repeatable)",
    )
    ap.add_argument("--tokenizer", default="xlm-roberta-base")
    args = ap.parse_args()

    encode = load_tokenizer(args.tokenizer)

    print(f"Tokenizer: {args.tokenizer}")
    print(f"{'lang':<8}{'total_tok':>10}{'tok/sent':>12}{'tok/word':>12}{'tok/char':>12}{'tok/byte':>12}")
    print("-" * 66)

    results = {}
    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        lines = read_lines(path)
        metrics = analyze_corpus(lines, encode)
        results[lang] = metrics
        print(f"{lang:<8}{metrics['total_tokens']:>10d}{metrics['tok_per_sentence']:>12.2f}"
              f"{metrics['tok_per_word']:>12.2f}{metrics['tok_per_char']:>12.3f}{metrics['tok_per_byte']:>12.3f}")

    if "eng" in results:
        base = "eng"
        base_sent_tok = results[base]["tok_per_sentence"]
        base_byte_tok = results[base]["tok_per_byte"]
        print("\n--- Relative Ratio to English (eng = 1.00x) ---")
        print(f"{'lang':<8}{'Parallel Sent Ratio':>22}{'tok/byte Ratio':>18}")
        print("-" * 48)
        for lang, metrics in results.items():
            sent_ratio = metrics["tok_per_sentence"] / base_sent_tok
            byte_ratio = metrics["tok_per_byte"] / base_byte_tok
            print(f"{lang:<8}{sent_ratio:>22.2f}x{byte_ratio:>18.2f}x")

if __name__ == "__main__":
    main()
