#!/usr/bin/env python3
"""
build_corpus.py -- Fetch and prepare 4-language parallel evaluation corpus from FLORES-200
Languages: English (eng), Hindi (hin), Tamil (tam), Kannada (kan)
"""

import os
import unicodedata
from datasets import load_dataset

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
os.makedirs(CORPUS_DIR, exist_ok=True)

NUM_SENTENCES = 100

LANG_MAP = {
    "eng": "eng_Latn",
    "hin": "hin_Deva",
    "tam": "tam_Taml",
    "kan": "kan_Knda",
}

def main():
    print(f"Fetching FLORES-200 parallel data for {list(LANG_MAP.keys())}...")
    
    corpora = {}
    for code, flores_code in LANG_MAP.items():
        ds = load_dataset("tomasmajercik/flores-parquet", flores_code, split="validation")
        sentences = []
        for i in range(NUM_SENTENCES):
            text = ds[i]["sentence"].strip()
            # Normalize to NFC
            text = unicodedata.normalize("NFC", text)
            sentences.append(text)
        corpora[code] = sentences
        out_path = os.path.join(CORPUS_DIR, f"{code}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")
        print(f"Saved {len(sentences)} sentences to {out_path}")

    print("Corpus build complete.")

if __name__ == "__main__":
    main()
