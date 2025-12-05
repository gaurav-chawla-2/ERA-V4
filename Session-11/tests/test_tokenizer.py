# Module-level imports and TestHindiBPETokenizer class header
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
import sys

# Make Session-11 importable despite hyphen in folder name
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hindi_bpe_tokenizer import build_tokenizer, build_synthetic_hindi_corpus

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

class TestHindiBPETokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        cls.tokenizer = build_tokenizer(vocab_size=6000, max_training_merges=1500)
    def test_vocab_size_exceeds_5000(self):
        vocab_total = len(self.tokenizer.token_to_id)
        self.assertGreaterEqual(vocab_total, 5001, f"vocab size must be ≥ 5001, got {vocab_total}")

    def test_roundtrip_encoding_decoding(self):
        text = "भारत में शिक्षा और स्वास्थ्य का विकास हो रहा है। लोग काम कर रहे हैं।"
        enc = self.tokenizer.encode_to_bytes(text, add_bos_eos=True)
        dec = self.tokenizer.decode_bytes(enc, strip_bos_eos=True)
        self.assertEqual(text, dec, "roundtrip must preserve the original text")

    def test_compression_ratio_minimum(self):
        # Use a longer sample based on synthetic corpus to ensure robust merges
        corpus = build_synthetic_hindi_corpus(n_sentences=200)
        text = " ".join(corpus)
        ratio = self.tokenizer.measure_compression_ratio(text)
        # Record artifact
        with open(os.path.join(ARTIFACTS_DIR, "compression_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"Compression ratio: {ratio:.3f}\n")
            f.write(f"Vocab summary: {self.tokenizer.describe_vocab()}\n")
        self.assertGreaterEqual(ratio, 3.0, f"compression ratio must be ≥ 3.0, got {ratio}")

    def test_special_tokens_exist(self):
        for tok in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", " "]:
            self.assertIn(tok, self.tokenizer.token_to_id)

if __name__ == "__main__":
    unittest.main()