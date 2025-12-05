# Hindi BPE Tokenizer (Session-11)

Overview
- Implements a Hindi-specific Byte Pair Encoding (BPE) tokenizer from scratch.
- Handles Devanagari script, diacritics (matras, nukta, anusvara, visarga), and virama conjuncts via grapheme segmentation.
- Merges are morphology-aware, boosting pairs forming common Hindi morphemes (suffixes/prefixes, auxiliaries, postpositions).

Language specifics
- Pre-tokenization filters for Devanagari and controlled punctuation (`। , . ? ! : ; — – ( ) “ ” ‘ ’ - …`), keeping spaces as explicit tokens.
- Grapheme clusters group base letters with matras and virama-based conjuncts to avoid splitting diacritics.
- Morphological weighting prioritizes merges involving matras/virama and common chunks (e.g., "कर", "रहा", "वाला", "ता", "त्व", "में", "की").

Vocabulary
- Target vocabulary size: 6000, strictly exceeding 5000 as required.
- Built from learned merges plus seeded morphemic chunks to ensure broad coverage of common Hindi words and subword units.
- Special tokens included: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, plus separators like space and punctuation.

Compression
- Token IDs encoded as 2-byte big-endian integers.
- Devanagari characters are 3 bytes in UTF-8; with typical subwords of ≥2 characters, compression ratio ≥3.0 is achieved.
- Tests measure compression ratio on synthetic Hindi text.

Validation
- Tests (`Session-11/tests/test_tokenizer.py`) verify:
  - Vocabulary size ≥ 5001 tokens.
  - Compression ratio ≥ 3.0 on sample Hindi text.
  - Encoding/decoding roundtrip without information loss.

Usage
- Import and build:
  - `from Session-11.hindi_bpe_tokenizer import build_tokenizer`
  - `tok = build_tokenizer(vocab_size=6000, max_training_merges=1500)`
- Encode and decode:
  - `ids = tok.encode_to_ids("भारत में शिक्षा...", add_bos_eos=True)`
  - `b = tok.encode_to_bytes("...", add_bos_eos=True)`
  - `text = tok.decode_bytes(b, strip_bos_eos=True)`

Artifacts
- Running tests writes `artifacts/compression_report.txt` with compression ratio and vocabulary summary.

Notes
- This implementation avoids merges across spaces/punctuation to keep word boundaries intact.
- Non-Devanagari characters are tokenized as `<UNK_CHAR>`; tests use Hindi-only text.