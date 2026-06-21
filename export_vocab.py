import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# ── Check required packages are available ─────────────────────────────────────
def _require(pkg: str, install_hint: str = ""):
    try:
        return __import__(pkg)
    except ImportError:
        hint = f"  pip install {install_hint or pkg}"
        print(f"\n✗  Missing package: '{pkg}'\n{hint}\n")
        sys.exit(1)

_require("nltk")
_require("datasets", "datasets")

import nltk
for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS  — must match notebook exactly
# ═════════════════════════════════════════════════════════════════════════════

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MIN_FREQ  = 2          # tokens appearing fewer than this times are dropped
DATASET   = "zeroshot/twitter-financial-news-sentiment"
CONFIG    = "default"
SPLIT     = "train"    # vocabulary built on training data ONLY

# Stopwords to KEEP because they carry financial sentiment signal
SENTIMENT_KEEP = {
    "up", "down", "not", "no", "nor",
    "against", "below", "above", "under", "over",
}

# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  — mirrors notebook Section 3 clean_tweet_rnn() exactly
# ═════════════════════════════════════════════════════════════════════════════

_STOP_WORDS = set(stopwords.words("english")) - SENTIMENT_KEEP
_LEM        = WordNetLemmatizer()


def clean_tweet_rnn(text: str) -> str:
    """
    Full cleaning pipeline for RNN-based models.

    Steps (must stay in sync with notebook Section 3):
      1. Lowercase
      2. Remove URLs  (http://…  www.…)
      3. Remove @mentions
      4. Normalise $TICKER → ticker   (drop $ symbol, keep the word)
      5. Remove # symbol, keep hashtag text
      6. Remove all non-alphabetic characters
      7. Collapse whitespace
      8. Remove stopwords (preserving financial signal words)
      9. Lemmatise each remaining token
      10. Drop tokens shorter than 3 characters
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+",  "",    text)   # step 2
    text = re.sub(r"@\w+",            "",    text)   # step 3
    text = re.sub(r"\$([A-Za-z]+)",   r"\1", text)   # step 4
    text = re.sub(r"#(\w+)",          r"\1", text)   # step 5
    text = re.sub(r"[^a-z\s]",        "",    text)   # step 6
    text = re.sub(r"\s+",             " ",   text).strip()  # step 7
    tokens = [
        _LEM.lemmatize(t)                            # step 9
        for t in text.split()
        if t not in _STOP_WORDS and len(t) > 2       # steps 8 & 10
    ]
    return " ".join(tokens)


# ═════════════════════════════════════════════════════════════════════════════
# VOCABULARY BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_vocab(verbose: bool = True) -> dict:
    """
    Build and return the word2idx mapping from the training split.

    Returns
    -------
    dict  {token: int_index}
          PAD → 0, UNK → 1, then sorted alphabetical vocab words.
    """
    if verbose:
        _section("STEP 1 — Load training data")
        print(f"  Dataset : {DATASET}  [{CONFIG} / {SPLIT}]")

    t0 = time.time()
    dataset  = load_dataset(DATASET, CONFIG)
    train_df = dataset[SPLIT].to_pandas()

    if verbose:
        print(f"  Rows loaded    : {len(train_df):,}")
        print(f"  Columns        : {train_df.columns.tolist()}")
        print(f"  Load time      : {time.time()-t0:.1f}s")
        _class_dist(train_df, "  Class distribution (raw)")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    if verbose:
        _section("STEP 2 — Apply preprocessing pipeline")
        print("  Running clean_tweet_rnn() on every tweet…")

    t1 = time.time()
    train_df["cleaned"] = train_df["text"].apply(clean_tweet_rnn)

    # Drop tweets that become empty after cleaning
    before = len(train_df)
    train_df = train_df[train_df["cleaned"].str.strip() != ""].reset_index(drop=True)
    after  = len(train_df)
    dropped = before - after

    if verbose:
        print(f"  Clean time     : {time.time()-t1:.1f}s")
        print(f"  Tweets dropped (empty after cleaning) : {dropped}")
        print(f"  Remaining tweets                      : {after:,}")
        # Show a few examples
        print("\n  Sample cleaned tweets:")
        for _, row in train_df.sample(4, random_state=42).iterrows():
            print(f"    Original : {row['text'][:70]}")
            print(f"    Cleaned  : {row['cleaned'][:70]}")
            print()

    # ── Step 3: Count token frequencies ──────────────────────────────────────
    if verbose:
        _section("STEP 3 — Count token frequencies (train only)")

    word_freq = Counter()
    for text in train_df["cleaned"]:
        word_freq.update(text.split())

    total_unique  = len(word_freq)
    singleton_cnt = sum(1 for f in word_freq.values() if f == 1)
    kept_cnt      = sum(1 for f in word_freq.values() if f >= MIN_FREQ)

    if verbose:
        print(f"  Total unique tokens   : {total_unique:,}")
        print(f"  Singletons (freq=1)   : {singleton_cnt:,}  ({singleton_cnt/total_unique*100:.1f}%)  ← dropped")
        print(f"  Tokens with freq≥{MIN_FREQ}    : {kept_cnt:,}  ({kept_cnt/total_unique*100:.1f}%)  ← kept")
        print(f"\n  Top-20 most frequent tokens:")
        for word, freq in word_freq.most_common(20):
            bar = "█" * min(30, int(freq / 50))
            print(f"    {word:<18} {freq:>5}  {bar}")

    # ── Step 4: Build word2idx ────────────────────────────────────────────────
    if verbose:
        _section("STEP 4 — Build word2idx mapping")

    vocab_words = [w for w, freq in word_freq.items() if freq >= MIN_FREQ]
    vocab_words = sorted(vocab_words)   # sorted → deterministic / reproducible

    # Reserve index 0 for PAD and 1 for UNK
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word in vocab_words:
        word2idx[word] = len(word2idx)

    if verbose:
        print(f"  PAD token  '{PAD_TOKEN}' → index {word2idx[PAD_TOKEN]}")
        print(f"  UNK token  '{UNK_TOKEN}' → index {word2idx[UNK_TOKEN]}")
        print(f"  Vocab words → indices {2} … {len(word2idx)-1}")
        print(f"  Total vocabulary size : {len(word2idx):,}")

        # Coverage: what % of train tokens are in vocab?
        all_tokens  = sum(len(t.split()) for t in train_df["cleaned"])
        unk_tokens  = sum(
            sum(1 for tok in t.split() if tok not in word2idx)
            for t in train_df["cleaned"]
        )
        coverage = (all_tokens - unk_tokens) / all_tokens * 100
        print(f"  Train token coverage  : {coverage:.2f}%  (tokens in vocab / total tokens)")

    return word2idx


# ═════════════════════════════════════════════════════════════════════════════
# CONSISTENCY VERIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def verify_against_models(vocab_path: str, models_dir: str) -> bool:
    """
    Sanity-check that the saved model weights are compatible with this vocab.

    We can't directly read vocab_size from a .pt state_dict without
    instantiating the model, but we CAN check the embedding weight shape.
    The embedding matrix shape is  (vocab_size, embed_dim).
    """
    import torch

    print()
    _section("STEP 6 — Verify vocab vs saved model weights")

    with open(vocab_path) as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    all_ok = True
    for model_name in ("SimpleRNN", "LSTM", "GRU"):
        pt_path = os.path.join(models_dir, f"{model_name}_best.pt")
        if not os.path.exists(pt_path):
            print(f"  ⚠  {model_name}_best.pt not found — skipping")
            continue
        try:
            state = torch.load(pt_path, map_location="cpu")
            emb_shape = state["embedding.weight"].shape
            saved_vocab = emb_shape[0]
            embed_dim   = emb_shape[1]
            match = "✓" if saved_vocab == vocab_size else "✗ MISMATCH"
            print(f"  {match}  {model_name:<12} "
                  f"embedding shape: {emb_shape}   "
                  f"vocab_size={saved_vocab}  "
                  f"(vocab.json has {vocab_size})")
            if saved_vocab != vocab_size:
                all_ok = False
                print(f"      ⚡ FIX: Re-run the notebook to retrain with the current vocab,")
                print(f"         or re-run export_vocab.py to rebuild vocab from the same")
                print(f"         training run that produced {model_name}_best.pt")
        except Exception as e:
            print(f"  ✗  {model_name}: could not read weights — {e}")
            all_ok = False

    if all_ok:
        print("\n  ✓ All available models are compatible with vocab.json")
    else:
        print("\n  ✗ Compatibility issues detected — see above")
    return all_ok


# ═════════════════════════════════════════════════════════════════════════════
# SAVE + REPORT
# ═════════════════════════════════════════════════════════════════════════════

def save_vocab(word2idx: dict, output_path: str, verbose: bool = True) -> None:
    """Write word2idx to JSON with a companion _meta.json for audit trail."""
    import hashlib

    if verbose:
        _section("STEP 5 — Save to disk")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(word2idx, f, indent=2)

    # Compute checksum for reproducibility audit
    with open(output_path, "rb") as f:
        checksum = hashlib.md5(f.read()).hexdigest()

    file_size_kb = os.path.getsize(output_path) / 1024

    # Write companion metadata file
    meta = {
        "vocab_size":  len(word2idx),
        "pad_token":   PAD_TOKEN,
        "pad_index":   word2idx[PAD_TOKEN],
        "unk_token":   UNK_TOKEN,
        "unk_index":   word2idx[UNK_TOKEN],
        "min_freq":    MIN_FREQ,
        "dataset":     DATASET,
        "config":      CONFIG,
        "split":       SPLIT,
        "file_size_kb": round(file_size_kb, 1),
        "md5":         checksum,
        "generated":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "Vocabulary built from training split ONLY. "
            "Indices: PAD=0, UNK=1, vocab words start at 2 (sorted alphabetically). "
            "Must match the vocab used when training the RNN models."
        ),
    }
    meta_path = output_path.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"  vocab.json      → {output_path}")
        print(f"  vocab_meta.json → {meta_path}")
        print(f"  File size       : {file_size_kb:.1f} KB")
        print(f"  MD5 checksum    : {checksum}")
        print(f"  Vocab size      : {len(word2idx):,} tokens")


def print_usage_example(vocab_path: str) -> None:
    """Print a copy-paste snippet showing how to use vocab.json in the app."""
    print()
    _section("HOW TO USE vocab.json IN app.py")
    print(f"""
  The dashboard's  load_vocab()  function already handles this automatically.
  For reference, here is the manual usage pattern:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  import json, torch                                                     │
  │                                                                         │
  │  # 1. Load vocab                                                        │
  │  with open("{vocab_path}") as f:                                        │
  │      word2idx = json.load(f)                                            │
  │                                                                         │
  │  # 2. Clean + tokenise your text                                        │
  │  cleaned = clean_tweet_rnn("$TSLA crushes Q4 earnings!")                │
  │  # → "tsla crush q4 earnings"                                           │
  │                                                                         │
  │  # 3. Convert to integer sequence (pad/truncate to 32)                  │
  │  UNK = word2idx["<UNK>"]                                                │
  │  PAD = word2idx["<PAD>"]                                                │
  │  seq = [word2idx.get(t, UNK) for t in cleaned.split()][:32]             │
  │  seq = seq + [PAD] * (32 - len(seq))                                    │
  │                                                                         │
  │  # 4. Model forward pass                                                │
  │  tensor = torch.tensor([seq], dtype=torch.long)                         │
  │  with torch.no_grad():                                                  │
  │      logits = model(tensor)          # shape: (1, 3)                    │
  │      probs  = torch.softmax(logits, dim=1).squeeze()                    │
  │  # → probs: [P(Bearish), P(Bullish), P(Neutral)]                        │
  └─────────────────────────────────────────────────────────────────────────┘
""")


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _section(title: str) -> None:
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


def _class_dist(df, label: str) -> None:
    dist = df["label"].value_counts().sort_index()
    label_names = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    total = len(df)
    print(f"\n  {label}:")
    for lbl_id, count in dist.items():
        name = label_names.get(int(lbl_id), str(lbl_id))
        pct  = count / total * 100
        bar  = "█" * int(pct / 3)
        print(f"    {name:<10} (label={lbl_id}): {count:>5}  ({pct:5.1f}%)  {bar}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Export vocabulary from the training dataset for use in the Streamlit dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_vocab.py                   # build and save vocab
  python export_vocab.py --verify          # also check model weight compatibility
  python export_vocab.py --force           # overwrite if vocab.json already exists
  python export_vocab.py --output models/vocab.json  # custom output path
        """
    )
    parser.add_argument("--output",  default=os.path.join("models", "vocab.json"),
                        help="Output path for vocab.json  (default: models/vocab.json)")
    parser.add_argument("--verify",  action="store_true",
                        help="After saving, verify vocab size matches saved model weights")
    parser.add_argument("--force",   action="store_true",
                        help="Overwrite vocab.json if it already exists")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress detailed output (just save and print final path)")
    args = parser.parse_args()

    verbose    = not args.quiet
    vocab_path = args.output
    models_dir = os.path.dirname(vocab_path) or "models"

    # ── Check if already exists ───────────────────────────────────────────────
    if os.path.exists(vocab_path) and not args.force:
        with open(vocab_path) as f:
            existing = json.load(f)
        print(f"\n  vocab.json already exists at: {vocab_path}")
        print(f"  Existing vocab size          : {len(existing):,} tokens")
        print(f"\n  Use --force to overwrite, or --verify to check compatibility.")
        print(f"  Exiting without changes.")
        sys.exit(0)

    t_start = time.time()

    if verbose:
        print()
        print("═" * 65)
        print("  export_vocab.py — Financial News Sentiment Project")
        print("═" * 65)
        print(f"  Output path : {vocab_path}")
        print(f"  Dataset     : {DATASET}  [{CONFIG}]")
        print(f"  Min freq    : {MIN_FREQ}")
        print(f"  Special tokens: PAD={PAD_TOKEN!r}(idx=0), UNK={UNK_TOKEN!r}(idx=1)")

    # ── Build ─────────────────────────────────────────────────────────────────
    word2idx = build_vocab(verbose=verbose)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_vocab(word2idx, vocab_path, verbose=verbose)

    # ── Verify ────────────────────────────────────────────────────────────────
    if args.verify:
        verify_against_models(vocab_path, models_dir)

    # ── Usage example ─────────────────────────────────────────────────────────
    if verbose:
        print_usage_example(vocab_path)

    elapsed = time.time() - t_start
    print()
    print("═" * 65)
    print(f"  ✓  Done in {elapsed:.1f}s")
    print(f"  ✓  vocab.json written to: {vocab_path}")
    print(f"  ✓  Vocab size: {len(word2idx):,} tokens")
    print()
    print("  Next step — launch the dashboard:")
    print("    streamlit run app.py")
    print("═" * 65)
    print()


if __name__ == "__main__":
    main()
