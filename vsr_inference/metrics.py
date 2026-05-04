"""
Helpers for computing WER/CER against a reference transcription read
from a metadata CSV.

Used by `inference.py` and `evaluation/avsr/inference_avsr.py` to print
per-clip metrics when a reference is available, without requiring the
user to pass the reference manually.
"""

import os
from typing import Optional

import pandas as pd


def lookup_reference(metadata_csv: str, fpath: str) -> Optional[str]:
    """
    Look up the reference transcription for `fpath` in `metadata_csv`.

    Matching is by basename: the row whose `file_path` column ends with
    the same basename as `fpath` is selected. This way the script works
    whether the user passes an absolute path, a relative path, or just
    the file name.

    Returns the reference string, or None if not found.
    """
    if not metadata_csv or not os.path.isfile(metadata_csv):
        return None

    target_basename = os.path.basename(fpath)
    try:
        df = pd.read_csv(metadata_csv)
    except Exception:
        return None

    if "file_path" not in df.columns or "reference" not in df.columns:
        return None

    matches = df[df["file_path"].apply(
        lambda p: os.path.basename(str(p)) == target_basename
    )]
    if len(matches) == 0:
        return None

    ref = matches.iloc[0]["reference"]
    if pd.isna(ref) or str(ref).strip().startswith("REPLACE_WITH"):
        return None
    return str(ref)


def compute_wer_cer(reference: str, hypothesis: str):
    """
    Compute WER and CER on already-normalized strings. Returns
    (wer, cer) as floats in [0, 1+], or (None, None) if jiwer is not
    installed or the reference is empty.
    """
    if not reference or not reference.strip():
        return None, None
    try:
        import jiwer
    except ImportError:
        return None, None
    try:
        wer = jiwer.wer(reference, hypothesis) if hypothesis.strip() else 1.0
        cer = jiwer.cer(reference, hypothesis) if hypothesis.strip() else 1.0
    except Exception:
        return None, None
    return wer, cer


def print_metrics_block(reference: Optional[str], hypothesis: str, normalize_fn):
    """
    Print a metrics block with reference, hypothesis, WER, and CER.
    `normalize_fn` is applied to both before scoring.
    """
    if reference is None:
        return
    print(f"Reference:      {reference}")
    ref_norm = normalize_fn(reference)
    hyp_norm = normalize_fn(hypothesis)
    try:
        import jiwer
    except ImportError:
        print("WER/CER:        (install `jiwer` to compute metrics: pip install jiwer)")
        return
    wer, cer = compute_wer_cer(ref_norm, hyp_norm)
    if wer is not None:
        print(f"WER:            {wer * 100:.2f}%")
        print(f"CER:            {cer * 100:.2f}%")
