"""
Evaluare batch pe un test set întreg.

Diferit de inference.py (care rulează pe un singur clip), acest script:
  - Citește un CSV cu coloanele `file_path` și `transcript`
  - Rulează inferență pe toate clipurile
  - Calculează WER și CER global
  - Salvează predicțiile per clip într-un CSV de output

Exemplu de utilizare:
    python evaluate.py \
        --test_csv data/test_seen.csv \
        --data_dir data/ \
        --model iulik-pisik/ro_vsr_125h_auto \
        --output predictions/test_seen.csv

Așteaptă structura: `<data_dir>/<podcast>/<file_path>.avi` unde `<podcast>`
se deduce din `file_path` (primele 2 părți după split pe `_`).
"""

import argparse
import os
import sys

import jiwer
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import (
    DEFAULT_MODEL, DEFAULT_VTP_PATH,
    load_models, normalize_text, run_inference,
)
from ro_vsr.tokenizer import get_tokenizer


def get_podcast(fname: str) -> str:
    parts = str(fname).split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 5 else parts[0]


def fname_to_avi_path(data_dir: str, fname: str) -> str:
    return os.path.join(data_dir, get_podcast(fname), fname + ".avi")


def load_video(avi_path: str, device: torch.device):
    """Încarcă un .avi și returnează tensor [1, 3, T, 96, 96]."""
    import numpy as np
    from decord import VideoReader, bridge
    bridge.set_bridge("native")

    with open(avi_path, "rb") as f:
        vr = VideoReader(f, width=160, height=160)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()

    frames = frames.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).to(device).unsqueeze(0)
    frames = frames.permute(0, 4, 1, 2, 3)

    crop_x = (frames.size(3) - 96) // 2
    crop_y = (frames.size(4) - 96) // 2
    return frames[:, :, :, crop_x:crop_x + 96, crop_y:crop_y + 96]


def compute_global_wer_cer(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return 1.0, 1.0
    refs_l, hyps_l = zip(*pairs)
    return jiwer.wer(list(refs_l), list(hyps_l)), jiwer.cer(list(refs_l), list(hyps_l))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluare batch pe un test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_csv", type=str, required=True,
                        help="CSV cu coloanele file_path, transcript")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder care conține <podcast>/<file>.avi")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--vtp_path", type=str, default=DEFAULT_VTP_PATH)
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=5,
                        help="Dimensiunea n-gram-ului blocat (0 = dezactivat)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    df = pd.read_csv(args.test_csv).dropna(subset=["transcript"])
    print(f"[eval] {len(df)} clipuri în {args.test_csv}")

    tokenizer = get_tokenizer()
    model, visual_encoder = load_models(args.model, args.vtp_path, device)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inferență"):
        fname = str(row["file_path"])
        avi_path = fname_to_avi_path(args.data_dir, fname)
        if not os.path.isfile(avi_path):
            continue

        try:
            faces = load_video(avi_path, device)
            hyp_raw = run_inference(
                faces=faces, model=model, visual_encoder=visual_encoder,
                tokenizer=tokenizer, device=device,
                beam_size=args.beam_size, max_len=args.max_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        except Exception as e:
            print(f"[eval] ⚠️  Eroare {fname}: {e}")
            hyp_raw = ""

        ref_clean = normalize_text(str(row["transcript"]))
        hyp_clean = normalize_text(hyp_raw)

        try:
            wer = jiwer.wer(ref_clean, hyp_clean) if ref_clean and hyp_clean else 1.0
            cer = jiwer.cer(ref_clean, hyp_clean) if ref_clean and hyp_clean else 1.0
        except Exception:
            wer, cer = 1.0, 1.0

        results.append({
            "file_path": fname,
            "reference": ref_clean,
            "hypothesis": hyp_clean,
            "wer": round(wer, 4),
            "cer": round(cer, 4),
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output, index=False)
    print(f"[eval] ✅ Predicții salvate: {args.output}")

    global_wer, global_cer = compute_global_wer_cer(
        df_out["reference"].tolist(), df_out["hypothesis"].tolist()
    )
    print(f"[eval] WER global: {global_wer:.4f}")
    print(f"[eval] CER global: {global_cer:.4f}")
    print(f"[eval] Clipuri:    {len(df_out)}")


if __name__ == "__main__":
    main()
