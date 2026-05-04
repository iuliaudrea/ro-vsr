"""
Single-clip inference for the Romanian VSR system (VSRo-200).

Example usage:
    python inference.py --fpath samples/sample_1.avi
    python inference.py --fpath samples/sample_1.avi --model vsro200/models-vsro200/checkpoints/model_200h_auto.pt
    python inference.py --fpath samples/sample_1.avi --no_repeat_ngram_size 0

Input clips must be .avi files with 224x224 frames. For raw video, see docs/PREPROCESSING.md.
"""

import argparse
import os
import re
import sys
import warnings

# Suppress autocast warnings when running on CPU (decorators in models.py
# are static and evaluated at import time)
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")

import numpy as np
import torch
from huggingface_hub import hf_hub_download

# Add the `vsr_inference/` package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vsr_inference.models import build_model, build_visual_encoder
from vsr_inference.tokenizer import get_tokenizer
from vsr_inference.beam_search_ngram import beam_search_with_rep_penalty
from vsr_inference.metrics import lookup_reference, print_metrics_block


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_MODEL_REPO = "vsro200/models-vsro200"
DEFAULT_MODEL_FILENAME = "model_200h_auto.pt"
DEFAULT_VTP_PATH = "checkpoints/feature_extractor.pth"
DEFAULT_METADATA = "samples/samples_metadata.csv"


# ============================================================
# UTILITIES
# ============================================================

def clean_prediction(text: str) -> str:
    """Remove Whisper-style special tokens from the prediction."""
    for token in [
        "<|startoftranscript|>", "<|ro|>", "<|transcribe|>",
        "<|notimestamps|>", "<|endoftext|>",
    ]:
        text = text.replace(token, "")
    return text.strip()


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation (keeping hyphens), normalize whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = " ".join(text.split())
    return text.strip()


# ============================================================
# VIDEO LOADING
# ============================================================

def read_video(fpath: str, device: torch.device) -> torch.Tensor:
    """
    Read .avi clip and return a tensor of shape [1, 3, T, 96, 96].

    Crops the central 96x96 region from each frame.
    """
    from decord import VideoReader, bridge
    bridge.set_bridge("native")

    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Video file not found: {fpath}")

    with open(fpath, "rb") as f:
        vr = VideoReader(f, width=160, height=160)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()

    # [T, H, W, C] -> [1, C, T, H, W], normalized to [0, 1]
    frames = frames.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).to(device).unsqueeze(0)
    frames = frames.permute(0, 4, 1, 2, 3)

    # Center crop 96x96
    crop_x = (frames.size(3) - 96) // 2
    crop_y = (frames.size(4) - 96) // 2
    faces = frames[:, :, :, crop_x:crop_x + 96, crop_y:crop_y + 96]

    return faces


# ============================================================
# MODEL LOADING
# ============================================================

def load_models(model_filename: str, vtp_path: str, device: torch.device):
    """
    Load the VTP feature extractor (from local .pth file) and the
    encoder-decoder model (from HuggingFace Hub).

    Returns (model, visual_encoder).
    """
    # --- VTP feature extractor ---
    if not os.path.isfile(vtp_path):
        raise FileNotFoundError(
            f"VTP checkpoint not found: {vtp_path}\n"
            f"   Run: bash scripts/setup.sh"
        )

    visual_encoder = build_visual_encoder().to(device).eval()
    s = torch.load(vtp_path, map_location=device)["state_dict"]
    new_s = {}
    for k, v in s.items():
        if "face_encoder" not in k:
            continue
        new_s[k.replace("module.face_encoder.", "")] = v
    visual_encoder.load_state_dict(new_s)
    for p in visual_encoder.parameters():
        p.requires_grad = False

    # --- Encoder-decoder from HuggingFace ---
    print(f"[load] Downloading encoder-decoder from {DEFAULT_MODEL_REPO}/checkpoints/{model_filename} ...")
    lm_path = hf_hub_download(
        repo_id=DEFAULT_MODEL_REPO,
        filename=f"checkpoints/{model_filename}",
        repo_type="model",
    )
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(lm_path, map_location=device))

    print(f"[load] ✅ Models loaded successfully")
    print(f"        VTP:             {vtp_path}")
    print(f"        Encoder-decoder: {DEFAULT_MODEL_REPO}/checkpoints/{model_filename}")
    return model, visual_encoder


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    faces: torch.Tensor,
    model,
    visual_encoder,
    tokenizer,
    device: torch.device,
    beam_size: int = 5,
    max_len: int = 256,
    no_repeat_ngram_size: int = 5,
) -> str:
    """Run a single forward pass and return the cleaned transcription."""
    start_prompt_ids = tokenizer.encode(
        "<|startoftranscript|><|ro|><|transcribe|><|notimestamps|>"
    )
    start_symbol = torch.tensor(start_prompt_ids).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            vid_emb = visual_encoder(faces)
            B, T, _ = vid_emb.size()
            src_mask = torch.ones((B, 1, T), device=device).bool()
            memory, _ = model.encode(vid_emb, src_mask)

            # Special tokens that should not be penalized by ngram blocking
            special = set()
            for attr in ["sot", "eot", "transcribe", "translate",
                         "no_timestamps", "no_speech", "timestamp_begin"]:
                if hasattr(tokenizer, attr):
                    tok = getattr(tokenizer, attr)
                    if isinstance(tok, int):
                        special.add(tok)
            for tok in start_symbol.tolist():
                special.add(int(tok))
            special.add(0)

            beam_outs, _ = beam_search_with_rep_penalty(
                model=model,
                bos_index=start_symbol,
                eos_index=tokenizer.eot,
                pad_index=0,
                encoder_output=memory,
                src_mask=src_mask,
                size=beam_size,
                max_output_length=max_len,
                n_best=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
                special_token_ids=special,
            )

    best_ids = beam_outs[0][0]
    if isinstance(best_ids, torch.Tensor):
        best_ids = best_ids.tolist()

    return clean_prediction(tokenizer.decode(best_ids))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Romanian VSR inference on a single clip",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fpath", type=str, required=True,
        help="Path to the input .avi clip",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_FILENAME,
        help="Checkpoint filename in vsro200/models-vsro200 (e.g. model_200h_auto.pt, "
            "model_150h_auto.pt, model_100h_annot.pt). See HF repo for the full list.",
    )
    parser.add_argument(
        "--vtp_path", type=str, default=DEFAULT_VTP_PATH,
        help="Path to the VTP feature extractor checkpoint",
    )
    parser.add_argument(
        "--beam_size", type=int, default=5,
        help="Beam size for decoding",
    )
    parser.add_argument(
        "--max_len", type=int, default=256,
        help="Maximum output length (in tokens)",
    )
    parser.add_argument(
        "--no_repeat_ngram_size", type=int, default=5,
        help="Block n-grams of this size from repeating (0 = disabled)",
    )
    parser.add_argument(
        "--metadata", type=str, default=DEFAULT_METADATA,
        help="Path to metadata CSV (used to look up reference and compute WER/CER if available)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Explicit device (cuda / cpu). Default: auto-detect.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")

    # Load
    tokenizer = get_tokenizer()
    model, visual_encoder = load_models(args.model, args.vtp_path, device)

    # Read
    faces = read_video(args.fpath, device)
    print(f"[video] Frames extracted: {tuple(faces.shape)}")

    # Inference
    print("[infer] Running inference ...")
    transcription = run_inference(
        faces=faces,
        model=model,
        visual_encoder=visual_encoder,
        tokenizer=tokenizer,
        device=device,
        beam_size=args.beam_size,
        max_len=args.max_len,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    # Output
    print("─" * 70)
    print(f"File:           {args.fpath}")
    print(f"Transcription:  {transcription}")
    reference = lookup_reference(args.metadata, args.fpath)
    print_metrics_block(reference, transcription, normalize_text)
    print("─" * 70)


if __name__ == "__main__":
    main()
