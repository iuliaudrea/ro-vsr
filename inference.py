"""
Inferență pe un singur clip pentru sistemul Romanian VSR.

Exemplu de utilizare:
    python inference.py --fpath samples/sample_1.avi
    python inference.py --fpath samples/sample_1.avi --model iulik-pisik/ro_vsr_150h_auto
    python inference.py --fpath samples/sample_1.avi --no-ngram-blocking

Clipul de input trebuie să fie un fișier .avi cu cadre de 160x160
(crop de față centrat pe gură). Pentru video brut, vezi docs/PREPROCESSING.md.
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download

# Adăugăm package-ul `ro_vsr/` în path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ro_vsr.models import build_model, build_visual_encoder
from ro_vsr.tokenizer import get_tokenizer
from ro_vsr.beam_search_ngram import beam_search_with_rep_penalty


# ============================================================
# CONFIG IMPLICIT
# ============================================================

DEFAULT_MODEL = "iulik-pisik/ro_vsr_175h_auto"
DEFAULT_VTP_URL = (
    "https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/"
    "checkpoints/extended_train_data/feature_extractor.pth"
)
DEFAULT_VTP_PATH = "checkpoints/feature_extractor.pth"


# ============================================================
# UTILITARE
# ============================================================

def clean_prediction(text: str) -> str:
    """Elimină token-urile speciale Whisper din predicție."""
    for token in [
        "<|startoftranscript|>", "<|ro|>", "<|transcribe|>",
        "<|notimestamps|>", "<|endoftext|>",
    ]:
        text = text.replace(token, "")
    return text.strip()


def normalize_text(text: str) -> str:
    """Lowercase + elimină punctuația (păstrează cratima) + spații normalizate."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = " ".join(text.split())
    return text.strip()


# ============================================================
# CITIRE VIDEO
# ============================================================

def read_video(fpath: str, device: torch.device) -> torch.Tensor:
    """
    Citește un .avi de 160x160 și returnează tensor [1, 3, T, 96, 96].

    Crop central 96x96 din mijlocul cadrului (zona gurii).
    """
    from decord import VideoReader, bridge
    bridge.set_bridge("native")

    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Fișier video lipsă: {fpath}")

    with open(fpath, "rb") as f:
        vr = VideoReader(f, width=160, height=160)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()

    # [T, H, W, C] -> [1, C, T, H, W], normalizat la [0, 1]
    frames = frames.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).to(device).unsqueeze(0)
    frames = frames.permute(0, 4, 1, 2, 3)  # [1, C, T, H, W]

    # Crop central 96x96 (zona gurii)
    crop_x = (frames.size(3) - 96) // 2
    crop_y = (frames.size(4) - 96) // 2
    faces = frames[:, :, :, crop_x:crop_x + 96, crop_y:crop_y + 96]

    return faces


# ============================================================
# ÎNCĂRCARE MODELE
# ============================================================

def load_models(model_repo: str, vtp_path: str, device: torch.device):
    """
    Încarcă VTP feature extractor (din .pth local) + encoder-decoder
    (din HuggingFace Hub).

    Returnează (model, visual_encoder).
    """
    # --- VTP feature extractor ---
    if not os.path.isfile(vtp_path):
        raise FileNotFoundError(
            f"Checkpoint VTP lipsă: {vtp_path}\n"
            f"   Rulează: bash scripts/download_checkpoints.sh"
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

    # --- Encoder-decoder de pe HuggingFace ---
    print(f"[load] Descarc enc-dec din {model_repo} ...")
    lm_path = hf_hub_download(
        repo_id=model_repo,
        filename="checkpoints/best_model.pt",
        repo_type="model",
    )
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(lm_path, map_location=device))

    print(f"[load] ✅ Modele încărcate cu succes")
    print(f"        VTP:     {vtp_path}")
    print(f"        Enc-dec: {model_repo}")
    return model, visual_encoder


# ============================================================
# INFERENȚĂ
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
    """
    Rulează un singur forward pass și returnează transcrierea curată.
    """
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

            # Token-uri speciale care nu trebuie penalizate
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
        description="Inferență Romanian VSR pe un singur clip",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fpath", type=str, required=True,
        help="Calea către clipul .avi de input (160x160)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Repo HuggingFace pentru modelul enc-dec",
    )
    parser.add_argument(
        "--vtp_path", type=str, default=DEFAULT_VTP_PATH,
        help="Calea către checkpoint-ul VTP feature extractor",
    )
    parser.add_argument(
        "--beam_size", type=int, default=5,
        help="Beam size pentru decodare",
    )
    parser.add_argument(
        "--max_len", type=int, default=256,
        help="Lungime maximă de output (în token-uri)",
    )
    parser.add_argument(
        "--no_repeat_ngram_size", type=int, default=5,
        help="Dimensiunea n-gram-ului blocat la repetare (0 = dezactivat)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device explicit (cuda / cpu). Implicit: auto.",
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
    print(f"[video] Frames extrase: {tuple(faces.shape)}")

    # Inference
    print("[infer] Rulez inferența ...")
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
    print(f"Fișier:       {args.fpath}")
    print(f"Transcriere:  {transcription}")
    print("─" * 70)


if __name__ == "__main__":
    main()
