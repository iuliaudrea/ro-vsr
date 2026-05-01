"""
Single-clip word classification on LRRo using our pre-trained encoder
and a lightweight attention-pooling + MLP head.

Pipeline:
  1. Load clip frames (preprocessing strategy chosen via --strategy)
  2. Pass through the (frozen) VTP visual encoder + VSR transformer encoder
  3. Pool temporally with attention; classify with MLP

The VTP visual encoder and the VSR encoder are FIXED (not configurable):
  - VTP: original feature extractor from VGG Oxford
  - VSR encoder: from iulik-pisik/ro_vsr_150h_auto

The MLP and preprocessing strategy ARE configurable. Each preprocessing
strategy has its own MLPs trained on it (separate for the LAB and WILD
splits of LRRo).

Example usage:
    # Default: 48_bottom strategy + LAB MLP
    python inference_lrro.py --clip_dir /path/to/lrro/clip_folder

    # Use 64_bottom + Wild MLP:
    python inference_lrro.py --clip_dir /path/to/clip --strategy 64_bottom --split wild

LRRo dataset must be obtained separately from the official source:
    https://bionescu.aimultimedialab.ro/LRRo.html
"""

import argparse
import json
import os
import sys
import warnings
from typing import Optional

# Suppress autocast warnings on CPU
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")

import torch
from huggingface_hub import hf_hub_download

# Add the repository root to path so we can import from ro_vsr/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from ro_vsr.models import build_model, build_visual_encoder

from preprocessing import PREPROCESSING_FNS
from model import MLP


# ============================================================
# DEFAULTS
# ============================================================

VSR_MODEL = "iulik-pisik/ro_vsr_150h_auto"           # fixed
MLP_REPO = "iulik-pisik/ro_vsr_classification_mlps"  # fixed
VTP_PATH = os.path.join(REPO_ROOT, "checkpoints/feature_extractor.pth")  # fixed

DEFAULT_STRATEGY = "48_bottom"
DEFAULT_SPLIT = "lab"

# MLP hyperparameters (must match training; see notebook Cell 26)
HIDDEN_DIM = 512
DROPOUT = 0.6


# ============================================================
# LRRo VOCABULARIES
# ============================================================
# These match the splits in the LRRo dataset (Jitaru et al., 2020).
# LAB has 48 word classes, WILD has 21.
#
# We list them alphabetically (matching `sorted(...)` from training)
# so that class index → word lookup matches the trained MLP exactly.

LRRO_LAB_WORDS = sorted([
    # Will be populated automatically from the MLP head if available;
    # see resolve_class_names() below.
])

LRRO_WILD_WORDS = sorted([])


def auto_detect_class_map(clip_dir: str, split: str) -> Optional[list]:
    """
    Try to detect the class-to-index mapping from the LRRo dataset
    structure, given a clip path.

    Expected LRRo layout:
        <lrro_root>/<DatasetName>/(train|val|test)/<word>/<clip_id>/*.jpg

    where <DatasetName> is 'Lab_LRRo_data_set' or 'Wild_LRRo_data_set'.

    Returns the alphabetically sorted list of word folders found in
    `train/` (matches the `sorted(...)` ordering used during training).
    Returns None if the structure does not match.
    """
    # Walk up: clip_id -> word -> split (train/val/test) -> dataset
    # clip_dir might end with a trailing slash, normalize first
    clip_dir = os.path.normpath(clip_dir)
    parts = clip_dir.split(os.sep)
    if len(parts) < 4:
        return None

    # The dataset folder is the one whose name contains 'LRRo_data_set'
    dataset_idx = None
    for i, p in enumerate(parts):
        if "LRRo_data_set" in p and (
            ("Lab" in p and split == "lab") or ("Wild" in p and split == "wild")
        ):
            dataset_idx = i
            break
    if dataset_idx is None:
        return None

    dataset_dir = os.sep.join(parts[: dataset_idx + 1])
    train_dir = os.path.join(dataset_dir, "train")
    if not os.path.isdir(train_dir):
        return None

    words = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    return words if words else None


def resolve_class_names(num_classes: int, split: str, clip_dir: str = None,
                       class_map_file: str = None) -> tuple:
    """
    Returns (class_names, source_description).

    Resolution order:
      1. --class_map JSON file if provided
      2. Auto-detect from LRRo folder structure (if clip_dir is in an LRRo tree)
      3. Fall back to generic 'class_<i>' labels

    The number of classes from the MLP is used to validate the resolved
    list — if it doesn't match, we fall back to generic labels.
    """
    # 1. Explicit mapping file
    if class_map_file and os.path.isfile(class_map_file):
        with open(class_map_file) as f:
            mapping = json.load(f)
        if all(str(k).isdigit() for k in mapping.keys()):
            names = [mapping[str(i)] for i in range(num_classes)]
        else:
            names = sorted(mapping.keys(), key=lambda w: mapping[w])
        if len(names) == num_classes:
            return names, f"loaded from {class_map_file}"

    # 2. Auto-detect from LRRo folder structure
    if clip_dir is not None:
        names = auto_detect_class_map(clip_dir, split)
        if names and len(names) == num_classes:
            return names, "auto-detected from LRRo folder structure"
        elif names:
            print(f"[warn] auto-detected {len(names)} classes from folder "
                  f"structure but MLP has {num_classes} — falling back")

    # 3. Generic labels
    return [f"class_{i}" for i in range(num_classes)], None


# ============================================================
# MODEL LOADING
# ============================================================

def load_encoder_models(device: torch.device):
    """Load the (frozen) VTP visual encoder and the VSR transformer encoder."""
    if not os.path.isfile(VTP_PATH):
        raise FileNotFoundError(
            f"VTP checkpoint not found: {VTP_PATH}\n"
            f"   Run: bash scripts/setup.sh (from the repository root)"
        )

    print(f"[load] Loading VTP visual encoder from {VTP_PATH}")
    visual_encoder = build_visual_encoder().to(device).eval()
    s = torch.load(VTP_PATH, map_location=device)["state_dict"]
    new_s = {
        k.replace("module.face_encoder.", ""): v
        for k, v in s.items() if "face_encoder" in k
    }
    visual_encoder.load_state_dict(new_s)
    for p in visual_encoder.parameters():
        p.requires_grad = False

    print(f"[load] Downloading VSR encoder from {VSR_MODEL}")
    enc_path = hf_hub_download(
        repo_id=VSR_MODEL, filename="checkpoints/best_model.pt", repo_type="model",
    )
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(enc_path, map_location=device))
    for p in model.parameters():
        p.requires_grad = False

    return visual_encoder, model


def load_mlp(strategy: str, split: str, emb_dim: int, device: torch.device):
    """
    Load the MLP head from HuggingFace. The MLP repo is structured as:
        <strategy>/best_lab_clf.pt
        <strategy>/best_wild_clf.pt

    We need to know the number of classes ahead of time to build the
    MLP. We infer it from the saved state_dict.
    """
    filename = f"{strategy}/best_{split}_clf.pt"
    print(f"[load] Downloading MLP from {MLP_REPO}/{filename}")
    mlp_path = hf_hub_download(
        repo_id=MLP_REPO, filename=filename, repo_type="model",
    )
    state_dict = torch.load(mlp_path, map_location=device)

    # The final classification layer is the last entry in `net`. Its
    # output dimension equals the number of classes.
    final_layer_key = next(
        k for k in reversed(list(state_dict.keys())) if k.endswith(".weight")
    )
    num_classes = state_dict[final_layer_key].shape[0]
    print(f"[load] MLP has {num_classes} output classes")

    clf = MLP(
        input_dim=emb_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device).eval()
    clf.load_state_dict(state_dict)
    return clf, num_classes


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    clip_dir: str,
    strategy: str,
    visual_encoder,
    encoder_model,
    classifier,
    device: torch.device,
    top_k: int = 5,
):
    """Returns a list of (class_idx, probability) tuples sorted by probability."""
    load_fn = PREPROCESSING_FNS[strategy]
    video = load_fn(clip_dir)
    if video is None:
        raise ValueError(f"No frames found in {clip_dir}")

    video_t = torch.FloatTensor(video).unsqueeze(0).to(device)  # (1, 3, T, 96, 96)
    print(f"[video] Frames extracted: {tuple(video_t.shape)}")

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            vid_emb = visual_encoder(video_t)
            B, T, _ = vid_emb.size()
            src_mask_enc = torch.ones((B, 1, T), device=device).bool()
            memory, _ = encoder_model.encode(vid_emb, src_mask_enc)
            # memory: (1, T_enc, D)

            # MLP expects (B, T, D) and a (B, 1, T) mask
            T_enc = memory.size(1)
            mlp_mask = torch.ones((1, 1, T_enc), dtype=torch.bool, device=device)
            logits, _ = classifier(memory, mlp_mask)
            probs = torch.softmax(logits.float(), dim=-1).squeeze(0).cpu()

    top_probs, top_idx = probs.topk(min(top_k, probs.size(0)))
    return [(int(i), float(p)) for i, p in zip(top_idx.tolist(), top_probs.tolist())]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LRRo word classification on a single clip",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clip_dir", type=str, required=True,
        help="Folder containing the clip's .jpg frames (numbered 0.jpg, 1.jpg, ...)",
    )
    parser.add_argument(
        "--strategy", type=str, default=DEFAULT_STRATEGY,
        choices=list(PREPROCESSING_FNS.keys()),
        help="Preprocessing strategy: how to place LRRo frames on the 96x96 canvas",
    )
    parser.add_argument(
        "--split", type=str, default=DEFAULT_SPLIT, choices=["lab", "wild"],
        help="Which MLP head to use (lab: 48 classes, wild: 21 classes)",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top predictions to display",
    )
    parser.add_argument(
        "--class_map", type=str, default=None,
        help="Optional path to a JSON file mapping class indices to words "
             "(e.g., {\"0\": \"buna\", \"1\": \"ziua\", ...}). If absent, "
             "predictions are shown as 'class_<i>'.",
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

    if not os.path.isdir(args.clip_dir):
        print(f"❌ Clip folder not found: {args.clip_dir}")
        sys.exit(1)

    # Load models
    visual_encoder, encoder_model = load_encoder_models(device)

    # We need EMB_DIM to build the MLP — get it via a dummy forward pass
    dummy_input = torch.zeros((1, 3, 25, 96, 96), device=device)
    with torch.no_grad():
        dummy_emb = visual_encoder(dummy_input)
        B, T, _ = dummy_emb.size()
        dummy_mask = torch.ones((B, 1, T), device=device).bool()
        dummy_mem, _ = encoder_model.encode(dummy_emb, dummy_mask)
    emb_dim = dummy_mem.size(-1)

    classifier, num_classes = load_mlp(args.strategy, args.split, emb_dim, device)

    # Detect mismatch between clip dataset and selected split (warning shown
    # AFTER inference so it's not lost in the download logs)
    split_mismatch = None
    if "Lab_LRRo" in args.clip_dir and args.split != "lab":
        split_mismatch = "lab"
    elif "Wild_LRRo" in args.clip_dir and args.split != "wild":
        split_mismatch = "wild"

    # Resolve class names (explicit map → auto-detect → generic fallback)
    class_names, source = resolve_class_names(
        num_classes=num_classes,
        split=args.split,
        clip_dir=args.clip_dir,
        class_map_file=args.class_map,
    )
    if source:
        print(f"[load] Class names {source}")

    # Run inference
    print("[infer] Running inference ...")
    predictions = run_inference(
        clip_dir=args.clip_dir,
        strategy=args.strategy,
        visual_encoder=visual_encoder,
        encoder_model=encoder_model,
        classifier=classifier,
        device=device,
        top_k=args.top_k,
    )

    # Output
    print("─" * 70)
    print(f"Clip:            {args.clip_dir}")
    print(f"Strategy:        {args.strategy}")
    print(f"MLP split:       {args.split}  ({num_classes} classes)")

    # If clip is in a folder named after its true word, show that
    true_word = None
    parts = os.path.normpath(args.clip_dir).split(os.sep)
    if len(parts) >= 2:
        candidate = parts[-2]  # parent folder name
        if candidate in class_names:
            true_word = candidate
            print(f"True label:      {true_word}")

    print(f"Top-{args.top_k} predictions:")
    for rank, (idx, prob) in enumerate(predictions, start=1):
        word = class_names[idx]
        marker = "  ←" if word == true_word else ""
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {rank}. {word:<20s} {bar} {prob*100:5.2f}%{marker}")
    print("─" * 70)

    # Surface mismatches at the end where they're easy to spot
    if split_mismatch is not None:
        dataset_name = "Lab_LRRo_data_set" if split_mismatch == "lab" else "Wild_LRRo_data_set"
        print()
        print(f"⚠️  WARNING: Clip is in {dataset_name} but you used --split={args.split}.")
        print(f"   The {args.split} MLP has different vocabulary than the {split_mismatch} MLP.")
        print(f"   Re-run with --split {split_mismatch} for meaningful predictions.")


if __name__ == "__main__":
    main()
