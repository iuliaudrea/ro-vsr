"""
Single-clip AVSR (Audio-Visual Speech Recognition) inference.

Combines our VSR model (video) with a fine-tuned Whisper model (audio)
via shallow fusion at the log-probability level.

The VSR model uses the MultiVSR architecture (Prajwal et al., 2025),
trained on our VSRo-200 dataset.

Example usage:
    python inference_avsr.py --fpath samples_avsr/sample_1_babble_SNR0.mp4
    python inference_avsr.py --fpath samples_avsr/sample_1_babble_SNR0.mp4 --mode whisper
    python inference_avsr.py --fpath samples_avsr/sample_1_babble_SNR0.mp4 --mode multivsr

Input clips must be MP4 files with:
  - 160x160 video (face crop)
  - audio track (any sample rate; resampled to 16 kHz internally)

For demo purposes, our `samples_avsr/` clips have noise pre-mixed into
the audio track. To evaluate on your own clips, simply provide an MP4
with the same video format.
"""

import argparse
import os
import re
import sys
import warnings

# Suppress autocast warnings on CPU (decorators in models.py are static)
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Add the repository root to path so we can import from ro_vsr/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from vsr_inference.models import build_model, build_visual_encoder
from vsr_inference.tokenizer import get_tokenizer
from vsr_inference.metrics import lookup_reference, print_metrics_block

from beam_search_fusion import beam_search_fusion


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_VSR_MODEL_REPO = "vsro200/models-vsro200"
DEFAULT_VSR_MODEL_FILENAME = "model_200h_auto.pt"
DEFAULT_WHISPER_MODEL = "vsro200/whisper-small-vsro200"
DEFAULT_VTP_PATH = os.path.join(REPO_ROOT, "checkpoints/feature_extractor.pth")
DEFAULT_METADATA = os.path.join(
    os.path.dirname(__file__), "samples_avsr", "samples_avsr_metadata.csv"
)


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
    return " ".join(text.split()).strip()


# ============================================================
# I/O
# ============================================================

def read_video(fpath: str, device: torch.device) -> torch.Tensor:
    """Read a 160x160 .mp4/.avi clip → tensor of shape [1, 3, T, 96, 96]."""
    from decord import VideoReader, bridge
    bridge.set_bridge("native")

    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Video file not found: {fpath}")

    with open(fpath, "rb") as f:
        vr = VideoReader(f, width=160, height=160)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()

    frames = frames.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).to(device).unsqueeze(0)
    frames = frames.permute(0, 4, 1, 2, 3)  # [1, C, T, H, W]

    crop_x = (frames.size(3) - 96) // 2
    crop_y = (frames.size(4) - 96) // 2
    return frames[:, :, :, crop_x:crop_x + 96, crop_y:crop_y + 96]


def read_audio(fpath: str, target_sr: int = 16000) -> torch.Tensor:
    """Read audio track from a video file → mono 16 kHz tensor of shape [T]."""
    waveform, sr = torchaudio.load(fpath)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    return waveform


# ============================================================
# MODEL LOADING
# ============================================================

def load_vsr_models(vsr_filename: str, vtp_path: str, device: torch.device):
    """Load the VSR encoder-decoder and the VTP visual encoder."""
    if not os.path.isfile(vtp_path):
        raise FileNotFoundError(
            f"VTP checkpoint not found: {vtp_path}\n"
            f"   Run: bash scripts/setup.sh (from the repository root)"
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

    print(f"[load] Downloading VSR model from {DEFAULT_VSR_MODEL_REPO}/checkpoints/{vsr_filename} ...")
    lm_path = hf_hub_download(
        repo_id=DEFAULT_VSR_MODEL_REPO,
        filename=f"checkpoints/{vsr_filename}",
        repo_type="model",
    )
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(lm_path, map_location=device))

    return model, visual_encoder


def load_whisper(whisper_repo: str, device: torch.device):
    """Load a fine-tuned Whisper model + its processor."""
    print(f"[load] Downloading Whisper from {whisper_repo} ...")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_repo).to(device)
    whisper_model.eval()
    whisper_processor = WhisperProcessor.from_pretrained(whisper_repo)
    return whisper_model, whisper_processor


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    video: torch.Tensor,        # [1, 3, T, 96, 96]
    audio: torch.Tensor,        # [T_audio]
    vsr_model,
    visual_encoder,
    whisper_model,
    whisper_processor,
    tokenizer,
    device: torch.device,
    mode: str = "hibrid_logp",
    beam_size: int = 5,
    max_len: int = 256,
) -> str:
    """Run a single AVSR forward pass and return the cleaned transcription."""
    start_prompt_ids = tokenizer.encode(
        "<|startoftranscript|><|ro|><|transcribe|><|notimestamps|>"
    )

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            # Audio path: Whisper encoder
            mel = whisper_processor(
                audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            audio_embeds = whisper_model.model.encoder(mel).last_hidden_state

            # Video path: VTP + VSR encoder
            vid_emb = visual_encoder(video)
            B, T, _ = vid_emb.size()
            src_mask = torch.ones((B, 1, T), device=device).bool()
            video_memory, _ = vsr_model.encode(vid_emb, src_mask)

            # Beam search with shallow fusion
            beam_outs, _ = beam_search_fusion(
                whisper_model=whisper_model.model,
                whisper_proj=whisper_model.proj_out,
                audio_embeds=audio_embeds,
                multivsr_model=vsr_model,
                video_memory=video_memory,
                src_mask=src_mask,
                bos_indices=start_prompt_ids,
                eos_index=tokenizer.eot,
                pad_index=0,
                max_output_length=max_len,
                size=beam_size,
                n_best=1,
                mode=mode,
            )

    if not beam_outs[0]:
        return ""
    best_ids = beam_outs[0][0]
    if isinstance(best_ids, torch.Tensor):
        best_ids = best_ids.tolist()

    return clean_prediction(tokenizer.decode(best_ids))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-clip AVSR inference with shallow fusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fpath", type=str, required=True,
                        help="Path to the input video file (.mp4 or .avi, 160x160 with audio)")
    parser.add_argument("--vsr_model", type=str, default=DEFAULT_VSR_MODEL_FILENAME,
                    help="Checkpoint filename in vsro200/models-vsro200 (e.g. model_200h_auto.pt)")
    parser.add_argument("--whisper_model", type=str, default=DEFAULT_WHISPER_MODEL,
                        help="HuggingFace repo for the Whisper model")
    parser.add_argument("--vtp_path", type=str, default=DEFAULT_VTP_PATH,
                        help="Path to the VTP feature extractor checkpoint")
    parser.add_argument("--mode", type=str, default="hibrid_logp",
                        choices=["hibrid_logp", "whisper", "multivsr"],
                        help="Decoding mode: shallow fusion, audio-only, or video-only")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA,
                        help="Path to metadata CSV (used to look up reference and compute WER/CER if available)")
    parser.add_argument("--device", type=str, default=None,
                        help="Explicit device (cuda / cpu). Default: auto.")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")

    # Load
    tokenizer = get_tokenizer()
    vsr_model, visual_encoder = load_vsr_models(args.vsr_model, args.vtp_path, device)
    whisper_model, whisper_processor = load_whisper(args.whisper_model, device)
    print(f"[load] ✅ All models loaded successfully")

    # Read
    video = read_video(args.fpath, device)
    audio = read_audio(args.fpath, target_sr=16000)
    print(f"[video] Frames extracted: {tuple(video.shape)}")
    print(f"[audio] Samples loaded:   {audio.shape[0]} ({audio.shape[0]/16000:.2f}s @ 16 kHz)")

    # Inference
    print(f"[infer] Running inference (mode={args.mode}) ...")
    transcription = run_inference(
        video=video,
        audio=audio,
        vsr_model=vsr_model,
        visual_encoder=visual_encoder,
        whisper_model=whisper_model,
        whisper_processor=whisper_processor,
        tokenizer=tokenizer,
        device=device,
        mode=args.mode,
        beam_size=args.beam_size,
        max_len=args.max_len,
    )

    print("─" * 70)
    print(f"File:           {args.fpath}")
    print(f"Mode:           {args.mode}")
    print(f"Transcription:  {transcription}")
    reference = lookup_reference(args.metadata, args.fpath)
    print_metrics_block(reference, transcription, normalize_text)
    print("─" * 70)


if __name__ == "__main__":
    main()
