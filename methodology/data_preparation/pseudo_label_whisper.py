"""
Generate pseudo-labels for face-tracked clips using a Romanian-tuned
Whisper-large model.

Reads .avi files from a directory tree (post-MultiVSR face crops),
extracts the audio track from each, transcribes it with Whisper-large,
and writes a CSV with `file_path,transcript`.

`file_path` is relative to `--clips_dir` and follows the convention
`<youtube_id>/<clip_index>` (no extension), matching the format used in
the VSRo-200 HuggingFace splits.

Example usage:
    python pseudo_label_whisper.py \\
        --clips_dir /path/to/face_tracks/test_seen/pycrop/ \\
        --output_csv pseudo_labels.csv

This was used to produce the `trainval_auto.csv` and `test_*.csv`
splits on HuggingFace.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_MODEL_ID = "alexandradiaconu/whisper-large-all-34-new3"


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio_from_avi(avi_path: str, target_sr: int = 16000) -> str:
    """Extract audio from an AVI to a temporary 16 kHz mono WAV.
    Returns the temp WAV path (caller must delete)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", avi_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(target_sr),
        tmp.name,
    ]
    subprocess.run(cmd, check=True)
    return tmp.name


def load_audio_as_array(wav_path: str):
    """Load a 16 kHz mono WAV into a numpy float32 array in [-1, 1]."""
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000, f"Expected 16 kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


# ============================================================
# WHISPER INFERENCE
# ============================================================

def load_whisper(model_id: str, device: torch.device):
    """Load Whisper-large with the prompt setup the team used during labeling."""
    print(f"[load] Downloading {model_id} ...")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)

    # Build the Romanian transcribe prompt (same prefix used during labeling)
    prompt_ids = [model.config.decoder_start_token_id]
    forced_ids = processor.get_decoder_prompt_ids(language="romanian", task="transcribe")
    prompt_ids.extend([token_id for _, token_id in forced_ids])
    decoder_input_ids = torch.tensor([prompt_ids], device=device)

    return model, processor, decoder_input_ids, torch_dtype


def transcribe_audio(audio, model, processor, decoder_input_ids,
                      torch_dtype, device: torch.device) -> str:
    """Transcribe a single audio array with Whisper."""
    inputs = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).to(device, dtype=torch_dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_features,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=440,
        )

    transcription = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return transcription.strip()


# ============================================================
# MAIN
# ============================================================

def find_avi_files(clips_dir: str):
    """Walk `clips_dir` and yield (relative_file_path, full_path) tuples,
    where relative_file_path is `<youtube_id>/<clip_index>` with no
    extension (matching the VSRo-200 CSV convention)."""
    clips_dir = os.path.abspath(clips_dir)
    for root, _, files in os.walk(clips_dir):
        for fn in sorted(files):
            if not fn.lower().endswith(".avi"):
                continue
            full_path = os.path.join(root, fn)
            rel_path = os.path.relpath(full_path, clips_dir)

            # Strip extension and `00000` prefix from MultiVSR pycrop/ output:
            # pycrop/<yid>/<clip_index>/00000.avi -> <yid>/<clip_index>
            rel_no_ext = os.path.splitext(rel_path)[0]
            if os.path.basename(rel_no_ext) == "00000":
                rel_no_ext = os.path.dirname(rel_no_ext)

            yield rel_no_ext, full_path


def main():
    parser = argparse.ArgumentParser(
        description="Pseudo-label face-cropped clips with Whisper-large.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clips_dir", type=str, required=True,
        help="Folder containing .avi clips (typically pycrop/ from MultiVSR)",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Output CSV with columns: file_path, transcript",
    )
    parser.add_argument(
        "--model_id", type=str, default=DEFAULT_MODEL_ID,
        help="HuggingFace repo ID of the Whisper model",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Explicit device (cuda / cpu). Default: auto.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If output_csv exists, skip clips that already have a transcript",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")

    if not os.path.isdir(args.clips_dir):
        print(f"❌ Clips folder not found: {args.clips_dir}", file=sys.stderr)
        sys.exit(1)

    # Resume support: load existing transcripts
    existing = {}
    if args.resume and os.path.isfile(args.output_csv):
        df_existing = pd.read_csv(args.output_csv)
        existing = dict(zip(df_existing["file_path"], df_existing["transcript"]))
        print(f"[resume] {len(existing)} clips already transcribed")

    # Discover clips
    clips = list(find_avi_files(args.clips_dir))
    print(f"[clips] Found {len(clips)} .avi files in {args.clips_dir}")

    to_process = [(fp, ap) for fp, ap in clips if fp not in existing]
    print(f"[clips] {len(to_process)} need transcription")

    if not to_process:
        print("[done] Nothing to do.")
        return

    # Load model
    model, processor, decoder_input_ids, torch_dtype = load_whisper(
        args.model_id, device
    )

    # Transcribe
    results = dict(existing)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".",
                exist_ok=True)

    for file_path, full_path in tqdm(to_process, desc="Transcribing"):
        wav_path = None
        try:
            wav_path = extract_audio_from_avi(full_path)
            audio = load_audio_as_array(wav_path)
            transcript = transcribe_audio(
                audio, model, processor, decoder_input_ids,
                torch_dtype, device,
            )
            results[file_path] = transcript
        except Exception as e:
            print(f"[warn] Failed on {file_path}: {e}")
            results[file_path] = ""
        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

        # Periodic save (every 50 clips)
        if len(results) % 50 == 0:
            pd.DataFrame(
                [{"file_path": fp, "transcript": t}
                 for fp, t in results.items()]
            ).to_csv(args.output_csv, index=False)

    # Final save
    df = pd.DataFrame(
        [{"file_path": fp, "transcript": t} for fp, t in results.items()]
    )
    df = df.sort_values("file_path").reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\n🏁 Wrote {len(df)} transcripts to {args.output_csv}")


if __name__ == "__main__":
    main()
