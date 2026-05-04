"""
Preprocessing pipeline for raw Romanian podcast videos: produces a CSV
of (start, end) timestamps marking guest-only speaking intervals.

The pipeline runs three filtering steps:

    Raw podcast video
       │
       ▼  Step 1: PySceneDetect — detect scene boundaries
    Scenes
       │
       ▼  Step 2: InsightFace — keep only scenes with a single face
       │           that is NOT the host (guest-only visual)
    Guest-only scenes
       │
       ▼  Step 3: pyannote — diarize audio, keep only segments where
       │           the guest speaks alone (no host overlap)
    Guest-speaking timestamps -> CSV

The resulting CSV contains start/end timestamps; use `cut_clips.py` to
actually extract the .mp4 clips from these timestamps.

Example usage:
    export HF_TOKEN=hf_xxxxxxxxxx

    python preprocess_pipeline.py \\
        --video_path videos/podcast_morar.mp4 \\
        --host_embeddings host_embeddings/ \\
        --output_csv timestamps/podcast_morar.csv
"""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchaudio
from insightface.app import FaceAnalysis
from pyannote.audio import Pipeline
from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import ContentDetector


# ============================================================
# STEP 1: SCENE DETECTION
# ============================================================

def extract_scenes(video_path: str, threshold: float = 27.0):
    """Detect scene boundaries with PySceneDetect."""
    print(f"\n [STEP 1] Detecting scenes in {os.path.basename(video_path)}")

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scenes = []
    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        for i, scene in enumerate(scene_list):
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            scenes.append({
                "scene_id": i + 1,
                "start": start_sec,
                "end": end_sec,
                "duration": end_sec - start_sec,
            })

        print(f"        ✅ {len(scenes)} scenes found")
    finally:
        video_manager.release()

    return scenes


# ============================================================
# STEP 2: FACE FILTERING (keep only guest scenes)
# ============================================================

def filter_guest_scenes(
    video_path: str,
    all_scenes: list,
    host_embeddings_dir: str,
    similarity_threshold: float = 0.5,
):
    """
    Keep scenes where exactly one face appears AND it does NOT match
    any of the known host embeddings.
    """
    print(f"\n🔍 [STEP 2] Filtering guest scenes ({len(all_scenes)} candidates)")

    if not os.path.isdir(host_embeddings_dir):
        raise FileNotFoundError(f"Host embeddings folder not found: {host_embeddings_dir}")

    host_files = [f for f in os.listdir(host_embeddings_dir) if f.endswith(".npy")]
    if not host_files:
        raise ValueError(f"No .npy files found in {host_embeddings_dir}")

    known_host_embeddings = [
        np.load(os.path.join(host_embeddings_dir, f)) for f in host_files
    ]
    print(f"        Loaded {len(known_host_embeddings)} host embeddings")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    guest_scenes = []
    for scene in all_scenes:
        midpoint_sec = (scene["start"] + scene["end"]) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, midpoint_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        faces = app.get(frame)
        if len(faces) != 1:
            continue  # zero or multiple faces -> skip

        face_emb = faces[0].embedding
        norm_face = np.linalg.norm(face_emb)

        # Cosine similarity vs each host
        is_host = False
        for host_emb in known_host_embeddings:
            sim = np.dot(host_emb, face_emb) / (np.linalg.norm(host_emb) * norm_face)
            if sim > similarity_threshold:
                is_host = True
                break

        if not is_host:
            guest_scenes.append(scene)

    cap.release()
    print(f"         Kept {len(guest_scenes)} / {len(all_scenes)} scenes (guest only)")
    return guest_scenes


# ============================================================
# STEP 3: AUDIO DIARIZATION (keep only guest-speaking segments)
# ============================================================

def prepare_audio_track(video_path: str) -> str:
    """Extract mono 16kHz WAV audio (pyannote's preferred format)."""
    base = os.path.splitext(video_path)[0]
    audio_path = f"{base}_temp_audio.wav"

    if os.path.exists(audio_path):
        return audio_path

    print(f"         Extracting audio (16kHz mono) ...")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        audio_path, "-y",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def merge_segments_with_history(raw_segments: list, max_pause: float = 1.0):
    """Merge consecutive segments by the same speaker if pause < max_pause.
    Stores 'safe_cuts' = the merge points so we can later split cleanly."""
    if not raw_segments:
        return []

    sorted_segments = sorted(raw_segments, key=lambda x: x["start"])
    merged = []
    current = copy.deepcopy(sorted_segments[0])
    current["safe_cuts"] = []

    for nxt in sorted_segments[1:]:
        pause = nxt["start"] - current["end"]
        if nxt["speaker"] == current["speaker"] and pause <= max_pause:
            current["safe_cuts"].append(current["end"])
            current["end"] = nxt["end"]
        else:
            merged.append(current)
            current = copy.deepcopy(nxt)
            current["safe_cuts"] = []

    merged.append(current)
    return merged


def atomic_decomposition(merged_segments: list):
    """Break the timeline into atomic intervals where exactly one speaker
    is active (drops overlapped speech)."""
    boundaries = set()
    for seg in merged_segments:
        boundaries.add(seg["start"])
        boundaries.add(seg["end"])

    sorted_boundaries = sorted(boundaries)
    atomic_intervals = []

    for i in range(len(sorted_boundaries) - 1):
        start = sorted_boundaries[i]
        end = sorted_boundaries[i + 1]
        if end <= start:
            continue

        active = [s for s in merged_segments if s["start"] <= start and s["end"] >= end]
        if len(active) == 1:
            parent = active[0]
            relevant_cuts = [c for c in parent["safe_cuts"] if start < c < end]
            atomic_intervals.append({
                "start": start, "end": end,
                "speaker": parent["speaker"],
                "safe_cuts": relevant_cuts,
            })

    return atomic_intervals


def split_long_clips(intervals: list, max_duration: float = 30.0):
    """Split intervals longer than `max_duration` seconds at safe_cuts."""
    result = []
    for interval in intervals:
        duration = interval["end"] - interval["start"]
        if duration <= max_duration:
            result.append(interval)
            continue

        cuts = sorted(interval["safe_cuts"])
        last_start = interval["start"]
        for cut in cuts:
            if cut - last_start > max_duration:
                continue
            result.append({
                "start": last_start,
                "end": cut,
                "speaker": interval["speaker"],
            })
            last_start = cut
        if interval["end"] - last_start <= max_duration:
            result.append({
                "start": last_start,
                "end": interval["end"],
                "speaker": interval["speaker"],
            })
    return result


def filter_min_duration(intervals: list, min_duration: float = 3.0):
    """Drop intervals shorter than `min_duration`."""
    return [i for i in intervals if (i["end"] - i["start"]) >= min_duration]


def process_pyannote_output(raw_segments: list):
    """Glue: merge -> decompose -> split long -> filter short."""
    merged = merge_segments_with_history(raw_segments, max_pause=1.0)
    atomic = atomic_decomposition(merged)
    split = split_long_clips(atomic, max_duration=30.0)
    filtered = filter_min_duration(split, min_duration=3.0)
    return filtered


def analyze_audio_segments(
    video_path: str,
    guest_scenes: list,
    hf_token: str,
):
    """Run pyannote diarization within each guest scene, keep only
    intervals where the guest speaks alone."""
    print(f"\n  [STEP 3] Audio diarization on {len(guest_scenes)} guest scenes")

    audio_path = prepare_audio_track(video_path)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    metadata = torchaudio.info(audio_path)
    sample_rate = metadata.sample_rate

    final_global_clips = []
    SAFETY_PAD = 0.05

    for scene in guest_scenes:
        scene_start = scene["start"]
        scene_end = scene["end"]

        frame_start = int(scene_start * sample_rate)
        num_frames = int((scene_end - scene_start) * sample_rate)

        try:
            waveform, sr = torchaudio.load(
                audio_path, frame_offset=frame_start, num_frames=num_frames,
            )
        except Exception as e:
            print(f"          Audio read error in scene {scene['scene_id']}: {e}")
            continue

        try:
            diarization = pipeline({"waveform": waveform, "sample_rate": sr})
        except Exception as e:
            print(f"          Pyannote error in scene {scene['scene_id']}: {e}")
            continue

        raw_segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, speaker in diarization.speaker_diarization
        ]
        if not raw_segments:
            continue

        local_clips = process_pyannote_output(raw_segments)

        for clip in local_clips:
            calculated_start = clip["start"] + scene_start
            calculated_end = clip["end"] + scene_start

            # Clamp + erode by SAFETY_PAD to avoid bleeding into adjacent scenes
            safe_start = max(calculated_start, scene_start + SAFETY_PAD)
            safe_end = min(calculated_end, scene_end - SAFETY_PAD)
            if safe_start >= safe_end:
                continue

            final_global_clips.append({
                "start": safe_start,
                "end": safe_end,
                "speaker": clip["speaker"],
            })

    print(f"         {len(final_global_clips)} guest-speaking intervals kept")
    return final_global_clips


# ============================================================
# CSV OUTPUT
# ============================================================

def save_timestamps_csv(clips: list, output_csv: str, video_path: str):
    """Save guest-speaking timestamps to CSV (compatible with cut_clips.py)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    base = Path(video_path).stem
    rows = []
    for i, clip in enumerate(clips, start=1):
        rows.append({
            "clip_id": f"{base}_clip_{i}",
            "start": round(clip["start"], 3),
            "end": round(clip["end"], 3),
            "duration": round(clip["end"] - clip["start"], 3),
            "speaker": clip.get("speaker", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n Saved {len(rows)} timestamps to {output_csv}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a podcast video, output a CSV of guest-only timestamps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_path", type=str, required=True,
        help="Path to the input podcast video (.mp4)",
    )
    parser.add_argument(
        "--host_embeddings", type=str, required=True,
        help="Folder containing .npy embeddings of the podcast's hosts",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Output CSV file with guest-speaking timestamps",
    )
    parser.add_argument(
        "--scene_threshold", type=float, default=27.0,
        help="PySceneDetect threshold (lower = more scene cuts)",
    )
    parser.add_argument(
        "--similarity_threshold", type=float, default=0.5,
        help="Cosine similarity above which a face is matched as host",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("   ERROR: HF_TOKEN environment variable is required for pyannote.")
        print("   Get a token at https://huggingface.co/settings/tokens")
        print("   Then run: export HF_TOKEN=hf_xxx")
        sys.exit(1)

    scenes = extract_scenes(args.video_path, threshold=args.scene_threshold)
    guest_scenes = filter_guest_scenes(
        args.video_path, scenes, args.host_embeddings,
        similarity_threshold=args.similarity_threshold,
    )
    final_clips = analyze_audio_segments(args.video_path, guest_scenes, hf_token)
    save_timestamps_csv(final_clips, args.output_csv, args.video_path)

    print(f"\n  Pipeline complete.")
    print(f"   Next: python cut_clips.py --video_path {args.video_path} \\")
    print(f"             --timestamps_csv {args.output_csv} --output_dir clips/")


if __name__ == "__main__":
    main()
