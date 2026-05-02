"""
Cut MP4 clips from a video, given a CSV of (start, end) timestamps.

Uses ffmpeg with hybrid seeking (fast input seek + 5-second pre-roll
buffer) for frame-accurate cuts without black frames at the start.

Example usage:
    python cut_clips.py \\
        --video_path videos/podcast_morar.mp4 \\
        --timestamps_csv timestamps/podcast_morar.csv \\
        --output_dir clips/podcast_morar/

The input CSV must have at least these columns:
    - clip_id (used as the output filename, .mp4 extension is added)
    - start   (start time in seconds)
    - end     (end time in seconds)

This is the format produced by `preprocess_pipeline.py`.
"""

import argparse
import os
import subprocess
import sys

import pandas as pd


PRE_ROLL = 5.0  # seconds of decoder warmup buffer for clean cuts


def cut_clip(video_path: str, start: float, end: float, output_path: str):
    """Cut a single clip with ffmpeg hybrid seek + ultrafast x264."""
    duration = end - start

    # Hybrid seek: coarse fast seek (input -ss) + fine slow seek (output -ss)
    if start > PRE_ROLL:
        coarse_seek = start - PRE_ROLL
        fine_seek = PRE_ROLL
    else:
        coarse_seek = 0.0
        fine_seek = start

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(coarse_seek),
        "-i", video_path,
        "-ss", str(fine_seek),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-tune", "zerolatency",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Cut MP4 clips from a video using a CSV of timestamps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_path", type=str, required=True,
        help="Path to the source video (.mp4)",
    )
    parser.add_argument(
        "--timestamps_csv", type=str, required=True,
        help="CSV with columns: clip_id, start, end",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Folder where clips will be saved",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"❌ Video not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.timestamps_csv):
        print(f"❌ Timestamps CSV not found: {args.timestamps_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.timestamps_csv)
    required_cols = {"clip_id", "start", "end"}
    if not required_cols.issubset(df.columns):
        print(f"❌ CSV must contain columns: {required_cols}", file=sys.stderr)
        print(f"   Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✂️  Cutting {len(df)} clips from {os.path.basename(args.video_path)}")

    n_success = 0
    for _, row in df.iterrows():
        clip_id = str(row["clip_id"])
        start = float(row["start"])
        end = float(row["end"])
        output_path = os.path.join(args.output_dir, f"{clip_id}.mp4")

        try:
            cut_clip(args.video_path, start, end, output_path)
            n_success += 1
            print(f"   ✅ {clip_id}.mp4  ({start:.2f}s -> {end:.2f}s)")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ {clip_id}: {e}")

    print(f"\n🏁 Cut {n_success}/{len(df)} clips into {args.output_dir}/")


if __name__ == "__main__":
    main()
