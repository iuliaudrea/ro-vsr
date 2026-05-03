"""
Cut MP4 clips from downloaded YouTube videos, using a timestamps CSV
from the VSRo-200 HuggingFace dataset.

`file_path` is `<youtube_id>/<5-digit-index>` and is used both to
locate the source video (videos/<youtube_id>.mp4) and to lay out the
output clips (output_dir/<youtube_id>/<index>.mp4).

Example usage:
    python cut_clips.py \\
        --csv test_seen.csv \\
        --videos_dir videos/ \\
        --output_dir clips/test_seen/

Uses ffmpeg with hybrid seek (5-second pre-roll buffer + ultrafast
x264) for frame-accurate cuts without black frames at clip boundaries.
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict

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
        description="Cut clips from YouTube videos using a VSRo-200 timestamps CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to the timestamps CSV (e.g. test_seen.csv from HuggingFace)",
    )
    parser.add_argument(
        "--videos_dir", type=str, required=True,
        help="Folder with downloaded source videos (videos/<youtube_id>.mp4)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Folder where clips will be saved (organized as <youtube_id>/<index>.mp4)",
    )
    parser.add_argument(
        "--skip_missing", action="store_true",
        help="Skip clips whose source video is missing instead of erroring out",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"❌ CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.videos_dir):
        print(f"❌ Videos folder not found: {args.videos_dir}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    required_cols = {"file_path", "start", "end"}
    if not required_cols.issubset(df.columns):
        print(f"❌ CSV must contain columns: {required_cols}", file=sys.stderr)
        print(f"   Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"✂️  Cutting {len(df)} clips from {len(df['file_path'].apply(lambda p: p.split('/')[0]).unique())} source videos")

    # Stats per video to give a useful summary
    n_success = defaultdict(int)
    n_failed = defaultdict(int)
    n_missing = defaultdict(int)

    for _, row in df.iterrows():
        file_path = str(row["file_path"]).strip()
        start = float(row["start"])
        end = float(row["end"])

        try:
            youtube_id, clip_index = file_path.split("/")
        except ValueError:
            print(f"   ⚠️  Skipping malformed file_path: {file_path}")
            continue

        # Resolve source video
        source_video = os.path.join(args.videos_dir, f"{youtube_id}.mp4")
        if not os.path.isfile(source_video):
            n_missing[youtube_id] += 1
            if args.skip_missing:
                continue
            else:
                print(f"   ❌ Source video missing: {source_video}")
                print(f"      Re-run download_videos.py or use --skip_missing")
                sys.exit(1)

        # Output: <output_dir>/<youtube_id>/<clip_index>.mp4
        output_subdir = os.path.join(args.output_dir, youtube_id)
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, f"{clip_index}.mp4")

        try:
            cut_clip(source_video, start, end, output_path)
            n_success[youtube_id] += 1
        except subprocess.CalledProcessError as e:
            print(f"   ❌ {file_path}: {e}")
            n_failed[youtube_id] += 1

    # Summary
    total_success = sum(n_success.values())
    total_failed = sum(n_failed.values())
    total_missing = sum(n_missing.values())
    print(f"\n🏁 Done. Cut {total_success}/{len(df)} clips into {args.output_dir}/")
    if total_failed:
        print(f"   ⚠️  {total_failed} clips failed during ffmpeg cutting")
    if total_missing:
        n_missing_videos = len(n_missing)
        print(f"   ⚠️  {total_missing} clips skipped because their source "
              f"video was missing ({n_missing_videos} unique YouTube IDs)")


if __name__ == "__main__":
    main()
