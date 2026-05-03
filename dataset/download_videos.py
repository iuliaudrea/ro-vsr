"""
Download all YouTube videos listed in a file (one ID per line).

Used to fetch the source videos for VSRo-200. Videos are saved with
their YouTube ID as the filename, e.g. videos/ZWNM2sZxtRg.mp4.

Example usage:
    python download_videos.py --ids youtube_ids.txt --output_dir videos/

To download just a subset (e.g., the IDs that appear in a specific split):
    awk -F/ '{print $1}' test_seen.csv | sort -u | tail -n +2 > test_seen_ids.txt
    python download_videos.py --ids test_seen_ids.txt --output_dir videos/

Requires:
    pip install yt-dlp
"""

import argparse
import os
import subprocess
import sys


def read_ids(ids_file: str):
    """Read YouTube IDs from a file (one per line, ignore '#' comments)."""
    ids = []
    with open(ids_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def download_one(video_id: str, output_dir: str, format_str: str) -> bool:
    """Download a single video. Returns True on success, False on failure."""
    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    # Skip if already downloaded
    if os.path.isfile(output_path):
        print(f"   ⏭️  {video_id}.mp4 already exists, skipping")
        return True

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "-f", format_str,
        "--merge-output-format", "mp4",
        "-o", os.path.join(output_dir, "%(id)s.%(ext)s"),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"   ✅ {video_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {video_id} failed: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube videos for VSRo-200 reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ids", type=str, default="youtube_ids.txt",
        help="Text file with one YouTube ID per line",
    )
    parser.add_argument(
        "--output_dir", type=str, default="videos",
        help="Folder where videos are saved",
    )
    parser.add_argument(
        "--format", type=str,
        default="bestvideo[height<=720]+bestaudio/best",
        help="yt-dlp format string (default: best up to 720p)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.ids):
        print(f"❌ IDs file not found: {args.ids}", file=sys.stderr)
        sys.exit(1)

    ids = read_ids(args.ids)
    print(f"Downloading {len(ids)} videos to {args.output_dir}/")
    os.makedirs(args.output_dir, exist_ok=True)

    n_success = 0
    n_failed = 0
    failed_ids = []

    for video_id in ids:
        if download_one(video_id, args.output_dir, args.format):
            n_success += 1
        else:
            n_failed += 1
            failed_ids.append(video_id)

    print(f"\n🏁 Done. {n_success}/{len(ids)} videos downloaded.")
    if failed_ids:
        print(f"   ⚠️  {n_failed} failed (likely removed from YouTube):")
        for vid in failed_ids:
            print(f"      - {vid}")


if __name__ == "__main__":
    main()
