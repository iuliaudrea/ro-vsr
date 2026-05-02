"""
Download a single YouTube video by ID, using yt-dlp.

Example usage:
    python download_video.py --video_id dQw4w9WgXcQ --output_dir videos/

To download every podcast in the dataset, loop over `podcast_ids.txt`:
    while IFS= read -r id; do
        [[ "$id" =~ ^#.*$ || -z "$id" ]] && continue
        python download_video.py --video_id "$id" --output_dir videos/
    done < podcast_ids.txt

Requires:
    pip install yt-dlp
"""

import argparse
import os
import subprocess
import sys


def download_video(video_id: str, output_dir: str, format_str: str):
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", format_str,
        "-o", output_template,
        url,
    ]
    print(f"[download] {video_id} -> {output_dir}/")
    try:
        subprocess.run(cmd, check=True)
        print(f"[download] ✅ {video_id}")
    except subprocess.CalledProcessError as e:
        print(f"[download] ❌ Failed for {video_id}: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("[download] ❌ yt-dlp not found. Install it with: pip install yt-dlp",
              file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download a single YouTube video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_id", type=str, required=True,
        help="YouTube video ID (the part after v= in the URL)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="videos",
        help="Folder where the video will be saved",
    )
    parser.add_argument(
        "--format", type=str,
        default="bestvideo[height<=720]+bestaudio/best",
        help="yt-dlp format string (default: best up to 720p)",
    )
    args = parser.parse_args()

    download_video(args.video_id, args.output_dir, args.format)


if __name__ == "__main__":
    main()
